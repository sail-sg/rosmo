# Copyright 2022 Garena Online Private Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent learner."""
import time
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple

import acme
import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import optax
import rlax
import tree
from absl import logging
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.utils import loggers

from rosmo.agent.network import Networks
from rosmo.agent.type import AgentOutput, Params
from rosmo.agent.utils import (
    inv_value_transform,
    logits_to_scalar,
    scalar_to_two_hot,
    scale_gradient,
    value_transform,
)
from rosmo.type import ActorOutput, Array


class TrainingState(NamedTuple):
    """Training state."""

    optimizer_state: optax.OptState
    params: Params
    target_params: Params

    step: int


class RosmoLearner(acme.core.Learner):
    """ROSMO learner."""

    def __init__(
        self,
        networks: Networks,
        demonstrations: Iterator[ActorOutput],
        config: Dict,
        logger: Optional[loggers.Logger] = None,
    ) -> None:
        """Init ROSMO learner.

        Args:
            networks (Networks): ROSMO networks.
            demonstrations (Iterator[ActorOutput]): Data loader.
            config (Dict): Configurations.
            logger (Optional[loggers.Logger], optional): Logger. Defaults to None.
        """
        discount_factor = config["discount_factor"]
        weight_decay = config["weight_decay"]
        value_coef = config["value_coef"]
        behavior_coef = config["behavior_coef"]
        policy_coef = config["policy_coef"]
        unroll_steps = config["unroll_steps"]
        td_steps = config["td_steps"]
        target_update_interval = config["target_update_interval"]
        log_interval = config["log_interval"]
        batch_size = config["batch_size"]
        max_grad_norm = config["max_grad_norm"]
        num_bins = config["num_bins"]
        sampling = config.get("sampling", False)
        num_simulations = config.get("num_simulations", -1)

        _batch_categorical_cross_entropy = jax.vmap(rlax.categorical_cross_entropy)

        def loss(
            params: Params,
            target_params: Params,
            trajectory: ActorOutput,
            rng_key: networks_lib.PRNGKey,
        ) -> Tuple[Array, Dict[str, Array]]:
            # Encode obs via learning and target networks, [T, S]
            state = networks.representation_network.apply(
                params.representation, trajectory.observation
            )
            target_state = networks.representation_network.apply(
                target_params.representation, trajectory.observation
            )

            # 1) Model unroll, sampling and estimation.
            root_state = jax.tree_map(lambda t: t[:1], state)
            learner_root = root_unroll(networks, params, num_bins, root_state)
            learner_root: AgentOutput = jax.tree_map(lambda t: t[0], learner_root)

            unroll_trajectory: ActorOutput = jax.tree_map(
                lambda t: t[: unroll_steps + 1], trajectory
            )
            random_action_mask = (
                jnp.cumprod(1.0 - unroll_trajectory.is_first[1:]) == 0.0
            )
            action_sequence = unroll_trajectory.action[:unroll_steps]
            num_actions = learner_root.policy_logits.shape[-1]
            rng_key, action_key = jax.random.split(rng_key)
            random_actions = jax.random.choice(
                action_key, num_actions, action_sequence.shape, replace=True
            )
            simulate_action_sequence = jax.lax.select(
                random_action_mask, random_actions, action_sequence
            )

            model_out: AgentOutput = model_unroll(
                networks,
                params,
                num_bins,
                learner_root.state,
                simulate_action_sequence,
            )

            # Model predictions.
            policy_logits = jnp.concatenate(
                [
                    learner_root.policy_logits[None],
                    model_out.policy_logits,
                ],
                axis=0,
            )

            value_logits = jnp.concatenate(
                [
                    learner_root.value_logits[None],
                    model_out.value_logits,
                ],
                axis=0,
            )

            # 2) Model learning targets.
            # a) Reward.
            rewards = trajectory.reward
            reward_target = jax.lax.select(
                random_action_mask,
                jnp.zeros_like(rewards[:unroll_steps]),
                rewards[:unroll_steps],
            )
            reward_target_transformed = value_transform(reward_target)
            reward_logits_target = scalar_to_two_hot(
                reward_target_transformed, num_bins
            )

            # b) Policy.
            target_roots: AgentOutput = root_unroll(
                networks, target_params, num_bins, target_state
            )
            search_roots: AgentOutput = jax.tree_map(
                lambda t: t[: unroll_steps + 1], target_roots
            )
            rng_key, improve_key = jax.random.split(rng_key)

            improve_keys = jax.random.split(improve_key, search_roots.state.shape[0])
            policy_target, improve_adv = jax.vmap(
                one_step_improve,
                (None, 0, None, 0, None, None, None, None),
            )(
                networks,
                improve_keys,
                target_params,
                search_roots,
                num_bins,
                discount_factor,
                num_simulations,
                sampling,
            )
            uniform_policy = jnp.ones_like(policy_target) / num_actions
            random_policy_mask = jnp.cumprod(1.0 - unroll_trajectory.is_last) == 0.0
            random_policy_mask = jnp.broadcast_to(
                random_policy_mask[:, None], policy_target.shape
            )
            policy_target = jax.lax.select(
                random_policy_mask, uniform_policy, policy_target
            )
            policy_target = jax.lax.stop_gradient(policy_target)

            # c) Value.
            discounts = (1.0 - trajectory.is_last[1:]) * discount_factor
            v_bootstrap = target_roots.value

            def n_step_return(i: int) -> jnp.ndarray:
                bootstrap_value = jax.tree_map(lambda t: t[i + td_steps], v_bootstrap)
                _rewards = jnp.concatenate(
                    [rewards[i : i + td_steps], bootstrap_value[None]], axis=0
                )
                _discounts = jnp.concatenate(
                    [jnp.ones((1,)), jnp.cumprod(discounts[i : i + td_steps])],
                    axis=0,
                )
                return jnp.sum(_rewards * _discounts)

            returns = []
            for i in range(unroll_steps + 1):
                returns.append(n_step_return(i))
            returns = jnp.stack(returns)
            # Value targets for the absorbing state and the states after are 0.
            zero_return_mask = jnp.cumprod(1.0 - unroll_trajectory.is_last) == 0.0
            value_target = jax.lax.select(
                zero_return_mask, jnp.zeros_like(returns), returns
            )
            value_target_transformed = value_transform(value_target)
            value_logits_target = scalar_to_two_hot(value_target_transformed, num_bins)
            value_logits_target = jax.lax.stop_gradient(value_logits_target)

            # 3) Behavior regularization.
            in_sample_action = trajectory.action[: unroll_steps + 1]
            log_prob = jax.nn.log_softmax(policy_logits)
            action_log_prob = log_prob[jnp.arange(unroll_steps + 1), in_sample_action]

            _target_value = target_roots.value[: unroll_steps + 1]
            _target_reward = target_roots.reward[1 : unroll_steps + 1 + 1]
            _target_value_prime = target_roots.value[1 : unroll_steps + 1 + 1]
            _target_adv = (
                _target_reward + discount_factor * _target_value_prime - _target_value
            )
            _target_adv = jax.lax.stop_gradient(_target_adv)
            behavior_loss = -action_log_prob * jnp.heaviside(_target_adv, 0.0)
            # Deal with cross-episode trajectories.
            invalid_action_mask = jnp.cumprod(1.0 - trajectory.is_first[1:]) == 0.0
            behavior_loss = jax.lax.select(
                invalid_action_mask[: unroll_steps + 1],
                jnp.zeros_like(behavior_loss),
                behavior_loss,
            )
            behavior_loss = jnp.mean(behavior_loss) * behavior_coef

            # 4) Compute the losses.
            reward_loss = jnp.mean(
                _batch_categorical_cross_entropy(
                    reward_logits_target, model_out.reward_logits
                )
            )

            value_loss = (
                jnp.mean(
                    _batch_categorical_cross_entropy(value_logits_target, value_logits)
                )
                * value_coef
            )

            policy_loss = (
                jnp.mean(_batch_categorical_cross_entropy(policy_target, policy_logits))
                * policy_coef
            )

            total_loss = reward_loss + value_loss + policy_loss + behavior_loss

            if sampling:
                # Unnormalized.
                entropy_fn = lambda p: distrax.Categorical(logits=p).entropy()
            else:
                entropy_fn = lambda p: distrax.Categorical(probs=p).entropy()
            policy_target_entropy = jax.vmap(entropy_fn)(policy_target)
            policy_entropy = jax.vmap(
                lambda l: distrax.Categorical(logits=l).entropy()
            )(policy_logits)

            log = {
                "reward_target": reward_target,
                "reward_prediction": model_out.reward,
                "value_target": value_target,
                "value_prediction": model_out.value,
                "policy_entropy": policy_entropy,
                "policy_target_entropy": policy_target_entropy,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "behavior_loss": behavior_loss,
                "improve_advantage": improve_adv,
                "total_loss": total_loss,
            }
            return total_loss, log

        def batch_loss(
            params: Params,
            target_params: Params,
            trajectory: ActorOutput,
            rng_key: networks_lib.PRNGKey,
        ) -> Tuple[Array, Dict[str, Array]]:
            bs = len(trajectory.reward)
            rng_keys = jax.random.split(rng_key, bs)
            losses, log = jax.vmap(loss, (None, None, 0, 0))(
                params,
                target_params,
                trajectory,
                rng_keys,
            )
            log_mean = {f"{k}_mean": jnp.mean(v) for k, v in log.items()}
            std_keys = [
                "reward_target",
                "reward_prediction",
                "q_val_target",
                "q_val_prediction",
                "value_target",
                "value_prediction",
                "improve_advantage",
            ]
            std_keys = [k for k in std_keys if k in log]
            log_std = {f"{k}_std": jnp.std(log[k]) for k in std_keys}
            log_mean.update(log_std)
            return jnp.mean(losses), log_mean

        def update_step(
            state: TrainingState,
            trajectory: ActorOutput,
            rng_key: networks_lib.PRNGKey,
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            params = state.params
            optimizer_state = state.optimizer_state

            grads, log = jax.grad(batch_loss, has_aux=True)(
                state.params, state.target_params, trajectory, rng_key
            )
            grads = jax.lax.pmean(grads, axis_name="i")
            network_updates, optimizer_state = optimizer.update(
                grads, optimizer_state, params
            )
            params = optax.apply_updates(params, network_updates)
            log.update(
                {
                    "grad_norm": optax.global_norm(grads),
                    "update_norm": optax.global_norm(network_updates),
                    "param_norm": optax.global_norm(params),
                }
            )
            new_state = TrainingState(
                optimizer_state=optimizer_state,
                params=params,
                target_params=state.target_params,
                step=state.step + 1,
            )
            return new_state, log

        # Logger.
        self._logger = logger or loggers.make_default_logger(
            "learner", asynchronous=True, serialize_fn=utils.fetch_devicearray
        )

        # Iterator on demonstration transitions.
        self._demonstrations = demonstrations

        # JIT compiler.
        self._batch_size = batch_size
        self._num_devices = jax.lib.xla_bridge.device_count()
        assert self._batch_size % self._num_devices == 0
        self._update_step = jax.pmap(update_step, axis_name="i")

        # Create initial state.
        random_key = jax.random.PRNGKey(config["seed"])
        self._rng_key: networks_lib.PRNGKey
        key_r, key_d, key_p, self._rng_key = jax.random.split(random_key, 4)
        representation_params = networks.representation_network.init(key_r)
        transition_params = networks.transition_network.init(key_d)
        prediction_params = networks.prediction_network.init(key_p)

        # Create and initialize optimizer.
        params = Params(
            representation_params,
            transition_params,
            prediction_params,
        )
        weight_decay_mask = Params(
            representation=hk.data_structures.map(
                lambda module_name, name, value: True if name == "w" else False,
                params.representation,
            ),
            transition=hk.data_structures.map(
                lambda module_name, name, value: True if name == "w" else False,
                params.transition,
            ),
            prediction=hk.data_structures.map(
                lambda module_name, name, value: True if name == "w" else False,
                params.prediction,
            ),
        )
        learning_rate = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=config["learning_rate"],
            warmup_steps=config["warmup_steps"],
            transition_steps=100_000,
            decay_rate=config["learning_rate_decay"],
            staircase=True,
        )
        optimizer = optax.adamw(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            mask=weight_decay_mask,
        )
        if max_grad_norm:
            optimizer = optax.chain(optax.clip_by_global_norm(max_grad_norm), optimizer)
        optimizer_state = optimizer.init(params)
        target_params = params

        # Learner state.
        self._state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            target_params=target_params,
            step=0,
        )
        self._target_update_interval = target_update_interval

        self._state = jax.device_put_replicated(self._state, jax.local_devices())

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online
        # and fill the replay buffer.
        self._timestamp: Optional[float] = None
        self._elapsed = 0.0
        self._log_interval = log_interval
        self._unroll_steps = unroll_steps

    def step(self) -> None:
        """Train step."""
        update_key, self._rng_key = jax.random.split(self._rng_key)
        update_keys = jax.random.split(update_key, self._num_devices)
        trajectory: ActorOutput = next(self._demonstrations)
        trajectory = tree.map_structure(
            lambda x: x.reshape(
                self._num_devices, self._batch_size // self._num_devices, *x.shape[1:]
            ),
            trajectory,
        )

        self._state, metrics = self._update_step(self._state, trajectory, update_keys)

        _step = self._state.step[0]  # type: ignore
        timestamp = time.time()
        elapsed: float = 0
        if self._timestamp:
            elapsed = timestamp - self._timestamp
        self._timestamp = timestamp
        self._elapsed += elapsed

        if _step % self._target_update_interval == 0:
            state: TrainingState = self._state
            self._state = TrainingState(
                optimizer_state=state.optimizer_state,
                params=state.params,
                target_params=state.params,
                step=state.step,
            )
        if _step % self._log_interval == 0:
            metrics = jax.tree_util.tree_map(lambda t: t[0], metrics)
            metrics = jax.device_get(metrics)
            self._logger.write(
                {
                    **metrics,
                    **{
                        "step": _step,
                        "elapsed_time": self._elapsed,
                    },
                }
            )

    def get_variables(self, names: List[str]) -> List[Any]:
        """Get network parameters."""
        state = self.save()
        variables = {
            "representation": state.params.representation,
            "dynamics": state.params.transition,
            "prediction": state.params.prediction,
        }
        return [variables[name] for name in names]

    def save(self) -> TrainingState:
        """Save the training state.

        Returns:
            TrainingState: State to be saved.
        """
        _state = utils.fetch_devicearray(jax.tree_map(lambda t: t[0], self._state))
        return _state

    def restore(self, state: TrainingState) -> None:
        """Restore the training state.

        Args:
            state (TrainingState): State to be resumed.
        """
        self._state = jax.device_put_replicated(state, jax.local_devices())


def root_unroll(
    networks: Networks,
    params: Params,
    num_bins: int,
    state: Array,
) -> AgentOutput:
    """Unroll the learned model from the root node."""
    (
        policy_logits,
        reward_logits,
        value_logits,
    ) = networks.prediction_network.apply(params.prediction, state)
    reward = logits_to_scalar(reward_logits, num_bins)
    reward = inv_value_transform(reward)
    value = logits_to_scalar(value_logits, num_bins)
    value = inv_value_transform(value)
    return AgentOutput(
        state=state,
        policy_logits=policy_logits,
        reward_logits=reward_logits,
        reward=reward,
        value_logits=value_logits,
        value=value,
    )


def model_unroll(
    networks: Networks,
    params: Params,
    num_bins: int,
    state: Array,
    action_sequence: Array,
) -> AgentOutput:
    """Unroll the learned model with a sequence of actions."""

    def fn(state: Array, action: Array) -> Tuple[Array, Array]:
        """Dynamics fun for scan."""
        next_state = networks.transition_network.apply(
            params.transition, action[None], state
        )
        next_state = scale_gradient(next_state, 0.5)
        return next_state, next_state

    _, state_sequence = jax.lax.scan(fn, state, action_sequence)
    (
        policy_logits,
        reward_logits,
        value_logits,
    ) = networks.prediction_network.apply(params.prediction, state_sequence)
    reward = logits_to_scalar(reward_logits, num_bins)
    reward = inv_value_transform(reward)
    value = logits_to_scalar(value_logits, num_bins)
    value = inv_value_transform(value)
    return AgentOutput(
        state=state_sequence,
        policy_logits=policy_logits,
        reward_logits=reward_logits,
        reward=reward,
        value_logits=value_logits,
        value=value,
    )


def model_simulate(
    networks: Networks,
    params: Params,
    num_bins: int,
    state: Array,
    actions_to_simulate: Array,
) -> AgentOutput:
    """Simulate the learned model using one-step look-ahead."""

    def fn(state: Array, action: Array) -> Array:
        """Dynamics fun for vmap."""
        next_state = networks.transition_network.apply(
            params.transition, action[None], state
        )
        return next_state

    states_imagined = jax.vmap(fn, (None, 0))(state, actions_to_simulate)

    (
        policy_logits,
        reward_logits,
        value_logits,
    ) = networks.prediction_network.apply(params.prediction, states_imagined)
    reward = logits_to_scalar(reward_logits, num_bins)
    reward = inv_value_transform(reward)
    value = logits_to_scalar(value_logits, num_bins)
    value = inv_value_transform(value)
    return AgentOutput(
        state=states_imagined,
        policy_logits=policy_logits,
        reward_logits=reward_logits,
        reward=reward,
        value_logits=value_logits,
        value=value,
    )


def one_step_improve(
    networks: Networks,
    rng_key: networks_lib.PRNGKey,
    params: Params,
    model_root: AgentOutput,
    num_bins: int,
    discount_factor: float,
    num_simulations: int = -1,
    sampling: bool = False,
) -> Tuple[Array, Array]:
    """Obtain the one-step look-ahead target policy."""
    environment_specs = networks.environment_specs

    pi_prior = jax.nn.softmax(model_root.policy_logits)
    value_prior = model_root.value

    if sampling:
        assert num_simulations > 0
        logging.info(
            f"[Sample] Using {num_simulations} samples to estimate improvement."
        )
        pi_sample = distrax.Categorical(probs=pi_prior)
        sample_acts = pi_sample.sample(seed=rng_key, sample_shape=num_simulations)
        sample_one_step_out: AgentOutput = model_simulate(
            networks, params, num_bins, model_root.state, sample_acts
        )
        sample_adv = (
            sample_one_step_out.reward
            + discount_factor * sample_one_step_out.value
            - value_prior
        )
        adv = sample_adv  # for log
        sample_exp_adv = jnp.exp(sample_adv)
        normalizer_raw = (jnp.sum(sample_exp_adv) + 1) / num_simulations
        coeff = jnp.zeros_like(pi_prior)

        def body(i: int, val: jnp.ndarray) -> jnp.ndarray:
            """Body fun for the loop."""
            normalizer_i = normalizer_raw - sample_exp_adv[i] / num_simulations
            delta = jnp.zeros_like(val)
            delta = delta.at[sample_acts[i]].set(sample_exp_adv[i] / normalizer_i)
            return val + delta

        coeff = jax.lax.fori_loop(0, num_simulations, body, coeff)
        pi_improved = coeff / num_simulations
    else:
        all_actions = jnp.arange(environment_specs.actions.num_values)
        model_one_step_out: AgentOutput = model_simulate(
            networks, params, num_bins, model_root.state, all_actions
        )
        chex.assert_equal_shape([model_one_step_out.reward, pi_prior])
        chex.assert_equal_shape([model_one_step_out.value, pi_prior])
        adv = (
            model_one_step_out.reward
            + discount_factor * model_one_step_out.value
            - value_prior
        )
        pi_improved = pi_prior * jnp.exp(adv)
        pi_improved = pi_improved / jnp.sum(pi_improved)

    chex.assert_equal_shape([pi_improved, pi_prior])
    # pi_improved here might not sum to 1, in which case we use CE
    # to conveniently calculate the policy gradients (Eq. 9)
    return pi_improved, adv
