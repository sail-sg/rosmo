# Copyright 2023 Garena Online Private Limited.
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

"""Improvement operators."""
from typing import Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
import mctx
from absl import logging
from acme.jax import networks as networks_lib

from rosmo.agent.network import Networks
from rosmo.agent.type import AgentOutput, Params
from rosmo.agent.utils import inv_value_transform, logits_to_scalar
from rosmo.type import Array

## -------------------------------- ##
## One-step Look-ahead Improvement. ##
## -------------------------------- ##


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
        sample_acts = pi_sample.sample(
            seed=rng_key, sample_shape=num_simulations)
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
            delta = delta.at[sample_acts[i]].set(
                sample_exp_adv[i] / normalizer_i)
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


def mcts_improve(
    networks: Networks,
    rng_key: networks_lib.PRNGKey,
    params: Params,
    num_bins: int,
    model_root: AgentOutput,
    discount_factor: float,
    num_simulations: int,
    search_depth: int,
) -> mctx.PolicyOutput:
    """Obtain the Monte-Carlo Tree Search target policy."""

    def recurrent_fn(
        params: Params, rng_key: networks_lib.PRNGKey, action: Array, state: Array
    ) -> Tuple[mctx.RecurrentFnOutput, Array]:
        del rng_key

        def fn(state: Array, action: Array):
            next_state = networks.transition_network.apply(
                params.transition, action[None], state
            )
            return next_state

        next_state = jax.vmap(fn, (0, 0))(state, action)

        (
            policy_logits,
            reward_logits,
            value_logits,
        ) = networks.prediction_network.apply(params.prediction, next_state)
        reward = logits_to_scalar(reward_logits, num_bins)
        reward = inv_value_transform(reward)
        value = logits_to_scalar(value_logits, num_bins)
        value = inv_value_transform(value)
        recurrent_fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.full_like(value, fill_value=discount_factor),
            prior_logits=policy_logits,
            value=value,
        )
        return recurrent_fn_output, next_state

    root = mctx.RootFnOutput(
        prior_logits=model_root.policy_logits,
        value=model_root.value,
        embedding=model_root.state,
    )

    return mctx.muzero_policy(
        params,
        rng_key,
        root,
        recurrent_fn,
        num_simulations,
        max_depth=search_depth,
    )
