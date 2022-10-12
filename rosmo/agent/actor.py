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

"""Evaluating actor."""
from typing import Dict, Optional, Tuple

import acme
import chex
import dm_env
import jax
import numpy as np
import rlax
from absl import logging
from acme import types
from acme.jax import networks as networks_lib

from rosmo.agent.learning import one_step_improve, root_unroll
from rosmo.agent.network import Networks
from rosmo.agent.type import AgentOutput, Params
from rosmo.type import ActorOutput, Array


class RosmoEvalActor(acme.core.Actor):
    """ROSMO evaluation actor."""

    def __init__(
        self,
        networks: Networks,
        config: Dict,
    ) -> None:
        """Init ROSMO evaluation actor."""
        self._networks = networks

        self._environment_specs = networks.environment_specs
        self._rng_key = jax.random.PRNGKey(config["seed"])
        self._random_action = False
        self._params: Optional[Params] = None
        self._timestep: Optional[ActorOutput] = None

        num_bins = config["num_bins"]
        discount_factor = config["discount_factor"]
        sampling = config.get("sampling", False)
        num_simulations = config.get("num_simulations", -1)

        def root_step(
            rng_key: chex.PRNGKey,
            params: Params,
            timesteps: ActorOutput,
        ) -> Tuple[Array, AgentOutput]:
            # Model one-step look-ahead for acting.
            trajectory = jax.tree_map(
                lambda t: t[None], timesteps
            )  # Add a dummy time dimension.
            state = networks.representation_network.apply(
                params.representation, trajectory.observation
            )
            agent_out: AgentOutput = root_unroll(
                self._networks, params, num_bins, state
            )
            improve_key, sample_key = jax.random.split(rng_key)

            agent_out: AgentOutput = jax.tree_map(
                lambda t: t.squeeze(axis=0), agent_out
            )  # Squeeze the dummy time dimension.
            if not sampling:
                logging.info("[Actor] Using onestep improvement.")
                improved_policy, _ = one_step_improve(
                    self._networks,
                    improve_key,
                    params,
                    agent_out,
                    num_bins,
                    discount_factor,
                    num_simulations,
                    sampling,
                )
            else:
                logging.info("[Actor] Using policy.")
                improved_policy = jax.nn.softmax(agent_out.policy_logits)
            action = rlax.categorical_sample(sample_key, improved_policy)
            return action, agent_out

        def batch_step(
            rng_key: chex.PRNGKey,
            params: Params,
            timesteps: ActorOutput,
        ) -> Tuple[networks_lib.PRNGKey, Array, AgentOutput]:
            batch_size = timesteps.reward.shape[0]
            rng_key, step_key = jax.random.split(rng_key)
            step_keys = jax.random.split(step_key, batch_size)
            batch_root_step = jax.vmap(root_step, (0, None, 0))
            actions, agent_out = batch_root_step(step_keys, params, timesteps)
            return rng_key, actions, agent_out

        self._agent_step = jax.jit(batch_step)

    def select_action(self, observation: types.NestedArray) -> types.NestedArray:
        """Select action to execute."""
        if self._random_action:
            return np.random.randint(0, self._environment_specs.actions.num_values, [1])
        batched_timestep = jax.tree_map(
            lambda t: t[None], jax.device_put(self._timestep)
        )
        self._rng_key, action, _ = self._agent_step(
            self._rng_key, self._params, batched_timestep
        )
        action = jax.device_get(action).item()
        return action

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        """Observe and record the first timestep."""
        self._timestep = ActorOutput(
            action=np.zeros((1,), dtype=np.int32),
            reward=np.zeros((1,), dtype=np.float32),
            observation=timestep.observation,
            is_first=np.ones((1,), dtype=np.float32),
            is_last=np.zeros((1,), dtype=np.float32),
        )

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ) -> None:
        """Observe and record a timestep."""
        self._timestep = ActorOutput(
            action=action,
            reward=next_timestep.reward,
            observation=next_timestep.observation,
            is_first=next_timestep.first(),  # previous last = this first.
            is_last=next_timestep.last(),
        )

    def update(self, wait: bool = False) -> None:
        """Update."""

    def update_params(self, params: Params) -> None:
        """Update parameters.

        Args:
            params (Params): Parameters.
        """
        self._params = params
