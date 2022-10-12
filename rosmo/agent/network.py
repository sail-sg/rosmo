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

"""Haiku neural network modules."""
import dataclasses
from typing import Any, Dict, List, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

from rosmo.type import Forwardable


def get_prediction_head_layers(
    reduced_channels_head: int,
    mlp_layers: List[int],
    num_predictions: int,
    w_init: Optional[hk.initializers.Initializer] = None,
) -> List[Forwardable]:
    """Get prediction head layers.

    Args:
        reduced_channels_head (int): Conv reduced channels.
        mlp_layers (List[int]): MLP layers' hidden units.
        num_predictions (int): Output size.
        w_init (Optional[hk.initializers.Initializer], optional): Weight
          initialization. Defaults to None.

    Returns:
        List[Forwardable]: List of layers.
    """
    layers: List[Forwardable] = [
        hk.Conv2D(
            reduced_channels_head,
            kernel_shape=1,
            stride=1,
            padding="SAME",
            with_bias=False,
        ),
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
        hk.Flatten(-3),
    ]
    for l in mlp_layers:
        layers.extend(
            [
                hk.Linear(l, with_bias=False),
                hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
                jax.nn.relu,
            ]
        )
    layers.append(hk.Linear(num_predictions, w_init=w_init))
    return layers


def get_ln_relu_layers() -> List[Forwardable]:
    """Get LN relu layers.

    Returns:
        List[Forwardable]: LayerNorm+relu.
    """
    return [
        hk.LayerNorm(axis=(-3, -2, -1), create_scale=True, create_offset=True),
        jax.nn.relu,
    ]


class ResConvBlock(hk.Module):
    """A residual convolutional block in pre-activation style."""

    def __init__(
        self,
        channels: int,
        stride: int,
        use_projection: bool,
        name: str = "res_conv_block",
    ):
        """Init residual block."""
        super().__init__(name=name)
        self._use_projection = use_projection
        if use_projection:
            self._proj_conv = hk.Conv2D(
                channels, kernel_shape=3, stride=stride, padding="SAME", with_bias=False
            )
        self._conv_0 = hk.Conv2D(
            channels, kernel_shape=3, stride=stride, padding="SAME", with_bias=False
        )
        self._ln_0 = hk.LayerNorm(
            axis=(-3, -2, -1), create_scale=True, create_offset=True
        )
        self._conv_1 = hk.Conv2D(
            channels, kernel_shape=3, stride=1, padding="SAME", with_bias=False
        )
        self._ln_1 = hk.LayerNorm(
            axis=(-3, -2, -1), create_scale=True, create_offset=True
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward ResBlock."""
        # NOTE: Using LayerNorm is fine
        # (https://arxiv.org/pdf/2104.06294.pdf Appendix A).
        shortcut = out = x
        out = self._ln_0(out)
        out = jax.nn.relu(out)
        if self._use_projection:
            shortcut = self._proj_conv(out)
        out = hk.Sequential(
            [
                self._conv_0,
                self._ln_1,
                jax.nn.relu,
                self._conv_1,
            ]
        )(out)
        return shortcut + out


class Representation(hk.Module):
    """Representation encoding module."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        name: str = "representation",
    ):
        """Init representatioin function."""
        super().__init__(name=name)
        self._channels = channels
        self._num_blocks = num_blocks

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        """Forward representation function."""
        # 1. Downsampling.
        torso: List[Forwardable] = [
            lambda x: x / 255.0,
            hk.Conv2D(
                self._channels // 2,
                kernel_shape=3,
                stride=2,
                padding="SAME",
                with_bias=False,
            ),
        ]
        torso.extend(
            [
                ResConvBlock(self._channels // 2, stride=1, use_projection=False)
                for _ in range(1)
            ]
        )
        torso.append(ResConvBlock(self._channels, stride=2, use_projection=True))
        torso.extend(
            [
                ResConvBlock(self._channels, stride=1, use_projection=False)
                for _ in range(1)
            ]
        )
        torso.append(
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")
        )
        torso.extend(
            [
                ResConvBlock(self._channels, stride=1, use_projection=False)
                for _ in range(1)
            ]
        )
        torso.append(
            hk.AvgPool(window_shape=(3, 3, 1), strides=(2, 2, 1), padding="SAME")
        )

        # 2. Encoding.
        torso.extend(
            [
                ResConvBlock(self._channels, stride=1, use_projection=False)
                for _ in range(self._num_blocks)
            ]
        )
        return hk.Sequential(torso)(observations)


class Transition(hk.Module):
    """Dynamics transition module."""

    def __init__(
        self,
        channels: int,
        num_blocks: int,
        name: str = "transition",
    ):
        """Init transition function."""
        super().__init__(name=name)
        self._channels = channels
        self._num_blocks = num_blocks

    def __call__(
        self, encoded_action: jnp.ndarray, prev_state: jnp.ndarray
    ) -> jnp.ndarray:
        """Forward transition function."""
        channels = prev_state.shape[-1]
        shortcut = prev_state

        prev_state = hk.LayerNorm(
            axis=(-3, -2, -1), create_scale=True, create_offset=True
        )(prev_state)
        prev_state = jax.nn.relu(prev_state)

        x_and_h = jnp.concatenate([prev_state, encoded_action], axis=-1)
        out = hk.Conv2D(
            self._channels,
            kernel_shape=3,
            stride=1,
            padding="SAME",
            with_bias=False,
        )(x_and_h)
        out += shortcut  # Residual link to maintain recurrent info flow.

        res_layers = [
            ResConvBlock(channels, stride=1, use_projection=False)
            for _ in range(self._num_blocks)
        ]
        out = hk.Sequential(res_layers)(out)
        return out


class Prediction(hk.Module):
    """Policy, value and reward prediction module."""

    def __init__(
        self,
        num_blocks: int,
        num_actions: int,
        num_bins: int,
        channel: int,
        fc_layers_reward: List[int],
        fc_layers_value: List[int],
        fc_layers_policy: List[int],
        output_init_scale: float,
        name: str = "prediction",
    ) -> None:
        """Init prediction function."""
        super().__init__(name=name)
        self._num_blocks = num_blocks
        self._num_actions = num_actions
        self._num_bins = num_bins
        self._channel = channel
        self._fc_layers_reward = fc_layers_reward
        self._fc_layers_value = fc_layers_value
        self._fc_layers_policy = fc_layers_policy
        self._output_init_scale = output_init_scale

    def __call__(
        self, states: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Forward prediction function."""
        output_init = hk.initializers.VarianceScaling(scale=self._output_init_scale)
        reward_head, value_head, policy_head = [], [], []

        # Add LN+Relu due to pre-activation.
        reward_head.extend(get_ln_relu_layers())
        value_head.extend(get_ln_relu_layers())
        policy_head.extend(get_ln_relu_layers())

        reward_head.extend(
            get_prediction_head_layers(
                self._channel,
                self._fc_layers_reward,
                self._num_bins,
                output_init,
            )
        )
        reward_logits = hk.Sequential(reward_head)(states)

        res_layers = [
            ResConvBlock(states.shape[-1], stride=1, use_projection=False)
            for _ in range(self._num_blocks)
        ]
        out = hk.Sequential(res_layers)(states)

        value_head.extend(
            get_prediction_head_layers(
                self._channel,
                self._fc_layers_value,
                self._num_bins,
                output_init,
            )
        )
        value_logits = hk.Sequential(value_head)(out)

        policy_head.extend(
            get_prediction_head_layers(
                self._channel,
                self._fc_layers_policy,
                self._num_actions,
                output_init,
            )
        )
        policy_logits = hk.Sequential(policy_head)(out)
        return policy_logits, reward_logits, value_logits


@dataclasses.dataclass
class Networks:
    """ROSMO Networks."""

    representation_network: networks_lib.FeedForwardNetwork
    transition_network: networks_lib.FeedForwardNetwork
    prediction_network: networks_lib.FeedForwardNetwork

    environment_specs: specs.EnvironmentSpec


def make_atari_networks(
    env_spec: specs.EnvironmentSpec,
    channels: int,
    num_bins: int,
    output_init_scale: float,
    blocks_representation: int,
    blocks_prediction: int,
    blocks_transition: int,
    reduced_channels_head: int,
    fc_layers_reward: List[int],
    fc_layers_value: List[int],
    fc_layers_policy: List[int],
) -> Networks:
    """Make Atari networks.

    Args:
        env_spec (specs.EnvironmentSpec): Environment spec.
        channels (int): Convolution channels.
        num_bins (int): Number of bins.
        output_init_scale (float): Weight init scale.
        blocks_representation (int): Number of blocks for representation.
        blocks_prediction (int): Number of blocks for prediction.
        blocks_transition (int): Number of blocks for transition.
        reduced_channels_head (int): Reduced conv channels for prediction head.
        fc_layers_reward (List[int]): Fully connected layers for reward prediction.
        fc_layers_value (List[int]): Fully connected layers for value prediction.
        fc_layers_policy (List[int]): Fully connected layers for policy prediction.

    Returns:
        Networks: Constructed networks.
    """
    action_space_size = env_spec.actions.num_values

    def _representation_fun(observations: jnp.ndarray) -> jnp.ndarray:
        network = Representation(channels, blocks_representation)
        state = network(observations)
        return state

    representation = hk.without_apply_rng(hk.transform(_representation_fun))
    hidden_channels = channels

    def _transition_fun(action: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        action = hk.one_hot(action, action_space_size)[None, :]
        encoded_action = jnp.broadcast_to(action, state.shape[:-1] + action.shape[-1:])

        network = Transition(hidden_channels, blocks_transition)
        next_state = network(encoded_action, state)
        return next_state

    transition = hk.without_apply_rng(hk.transform(_transition_fun))
    prediction = hk.without_apply_rng(
        hk.transform(
            lambda states: Prediction(
                blocks_prediction,
                action_space_size,
                num_bins,
                reduced_channels_head,
                fc_layers_reward,
                fc_layers_value,
                fc_layers_policy,
                output_init_scale,
            )(states)
        )
    )

    dummy_action = jnp.array([env_spec.actions.generate_value()])
    dummy_obs = utils.zeros_like(env_spec.observations)

    def _dummy_state(key: networks_lib.PRNGKey) -> jnp.ndarray:
        encoder_params = representation.init(key, dummy_obs)
        dummy_state = representation.apply(encoder_params, dummy_obs)
        return dummy_state

    return Networks(
        representation_network=networks_lib.FeedForwardNetwork(
            lambda key: representation.init(key, dummy_obs), representation.apply
        ),
        transition_network=networks_lib.FeedForwardNetwork(
            lambda key: transition.init(key, dummy_action, _dummy_state(key)),
            transition.apply,
        ),
        prediction_network=networks_lib.FeedForwardNetwork(
            lambda key: prediction.init(key, _dummy_state(key)),
            prediction.apply,
        ),
        environment_specs=env_spec,
    )


def get_bsuite_networks(
    env_spec: specs.EnvironmentSpec, config: Dict[str, Any]
) -> Networks:
    """Make BSuite networks.

    Args:
        env_spec (specs.EnvironmentSpec): Environment specifications.
        config (Dict[str, Any]): Configurations.

    Returns:
        Networks: Constructed networks.
    """
    action_space_size = env_spec.actions.num_values

    def _representation_fun(observations: jnp.ndarray) -> jnp.ndarray:
        network = hk.Sequential([hk.Flatten(), hk.nets.MLP(config["encoder_layers"])])
        state = network(observations)
        return state

    representation = hk.without_apply_rng(hk.transform(_representation_fun))

    def _transition_fun(action: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        action = hk.one_hot(action, action_space_size)
        network = hk.nets.MLP(config["dynamics_layers"])
        sa = jnp.concatenate(
            [jnp.reshape(state, (-1, state.shape[-1])), action], axis=-1
        )
        next_state = network(sa).squeeze()
        return next_state

    transition = hk.without_apply_rng(hk.transform(_transition_fun))

    def _prediction_fun(state: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        network = hk.nets.MLP(config["prediction_layers"], activate_final=True)
        head_state = network(state)
        output_init = hk.initializers.VarianceScaling(scale=0.0)
        head_policy = hk.nets.MLP([action_space_size], w_init=output_init)
        head_value = hk.nets.MLP([config["num_bins"]], w_init=output_init)
        head_reward = hk.nets.MLP([config["num_bins"]], w_init=output_init)

        return (
            head_policy(head_state),
            head_reward(head_state),
            head_value(head_state),
        )

    prediction = hk.without_apply_rng(hk.transform(_prediction_fun))

    dummy_action = utils.add_batch_dim(jnp.array(env_spec.actions.generate_value()))

    dummy_obs = utils.add_batch_dim(utils.zeros_like(env_spec.observations))

    def _dummy_state(key: networks_lib.PRNGKey) -> jnp.ndarray:
        encoder_params = representation.init(key, dummy_obs)
        dummy_state = representation.apply(encoder_params, dummy_obs)
        return dummy_state

    return Networks(
        representation_network=networks_lib.FeedForwardNetwork(
            lambda key: representation.init(key, dummy_obs), representation.apply
        ),
        transition_network=networks_lib.FeedForwardNetwork(
            lambda key: transition.init(key, dummy_action, _dummy_state(key)),
            transition.apply,
        ),
        prediction_network=networks_lib.FeedForwardNetwork(
            lambda key: prediction.init(key, _dummy_state(key)),
            prediction.apply,
        ),
        environment_specs=env_spec,
    )
