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

"""RL Unplugged Atari datasets."""
import json
import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

import dm_env
import gym
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from acme import wrappers
from dm_env import specs
from dopamine.discrete_domains import atari_lib

from rosmo.type import Array

with open(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "rl_unplugged_atari_baselines.json"
    ),
    "r",
) as f:
    BASELINES = json.load(f)


class _BatchToTransition:
    """Creates (s,a,r,f,l) transitions."""

    @staticmethod
    def create_transitions(batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Create stacked transitions.

        Args:
            batch (Dict[str, tf.Tensor]): Data batch

        Returns:
            Dict[str, tf.Tensor]: Stacked data batch.
        """
        observation = tf.squeeze(batch[rlds.OBSERVATION], axis=-1)
        observation = tf.transpose(observation, perm=[1, 2, 0])
        action = batch[rlds.ACTION][-1]
        reward = batch[rlds.REWARD][-1]
        discount = batch[rlds.DISCOUNT][-1]
        return {
            "observation": observation,
            "action": action,
            "reward": reward,
            "discount": discount,
            "is_first": batch[rlds.IS_FIRST][0],
            "is_last": batch[rlds.IS_LAST][-1],
        }


def _get_trajectory_dataset_fn(
    stack_size: int,
    trajectory_length: int = 1,
) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    batch_fn = _BatchToTransition().create_transitions

    def make_trajectory_dataset(episode: tf.data.Dataset) -> tf.data.Dataset:
        """Converts an episode of steps to a dataset of custom transitions.

        Episode spec: {
          'checkpoint_id': <tf.Tensor: shape=(), dtype=int64, numpy=0>,
          'episode_id': <tf.Tensor: shape=(), dtype=int64, numpy=0>,
          'episode_return': <tf.Tensor: shape=(), dtype=float32, numpy=0>,
          'steps': <_VariantDataset element_spec={
            'action': TensorSpec(shape=(), dtype=tf.int64, name=None),
            'discount': TensorSpec(shape=(), dtype=tf.float32, name=None),
            'is_first': TensorSpec(shape=(), dtype=tf.bool, name=None),
            'is_last': TensorSpec(shape=(), dtype=tf.bool, name=None),
            'is_terminal': TensorSpec(shape=(), dtype=tf.bool, name=None),
            'observation': TensorSpec(shape=(84, 84, 1), dtype=tf.uint8,
              name=None),
            'reward': TensorSpec(shape=(), dtype=tf.float32, name=None)
            }
          >}
        """
        # Create a dataset of 2-step sequences with overlap of 1.
        timesteps: tf.data.Dataset = episode[rlds.STEPS]
        batched_steps = rlds.transformations.batch(
            timesteps,
            size=stack_size,
            shift=1,
            drop_remainder=True,
        )
        transitions = batched_steps.map(batch_fn)
        # Batch trajectory.
        if trajectory_length > 1:
            transitions = transitions.repeat(2)
            transitions = transitions.skip(
                tf.random.uniform([], 0, trajectory_length, dtype=tf.int64)
            )
            trajectory = transitions.batch(trajectory_length, drop_remainder=True)
        else:
            trajectory = transitions
        return trajectory

    return make_trajectory_dataset


def _uniformly_subsampled_atari_data(
    dataset_name: str,
    data_percent: int,
    data_dir: str,
) -> tf.data.Dataset:
    ds_builder = tfds.builder(dataset_name)
    data_splits = []
    total_num_episode = 0
    for split, info in ds_builder.info.splits.items():
        # Convert `data_percent` to number of episodes to allow
        # for fractional percentages.
        num_episodes = int((data_percent / 100) * info.num_examples)
        total_num_episode += num_episodes
        if num_episodes == 0:
            raise ValueError(f"{data_percent}% leads to 0 episodes in {split}!")
        # Sample first `data_percent` episodes from each of the data split.
        data_splits.append(f"{split}[:{num_episodes}]")
    # Interleave episodes across different splits/checkpoints.
    # Set `shuffle_files=True` to shuffle episodes across files within splits.
    read_config = tfds.ReadConfig(
        interleave_cycle_length=len(data_splits),
        shuffle_reshuffle_each_iteration=True,
        enable_ordering_guard=False,
    )
    logging.info(f"Total number of episode = {total_num_episode}")
    return tfds.load(
        dataset_name,
        data_dir=data_dir,
        split="+".join(data_splits),
        read_config=read_config,
        shuffle_files=True,
    )


def create_atari_ds_loader(
    env_name: str,
    run_number: int,
    dataset_dir: str,
    stack_size: int = 4,
    data_percentage: int = 10,
    trajectory_fn: Optional[Callable] = None,
    shuffle_num_episodes: int = 1000,
    shuffle_num_steps: int = 50000,
    trajectory_length: int = 10,
    **_: Any,
) -> tf.data.Dataset:
    """Create Atari dataset loader.

    Args:
        env_name (str): Environment name.
        run_number (int): Run number.
        dataset_dir (str): Directory to the dataset.
        stack_size (int, optional): Stack size. Defaults to 4.
        data_percentage (int, optional): Fraction of data to be used. Defaults to 10.
        trajectory_fn (Optional[Callable], optional): Function to form trajectory.
          Defaults to None.
        shuffle_num_episodes (int, optional): Number of episodes to shuffle.
          Defaults to 1000.
        shuffle_num_steps (int, optional): Number of steps to shuffle.
          Defaults to 50000.
        trajectory_length (int, optional): Trajectory length. Defaults to 10.
        **_: Other keyword arguments.

    Returns:
        tf.data.Dataset: Dataset.
    """
    if trajectory_fn is None:
        trajectory_fn = _get_trajectory_dataset_fn(stack_size, trajectory_length)
    dataset_name = f"rlu_atari_checkpoints_ordered/{env_name}_run_{run_number}"
    # Create a dataset of episodes sampling `data_percent`% episodes
    # from each of the data split.
    dataset = _uniformly_subsampled_atari_data(
        dataset_name, data_percentage, dataset_dir
    )
    # Shuffle the episodes to avoid consecutive episodes.
    dataset = dataset.shuffle(shuffle_num_episodes)
    # Interleave=1 keeps ordered sequential steps.
    dataset = dataset.interleave(
        trajectory_fn,
        cycle_length=100,
        block_length=1,
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    # Shuffle trajectories in the dataset.
    dataset = dataset.shuffle(
        shuffle_num_steps // trajectory_length,
        reshuffle_each_iteration=True,
    )
    return dataset


class _AtariDopamineWrapper(dm_env.Environment):
    """Wrapper for Atari Dopamine environmnet."""

    def __init__(self, env: gym.Env, max_episode_steps: int = 108000):
        self._env = env
        self._max_episode_steps = max_episode_steps
        self._episode_steps = 0
        self._reset_next_episode = True
        self._reset_next_step = True

    def reset(self) -> dm_env.TimeStep:
        self._episode_steps = 0
        self._reset_next_step = False
        observation = self._env.reset()
        return dm_env.restart(observation.squeeze(-1))  # type: ignore

    def step(self, action: Union[int, Array]) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()
        if not isinstance(action, int):
            action = action.item()
        observation, reward, terminal, _ = self._env.step(action)  # type: ignore
        observation = observation.squeeze(-1)
        discount = 1 - float(terminal)
        self._episode_steps += 1
        if terminal:
            self._reset_next_episode = True
            return dm_env.termination(reward, observation)
        if self._episode_steps == self._max_episode_steps:
            self._reset_next_episode = True
            return dm_env.truncation(reward, observation, discount)
        return dm_env.transition(reward, observation, discount)

    def observation_spec(self) -> specs.Array:
        space = self._env.observation_space
        return specs.Array(space.shape[:-1], space.dtype)  # type: ignore

    def action_spec(self) -> specs.DiscreteArray:
        return specs.DiscreteArray(self._env.action_space.n)  # type: ignore

    def render(self, mode: str = "rgb_array") -> Any:
        """Render the environment.

        Args:
            mode (str, optional): Mode of rendering. Defaults to "rgb_array".

        Returns:
            Any: Rendered result.
        """
        return self._env.render(mode)


def environment(game: str, stack_size: int) -> dm_env.Environment:
    """Atari environment."""
    env = atari_lib.create_atari_environment(game_name=game, sticky_actions=True)
    env = _AtariDopamineWrapper(env, max_episode_steps=20_000)
    env = wrappers.FrameStackingWrapper(env, num_frames=stack_size)
    return wrappers.SinglePrecisionWrapper(env)


def env_loader(
    env_name: str,
    run_number: int,
    dataset_dir: str,
    stack_size: int = 4,
    data_percentage: int = 10,
    trajectory_fn: Optional[Callable] = None,
    shuffle_num_episodes: int = 1000,
    shuffle_num_steps: int = 50000,
    trajectory_length: int = 10,
    **_: Any,
) -> Tuple[dm_env.Environment, tf.data.Dataset]:
    """Get the environment and dataset.

    Args:
        env_name (str): Name of the environment.
        run_number (int): Run number of the dataset.
        dataset_dir (str): Directory storing the dataset.
        stack_size (int, optional): Number of frame stacking. Defaults to 4.
        data_percentage (int, optional): Fraction of data to be used. Defaults to 10.
        trajectory_fn (Optional[Callable], optional): Function to form trajectory.
          Defaults to None.
        shuffle_num_episodes (int, optional): Number of episodes to shuffle.
          Defaults to 1000.
        shuffle_num_steps (int, optional): Number of steps to shuffle.
          Defaults to 50000.
        trajectory_length (int, optional): Trajectory length. Defaults to 10.
        **_: Other keyword arguments.

    Returns:
        Tuple[dm_env.Environment, tf.data.Dataset]: Environment and dataset.
    """
    return environment(game=env_name, stack_size=stack_size), create_atari_ds_loader(
        env_name=env_name,
        run_number=run_number,
        dataset_dir=dataset_dir,
        stack_size=stack_size,
        data_percentage=data_percentage,
        trajectory_fn=trajectory_fn,
        shuffle_num_episodes=shuffle_num_episodes,
        shuffle_num_steps=shuffle_num_steps,
        trajectory_length=trajectory_length,
    )
