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

"""BSuite datasets."""
import os
from typing import Any, Dict, Tuple

import dm_env
import numpy as np
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging
from bsuite.environments.cartpole import Cartpole as _Cartpole
from bsuite.environments.catch import Catch as _Catch
from bsuite.environments.mountain_car import MountainCar as _MountainCar

from rosmo.data.buffer import UniformBuffer
from rosmo.type import ActorOutput


class Cartpole(_Cartpole):
    """Carpole environment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init env."""
        super().__init__(*args, **kwargs)
        self.episode_id = 0
        self.episode_return = 0
        self.bsuite_id = "cartpole/0"

    def reset(self) -> dm_env.TimeStep:
        """Reset env."""
        self.episode_id += 1
        self.episode_return = 0
        return super().reset()

    def step(self, action: int) -> dm_env.TimeStep:
        """Step env."""
        timestep = super().step(action)
        if timestep.reward is not None:
            self.episode_return += timestep.reward
        return timestep


class Catch(_Catch):
    """Catch environment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init env."""
        super().__init__(*args, **kwargs)
        self.episode_id = 0
        self.episode_return = 0
        self.bsuite_id = "catch/0"

    def _reset(self) -> dm_env.TimeStep:
        self.episode_id += 1
        self.episode_return = 0
        return super()._reset()

    def _step(self, action: int) -> dm_env.TimeStep:
        timestep = super()._step(action)
        if timestep.reward is not None:
            self.episode_return += timestep.reward
        return timestep


class MountainCar(_MountainCar):
    """Mountain Car environment."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Init env."""
        super().__init__(*args, **kwargs)
        self.episode_id = 0
        self.episode_return = 0
        self.bsuite_id = "mountain_car/0"

    def _reset(self) -> dm_env.TimeStep:
        self.episode_id += 1
        self.episode_return = 0
        return super()._reset()

    def _step(self, action: int) -> dm_env.TimeStep:
        timestep = super()._step(action)
        if timestep.reward is not None:
            self.episode_return += timestep.reward
        return timestep


_ENV_FACTORY: Dict[str, Tuple[dm_env.Environment, int]] = {
    "cartpole": (Cartpole, 1000),
    "catch": (Catch, 2000),
    "mountain_car": (MountainCar, 500),
}

_LOAD_SIZE = 1e7

SCORES = {
    "cartpole": {
        "random": 64.833,
        "online_dqn": 1001.0,
    },
    "catch": {
        "random": -0.667,
        "online_dqn": 1.0,
    },
    "mountain_car": {
        "random": -1000.0,
        "online_dqn": -102.167,
    },
}


def create_bsuite_ds_loader(
    env_name: str, dataset_name: str, dataset_percentage: int
) -> tf.data.Dataset:
    """Create BSuite dataset loader.

    Args:
        env_name (str): Environment name.
        dataset_name (str): Dataset name.
        dataset_percentage (int): Fraction of data to be used

    Returns:
        tf.data.Dataset: Dataset.
    """
    dataset = tfds.builder_from_directory(dataset_name).as_dataset(split="all")
    num_trajectory = _ENV_FACTORY[env_name][1]
    if dataset_percentage < 100:
        idx = np.arange(0, num_trajectory, (100 // dataset_percentage))
        idx += np.random.randint(0, 100 // dataset_percentage, idx.shape) + 1
        idx = tf.convert_to_tensor(idx, "int32")
        filter_fn = lambda episode: tf.math.equal(
            tf.reduce_sum(tf.cast(episode["episode_id"] == idx, "int32")), 1
        )
        dataset = dataset.filter(filter_fn)
    parse_fn = lambda episode: episode[rlds.STEPS]
    dataset = dataset.interleave(
        parse_fn,
        cycle_length=1,
        block_length=1,
        deterministic=False,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset


def env_loader(
    env_name: str,
    dataset_dir: str,
    data_percentage: int = 100,
    batch_size: int = 8,
    trajectory_length: int = 1,
    **_: Any,
) -> Tuple[dm_env.Environment, tf.data.Dataset]:
    """Get the environment and dataset.

    Args:
        env_name (str): Name of the environment.
        dataset_dir (str): Directory storing the dataset.
        data_percentage (int, optional): Fraction of data to be used. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 8.
        trajectory_length (int, optional): Trajectory length. Defaults to 1.
        **_: Other keyword arguments.

    Returns:
        Tuple[dm_env.Environment, tf.data.Dataset]: Environment and dataset.
    """
    data_name = env_name
    if env_name not in _ENV_FACTORY:
        _env_setting = env_name.split("_")
        if len(_env_setting) > 1:
            env_name = "_".join(_env_setting[:-1])
    assert env_name in _ENV_FACTORY, f"env {env_name} not supported"

    dataset_name = os.path.join(dataset_dir, f"{data_name}")
    print(dataset_name)
    dataset = create_bsuite_ds_loader(env_name, dataset_name, data_percentage)
    dataloader = dataset.batch(int(_LOAD_SIZE)).as_numpy_iterator()
    data = next(dataloader)

    data_buffer = {}
    data_buffer["observation"] = data["observation"]
    data_buffer["reward"] = data["reward"]
    data_buffer["is_first"] = data["is_first"]
    data_buffer["is_last"] = data["is_last"]
    data_buffer["action"] = data["action"]

    timesteps = ActorOutput(**data_buffer)
    data_size = len(timesteps.reward)
    assert data_size < _LOAD_SIZE

    iterator = UniformBuffer(
        0,
        data_size,
        trajectory_length,
        batch_size,
    )
    logging.info(f"[Data] {data_size} transitions totally.")
    iterator.init_storage(timesteps)
    return _ENV_FACTORY[env_name][0](), iterator
