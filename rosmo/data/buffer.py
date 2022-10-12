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

"""In-memory buffer."""
from typing import Any, Iterator

import numpy as np
import tree


class UniformBuffer(Iterator):
    """Buffer that supports uniform sampling."""

    def __init__(
        self, min_size: int, max_size: int, traj_len: int, batch_size: int = 2
    ) -> None:
        """Init the buffer."""
        self._min_size = min_size
        self._max_size = max_size
        self._traj_len = traj_len
        self._timestep_storage = None
        self._n = 0
        self._idx = 0
        self._bs = batch_size
        self._static_buffer = False

    def __next__(self) -> Any:
        """Get the next sample.

        Returns:
            Any: Sampled data.
        """
        return self.sample(self._bs)

    def init_storage(self, timesteps: Any) -> None:
        """Initialize the buffer.

        Args:
            timesteps (Any): Timesteps that contain the whole dataset.
        """
        assert self._timestep_storage is None
        size = timesteps.observation.shape[0]
        assert self._min_size <= size <= self._max_size
        self._n = size
        self._timestep_storage = timesteps
        self._static_buffer = True

    def sample(self, batch_size: int) -> Any:
        """Sample a batch of data.

        Args:
            batch_size (int): Batch size to sample.

        Returns:
            Any: Sampled data.
        """
        if batch_size + self._traj_len > self._n:
            return None
        start_indices = np.random.choice(
            self._n - self._traj_len, batch_size, replace=False
        )
        all_indices = start_indices[:, None] + np.arange(self._traj_len + 1)[None]
        base_idx = 0 if self._n < self._max_size else self._idx
        all_indices = (all_indices + base_idx) % self._max_size
        trajectories = tree.map_structure(
            lambda a: a[all_indices], self._timestep_storage
        )
        return trajectories

    def full(self) -> bool:
        """Test if the buffer is full.

        Returns:
            bool: True if the buffer is full.
        """
        return self._n == self._max_size

    def ready(self) -> bool:
        """Test if the buffer has minimum size.

        Returns:
            bool: True if the buffer is ready.
        """
        return self._n >= self._min_size

    @property
    def size(self) -> int:
        """Get the size of the buffer.

        Returns:
            int: Buffer size.
        """
        return self._n

    def _preallocate(self, item: Any) -> Any:
        return tree.map_structure(
            lambda t: np.empty((self._max_size,) + t.shape, t.dtype), item
        )


def assign(array: Any, index: Any, data: Any) -> None:
    """Update array.

    Args:
        array (Any): Array to be updated.
        index (Any): Index of updates.
        data (Any): Update data.
    """
    array[index] = data
