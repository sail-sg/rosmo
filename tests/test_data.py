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

"""Testing data loading."""
import os

from absl import logging
from absl.testing import absltest, parameterized

from rosmo.data import atari_env_loader, bsuite_env_loader

_DATASET_DIR = "./datasets"


class RLUAtari(parameterized.TestCase):
    """Test RL Unplugged Atari data loader."""

    @staticmethod
    def test_data_loader():
        """Test data loader."""
        dataset_dir = os.path.join(_DATASET_DIR, "atari")
        _, dataloader = atari_env_loader(
            env_name="Asterix",
            run_number=1,
            dataset_dir=dataset_dir,
        )
        iterator = iter(dataloader)
        data = next(iterator)
        logging.info(data)


class BSuite(parameterized.TestCase):
    """Test BSuite data loader."""

    @staticmethod
    def test_data_loader():
        """Test data loader."""
        dataset_dir = os.path.join(_DATASET_DIR, "bsuite")
        _, dataloader = bsuite_env_loader(
            env_name="catch",
            dataset_dir=dataset_dir,
        )
        iterator = iter(dataloader)
        _ = next(iterator)


if __name__ == "__main__":
    absltest.main()
