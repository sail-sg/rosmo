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

"""Environment loop extensions."""
import time
from abc import abstractmethod
from typing import Dict, Optional, Union

import acme
import dm_env
import numpy as np
from acme.utils import observers as observers_lib
from acme.utils import signals
from importlib_metadata import collections

Number = Union[int, float]


class EvaluationLoop(acme.EnvironmentLoop):
    """Evaluation env-actor loop."""

    def run(
        self, num_episodes: Optional[int] = None, num_steps: Optional[int] = None
    ) -> None:
        """Run the evaluation loop."""
        if not (num_episodes is None or num_steps is None):
            raise ValueError('Either "num_episodes" or "num_steps" should be None.')

        def should_terminate(episode_count: int, step_count: int) -> bool:
            return (num_episodes is not None and episode_count >= num_episodes) or (
                num_steps is not None and step_count >= num_steps
            )

        episode_count, step_count = 0, 0
        all_results: Dict[str, list] = collections.defaultdict(list)
        with signals.runtime_terminator():
            while not should_terminate(episode_count, step_count):
                result = self.run_episode()
                episode_count += 1
                step_count += result["episode_length"]
                for k, v in result.items():
                    all_results[k].append(v)
            # Log the averaged results from all episodes.
            self._logger.write({k: np.mean(v) for k, v in all_results.items()})


class ExtendedEnvLoopObserver(observers_lib.EnvLoopObserver):
    """Extended env loop observer."""

    @abstractmethod
    def step(self) -> None:
        """Steps the observer."""

    @abstractmethod
    def restore(self, learning_step: int) -> None:
        """Restore the observer state."""


class LearningStepObserver(ExtendedEnvLoopObserver):
    """Observer to record the learning steps."""

    def __init__(self) -> None:
        """Init observer."""
        super().__init__()
        self._learning_step = 0
        self._eval_step = 0
        self._status = 1  # {0: train, 1: eval}
        self._train_elapsed = 0.0
        self._last_time: Optional[float] = None

    def step(self) -> None:
        """Steps the observer."""
        self._learning_step += 1

        if self._status == 0 and self._last_time:
            self._train_elapsed += time.time() - self._last_time
        if self._status == 1:
            self._status = 0

        self._last_time = time.time()

    def observe_first(self, env: dm_env.Environment, timestep: dm_env.TimeStep) -> None:
        """Observes the initial state, setting states."""
        self._status = 1
        self._eval_step += 1

    def observe(
        self, env: dm_env.Environment, timestep: dm_env.TimeStep, action: np.ndarray
    ) -> None:
        """Records one environment step, dummy."""

    def get_metrics(self) -> Dict[str, Number]:
        """Returns metrics collected for the current episode."""
        return {
            "step": self._learning_step,
            "eval_step": self._eval_step,
            "learning_time": self._train_elapsed,
        }

    def restore(self, learning_step: int) -> None:
        """Restore."""
        self._learning_step = learning_step
