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

"""Profiling utils."""
import os

import jax
from viztracer import VizTracer


class Profiler:
    """Profiler for python and jax (optional)."""

    def __init__(self, folder: str, name: str, with_jax: bool = False) -> None:
        """Initial method."""

        super().__init__()
        self._name = name
        self._folder = folder
        self._with_jax = with_jax
        self._vistracer = VizTracer(
            output_file=os.path.join(folder, "viztracer", name + ".html"),
            max_stack_depth=3,
        )
        self._jax_folder = os.path.join(folder, "jax_profiler/" + name)

    def start(self) -> None:
        """Start to trace."""
        if self._with_jax:
            jax.profiler.start_trace(self._jax_folder)
        self._vistracer.start()

    def stop(self) -> None:
        """Stop tracing."""
        self._vistracer.stop()
        if self._with_jax:
            jax.profiler.stop_trace()

    def save(self) -> None:
        """Save the results."""
        self._vistracer.save()

    def stop_and_save(self) -> None:
        """Combine stop and save."""
        self.stop()
        self.save()
