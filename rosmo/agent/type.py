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

"""Types."""
from typing import NamedTuple

from acme.jax import networks as networks_lib

from rosmo.type import Array


class Params(NamedTuple):
    """Agent parameters."""

    representation: networks_lib.Params
    transition: networks_lib.Params
    prediction: networks_lib.Params


class AgentOutput(NamedTuple):
    """Agent prediction output."""

    state: Array
    policy_logits: Array
    value_logits: Array
    value: Array
    reward_logits: Array
    reward: Array
