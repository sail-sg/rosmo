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

"""Atari experiment configs."""
from copy import deepcopy
from typing import Dict

from absl import flags, logging
from ml_collections import ConfigDict
from wandb.util import generate_id

FLAGS = flags.FLAGS


# ===== Configurations ===== #
def get_config(game_name: str) -> Dict:
    """Get experiment configurations."""
    config = deepcopy(CONFIG)
    config["seed"] = FLAGS.seed
    config["benchmark"] = "atari"
    config["sampling"] = FLAGS.sampling
    config["use_mcts"] = FLAGS.use_mcts
    config["game_name"] = game_name
    config["num_simulations"] = FLAGS.num_simulations
    config["batch_size"] = 16 if FLAGS.debug else config["batch_size"]
    exp_full_name = f"{FLAGS.exp_id}_{game_name}_" + generate_id()
    config["exp_full_name"] = exp_full_name
    logging.info(f"Configs: {config}")
    return config


CONFIG = ConfigDict(
    {
        "data_dir": "./datasets/rl_unplugged/tensorflow_datasets",
        "run_number": 1,
        "data_percentage": 100,
        "batch_size": 512,
        "stack_size": 4,
        "unroll_steps": 5,
        "td_steps": 5,
        "num_bins": 601,
        "channels": 64,
        "blocks_representation": 6,
        "blocks_prediction": 2,
        "blocks_transition": 2,
        "reduced_channels_head": 128,
        "fc_layers_reward": [128, 128],
        "fc_layers_value": [128, 128],
        "fc_layers_policy": [128, 128],
        "output_init_scale": 0.0,
        "discount_factor": 0.997**4,
        "clipping_threshold": 1.0,
        "evaluate_episodes": 2,
        "log_interval": 400,
        "learning_rate": 7e-4,
        "warmup_steps": 1_000,
        "learning_rate_decay": 0.1,
        "weight_decay": 1e-4,
        "max_grad_norm": 5.0,
        "target_update_interval": 200,
        "value_coef": 0.25,
        "policy_coef": 1.0,
        "behavior_coef": 0.2,
        "save_period": 10_000,
        "eval_period": 1_000,
        "total_steps": 200_000,
    }
)
