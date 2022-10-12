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

"""BSuite experiment entry."""
import random
import time
from typing import Dict, List, Optional, Tuple

import jax
import numpy as np
import wandb
from absl import app, flags, logging
from acme import EnvironmentLoop
from acme.specs import make_environment_spec
from acme.utils.loggers import Logger

from experiment.bsuite.config import get_config
from rosmo.agent.actor import RosmoEvalActor
from rosmo.agent.learning import RosmoLearner
from rosmo.agent.network import get_bsuite_networks
from rosmo.data import bsuite_env_loader
from rosmo.env_loop_observer import (
    EvaluationLoop,
    ExtendedEnvLoopObserver,
    LearningStepObserver,
)
from rosmo.loggers import logger_fn

# ===== Flags. ===== #
FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", True, "Debug run.")
flags.DEFINE_boolean("use_wb", False, "Use WB to log.")
flags.DEFINE_string("user", "username", "Wandb user id.")
flags.DEFINE_string("project", "rosmo", "Wandb project id.")

flags.DEFINE_string("exp_id", None, "Experiment id.", required=True)
flags.DEFINE_string("env", None, "Environment name to run.", required=True)
flags.DEFINE_integer("seed", int(time.time()), "Random seed.")


# ===== Learner. ===== #
def get_learner(config, networks, data_iterator, logger) -> RosmoLearner:
    """Get ROSMO learner."""
    learner = RosmoLearner(
        networks,
        demonstrations=data_iterator,
        config=config,
        logger=logger,
    )
    return learner


# ===== Eval Actor-Env Loop. ===== #
def get_actor_env_eval_loop(
    config, networks, environment, observers, logger
) -> Tuple[RosmoEvalActor, EnvironmentLoop]:
    """Get actor, env and evaluation loop."""
    actor = RosmoEvalActor(
        networks,
        config,
    )
    eval_loop = EvaluationLoop(
        environment=environment,
        actor=actor,
        logger=logger,
        should_update=False,
        observers=observers,
    )
    return actor, eval_loop


def get_env_loop_observers() -> List[ExtendedEnvLoopObserver]:
    """Get environment loop observers."""
    observers = []
    learning_step_ob = LearningStepObserver()
    observers.append(learning_step_ob)
    return observers


# ===== Misc. ===== #
def get_logger_fn(
    exp_full_name: str,
    job_name: str,
    is_eval: bool = False,
    config: Optional[Dict] = None,
) -> Logger:
    """Get logger function."""
    save_data = is_eval
    return logger_fn(
        exp_name=exp_full_name,
        label=job_name,
        save_data=save_data and not FLAGS.debug,
        use_tb=False,
        use_wb=FLAGS.use_wb and not FLAGS.debug,
        config=config,
    )


def main(_):
    """Main program."""
    logging.info(f"Debug mode: {FLAGS.debug}")
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    platform = jax.lib.xla_bridge.get_backend().platform
    num_devices = jax.lib.xla_bridge.device_count()
    logging.warn(f"Compute platform: {platform} with {num_devices} devices.")

    # ===== Setup. ===== #
    cfg = get_config(FLAGS.env)

    env, dataloader = bsuite_env_loader(
        env_name=FLAGS.env,
        dataset_dir=cfg["data_dir"],
        data_percentage=cfg["data_percentage"],
        batch_size=cfg["batch_size"],
        trajectory_length=cfg["td_steps"] + cfg["unroll_steps"] + 1,
    )
    networks = get_bsuite_networks(make_environment_spec(env), cfg)

    # ===== Essentials. ===== #
    learner = get_learner(
        cfg,
        networks,
        dataloader,
        get_logger_fn(
            cfg["exp_full_name"],
            "learner",
            config=cfg,
        ),
    )
    observers = get_env_loop_observers()
    actor, eval_loop = get_actor_env_eval_loop(
        cfg,
        networks,
        env,
        observers,
        get_logger_fn(cfg["exp_full_name"], "evaluator", is_eval=True, config=cfg),
    )
    evaluate_episodes = 2 if FLAGS.debug else cfg["evaluate_episodes"]

    # ===== Restore. ===== #
    init_step = 0
    if FLAGS.use_wb and not FLAGS.debug:
        wb_name = cfg["exp_full_name"]
        wb_cfg = cfg.to_dict()
        wandb.init(
            project=FLAGS.project,
            entity=FLAGS.user,
            name=wb_name,
            config=wb_cfg,
            sync_tensorboard=False,
        )

    # ===== Training Loop. ===== #
    for i in range(init_step + 1, cfg["total_steps"]):
        learner.step()
        for ob in observers:
            ob.step()

        if FLAGS.debug or (i + 1) % cfg["eval_period"] == 0:
            actor.update_params(learner.save().params)
            eval_loop.run(evaluate_episodes)

        if FLAGS.debug:
            break

    # ===== Cleanup. ===== #
    learner._logger.close()
    eval_loop._logger.close()
    del env, networks, dataloader, learner, observers, actor, eval_loop


if __name__ == "__main__":
    app.run(main)
