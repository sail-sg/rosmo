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

"""Atari experiment entry."""

import os
import pickle
import random
import time
from typing import Dict, Iterator, List, Optional, Tuple

import dm_env
import jax
import numpy as np
import tensorflow as tf
import wandb
from absl import app, flags, logging
from acme import EnvironmentLoop
from acme.specs import make_environment_spec
from acme.utils.loggers import Logger

from experiment.atari.config import get_config
from rosmo.agent.actor import RosmoEvalActor
from rosmo.agent.learning import RosmoLearner
from rosmo.agent.network import Networks, make_atari_networks
from rosmo.data import atari_env_loader
from rosmo.env_loop_observer import (
    EvaluationLoop,
    ExtendedEnvLoopObserver,
    LearningStepObserver,
)
from rosmo.loggers import logger_fn
from rosmo.profiler import Profiler
from rosmo.type import ActorOutput

# ===== Flags. ===== #
FLAGS = flags.FLAGS
flags.DEFINE_boolean("debug", True, "Debug run.")
flags.DEFINE_boolean("profile", False, "Profile codes.")
flags.DEFINE_boolean("use_wb", False, "Use WB to log.")
flags.DEFINE_string("user", "username", "Wandb user id.")
flags.DEFINE_string("project", "rosmo", "Wandb project id.")

flags.DEFINE_string("exp_id", None, "Experiment id.", required=True)
flags.DEFINE_string("env", None, "Environment name to run.", required=True)
flags.DEFINE_integer("seed", int(time.time()), "Random seed.")

flags.DEFINE_boolean("sampling", False, "Whether to sample policy target.")
flags.DEFINE_integer("num_simulations", 4, "Simulation budget.")
flags.DEFINE_enum("algo", "rosmo", ["rosmo", "mzu"], "Algorithm to use.")
flags.DEFINE_integer(
    "search_depth",
    0,
    "Depth of Monte-Carlo Tree Search (only for mzu), \
        defaults to num_simulations.",
)

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


# ===== Eval Actor-Env Loop & Observer. ===== #
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


# ===== Environment & Dataloader. ===== #
def get_env_data_loader(config) -> Tuple[dm_env.Environment, Iterator]:
    """Get environment and trajectory data loader."""
    trajectory_length = config["unroll_steps"] + config["td_steps"] + 1
    environment, dataset = atari_env_loader(
        env_name=config["game_name"],
        run_number=config["run_number"],
        dataset_dir=config["data_dir"],
        stack_size=config["stack_size"],
        data_percentage=config["data_percentage"],
        trajectory_length=trajectory_length,
        shuffle_num_steps=5000 if FLAGS.debug else 50000,
    )

    def transform_timesteps(steps: Dict[str, np.ndarray]) -> ActorOutput:
        return ActorOutput(
            observation=steps["observation"],
            reward=steps["reward"],
            is_first=steps["is_first"],
            is_last=steps["is_last"],
            action=steps["action"],
        )

    dataset = (
        dataset.repeat()
        .batch(config["batch_size"])
        .map(transform_timesteps)
        .prefetch(tf.data.AUTOTUNE)
    )
    options = tf.data.Options()
    options.threading.max_intra_op_parallelism = 1
    dataset = dataset.with_options(options)
    iterator = dataset.as_numpy_iterator()
    return environment, iterator


# ===== Network. ===== #
def get_networks(config, environment) -> Networks:
    """Get environment-specific networks."""
    environment_spec = make_environment_spec(environment)
    logging.info(environment_spec)
    networks = make_atari_networks(
        env_spec=environment_spec,
        channels=config["channels"],
        num_bins=config["num_bins"],
        output_init_scale=config["output_init_scale"],
        blocks_representation=config["blocks_representation"],
        blocks_prediction=config["blocks_prediction"],
        blocks_transition=config["blocks_transition"],
        reduced_channels_head=config["reduced_channels_head"],
        fc_layers_reward=config["fc_layers_reward"],
        fc_layers_value=config["fc_layers_value"],
        fc_layers_policy=config["fc_layers_policy"],
    )
    return networks


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
    platform = jax.lib.xla_bridge.get_backend().platform
    num_devices = jax.device_count()
    logging.warn(f"Compute platform: {platform} with {num_devices} devices.")
    logging.info(f"Debug mode: {FLAGS.debug}")
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # ===== Setup. ===== #
    cfg = get_config(FLAGS.env)
    env, dataloader = get_env_data_loader(cfg)
    networks = get_networks(cfg, env)

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
        get_logger_fn(
            cfg["exp_full_name"],
            "evaluator",
            is_eval=True,
            config=cfg,
        ),
    )
    evaluate_episodes = 2 if FLAGS.debug else cfg["evaluate_episodes"]

    init_step = 0
    save_path = os.path.join("./checkpoint", cfg["exp_full_name"])
    os.makedirs(save_path, exist_ok=True)
    if FLAGS.profile:
        profile_dir = "./profile"
        os.makedirs(profile_dir, exist_ok=True)
        profiler = Profiler(profile_dir, cfg["exp_full_name"], with_jax=True)

    if FLAGS.use_wb and not (FLAGS.debug or FLAGS.profile):
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

        if (i + 1) % cfg["save_period"] == 0:
            with open(os.path.join(save_path, f"ckpt_{i}.pkl"), "wb") as f:
                pickle.dump(learner.save(), f)
        if (i + 1) % cfg["eval_period"] == 0:
            actor.update_params(learner.save().params)
            eval_loop.run(evaluate_episodes)

        if FLAGS.profile:
            if i == 100:
                profiler.start()
            if i == 200:
                profiler.stop_and_save()
                break
        elif FLAGS.debug:
            actor.update_params(learner.save().params)
            eval_loop.run(evaluate_episodes)
            break

    # ===== Cleanup. ===== #
    learner._logger.close()
    eval_loop._logger.close()
    del env, networks, dataloader, learner, observers, actor, eval_loop


if __name__ == "__main__":
    app.run(main)
