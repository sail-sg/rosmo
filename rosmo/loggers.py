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

"""Logger utils."""
import os
import re
import threading
from typing import Any, Callable, Dict, Mapping, Optional

import wandb
from acme.utils.loggers import Logger, aggregators
from acme.utils.loggers import asynchronous as async_logger
from acme.utils.loggers import base, csv, filters, terminal
from acme.utils.loggers.tf_summary import TFSummaryLogger

from rosmo.data.bsuite import SCORES
from rosmo.data.rlu_atari import BASELINES


class WBLogger(base.Logger):
    """Logger for W&B."""

    def __init__(
        self,
        scope: Optional[str] = None,
    ) -> None:
        """Init WB logger."""
        self._lock = threading.Lock()
        self._scope = scope

    def write(self, data: Dict[str, Any]) -> None:
        """Log the data."""
        step = data.pop("step", None)
        if step is not None:
            step = int(step)
        with self._lock:
            if self._scope is None:
                wandb.log(
                    data,
                    step=step,
                )
            else:
                wandb.log(
                    {f"{self._scope}/{k}": v for k, v in data.items()},
                    step=step,
                )

    def close(self) -> None:
        """Close WB logger."""
        with self._lock:
            wandb.finish()


class ResultFilter(base.Logger):
    """Postprocessing for normalized score."""

    def __init__(self, to: base.Logger, game_name: str):
        """Init result filter."""
        self._to = to
        game_name = re.sub(r"(?<!^)(?=[A-Z])", "_", game_name).lower()
        if game_name in BASELINES:
            # Atari
            random_score = BASELINES[game_name]["random"]
            dqn_score = BASELINES[game_name]["online_dqn"]
        elif game_name in SCORES:
            # BSuite
            random_score = SCORES[game_name]["random"]
            dqn_score = SCORES[game_name]["online_dqn"]

        def normalizer(score: float) -> float:
            return (score - random_score) / (dqn_score - random_score)

        self._normalizer = normalizer

    def write(self, data: base.LoggingData) -> None:
        """Write to logger."""
        if "episode_return" in data:
            data = {
                **data,
                "normalized_score": self._normalizer(data.get("episode_return", 0)),
            }
        self._to.write(data)

    def close(self) -> None:
        """Close logger."""
        self._to.close()


def make_sail_logger(
    exp_name: str,
    label: str,
    save_data: bool = True,
    save_dir: str = "./logs",
    use_tb: bool = False,
    tb_dir: Optional[str] = None,
    use_wb: bool = False,
    config: Optional[dict] = None,
    time_delta: float = 1.0,
    asynchronous: bool = False,
    print_fn: Optional[Callable[[str], None]] = None,
    serialize_fn: Optional[Callable[[Mapping[str, Any]], str]] = base.to_numpy,
) -> base.Logger:
    """Makes a logger for SAILors.

    Args:
      exp_name: Name of the experiment.
      label: Name to give to the logger.
      save_data: Whether to persist data.
      save_dir: Directory to save log data.
      use_tb: Whether to use TensorBoard.
      tb_dir: Tensorboard directory.
      use_wb: Whether to use Weights and Biases.
      config: Experiment configurations.
      time_delta: Time (in seconds) between logging events.
      asynchronous: Whether the write function should block or not.
      print_fn: How to print to terminal (defaults to print).
      serialize_fn: An optional function to apply to the write inputs before
        passing them to the various loggers.

    Returns:
      A logger object that responds to logger.write(some_dict).
    """
    if not print_fn:
        print_fn = print
    terminal_logger = terminal.TerminalLogger(label=label, print_fn=print_fn)

    loggers = [terminal_logger]

    if save_data:
        os.makedirs(save_dir, exist_ok=True)
        fd = open(os.path.join(save_dir, f"{exp_name}.csv"), "a")
        loggers.append(
            csv.CSVLogger(
                directory_or_file=fd,
                label=exp_name,
                add_uid=False,
                flush_every=2,
            )
        )

    if use_wb:
        wb_logger = WBLogger(scope=label)
        wb_logger = filters.TimeFilter(wb_logger, time_delta)
        loggers.append(wb_logger)

    if use_tb:
        if tb_dir is None:
            tb_dir = "./tblogs"
        loggers.append(TFSummaryLogger(tb_dir, label))

    # Dispatch to all writers and filter Nones and by time.
    logger = aggregators.Dispatcher(loggers, serialize_fn)
    logger = filters.NoneFilter(logger)

    if config:
        logger = ResultFilter(logger, game_name=config["game_name"])

    if asynchronous:
        logger = async_logger.AsyncLogger(logger)
    logger = filters.TimeFilter(logger, 5.0)

    return logger


def logger_fn(
    exp_name: str,
    label: str,
    save_data: bool = False,
    use_tb: bool = True,
    use_wb: bool = True,
    config: Optional[dict] = None,
    time_delta: float = 15.0,
) -> Logger:
    """Get logger function.

    Args:
        exp_name (str): Experiment name.
        label (str): Experiment label.
        save_data (bool, optional): Whether to save data. Defaults to False.
        use_tb (bool, optional): Whether to use TB. Defaults to True.
        use_wb (bool, optional): Whether to use WB. Defaults to True.
        config (Optional[dict], optional): Experiment configurations. Defaults to None.
        time_delta (float, optional): Time delta to emit logs. Defaults to 15.0.

    Returns:
        Logger: Logger.
    """
    tb_path = os.path.join("./tblogs", exp_name)
    return make_sail_logger(
        exp_name=exp_name,
        label=label,
        save_data=save_data,
        save_dir="./logs",
        use_tb=use_tb,
        tb_dir=tb_path,
        use_wb=use_wb,
        config=config,
        time_delta=time_delta,  # Applied to W&B.
        print_fn=print,
    )
