from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, asdict, fields, field
from datetime import datetime
from pathlib import Path

import yaml
from cattrs import structure

from local_information.mpi.mpi_funcs import get_mpi_variables

logger = logging.getLogger()
COMM, RANK, SIZE, NAME, PARALLEL = get_mpi_variables()


@dataclass
class Config:
    def __str__(self):
        string = "\n"
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            string += "\t{:<25}: {}\n".format(name, value.__str__())
        return string


@dataclass
class LoggingConfig(Config):
    """Logging configuration:
    logging_folder: folder to save log files
    log_level_console: set the level of logging to the console
    log_level_file: set the level of logging to log files"""

    # logging directory
    logging_folder: str = ""
    # log levels console output
    log_level_console: int = logging.INFO
    # log files output
    log_level_file: int = logging.DEBUG


@dataclass
class RungeKuttaConfig(Config):
    """Dataclass collecting Runge-Kutta parameters"""

    # typ of Petz map to be used: default 'sqrt'. Other options: 'exponential'
    petz_map: str = "sqrt"
    # RK order to which the von-Neumann equation is solved: choices are '45' (i.e. 4 (5)),
    # '23' (i.e. 3 (2)) and '1012' (i.e. 10 (12))
    RK_order: str = "45"
    # maximum allowed error in the RK method
    max_error: float = 1e-8
    # adaptive step size
    step_size: float = 0.015

    def __post_init__(self):
        assert self.max_error > 0, ValueError("max_error must be >0")


@dataclass
class MinimizationConfig(Config):
    """
    Dataclass collecting all parameters related to minimization of local information.
    """

    # percent of information allowed at max_l before the minimization is triggered
    minimization_threshold: float = 1e-2
    # parameter used while minimizing the information. Measures the relative reduction in information during one
    # optimisation step (with respect to the initial total information at the level of minimization). Default is 1e-5
    minimization_tolerance: float = 1e-5
    # parameter used in the conjugate gradient routine to accept convergence
    conjugate_gradient_tolerance: float = 1e-5
    # damping used to improve convergence in during conjugate gradient optimization
    conjugate_gradient_damping: float = 0.1

    def __post_init__(self):
        assert all(
            [
                0 < getattr(MinimizationConfig, field.name) < 1
                for field in fields(MinimizationConfig)
            ]
        )


@dataclass
class TimeEvolutionConfig(Config):
    """
    Dataclass collecting the relevant parameters for time evolution.
    """

    # level time evolution is carried out
    max_l: int = 5
    # level of minimization
    min_l: int = 3
    # parameter to specify the precision of the calculation: when information reaches
    # a value > threshold on some scale, dyn_max_l is updated.
    # This parameter should only be set to values >0 if you are interested in *short time* dynamics
    # Otherwise a competition of scales can arise and minimization might not be triggered!
    update_dyn_max_l_threshold: float = 0.0
    # maximum decimal value for which to density matrices are assumed to be identical.
    system_size_tol: float = 1e-12
    # shift to ensure stability in the presence of density matrices with small eigenvalues
    shift: float = 10.0
    # save configs
    save_checkpoint: bool = True
    # folder to store data
    checkpoint_folder: str = "./data"
    # logging configuration
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    # runge kutta parameters
    runge_kutta_config: RungeKuttaConfig = field(default_factory=RungeKuttaConfig)
    # parameters used in minimization of local information
    minimization_config: MinimizationConfig = field(default_factory=MinimizationConfig)

    def __post_init__(self):
        # enforce meaningful values
        assert 1 > self.update_dyn_max_l_threshold >= 0, ValueError(
            "threshold must be <1 and >0"
        )
        assert self.system_size_tol > 0, ValueError("system_size_tol must be >0")
        assert self.max_l != self.min_l, ValueError("max_l and min_l must be different")
        assert self.shift >= 0, ValueError("shift must no be <=0")

        # setup logging
        if not self.logging_config.logging_folder:
            self.logging_config.logging_folder = self.checkpoint_folder + "/logging"

        if not logger.hasHandlers():
            configure_loging(
                logger,
                folder=self.logging_config.logging_folder,
                log_level_file=self.logging_config.log_level_file,
                log_level_console=self.logging_config.log_level_console,
            )

        if self.save_checkpoint:
            if Path(self.checkpoint_folder).is_dir():
                logger.warning(
                    f"checkpoint_folder '{self.checkpoint_folder}' already exists"
                )

        if self.update_dyn_max_l_threshold > 0:
            logger.warning(
                "setting update_dyn_max_l_threshold > 0 can lead to obscured dynamics at late times"
            )

    @classmethod
    def from_yaml(cls, directory: str) -> TimeEvolutionConfig:
        p = Path(directory)
        config_path = p / "config.yaml"

        if config_path.is_file():
            with open(config_path, "r") as file:
                loaded = yaml.safe_load(file)

        return structure(loaded, cls)

    def to_yaml(self, directory: str | None = None):
        if directory:
            p = Path(directory)
        else:
            p = Path(self.checkpoint_folder)

        p.mkdir(parents=True, exist_ok=True)
        config_path = p / "config.yaml"
        config = asdict(self)
        with open(config_path, "w") as file:
            yaml.dump(config, file)


def configure_loging(
    logger, folder: str, log_level_console=logging.INFO, log_level_file=logging.DEBUG
):
    # set up logging to file and console
    # https://stackoverflow.com/questions/48304157/seeing-empty-log-file-after-logging
    # set the log level to ERROR for all but root
    msg = "starting process {0} of {1} on {2}.\n"
    sys.stdout.write(msg.format(RANK + 1, SIZE, NAME))
    sys.stdout.flush()

    if RANK != 0:
        log_level_console = logging.ERROR

    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    date = now.strftime("%d-%m-%Y-%H-%M-%S")
    filename = "run-" + date + ".log"
    filepath = p / filename

    logger.setLevel(logging.DEBUG)

    # create file handler
    fh = logging.FileHandler(filepath, mode="w")
    fh.setLevel(log_level_file)

    # create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(log_level_console)

    # create formatter and add it to the handlers
    console_formatter = logging.Formatter(
        "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if SIZE > 1:
        # logs each mpi process in log files
        rank_filter = RankContextFilter()
        logger.addFilter(rank_filter)
        fh.addFilter(rank_filter)
        ch.addFilter(rank_filter)
        file_formatter = logging.Formatter(
            "%(asctime)s,%(msecs)03d %(levelname)-8s [RANK %(RANK)d :%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(file_formatter)
        # add the handlers to the logger
        logger.addHandler(fh)
    else:
        file_formatter = logging.Formatter(
            "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(file_formatter)
        # add the handlers to the logger
        logger.addHandler(fh)

    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    pass


class RankContextFilter(logging.Filter):
    """
    This is a filter which injects contextual information into the log. Adds the RANK variable to each log
    message
    """

    def filter(self, record):
        record.RANK = RANK
        return True
