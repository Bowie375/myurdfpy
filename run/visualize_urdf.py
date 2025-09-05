import os
import time
from pathlib import Path

import hydra
import logging
import omegaconf

from myurdfpy.viz import Visualizer
from myurdfpy.utils import dump_cfg

_logger = logging.getLogger(__name__)

@hydra.main(config_path="../config", config_name=Path(__file__).stem, version_base="1.1")
def main(cfg: omegaconf.DictConfig):

    for logger_name in cfg.disable_loggers:
        logging.getLogger(logger_name).disabled = True
    _logger.info(f"pid:{os.getpid()}")

    dump_cfg(".hydra/visualize_robot.yaml", cfg.visualize)

    viz = Visualizer(**cfg.visualize)
    viz.run()

    # Keep the server alive
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
