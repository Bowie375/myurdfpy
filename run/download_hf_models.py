import logging
import os
from pathlib import Path
from typing import Optional

import hydra
import omegaconf
from huggingface_hub import snapshot_download

from myurdfpy.utils import dump_cfg

_logger = logging.getLogger(__name__)


def _optional_list(value: Optional[omegaconf.ListConfig]):
    if value is None:
        return None
    return list(value)


@hydra.main(config_path="../config", config_name=Path(__file__).stem, version_base="1.1")
def main(cfg: omegaconf.DictConfig):
    for logger_name in cfg.disable_loggers:
        logging.getLogger(logger_name).disabled = True

    dump_cfg(".hydra/download_hf_models.yaml", cfg)

    token = None
    if cfg.hf.token_env:
        token = os.getenv(cfg.hf.token_env)
        if token is None:
            _logger.warning(
                "Environment variable '%s' is not set. Proceeding without auth token.",
                cfg.hf.token_env,
            )

    local_dir = os.path.abspath(os.path.expanduser(cfg.download.local_dir))
    os.makedirs(local_dir, exist_ok=True)

    _logger.info("Downloading from %s (%s) to %s", cfg.hf.repo_id, cfg.hf.repo_type, local_dir)
    path = snapshot_download(
        repo_id=cfg.hf.repo_id,
        repo_type=cfg.hf.repo_type,
        revision=cfg.hf.revision,
        allow_patterns=_optional_list(cfg.download.include_patterns),
        ignore_patterns=_optional_list(cfg.download.ignore_patterns),
        local_dir=local_dir,
        local_dir_use_symlinks=cfg.download.local_dir_use_symlinks,
        token=token,
    )
    _logger.info("Download complete: %s", path)


if __name__ == "__main__":
    main()
