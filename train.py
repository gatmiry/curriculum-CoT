import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional
import wandb
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

@hydra.main(version_base="1.3", config_path="./configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    print('Starting training...')
    if cfg.get("debug", False):
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger to attach...")
        debugpy.wait_for_client()

    print(OmegaConf.to_yaml(cfg, resolve=True))

    #wandb.init(
    #    config=OmegaConf.to_container(cfg),
    #    **OmegaConf.to_container(cfg.wandb_config, resolve=True)
    #)

    train=hydra.utils.instantiate(cfg.trainer)
    train.fit()


if __name__ == "__main__":
    main()