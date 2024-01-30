import dotenv
import hydra
import glob
from pathlib import Path
import torch
import os
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
from src import utils

log = utils.get_logger(__name__)

@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.utils import metrics
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)
    # Evaluate model
    chkpts = []
    os.chdir("/app")
    path = config.ckpt_folder
    for ck in Path(path).rglob("*.ckpt"):
        if not "last" in str(ck):
            chkpts.append(ck)
    
    # use last chkpt for the settings
    config.ckpt_path = chkpts[-1]
    
    return test(config)


if __name__ == "__main__":
    
    
    main()
