
import logging

from omegaconf import OmegaConf
from nemo.core.config import hydra_runner

from morsecodetoolkit.data import SyntheticMorseDataset


@hydra_runner(config_path="conf", config_name="english")
def main(cfg):
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    logging.info(f"Hydra config: {OmegaConf.to_yaml(cfg)}")

    dataset: SyntheticMorseDataset = SyntheticMorseDataset(**cfg.dataset)

    dataset.synthesize(cfg.output_dir)


if __name__ == '__main__':
    main()

