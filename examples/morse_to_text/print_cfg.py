
import argparse
import logging

from omegaconf import OmegaConf, DictConfig

from morsecodetoolkit.models.ctc_models import EncDecCTCModel


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Prints a model's configuration to stdout (for quick inspection).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model")

    args: argparse.Namespace = parser.parse_args()
    return args


def main():
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    args = get_args()

    logging.info(f"Restoring from {args.model}")
    cfg: DictConfig
    if args.model.endswith(".nemo"):
        cfg = EncDecCTCModel.restore_from(args.model, return_config=True)
    else:
        m = EncDecCTCModel.load_from_checkpoint(args.model, map_location="cpu")
        cfg = m.cfg

    logging.info("Model config:")
    logging.info(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
