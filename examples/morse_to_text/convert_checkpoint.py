
import argparse
import logging
import os
from typing import Dict

from omegaconf import ListConfig, DictConfig
import torch

from morsecodetoolkit.models.ctc_models import EncDecCTCModel


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Converts a .ckpt or .nemo file to a half-precision .nemo file to minimize file size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model")
    parser.add_argument("output_file")

    args: argparse.Namespace = parser.parse_args()
    return args


def _remove_local_paths(cfg: Dict) -> None:
    """Removes any file names or directory paths from a configuration, presumably for releasing models.
    """
    for k, v in cfg.items():
        if isinstance(v, (dict, DictConfig)):
            _remove_local_paths(v)
        elif isinstance(v, str):
            if "/" in v:
                cfg[k] = None
        elif isinstance(v, (list, ListConfig)):
            for i, entry in enumerate(v):
                if isinstance(entry, str) and "/" in entry:
                    cfg[k][i] = None


def main():
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    args = get_args()

    logging.info(f"Restoring from {args.model}")
    m: EncDecCTCModel
    if args.model.endswith(".nemo"):
        m = EncDecCTCModel.restore_from(args.model, map_location=torch.device("cpu"))
    else:
        m = EncDecCTCModel.load_from_checkpoint(args.model, map_location="cpu")

    m.eval().half()

    _remove_local_paths(m.cfg)

    logging.info(f"Saving model to {args.output_file}")
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    m.save_to(args.output_file)


if __name__ == "__main__":
    main()
