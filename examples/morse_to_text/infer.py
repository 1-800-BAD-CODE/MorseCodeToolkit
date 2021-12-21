
import os
import json
import argparse
import logging
from typing import List, Dict

import torch

from nemo.collections.asr.models import ASRModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs inference with a trained morse decoder model given a manifest of morse audio file paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", help="Path to morse decoder .nemo or .ckpt model.")
    parser.add_argument(
        "manifest",
        help="Path to a manifest, where each line contains '{\"audio_filepath\": \"/path/to/audio.wav\"}'."
    )
    parser.add_argument("output_file", help="Path to output file to write.")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")

    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    m: ASRModel
    if args.model.endswith(".nemo"):
        m = ASRModel.restore_from(args.model, map_location=torch.device("cpu"))
    else:
        m = ASRModel.load_from_checkpoint(args.model, map_location="cpu")
    m.eval()
    m.to(device)
    logging.info(f"Successfully loaded model '{args.model}'")
    logging.info(f"Using device {device}")

    logging.info(f"Loading audio files from manifest '{args.manifest}'")
    audio_filepaths: List[str] = []
    with open(args.manifest) as f:
        for line in f:
            d: Dict[str, str] = json.loads(line.strip())
            audio_filepaths.append(d["audio_filepath"])

    logging.info(f"Transcribing {len(audio_filepaths)} audio files.")
    outputs = m.transcribe(paths2audio_files=audio_filepaths)

    logging.info(f"Writing outputs to '{args.output_file}'")
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_file, "w") as writer:
        for output in outputs:
            writer.write(f"{output}\n")


if __name__ == '__main__':
    main()
