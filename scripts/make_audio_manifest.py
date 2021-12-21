
import argparse
import glob
import json
import logging
import os
from pathlib import Path
from typing import Dict


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Creates a simple manifest of audio files, sufficient for being used as background noise in the "
                    "synthetic data sets used in this project.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "audio_dir",
        help="Path to a directory containing audio files. All files found will be included in the output manifest."
    )
    parser.add_argument("--file-ext", default="wav", help="File extension to search for.")
    parser.add_argument("--output-file", default=None, help="If none, use <audio_dir>/../manifest.json")

    args: argparse.Namespace = parser.parse_args()
    return args


def main():
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    args = get_args()
    audio_dir = Path(args.audio_dir).absolute()

    output_file = args.output_file
    if output_file is None:
        audio_parent = audio_dir.parent.absolute()
        output_file = os.path.join(audio_parent, "manifest.json")

    with open(output_file, "w") as writer:
        glob_ptn = f"{audio_dir}/**/*.{args.file_ext}"
        p: str
        for p in glob.glob(glob_ptn, recursive=True):
            manifest_entry: Dict[str, str] = {
                "audio_filepath": p
            }
            writer.write(f"{json.dumps(manifest_entry)}\n")


if __name__ == "__main__":
    main()
