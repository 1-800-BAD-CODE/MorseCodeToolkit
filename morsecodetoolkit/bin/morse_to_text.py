
import torch
import logging
import argparse

from nemo.collections.asr.models import ASRModel


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Simple tool to transcribe a single morse audio file. For large-scale decoding and evaluation, see "
                    "`examples/morse_to_text/`.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", help="Path to a .nemo model. Model should be a subclass of NeMo's ASRModel.")
    parser.add_argument("audio_file", help="Path to an audio file.")

    args: argparse.Namespace = parser.parse_args()
    return args


def main():
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    args = get_args()

    # PTL checkpoints will not load as the abstract base class ASRModel
    if not args.model.endswith(".nemo"):
        raise ValueError(f"Expected a .nemo ASRModel; got '{args.model}'.")

    # Load the model, use CPU always since this script only decodes one audio file.
    m: ASRModel = ASRModel.restore_from(args.model, map_location=torch.device("cpu"))
    m = m.eval()
    logging.info(f"Successfully loaded model '{args.model}'")

    # Transcribe using model's abstract method
    logging.info(f"Transcribing '{args.audio_file}'")
    outputs = m.transcribe(paths2audio_files=[args.audio_file])
    outputs = outputs[0]

    # Dump transcription
    logging.info(f"Transcription: '{outputs}'")


if __name__ == "__main__":
    main()
