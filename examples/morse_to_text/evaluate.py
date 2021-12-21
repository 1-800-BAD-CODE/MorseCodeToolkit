
"""
This script may contain excerpts from https://github.com/NVIDIA/NeMo/blob/main/examples/asr/speech_to_text_infer.py
"""

import argparse
import logging

import torch
from tqdm import tqdm

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.metrics.wer import WER, word_error_rate


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Runs inference and computes WER of a morse corpus given a trained morse decoder model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("model", help="Path to morse decoder .nemo or .ckpt model.")
    parser.add_argument("manifest", help="Path to a labeled morse corpus.")

    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--cer", action='store_true', help="Compute CER instead of WER.")

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

    # Use an AudioToCharDataset from NeMo library to map morse signals to predicted characters
    m.setup_test_data(
        test_data_config={
            "batch_size": args.batch_size,
            "dataset_params": {
                "_target_": "nemo.collections.asr.data.audio_to_text.AudioToCharDataset",
                "sample_rate": m.cfg.preprocessor.sample_rate,
                "manifest_filepath": args.manifest,
                "labels": m.decoder.vocabulary,
                "normalize": False,
                "trim": False
            }
        }
    )

    # Resolve label map and set up WER computer
    labels_map = {i: token for i, token in enumerate(m.decoder.vocabulary)}
    wer = WER(vocabulary=m.decoder.vocabulary)
    # Run greedy decoding and collect all hyp/refs
    hypotheses = []
    references = []
    for test_batch in tqdm(m.test_dataloader()):
        test_batch = [x.to(device) for x in test_batch]
        signals, signal_lengths, all_seq_ids, all_seq_lengths = test_batch
        log_probs, encoded_len, greedy_predictions = m(input_signal=signals, input_signal_length=signal_lengths)
        hypotheses += wer.ctc_decoder_predictions_tensor(greedy_predictions)
        for batch_ind in range(len(greedy_predictions)):
            seq_len = all_seq_lengths[batch_ind].cpu().detach().numpy()
            seq_ids = all_seq_ids[batch_ind].cpu().detach().numpy()
            reference = ''.join([labels_map[c] for c in seq_ids[0:seq_len]])
            references.append(reference)
    wer_value = word_error_rate(hypotheses=hypotheses, references=references, use_cer=args.cer)

    tag = "CER" if args.cer else "WER"
    logging.info(f"{tag}: {wer_value:0.1%}")


if __name__ == '__main__':
    main()
