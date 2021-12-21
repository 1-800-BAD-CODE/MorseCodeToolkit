
import torch
import os
import logging
import argparse
import librosa
import torchaudio
import torch.nn as nn
import matplotlib.pyplot as plt


def get_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=".",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("audio_path", help="Audio file to extract features from.")
    parser.add_argument("output_path", help="Path to save the resulting image to.")

    args: argparse.Namespace = parser.parse_args()
    return args


def main():
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    args = get_args()

    np_signal, sr = librosa.load(path=args.audio_path, sr=None)
    signal: torch.Tensor = torch.tensor(np_signal).unsqueeze(0)

    extractor: nn.Module = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=80,
        n_fft=256 if sr == 8000 else 512,
        win_length=int(sr * 0.02),
        hop_length=int(sr * 0.01),
        normalized=True
    )
    mel_spec = extractor(signal)
    mel_spec = mel_spec.squeeze(0)
    mel_spec = mel_spec.clamp(min=1e-10).log()

    plt.imsave(fname=args.output_path, arr=mel_spec.numpy(), origin="lower")


if __name__ == "__main__":
    main()
