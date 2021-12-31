
import logging
import argparse
import random
from typing import List

import soundfile
from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

from morsecodetoolkit.util import functional
from morsecodetoolkit.alphabet import MorseAlphabet, Symbol


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generates one morse audio signal and saves it to a .wav file. Intended to be a tool for quick "
                    "tests and usage demonstration. For synthesizing large corpora see `examples/synthesize_dataset/`."
    )

    parser.add_argument("text", help="Sentence to generate morse signal from.")
    parser.add_argument("output_filepath")

    alphabet_opts = parser.add_argument_group("Alphabet-related options")
    alphabet_opts.add_argument(
        "--alphabet-name", default="international",
        help="Load build-in alphabet from resource based on this name. If --alphabet-yaml is given, it has priority "
             "over this option and this option is ignored."
    )
    alphabet_opts.add_argument(
        "--alphabet-yaml", default=None,
        help="If set, ignore --alphabet-name and instead load alphabet from this .yaml file."
    )
    alphabet_opts.add_argument("--prosign", action="store_true", help="If set, treat the input text as a prosign.")

    data_opts = parser.add_argument_group("Data options")
    data_opts.add_argument("--background-audio", help="If given, use this audio file as background noise.")
    data_opts.add_argument("--sample-rate", type=int, default=16000, help="Sample rate to generate at.")
    data_opts.add_argument("--tone-freq", type=float, default=500, help="Frequency to create the tones.")
    data_opts.add_argument("--snr-db", type=float, default=10, help="SNR of morse/background noise, in dB.")
    data_opts.add_argument("--gain-db", type=float, default=-10, help="Gain of morse signal, in dB.")
    data_opts.add_argument("--pad-left", type=int, default=500, help="Left-side padding, in ms.")
    data_opts.add_argument("--pad-right", type=int, default=500, help="Right-side padding, in ms.")
    data_opts.add_argument("--rng-seed", type=int, default=1111, help="Seed for RNG.")
    data_opts.add_argument("--window-name", default="hann", help="Window function to apply to tones.")
    data_opts.add_argument("--window-rise-time-ms", type=int, default=12, help="Window rise time, in ms.")
    data_opts.add_argument(
        "--dit-duration", type=int, default=60,
        help="Mean duration of a dit, and the basic unit of length for all other durations."
    )
    data_opts.add_argument(
        "--duration-sigma", type=float, default=5,
        help="The standard deviation of duration lengths, such that when randomly choosing a duration with mean mu, "
             "choose N(mu, sigma). This will be scaled up with the duration; e.g., for a DASH this value will be "
             "multiplied by 3 (as the DASH duration is 3x the DIT duration)."
    )

    args = parser.parse_args()
    return args


def main():
    logging.basicConfig(
         level=logging.INFO,
         format='[%(asctime)s] %(levelname)s : %(message)s',
         datefmt='%H:%M:%S'
    )
    args = get_args()

    # We'll pass this RNG to the functions that use randomness, for reproducable results.
    rng: random.Random = random.Random(args.rng_seed)

    # Resolve alphabet
    alphabet: MorseAlphabet = MorseAlphabet(name=args.alphabet_name, yaml_filepath=args.alphabet_yaml)

    # Convert text to dit/dash sequence
    symbols: List[Symbol] = alphabet.text_to_symbols(args.text, clean=True, is_prosign=args.prosign)

    # Generate the clean (morse-only) audio signal
    morse_signal: AudioSegment = functional.symbols_to_signal(
        symbols=symbols,
        sample_rate=args.sample_rate,
        gain_db=args.gain_db,
        tone_frequency_hz=args.tone_freq,
        dit_duration_ms=args.dit_duration,
        duration_sigma=args.duration_sigma,
        pad_left_ms=args.pad_left,
        pad_right_ms=args.pad_right,
        window_name=args.window_name,
        window_rise_time_ms=args.window_rise_time_ms,
        rng=rng
    )

    # Maybe add some noise to the morse-only audio signal
    if args.background_audio is not None:
        # Load background noise
        noise_signal: AudioSegment = AudioSegment.from_file(
            audio_file=args.background_audio,
            target_sr=args.sample_rate
        )
        # Mix the two together
        morse_signal = functional.mix_background_signal(
            morse_signal=morse_signal,
            noise_signal=noise_signal,
            snr_db=args.snr_db
        )

    # Save final audio file
    soundfile.write(
        file=args.output_filepath,
        data=morse_signal.samples,
        samplerate=morse_signal.sample_rate,
        format="WAV",
        subtype="PCM_16"
    )


if __name__ == "__main__":
    main()
