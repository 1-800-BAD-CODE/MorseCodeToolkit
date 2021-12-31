
import json
import os
import random
import re
from typing import List, Dict, Optional, Tuple, Union

from nemo.core import Dataset, typecheck
from nemo.core.neural_types import NeuralType, AudioSignal, LengthsType, LabelsType
from nemo.collections.asr.parts.preprocessing import AudioAugmentor, process_augmentations, AudioSegment
import soundfile
import torch
from tqdm import tqdm

from morsecodetoolkit.alphabet import MorseAlphabet, Symbol
from morsecodetoolkit.util.functional import mix_background_signal, symbols_to_signal


class SyntheticMorseDataset(Dataset):
    """Generates synthetic morse signals from text and background noises.

    Given a list of text sentences, generates morse audio. Optionally mixes in background noises (e.g. speech, noise).

    The following properties are generated randomly per example:
        * DIT/DASH/PAUSE mean duration (each individual symbol length is randomly drawn using one mean per example)
        * Tone frequency
        * Tone volume
        * SNR (morse/noise)
    To fix these properties, set the min/max values to the same value.

    Furthermore, each generated symbol (dit, dash, or pause) is drawn from a Gaussian distribution based on the
    randomly-selected dit duration for that example.


    Args:
        alphabet: Either a MorseAlphabet instance or a path to a yaml file containing an alphabet. For details on
        the format of this file, see examples in ``morsecodetoolkit.alphabet.data``. Alphabet must be a superset of the
        labels to ensure each label has a known morse encoding.
        labels: List of strings representing the model's labels. Must be a subset of the alphabet.
        text_files: List of files containing one sentence per line. Each line will be used as one example to generate a
        morse signal.
        background_audio_manifests: Optionally a list of text files where each line is a JSON element of the form
        {"audio_filepath": "/path/to/x.wav"} (extra keys are OK, and are ignored). Same as NeMo's ASR manifests but only
        requiring the audio_filepath key.
        sample_rate: Audio sample rate.
        max_words_per_eg: Maximum number of words per example to use. If this is used, the first N words of each
        example will be used. N.B. morse code signals are considerably longer than their speech equivalent.
        prosign_count: If the alphabet contains prosigns, use this many per epoch as examples. E.g. if we load 1000
        texts and this value is set to 100, dataset length will be 1100, and the final 100 will be the alphabet's
        prosigns, repeated as needed to reach 100.
        min_tone_frequency: Lower bound of frequency, in Hz, used for the tones.
        max_tone_frequency: Upper bound of tone, in Hz.
        min_tone_gain_db: Lower bound of tone gain, in dB. Tones are generated with a maximum value of 1.0, so this
        value should be negative to allow reduction of volume.
        max_tone_gain_db: Upper bound of tone gain, in dB. Tones are generated with a maximum value of 1.0, so this
        should be no higher than 0.0.
        min_dit_length_ms: Lower bound of dit length, in ms. ~80ms is common dit length.
        max_dit_length_ms: Upper bound of dir length, in ms.
        duration_sigma_pct: Standard deviation of tone/pause lengths, as a percentage of the tone/pause length
        actually used. E.g. if this is 0.1 and dit duration is 50ms, the actual dit duration will be N(50, 50*0.1).
        mix_background_percent: If using background noise, choose to mix clean signals with given noise this percentage
        of the time (1.0 == 100%). If mixing 100% of the time, the models may get confused when decoding simple pure
        morse signals (e.g. the Wikipedia examples).
        rng_seed: Seed for the random number generator, which is used to decide dit durations, frequencies, etc.
        min_pad_ms: Lower bound of padding added to the beginning of each morse audio signal, in ms.
        max_pad_ms: Upper bound of padding added to the end of each morse audio signal, in ms.
        min_snr_db: Lower bound of SNR, with signal being the morse signal and noise being any added signals.
        max_snr_db: Upper bound of SNR.
        window_names: List of all possible window functions to use. These are applied to the edges of the tones to make
        them smooth. If more than one, uniformly sample one per audio signal. This alters the features (typically
        spectrograms) at the edges of the tones.
        min_rise_time_ms: Lower bound of the window function rise time (essentially half the window function).
        max_rise_time_ms: Upper bound of the window function rise time (essentially half the window function).
        augment_config: A list of dictionaries which can be interpreted by the function
        ``nemo.collections.asr.parts.preprocessing.perturb.process_augmentations``.
    """
    def __init__(
            self,
            alphabet: Union[str, MorseAlphabet],
            labels: List[str],
            text_files: List[str],
            background_audio_manifests: Optional[List[str]] = None,
            max_words_per_eg: Optional[int] = None,
            sample_rate: int = 16000,
            prosign_count: int = 0,
            min_tone_frequency: int = 200,
            max_tone_frequency: int = 3000,
            min_tone_gain_db: float = -20,
            max_tone_gain_db: float = 0,
            min_dit_length_ms: int = 40,
            max_dit_length_ms: int = 130,
            duration_sigma_pct: float = 0.1,
            mix_background_percent: float = 0.75,
            rng_seed: int = 12345,
            min_pad_ms: int = 0,
            max_pad_ms: int = 2000,
            min_snr_db: float = 10.0,
            max_snr_db: float = 50.0,
            window_names: Tuple[str] = ("hann", "bartlett", "cosine"),
            min_rise_time_ms: int = 5,
            max_rise_time_ms: int = 16,
            augment_config: List[Dict] = None
    ):
        super().__init__()
        self._sample_rate: int = sample_rate
        self._labels = labels
        self._max_words_per_eg = max_words_per_eg
        self._min_tone_freq = min_tone_frequency
        self._max_tone_freq = max_tone_frequency
        self._min_tone_gain_db = min_tone_gain_db
        self._max_tone_gain_db = max_tone_gain_db
        self._min_snr_db = min_snr_db
        self._max_snr_db = max_snr_db
        self._min_dit_length_ms = min_dit_length_ms
        self._max_dit_length_ms = max_dit_length_ms
        self._min_pad_ms = min_pad_ms
        self._max_pad_ms = max_pad_ms
        self._min_rise_time_ms = min_rise_time_ms
        self._max_rise_time_ms = max_rise_time_ms
        self._window_names = window_names
        self._duration_sigma_pct = duration_sigma_pct
        self._prosign_count = prosign_count
        self._mix_background_percent = mix_background_percent

        self._rng: random.Random = random.Random(rng_seed)
        self._rng_init_state = self._rng.getstate()
        # Map labels to their class index
        self._label_to_index: Dict[str, int] = {x: i for i, x in enumerate(self._labels)}

        # Pattern to remove all OOV characters, if user requests to clean text
        labels_str = "".join(self._labels)
        self._oov_ptn = re.compile(rf"[^\s{labels_str}]")

        # Load data and prepare augmenter
        self._texts: List[str] = self._load_texts(text_files)
        self._background_manifest = self._load_background_manifests(background_audio_manifests)
        self._augmenter: Optional[AudioAugmentor] = process_augmentations(augment_config)

        # Resolve the alphabet and make sure it covers the labels
        self._alphabet: MorseAlphabet
        if isinstance(alphabet, str):
            self._alphabet = MorseAlphabet(alphabet)
        else:
            self._alphabet = alphabet
        self._verify_alphabet()

        # Get the alphabet's prosigns and verify that we have some if the user wants to include them.
        self._prosigns: List[str] = self._alphabet.prosigns
        if self._prosign_count > 0:
            if not self._prosigns:
                raise ValueError(
                    f"Specified to append {self._prosign_count} prosigns to the dataset but alphabet has no prosigns."
                )

    def __len__(self):
        # One example per text, plus lazily-appended prosigns, if applicable.
        return len(self._texts) + self._prosign_count

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        is_prosign = idx >= len(self._texts)
        text: str
        if not is_prosign:
            text = self._texts[idx]
        else:
            text = self._prosigns[idx % len(self._prosigns)]
        signal = self._text_to_signal(text, is_prosign=is_prosign)
        targets = self._text_to_targets(text)
        return signal, targets

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            "signals": NeuralType(("B", "T"), AudioSignal(freq=self._sample_rate)),
            "signal_lengths": NeuralType(("B",), LengthsType()),
            'transcripts': NeuralType(("B", "T"), LabelsType()),
            'transcript_lengths': NeuralType(("B",), LengthsType())
        }

    @typecheck()
    def collate_fn(self, batch):
        # __get_item__ produces (signal, transcription)
        signals_list: List[torch.Tensor]
        targets_list: List[torch.Tensor]
        signals_list, targets_list = zip(*batch)

        # Make lengths tensors
        signal_lengths = torch.tensor([len(x) for x in signals_list], dtype=torch.long)
        target_lengths = torch.tensor([len(x) for x in targets_list], dtype=torch.long)

        # Pack signals into a padded tensor
        max_signal_length = signal_lengths.max().item()
        signals: torch.Tensor = torch.zeros((len(signal_lengths), max_signal_length))
        for i, (signal, length) in enumerate(zip(signals_list, signal_lengths)):
            signals[i][:length] = signal

        # Pack targets into a padded tensor
        max_target_length = target_lengths.max().item()
        targets: torch.Tensor = torch.zeros((len(target_lengths), max_target_length))
        for i, (target, length) in enumerate(zip(targets_list, target_lengths)):
            targets[i][:length] = target

        return signals, signal_lengths, targets, target_lengths

    def _verify_alphabet(self):
        """
        Verifies that this class's labels are a subset of the alphabet, i.e., all labels have a
        morse encoding.
        """
        unknown_labels = set(self._labels) - set(self._alphabet.vocabulary + [" "])
        if unknown_labels:
            m = f"Dataset is using labels that are not in the alphabet, and therefore cannot " \
                f"be mapped to morse. Unknown labels = {unknown_labels}; alphabet is " \
                f"{self._alphabet.vocabulary}"
            raise ValueError(m)

    def _text_to_targets(self, text) -> torch.Tensor:
        """Generates training targets from text.

        Args:
            text: Input text. Should be cleaned and in-vocabulary.

        Returns:
            A torch.Tensor containing the integer values of the target labels.
        """
        targets_list: List[int] = [self._label_to_index[x] for x in text]
        targets = torch.tensor(targets_list, dtype=torch.long)
        return targets

    def synthesize(self, output_dir) -> None:
        """Synthesizes all text and saves a corpus to the specified output directory.

        This method should be called to create a synthetic corpus on disk, and not called during training.

        The output directory will contain a directory called ``wav/`` containing all audio files and a text file
        named ``manifest.json`` containing the corpus manifest.

        Args:
            output_dir: Path to an output directory.
        """
        # Audio files will be in <output>/wav/. Use absolute paths.
        output_dir = os.path.abspath(output_dir)
        wav_dir = os.path.join(output_dir, "wav")
        os.makedirs(wav_dir, exist_ok=True)
        manifest_file = os.path.join(output_dir, "manifest.json")
        # For pretty-printing file names that are numbered sequentially
        num_digits = len(str(len(self._texts)))
        with open(manifest_file, "w") as writer:
            for i, text in tqdm(
                    enumerate(self._texts), total=len(self._texts), unit="sentence", desc="Synthesizing"
            ):
                signal: torch.Tensor = self._text_to_signal(text)
                audio_filepath = os.path.join(wav_dir, f"audio_{i:0{num_digits}d}.wav")
                soundfile.write(
                    file=audio_filepath,
                    data=signal.numpy(),
                    samplerate=self._sample_rate,
                    format="WAV",
                    subtype="PCM_16"
                )
                manifest_data = {
                    "audio_filepath": audio_filepath,
                    "text": text,
                    "duration": len(signal) / self._sample_rate
                }
                manifest_str = json.dumps(manifest_data, ensure_ascii=False)
                writer.write(f"{manifest_str}\n")

    def on_validation_epoch_start(self):
        self._rng.setstate(self._rng_init_state)

    def clean_text(self, text: str) -> str:
        """Cleans a line according to this alphabet.

        Cleans a line of text to bring it in accordance to this dataset's labels, which are assumed to all be
        uppercase.

        N.B. OOV characters will be silently removed.

        Args:
            text: String to clean

        Returns:
            Cleaned string.
        """
        text = text.upper()
        text = re.sub(self._oov_ptn, "", text)
        text = text.strip()
        return text

    def _load_texts(self, text_file_paths: List[str]) -> List[str]:
        """Loads text files containing one sentence per line.

        Args:
            text_file_paths: List of file paths. Each file should contain one sentence per line.

        Returns:
            A list of strings, one string for each line of each input file.
        """
        texts: List[str] = []
        for text_filepath in text_file_paths:
            with open(text_filepath) as f:
                for line in f:
                    text = line.strip().upper()
                    text = self.clean_text(text)
                    if not text:
                        continue
                    # Maybe keep only the first N words
                    if self._max_words_per_eg is not None:
                        tokens = text.split()
                        if len(tokens) > self._max_words_per_eg:
                            text = " ".join(tokens[:self._max_words_per_eg])
                    texts.append(text)
        return texts

    def _load_background_manifests(self, manifests: Optional[List[str]]) -> List[Dict]:
        """Loads zero or more manifests containing audio to use as background noise.

        Each line of each file should be of the following format:

            {"audio_filepath": "/path/to/x.wav"}

        Extra keys are OK, and are ignored. Same as NeMo's ASR manifests but only the ``audio_filepath`` key is needed.

        Args:
            manifests: List of file paths.

        Returns:
            List of Dicts, each containing the parsed data from one line of a manifest.
        """
        data: List[Dict] = []
        if manifests is not None:
            for manifest in manifests:
                with open(manifest) as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
        return data

    def _add_bg_noise(self, morse_signal: AudioSegment) -> AudioSegment:
        """Adds a noise signal to a morse signal.

        If this class was given a manifest of background noise audio, this method will randomly choose one of those
        background noise files and mix it with the clean morse signal.

        Only one background noise file is used. If the background noise is longer than the input, it is truncated. If
        the noise signal is shorter than the morse signal, the noise signal is repeated to cover the morse signal
        entirely.

        Args:
            morse_signal: An AudioSegment with a clean morse signal.

        Returns:
            A new AudioSegment containing the morse signal mixed with a noise signal.
        """
        if not self._background_manifest:
            return morse_signal
        # Load a random noise segment
        manifest_line: Dict = self._rng.choice(self._background_manifest)
        noise_signal: AudioSegment = AudioSegment.from_file(
            manifest_line["audio_filepath"],
            target_sr=self._sample_rate,
            offset=manifest_line.get("offset", 0.0),
            duration=manifest_line.get("duration", 0.0)
        )
        # Randomly choose an SNR
        snr_db = self._rng.uniform(self._min_snr_db, self._max_snr_db + 1)
        mixed_signal = mix_background_signal(
            morse_signal=morse_signal,
            noise_signal=noise_signal,
            snr_db=snr_db
        )
        return mixed_signal

    def _generate_morse_signal(self, symbols: List[Symbol]) -> AudioSegment:
        """Creates a morse audio signal.

        Given a sequence of morse symbols, generates the corresponding audio signal. Uniformly chooses values for all
        parameters based on the dataset's min/max values for each.

        Args:
            symbols: Sequence of dits, dashes, char spaces, and word spaces.

        Returns:
            An AudioSegment whose samples represent the morse signal of the given symbols.
        """
        # Randomly draw all parameters
        dit_duration = self._rng.randint(self._min_dit_length_ms, self._max_dit_length_ms + 1)
        duration_sigma = int(dit_duration * self._duration_sigma_pct)
        tone_freq = self._rng.randint(self._min_tone_freq, self._max_tone_freq + 1)
        gain_db = self._rng.uniform(self._min_tone_gain_db, self._max_tone_gain_db + 1)
        padding_left = self._rng.randint(self._min_pad_ms, self._max_pad_ms + 1)
        padding_right = self._rng.randint(self._min_pad_ms, self._max_pad_ms + 1)
        rise_time_ms = self._rng.randint(self._min_rise_time_ms, self._max_rise_time_ms)
        window_fn_name = self._rng.choice(self._window_names)
        # Use functional interface to do the rest of the work
        return symbols_to_signal(
            symbols=symbols,
            tone_frequency_hz=tone_freq,
            dit_duration_ms=dit_duration,
            duration_sigma=duration_sigma,
            sample_rate=self._sample_rate,
            gain_db=gain_db,
            pad_left_ms=padding_left,
            pad_right_ms=padding_right,
            window_rise_time_ms=rise_time_ms,
            window_name=window_fn_name,
            rng=self._rng
        )

    def _text_to_signal(self, text, is_prosign: bool = False) -> torch.Tensor:
        """Converts a text into an audio signal containing the morse code.

        Given an input sentence, converts the characters into a sequence of DITs and DASHs along with character and
        word spaces. Then drawing from the specified distribution to determine the length of each tone and space,
        generates an audio signal at a randomly-drawn frequency. If background noise has been loaded, it is added to the
        signal within this function.

        Args:
            text: Text to encode.
            is_prosign: Whether to treat this text as a prosign, which usually has an abbreviated encoding.

        Returns:
            An audio signal containing the morse code with optional background noise mixed in.
        """
        # Note: don't clean here or signal may not match true text
        symbols: List[Symbol] = self._alphabet.text_to_symbols(text, clean=False, is_prosign=is_prosign)
        audio_segment: AudioSegment = self._generate_morse_signal(symbols)
        # Add background noise some percentage of the time
        if self._rng.random() < self._mix_background_percent:
            audio_segment = self._add_bg_noise(audio_segment)
        signal = torch.tensor(audio_segment.samples)
        return signal
