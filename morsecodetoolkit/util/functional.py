
import random
import librosa
import numpy as np
from typing import List, Optional, Tuple

from nemo.collections.asr.parts.preprocessing.segment import AudioSegment

from morsecodetoolkit.alphabet.alphabet import Symbol


def symbols_to_signal(
        symbols: List[Symbol],
        sample_rate: int,
        tone_frequency_hz: float,
        gain_db: float = 0.0,
        dit_duration_ms: int = 50,
        duration_sigma: float = 5,
        pad_left_ms: int = 500,
        pad_right_ms: int = 500,
        window_name: str = "hann",
        window_rise_time_ms: int = 12,
        rng: Optional[random.Random] = None
) -> AudioSegment:
    """Converts a sequence of morse symbols to an audio signal.

    Given a sequence of morse symbols from the set {DIT, DASH, CHAR_SPACE, WORD_SPACE}, generates an audio signal
    representing the morse encoding.

    Note that this produces a "clean" signal, i.e., morse tones only.

    Args:
        symbols: List of morse symbols to generate.
        sample_rate: Audio sample rate, in Hz.
        tone_frequency_hz: Frequency of the tone used, in Hz.
        gain_db: Gain applied to the tone, in dB. Should be <= 0.
        dit_duration_ms: Mean duration of a DIT, which implies the mean duration
        of DASH and pauses, as a scalar of this value, according to morse standards.
        duration_sigma: Durations are drawn from a Gaussian distribution with this sigma value.
        pad_left_ms: Amount of padding (silence) to add to the beginning of the signal, in ms.
        pad_right_ms: Amount of padding (silence) to add to the end of the signal, in ms.
        rng: Random number generator to use, if seeking reproducibility.
        window_rise_time_ms: Length, in milliseconds, of the rise and fall of the tone.
        window_name: Name of the window used during rise and lower times.

    Returns:
        An AudioSegment containing the morse audio signal.
    """
    # Note: all times are in ms
    if rng is None:
        rng = random.Random()
    # Compute start/stop times of all tone segments
    tone_segments: List[Tuple[int, int]] = []
    start = pad_left_ms
    for i, symbol in enumerate(symbols):
        if symbol in {Symbol.DIT, Symbol.DASH}:
            # If previous symbol was a tone, generate time stamps for a pause before this tone
            if i > 0 and symbols[i-1] in {Symbol.DIT, Symbol.DASH}:
                pause_duration = int(rng.gauss(dit_duration_ms, duration_sigma))
                start += pause_duration
            # Generate time stamps for this tone
            scale = 1 if symbol == Symbol.DIT else 3
            tone_duration = int(rng.gauss(dit_duration_ms * scale, duration_sigma * scale))
            tone_segments.append((start, start + tone_duration))
            start += tone_duration
        elif symbol in {Symbol.CHAR_SPACE, Symbol.WORD_SPACE}:
            # Generate time stamps for a pause
            scale = 3 if symbol == Symbol.CHAR_SPACE else 7
            pause_duration = int(rng.gauss(dit_duration_ms * scale, duration_sigma * scale))
            start += pause_duration
        else:
            raise ValueError(f"Couldn't interpret symbol type: {symbol}")
    # Determine the length of the total signal (last tone + specified padding)
    signal_length = tone_segments[-1][1] + pad_right_ms
    num_samples = int(signal_length / 1000 * sample_rate)
    # Create an empty signal and add the tone where the segments should appear
    signal = np.zeros(shape=[num_samples], dtype=np.float)
    max_tone_duration = max(x[1] - x[0] for x in tone_segments) + 0.1
    tone_samples = librosa.tone(tone_frequency_hz, duration=max_tone_duration/1000., sr=sample_rate)

    # Make sure rise time is <= half dit duration to prevent windowing issues.
    window_rise_time_ms = min(window_rise_time_ms, dit_duration_ms // 2)
    # Create the window for smooth rise/lower transitions on the tones
    tone_rise_num_samples = int(sample_rate / 1000.0 * window_rise_time_ms)
    window = librosa.filters.get_window(window_name, tone_rise_num_samples * 2 - 1)
    rise_window: np.array = window[:tone_rise_num_samples]
    fall_window: np.array = window[-tone_rise_num_samples:]
    # Generate each tone and insert into the otherwise-empty signal
    for start_ms, stop_ms in tone_segments:
        start_sample = int(start_ms/1000. * sample_rate)
        stop_sample = int(stop_ms/1000. * sample_rate)
        # Select tone with random phase from the pre-generated cosine wave
        num_samples = stop_sample - start_sample
        subsample_start = np.random.randint(0, tone_samples.shape[0] - num_samples)
        tone_subsamples = tone_samples[subsample_start:subsample_start + num_samples]
        # Stretch the window by inserting all 1's in the middle
        tone_window: np.array = np.ones_like(tone_subsamples)
        tone_window[:tone_rise_num_samples] = rise_window
        tone_window[-tone_rise_num_samples:] = fall_window
        # Apply window to tone and insert in signal
        signal[start_sample:stop_sample] = tone_subsamples * tone_window
    # Create an AudioSegment to hold the signal; apply gain
    seg: AudioSegment = AudioSegment(samples=signal, sample_rate=sample_rate)
    seg.gain_db(gain_db)
    return seg


def mix_background_signal(
        morse_signal: AudioSegment,
        noise_signal: AudioSegment,
        snr_db: float
) -> AudioSegment:
    """Adds a background audio to a morse signal.

    Adds to a morse signal a background (noise) audio. Intended for augmentation purposes and to make synthetic morse
    signals more realistic.

    If the noise signal is longer than the morse signal, the noise signal will be truncated and the beginning portion
    will be kept. If the noise signal is shorter than the morse signal, it is repeated to cover the entire morse signal.

    Args:
         morse_signal: Clean morse audio.
         noise_signal: Speech or noise audio to mix in.
         snr_db: SNR, in dB, for the morse/noise ratio.

    Returns:
        An AudioSegment containing the noisy morse signal.
    """
    # Adjust volume of noise signal
    gain_db = morse_signal.rms_db - noise_signal.rms_db - snr_db
    noise_signal.gain_db(gain_db)
    # Possibly resample noise signal; get raw samples from data structure
    sample_rate = morse_signal.sample_rate
    if noise_signal.sample_rate != sample_rate:
        noise_samples = librosa.resample(
            y=noise_signal.samples,
            orig_sr=noise_signal.sample_rate,
            target_sr=sample_rate
        )
    else:
        noise_samples = noise_signal.samples
    # Add volume-adjusted and resampled noise to morse signal
    morse_samples = morse_signal.samples
    # Truncate or repeat the noise samples to fit the morse signal
    if len(noise_samples) > len(morse_samples):
        morse_samples = morse_samples + noise_samples[:len(morse_samples)]
    else:
        for start_sample in range(0, len(morse_samples), len(noise_samples)):
            stop_sample = min(len(morse_samples), start_sample + len(noise_samples))
            num_samples = stop_sample - start_sample
            morse_samples[start_sample:stop_sample] += noise_samples[:num_samples]
    return AudioSegment(samples=morse_samples, sample_rate=sample_rate)
