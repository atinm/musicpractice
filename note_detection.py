"""
Note detection using librosa CQT (Constant-Q Transform) for accurate frequency analysis.
"""

import numpy as np

def _cqt_to_midi_map(sr, hop_length, fmin, n_bins, bins_per_octave):
    """Map CQT bins to nearest MIDI note."""
    try:
        import librosa
        freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
        midi = librosa.hz_to_midi(freqs)
        midi_round = np.round(midi).astype(int)
        return midi_round, freqs
    except ImportError:
        # Fallback if librosa not available
        return np.array([]), np.array([])

def compute_note_confidence(
    audio_data: np.ndarray,
    sr: int = 44100,
    hop_length: int = 512,
    bins_per_octave: int = 36,
    n_octaves: int = 8,
    fmin: float = 27.5,  # A0 frequency
    ema_alpha: float = 0.4,
    medfilt_width: int = 3,
):
    """Compute note confidence using librosa CQT."""
    try:
        import librosa

        # Ensure audio is mono
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)

        n_bins = bins_per_octave * n_octaves
        C = np.abs(librosa.cqt(
            audio_data, sr=sr, hop_length=hop_length, fmin=fmin,
            n_bins=n_bins, bins_per_octave=bins_per_octave, window='hann'
        )).astype(np.float32)  # (n_bins, n_frames)

        # Map each CQT bin -> nearest MIDI note
        midi_map, _ = _cqt_to_midi_map(sr, hop_length, fmin, n_bins, bins_per_octave)

        # We'll keep only real piano range A0(21)–C8(108)
        MIDI_MIN, MIDI_MAX = 21, 108
        n_frames = C.shape[1]
        note_conf = np.zeros((n_frames, 88), dtype=np.float32)  # 88 keys

        for b in range(n_bins):
            midi = midi_map[b]
            if MIDI_MIN <= midi <= MIDI_MAX:
                note_idx = midi - MIDI_MIN
                note_conf[:, note_idx] += C[b, :]

        # Per-frame robust normalization (95th percentile)
        q = np.percentile(note_conf, 95, axis=1, keepdims=True)
        q[q == 0.0] = 1.0
        note_conf = note_conf / q

        # Temporal EMA smoothing
        sm = np.copy(note_conf)
        for i in range(1, n_frames):
            sm[i] = ema_alpha * note_conf[i] + (1.0 - ema_alpha) * sm[i - 1]

        # Optional tiny median filter
        if medfilt_width > 1:
            try:
                from scipy.ndimage import median_filter
                sm = median_filter(sm, size=(medfilt_width, 1), mode='nearest')
            except ImportError:
                pass  # Skip median filter if scipy not available

        return sm

    except ImportError:
        # Fallback to simple FFT if librosa not available
        return _fallback_note_confidence(audio_data, sr, hop_length)
    except Exception as e:
        print(f"Error in compute_note_confidence: {e}")
        return _fallback_note_confidence(audio_data, sr, hop_length)

def _fallback_note_confidence(audio_data: np.ndarray, sr: int, hop_length: int):
    """Fallback note confidence using simple FFT."""
    # Simple FFT-based approach as fallback
    window_size = 2048
    n_frames = max(1, len(audio_data) // hop_length)
    note_conf = np.zeros((n_frames, 88), dtype=np.float32)

    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = min(start_idx + window_size, len(audio_data))
        if end_idx - start_idx < window_size // 2:
            break

        window = audio_data[start_idx:end_idx]
        if len(window) < window_size:
            window = np.pad(window, (0, window_size - len(window)))

        # Apply window function
        window = window * np.hanning(len(window))

        # FFT
        fft = np.fft.rfft(window)
        magnitude = np.abs(fft)
        freqs = np.fft.rfftfreq(len(window), 1/sr)

        # Map to MIDI notes
        for j in range(88):
            midi_note = 21 + j  # A0 to C8
            note_freq = 440.0 * (2 ** ((midi_note - 69) / 12.0))
            closest_idx = np.argmin(np.abs(freqs - note_freq))
            note_conf[i, j] = magnitude[closest_idx] if closest_idx < len(magnitude) else 0

    # Normalize
    max_val = np.max(note_conf)
    if max_val > 0:
        note_conf = note_conf / max_val

    return note_conf

def save_note_confidence(stem_wav_path: str, output_path: str = None):
    """Save note confidence data to file for caching."""
    try:
        import librosa
        y, sr = librosa.load(stem_wav_path, sr=44100, mono=True)
        note_conf = compute_note_confidence(y, sr=sr)

        if output_path is None:
            from pathlib import Path
            output_path = str(Path(stem_wav_path).with_suffix('.noteconf.npz'))

        np.savez_compressed(output_path, note=note_conf, sr=sr, hop_length=512)
        return output_path
    except Exception as e:
        print(f"Error saving note confidence: {e}")
        return None

def load_note_confidence(file_path: str):
    """Load cached note confidence data."""
    try:
        data = np.load(file_path)
        return {
            'note_conf': data['note'],
            'sr': int(data['sr']),
            'hop_length': int(data['hop_length'])
        }
    except Exception as e:
        print(f"Error loading note confidence: {e}")
        return None
