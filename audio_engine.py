# audio_engine.py — MP3/M4A-capable loop player (soundfile → ffmpeg → librosa fallback)
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import warnings
import shutil
import tempfile
import subprocess
import os

try:
    import librosa  # final fallback loader if ffmpeg unavailable
except Exception:
    librosa = None

# Prefer ffmpeg decode over audioread to avoid deprecation warnings and speed up loads
_DEF_FFMPEG = shutil.which("ffmpeg")

def _decode_via_ffmpeg(path: str, stereo: bool = True):
    """Decode any format via ffmpeg → temp WAV, then read with soundfile.
    Returns (y, sr) with y as float32 (N, C).
    """
    if not _DEF_FFMPEG:
        raise RuntimeError("ffmpeg not found on PATH")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    channels = 2 if stereo else 1
    cmd = [
        _DEF_FFMPEG,
        "-y",
        "-i", path,
        "-ac", str(channels),
        "-vn",
        "-map_metadata", "-1",
        tmp_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    try:
        y, sr = sf.read(tmp_path, dtype='float32', always_2d=True)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    return y, sr


def _load_audio_any(path: str):
    """Load audio and return (y, sr) where y is float32 (N, C).
    Try soundfile (WAV/FLAC/AIFF), then ffmpeg (MP3/M4A/AAC/etc), then librosa as last resort.
    """
    # 1) Fast path: wav/flac/aiff via soundfile
    try:
        y, sr = sf.read(path, dtype='float32', always_2d=True)
        return y, sr
    except Exception:
        pass

    # 2) ffmpeg decode for compressed formats (avoids audioread warnings)
    try:
        return _decode_via_ffmpeg(path, stereo=True)
    except Exception:
        pass

    # 3) Final fallback to librosa/audioread (silence deprecation warnings)
    if librosa is None:
        raise RuntimeError(
            "Could not decode audio. Install ffmpeg or librosa to handle compressed formats."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, sr = librosa.load(path, sr=None, mono=False)
    if y.ndim == 1:
        y = y.astype(np.float32)[:, None]
    else:
        y = y.T.astype(np.float32)
    return y, sr


class LoopPlayer:
    """Gapless loop player using sounddevice callback.
    Loads entire file into RAM for responsive looping.
    """
    def __init__(self, path: str):
        self._load(path)
        self.lock = threading.Lock()
        self.lo = 0
        self.hi = self.n
        self.pos = 0
        self._playing = False
        self._pos_samples = 0
        self.stream = sd.OutputStream(
            samplerate=self.sr,
            channels=self.y.shape[1],
            dtype='float32',
            callback=self._cb,
        )

    def _load(self, path: str):
        y, sr = _load_audio_any(path)
        self.y = y  # (N, C) float32
        self.sr = sr
        self.n = y.shape[0]
        self.path = path

    def reload(self, path: str):
        was_playing = self._playing
        if was_playing:
            self.stop()
        self._load(path)
        with self.lock:
            self.lo, self.hi, self.pos = 0, self.n, 0
        if was_playing:
            self.start()

    def set_loop_seconds(self, start_s: float, end_s: float):
        lo = int(max(0, min(self.n - 1, start_s * self.sr)))
        hi = int(max(lo + 1, min(self.n, end_s * self.sr)))
        self.lo, self.hi = lo, hi

    def duration_seconds(self) -> float:
        """Total duration of the loaded buffer in seconds."""
        return self.n / float(self.sr)

    def position_seconds(self) -> float:
        return float(self._pos_samples) / float(self.sr)

    def set_position_seconds(self, t: float, within_loop: bool = False):
        t = float(max(0.0, t))
        total = self.y.shape[0] if self.y is not None else 0
        pos = int(t * self.sr)
        if within_loop and getattr(self, 'hi', 0) > getattr(self, 'lo', 0):
            a = max(0, int(self.lo))
            b = min(total, int(self.hi))
            pos = max(a, min(max(a, b - 1), pos))
        else:
            pos = max(0, min(max(0, total - 1), pos))
        self._pos_samples = pos

    def clear_loop(self):
        self.lo, self.hi = 0, 0

    def _cb(self, outdata, frames, time, status):
        if status:
            # print(status)  # optional
            pass
        # Zero fill by default
        outdata[:] = 0.0
        if self.y is None:
            return

        n_total = self.y.shape[0]
        ch_file = self.y.shape[1] if self.y.ndim == 2 else 1
        ch_out = outdata.shape[1]

        # Loop bounds in samples (use lo/hi exclusively)
        if getattr(self, 'hi', 0) > getattr(self, 'lo', 0):
            loop_a, loop_b = int(self.lo), int(self.hi)
        else:
            loop_a, loop_b = 0, n_total

        pos = int(self._pos_samples)
        wrote = 0
        while wrote < frames:
            if pos >= loop_b:
                # wrap if a loop exists; otherwise leave remaining silence
                if loop_b > loop_a:
                    pos = loop_a
                else:
                    break
            take = min(frames - wrote, loop_b - pos)
            if take <= 0:
                break
            if ch_file == 1:
                src = self.y[pos:pos+take]
                if src.ndim == 1:
                    src = src.reshape(-1, 1)
            else:
                src = self.y[pos:pos+take, :ch_file]
            # Fit channels to device
            if ch_file >= ch_out:
                outdata[wrote:wrote+take, :] = src[:, :ch_out]
            else:
                outdata[wrote:wrote+take, :ch_file] = src
                if ch_file == 1 and ch_out > 1:
                    outdata[wrote:wrote+take, 1:] = src[:, :1]
            wrote += take
            pos += take
        self._pos_samples = pos

    def play(self):
        if self.stream is None:
            import sounddevice as sd
            self.stream = sd.OutputStream(
                samplerate=self.sr,
                channels=(self.y.shape[1] if self.y.ndim == 2 else 1),
                dtype='float32',
                callback=self._cb,
                blocksize=0,
            )
        if not self.stream.active:
            self.stream.start()

    def pause(self):
        if self.stream is not None and self.stream.active:
            self.stream.stop()

    def start(self):
        if not self._playing:
            self.stream.start()
            self._playing = True

    def stop(self):
        if self._playing:
            self.stream.stop()
            self._playing = False

    def close(self):
        try:
            self.stream.close()
        except Exception:
            pass
