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
    import librosa
except Exception as e:
    librosa = None
    _LIBROSA_IMPORT_ERROR = e
else:
    _LIBROSA_IMPORT_ERROR = None

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
        # === Time‑stretch / playback state ===
        self._rate = 1.0                    # original speed by default
        self._y_base = self.y               # original audio (N, C)
        self._y_play = self.y               # buffer currently sent to the stream (stretched)
        self._frames_out = 0                # cursor into _y_play (output domain)
        self._loop_seconds = None           # (a, b) in ORIGINAL seconds
        self._loop_frames_out = None        # (a_out, b_out) in output frames
        self.lock = threading.Lock()
        self.lo = 0
        self.hi = self.n
        self.pos = 0
        self._playing = False
        self._pos_samples = 0
        self.last_rate_error = None
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
        # Reset stretch buffers to newly loaded audio
        self._y_base = self.y
        self._y_play = self.y
        self._frames_out = 0

    def reload(self, path: str):
        was_playing = self._playing
        if was_playing:
            self.stop()
        self._load(path)
        with self.lock:
            self.lo, self.hi, self.pos = 0, self.n, 0
        # Reapply current rate to new buffer
        try:
            self._rebuild_rate(self._rate)
        except Exception:
            self._rate = 1.0
            self._y_play = self._y_base
            self._frames_out = 0
        if was_playing:
            self.start()

    def set_loop_seconds(self, start_s: float, end_s: float):
        a = float(min(start_s, end_s))
        b = float(max(start_s, end_s))
        # Keep original-sample loop for backwards compatibility (unused when stretched)
        self.lo = int(max(0, min(self.n - 1, a * self.sr)))
        self.hi = int(max(self.lo + 1, min(self.n, b * self.sr)))
        # Store seconds and compute output-domain loop frames for stretched playback
        self._loop_seconds = (a, b)
        self._loop_frames_out = (int(a * self.sr / max(self._rate, 1e-6)),
                                 int(b * self.sr / max(self._rate, 1e-6)))

    def duration_seconds(self) -> float:
        """Total duration of the loaded buffer in seconds."""
        # Original timeline duration
        n_out = int(self._y_play.shape[0])
        return (n_out / float(self.sr)) * max(self._rate, 1e-6)

    def position_seconds(self) -> float:
        return (float(self._frames_out) / float(self.sr)) * max(self._rate, 1e-6)

    def set_position_seconds(self, t: float, within_loop: bool = False):
        t = float(max(0.0, t))
        # Convert original‑timeline seconds → output frames
        i_out = int(t * self.sr / max(self._rate, 1e-6))
        n_out = int(self._y_play.shape[0]) if self._y_play is not None else 0
        if within_loop and self._loop_frames_out:
            a_out, b_out = self._loop_frames_out
            i_out = max(a_out, min(max(a_out, b_out - 1), i_out))
        else:
            i_out = max(0, min(max(0, n_out - 1), i_out))
        self._frames_out = i_out

    def clear_loop(self):
        self.lo, self.hi = 0, 0
        self._loop_seconds = None
        self._loop_frames_out = None

    def _rebuild_rate(self, rate: float) -> bool:
        """Rebuild the playback buffer for a new rate, preserving pitch.
        Keeps UI on original timeline by mapping output frames ⇄ original seconds.
        """
        self.last_rate_error = None
        rate = float(max(0.5, min(1.5, rate)))
        y = self._y_base
        if abs(rate - 1.0) < 1e-6:
            self._y_play = y
            self._rate = 1.0
        else:
            # Prefer librosa fallback if no custom stretcher is available
            if librosa is None:
                if '_LIBROSA_IMPORT_ERROR' in globals() and _LIBROSA_IMPORT_ERROR is not None:
                    self.last_rate_error = f"librosa import failed: {type(_LIBROSA_IMPORT_ERROR).__name__}: {_LIBROSA_IMPORT_ERROR}"
                else:
                    self.last_rate_error = "librosa not available"
                try:
                    print(f"[rate] rebuild failed → {self.last_rate_error}")
                except Exception:
                    pass
                return False
            try:
                if y.ndim == 1 or y.shape[1] == 1:
                    mono = y[:, 0] if y.ndim == 2 else y
                    y_st = librosa.effects.time_stretch(y=mono.astype(float), rate=rate)
                    y_st = y_st.astype(np.float32, copy=False)
                    self._y_play = np.column_stack([y_st, y_st]) if (y.ndim == 2 and y.shape[1] == 2) else y_st.reshape(-1, 1)
                else:
                    # per‑channel stretch and pad to equal length
                    chans = []
                    for c in range(y.shape[1]):
                        ch = librosa.effects.time_stretch(y=y[:, c].astype(float), rate=rate)
                        chans.append(ch)
                    L = max(len(cch) for cch in chans)
                    chans = [np.pad(cch, (0, L - len(cch))) for cch in chans]
                    y_st = np.stack(chans, axis=1).astype(np.float32, copy=False)
                    self._y_play = y_st
                self._rate = rate
            except Exception as e:
                self.last_rate_error = f"time_stretch failed: {type(e).__name__}: {e}"
                try:
                    print(f"[rate] rebuild failed → {self.last_rate_error}")
                except Exception:
                    pass
                return False
        # Update loop frames in output domain
        if self._loop_seconds is not None:
            a, b = self._loop_seconds
            self._loop_frames_out = (int(a * self.sr / max(self._rate, 1e-6)),
                                     int(b * self.sr / max(self._rate, 1e-6)))
        else:
            self._loop_frames_out = None
        # Keep transport position stable in original time
        t_orig = self.position_seconds()
        self._frames_out = int(t_orig * self.sr / max(self._rate, 1e-6))
        return True

    def set_rate(self, rate: float) -> bool:
        """Set playback rate (0.5–1.5x), pitch‑preserving."""
        return bool(self._rebuild_rate(rate))

    def _cb(self, outdata, frames, time, status):
        if status:
            pass
        outdata[:] = 0.0
        y = self._y_play
        if y is None:
            return
        n_total = int(y.shape[0])
        ch_file = y.shape[1] if y.ndim == 2 else 1
        ch_out = outdata.shape[1]

        # Output‑domain loop bounds if present, else fall back to original indices
        if self._loop_frames_out:
            loop_a, loop_b = int(self._loop_frames_out[0]), int(self._loop_frames_out[1])
        elif getattr(self, 'hi', 0) > getattr(self, 'lo', 0):
            # legacy: map original sample indices approximately to output domain by rate
            loop_a = int(self.lo / max(self._rate, 1e-6))
            loop_b = int(self.hi / max(self._rate, 1e-6))
        else:
            loop_a, loop_b = 0, n_total

        pos = int(self._frames_out)
        wrote = 0
        while wrote < frames:
            if pos >= loop_b:
                if loop_b > loop_a:
                    pos = loop_a
                else:
                    break
            take = min(frames - wrote, loop_b - pos)
            if take <= 0:
                break
            if ch_file == 1:
                src = y[pos:pos+take]
                if src.ndim == 1:
                    src = src.reshape(-1, 1)
            else:
                src = y[pos:pos+take, :ch_file]
            if ch_file >= ch_out:
                outdata[wrote:wrote+take, :] = src[:, :ch_out]
            else:
                outdata[wrote:wrote+take, :ch_file] = src
                if ch_file == 1 and ch_out > 1:
                    outdata[wrote:wrote+take, 1:] = src[:, :1]
            wrote += take
            pos += take
        self._frames_out = pos

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
