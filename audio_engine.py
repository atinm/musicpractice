# audio_engine.py — MP3/M4A-capable loop player (soundfile → ffmpeg → librosa fallback)
import numpy as np
import sounddevice as sd
import soundfile as sf
import threading
import warnings
# Silence noisy librosa warning when buffers are shorter than n_fft during transient operations
warnings.filterwarnings(
    "ignore",
    message=r"n_fft=\d+ is too large for input signal of length=\d+",
    category=UserWarning,
    module=r"librosa\.core\.spectrum"
)
import shutil
import tempfile
import subprocess
import os

# --- Per-file decode serialization to avoid concurrent opens of the same file ---
_DECODE_LOCKS: dict[str, threading.Lock] = {}
_DECODE_LOCKS_GUARD = threading.Lock()

def _decode_lock_for(path: str) -> threading.Lock:
    key = str(path)
    with _DECODE_LOCKS_GUARD:
        lk = _DECODE_LOCKS.get(key)
        if lk is None:
            lk = threading.Lock()
            _DECODE_LOCKS[key] = lk
        return lk


try:
    import librosa
except Exception as e:
    librosa = None
    _LIBROSA_IMPORT_ERROR = e
else:
    _LIBROSA_IMPORT_ERROR = None

# Guard lengths for librosa phase‑vocoder to avoid n_fft warnings on tiny buffers
_MIN_STRETCH_SAMPLES = 2048

def _safe_time_stretch(y: np.ndarray, rate: float) -> np.ndarray:
    """Phase‑vocoder stretch with guard for very short signals.
    If input is shorter than _MIN_STRETCH_SAMPLES, pad before stretching and then
    trim the output back to round(len(y)/rate). Returns float32 1‑D array.
    """
    if librosa is None or y is None:
        return y.astype(np.float32, copy=False) if isinstance(y, np.ndarray) else y
    yf = y.astype(float, copy=False)
    L = int(yf.shape[0])
    target = max(1, int(round(L / max(rate, 1e-6))))
    if L < _MIN_STRETCH_SAMPLES:
        pad = _MIN_STRETCH_SAMPLES - L
        yf = np.pad(yf, (0, pad))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = librosa.effects.time_stretch(y=yf, rate=rate)
        out = out[:target]
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = librosa.effects.time_stretch(y=yf, rate=rate)
        if out.shape[0] != target:
            if out.shape[0] > target:
                out = out[:target]
            else:
                out = np.pad(out, (0, target - out.shape[0]))
    return out.astype(np.float32, copy=False)

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
    Order of preference:
      • Compressed (mp3/m4a/aac/ogg/opus): ffmpeg → WAV, then librosa/audioread, then soundfile as last resort.
      • PCM (wav/flac/aiff): soundfile first, then ffmpeg, then librosa.
    Also serializes decoding per path to avoid crashy concurrent opens in underlying libs.
    """
    import os
    ext = os.path.splitext(str(path))[1].lower()
    compressed_exts = {'.mp3', '.m4a', '.aac', '.ogg', '.opus'}

    lk = _decode_lock_for(path)
    with lk:
        if ext in compressed_exts:
            # 1) Prefer ffmpeg for compressed formats (avoid libsndfile mpeg_init path)
            try:
                return _decode_via_ffmpeg(path, stereo=True)
            except Exception:
                pass
            # 2) librosa/audioread (CoreAudio/ffmpeg backend)
            if librosa is not None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y, sr = librosa.load(path, sr=None, mono=False)
                if y.ndim == 1:
                    y = y.astype(np.float32)[:, None]
                else:
                    y = y.T.astype(np.float32)
                return y, sr
            # 3) Last resort: soundfile
            y, sr = sf.read(path, dtype='float32', always_2d=True)
            return y, sr
        else:
            # Likely PCM container → soundfile first
            try:
                y, sr = sf.read(path, dtype='float32', always_2d=True)
                return y, sr
            except Exception:
                pass
            # Fallback to ffmpeg → WAV
            try:
                return _decode_via_ffmpeg(path, stereo=True)
            except Exception:
                pass
            # Final fallback to librosa/audioread
            if librosa is None:
                raise RuntimeError(
                    "Could not decode audio. Install ffmpeg or librosa to handle this format."
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
        # === Stems mixing state ===
        self._stems_base = {}     # name -> np.ndarray (N, C) original timeline
        self._stems_play = {}     # name -> np.ndarray (N, C) stretched/output domain
        self._stem_gains = {}     # name -> float in [0.0, 1.0]
        self._stem_mute = {}      # name -> bool
        self._stem_solo = {}      # name -> bool
        self._soloed_stem = None  # name of currently soloed stem, or None
        self._use_stems_only = True  # if True and stems exist, ignore full mix and play stems sum
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
        self._stems_base = {}
        self._stems_play = {}
        self._stem_gains = {}
        self._stem_mute = {}
        self._stem_solo = {}
        self._soloed_stem = None

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

    # ====== Stems API ======
    def set_stems_arrays(self, stems: dict):
        """Provide stems as name -> array (N,) or (N, C). sr must match self.sr.
        Rebuilds stretched buffers for current rate and resets per‑stem gains to 1.0 (unmuted)."""
        self._stems_base = {}
        self._stems_play = {}
        self._stem_gains = {}
        self._stem_mute = {}
        self._stem_solo = {}
        self._soloed_stem = None
        if not stems:
            return
        for name, arr in stems.items():
            if arr is None:
                continue
            a = arr
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            a = a.astype(np.float32, copy=False)
            self._stems_base[str(name)] = a
            self._stem_gains[str(name)] = 1.0
            self._stem_mute[str(name)] = False
            self._stem_solo[str(name)] = False
        self._rebuild_stems_for_rate()

    def clear_stems(self):
        self._stems_base.clear()
        self._stems_play.clear()
        self._stem_gains.clear()
        self._stem_mute.clear()
        self._stem_solo.clear()
        self._soloed_stem = None

    def set_stem_gain(self, name: str, gain01: float):
        self._stem_gains[str(name)] = float(max(0.0, min(1.0, gain01)))

    def set_stem_mute(self, name: str, muted: bool):
        self._stem_mute[str(name)] = bool(muted)

    def set_stem_solo(self, name: str, soloed: bool):
        """Set solo state for a stem. When soloed, all other stems are muted."""
        name = str(name)
        self._stem_solo[name] = bool(soloed)

        if soloed:
            # Solo this stem: mute all others, unmute this one
            self._soloed_stem = name
            for stem_name in self._stem_solo.keys():
                if stem_name != name:
                    self._stem_mute[stem_name] = True
                    self._stem_solo[stem_name] = False  # Clear other solo states
                else:
                    self._stem_mute[stem_name] = False
        else:
            # Unsolo this stem: if it was the soloed one, clear solo state
            if self._soloed_stem == name:
                self._soloed_stem = None
                # Unmute all stems and clear all solo states
                for stem_name in self._stem_solo.keys():
                    self._stem_mute[stem_name] = False
                    self._stem_solo[stem_name] = False

    def get_soloed_stem(self) -> str | None:
        """Get the name of the currently soloed stem, or None if none."""
        return self._soloed_stem

    def use_stems_only(self, enabled: bool):
        self._use_stems_only = bool(enabled)

    def _rebuild_stems_for_rate(self):
        """Stretch stems into output domain for current rate (uses librosa if available)."""
        self._stems_play = {}
        if not self._stems_base:
            return
        if abs(self._rate - 1.0) < 1e-6 or librosa is None:
            # No stretch needed or no stretcher; use base
            for k, a in self._stems_base.items():
                self._stems_play[k] = a
            return
        for k, a in self._stems_base.items():
            try:
                if a.ndim == 1 or a.shape[1] == 1:
                    mono = a[:, 0] if a.ndim == 2 else a
                    st = _safe_time_stretch(mono, self._rate)
                    self._stems_play[k] = st.reshape(-1, 1)
                else:
                    chans = []
                    for c in range(a.shape[1]):
                        ch = _safe_time_stretch(a[:, c], self._rate)
                        chans.append(ch)
                    L = max(len(ch) for ch in chans)
                    chans = [np.pad(ch, (0, L - len(ch))) for ch in chans]
                    st = np.stack(chans, axis=1).astype(np.float32, copy=False)
                    self._stems_play[k] = st
            except Exception:
                # On any failure, fall back to base for this stem
                self._stems_play[k] = a

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
                    y_st = _safe_time_stretch(mono, rate)
                    self._y_play = np.column_stack([y_st, y_st]) if (y.ndim == 2 and y.shape[1] == 2) else y_st.reshape(-1, 1)
                else:
                    # per‑channel stretch and pad to equal length
                    chans = []
                    for c in range(y.shape[1]):
                        ch = _safe_time_stretch(y[:, c], rate)
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
        # Rebuild stems stretched buffers for new rate
        self._rebuild_stems_for_rate()
        return True

    def set_rate(self, rate: float) -> bool:
        """Set playback rate (0.5–1.5x), pitch‑preserving."""
        return bool(self._rebuild_rate(rate))

    def _cb(self, outdata, frames, time, status):
        if status:
            pass
        outdata[:] = 0.0

        # Output‑domain loop bounds if present, else fall back to original indices
        if self._loop_frames_out:
            loop_a, loop_b = int(self._loop_frames_out[0]), int(self._loop_frames_out[1])
        elif getattr(self, 'hi', 0) > getattr(self, 'lo', 0):
            loop_a = int(self.lo / max(self._rate, 1e-6))
            loop_b = int(self.hi / max(self._rate, 1e-6))
        else:
            loop_a, loop_b = 0, (self._y_play.shape[0] if self._y_play is not None else 0)

        pos = int(self._frames_out)
        wrote = 0
        ch_out = outdata.shape[1]

        # If stems are present and enabled, mix stems; else play the full mix buffer
        if self._stems_play and self._use_stems_only:
            # determine a safe loop_b from stems too
            if not self._loop_frames_out:
                lens = [v.shape[0] for v in self._stems_play.values()]
                if lens:
                    loop_b = min(loop_b, min(lens)) if loop_b else min(lens)
            while wrote < frames:
                if pos >= loop_b:
                    if loop_b > loop_a:
                        pos = loop_a
                    else:
                        break
                take = min(frames - wrote, loop_b - pos)
                if take <= 0:
                    break
                mix = None
                for name, buf in self._stems_play.items():
                    if self._stem_mute.get(name):
                        continue
                    g = float(self._stem_gains.get(name, 1.0))
                    if g <= 0.0:
                        continue
                    sl = buf[pos:pos+take]
                    if sl.ndim == 1:
                        sl = sl.reshape(-1, 1)
                    sl = sl * g
                    mix = sl if mix is None else (mix + sl)
                if mix is None:
                    block = np.zeros((take, ch_out), dtype=np.float32)
                else:
                    if mix.shape[1] >= ch_out:
                        block = mix[:, :ch_out]
                    else:
                        pad = np.zeros((mix.shape[0], ch_out - mix.shape[1]), dtype=mix.dtype)
                        block = np.concatenate([mix, pad], axis=1)
                outdata[wrote:wrote+take, :] = block
                wrote += take
                pos += take
            self._frames_out = pos
            return
        else:
            y = self._y_play
            n_total = int(y.shape[0]) if y is not None else 0
            while wrote < frames:
                if pos >= loop_b:
                    if loop_b > loop_a:
                        pos = loop_a
                    else:
                        break
                take = min(frames - wrote, loop_b - pos)
                if take <= 0:
                    break
                if y is None or n_total == 0:
                    block = np.zeros((take, ch_out), dtype=np.float32)
                else:
                    j = min(pos + take, n_total)
                    chunk = y[pos:j]
                    if chunk.ndim == 1:
                        out = np.column_stack([chunk, chunk]).astype(np.float32, copy=False)
                    else:
                        if ch_out <= chunk.shape[1]:
                            out = chunk[:, :ch_out].astype(np.float32, copy=False)
                        else:
                            pad = np.zeros((chunk.shape[0], ch_out - chunk.shape[1]), dtype=chunk.dtype)
                            out = np.concatenate([chunk, pad], axis=1).astype(np.float32, copy=False)
                    # pad if needed
                    if out.shape[0] < take:
                        pad = np.zeros((take - out.shape[0], out.shape[1]), dtype=out.dtype)
                        out = np.vstack([out, pad])
                    block = out
                outdata[wrote:wrote+take, :] = block
                wrote += take
                pos += take
            self._frames_out = pos

    def is_playing(self) -> bool:
        try:
            return bool(self.stream is not None and self.stream.active)
        except Exception:
            return False

    def is_paused(self) -> bool:
        return not self.is_playing()

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
        self._playing = True

    def pause(self):
        if self.stream is not None and self.stream.active:
            self.stream.stop()
        self._playing = False

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
            self._playing = False
        except Exception:
            pass
