# chords.py
import numpy as np
import librosa
import soundfile as sf
import vamp  # from vamphost

# Enharmonic naming: choose sharps or flats later based on key context
SHARP_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
FLAT_NAMES  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]

def chord_templates():
    """Return normalized templates for: 12 maj, 12 min, 12 dom7, 12 maj7, 12 min7.
    Shapes returned as a tuple of 5 arrays, each (12, 12): maj, min, dom7, maj7, min7.
    """
    maj  = np.zeros((12, 12), dtype=float)
    mino = np.zeros((12, 12), dtype=float)
    dom7 = np.zeros((12, 12), dtype=float)
    maj7 = np.zeros((12, 12), dtype=float)
    min7 = np.zeros((12, 12), dtype=float)

    for root in range(12):
        # Triads
        for t in (0, 4, 7):      # major triad
            maj[root, (root + t) % 12] = 1
        for t in (0, 3, 7):      # minor triad
            mino[root, (root + t) % 12] = 1
        # 7th chords
        for t in (0, 4, 7, 10):  # dominant 7th (1,3,5,b7)
            dom7[root, (root + t) % 12] = 1
        for t in (0, 4, 7, 11):  # major 7th (1,3,5,7)
            maj7[root, (root + t) % 12] = 1
        for t in (0, 3, 7, 10):  # minor 7th (1,b3,5,b7)
            min7[root, (root + t) % 12] = 1

    # L2 normalize each row so template matching is cosine-like
    for M in (maj, mino, dom7, maj7, min7):
        M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return maj, mino, dom7, maj7, min7

def chord_likelihoods(chroma):
    maj, mino, dom7, maj7, min7 = chord_templates()
    chroma_n = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-12)
    maj_scores  = maj  @ chroma_n  # (12,T)
    min_scores  = mino @ chroma_n  # (12,T)
    dom7_scores = dom7 @ chroma_n  # (12,T)
    maj7_scores = maj7 @ chroma_n  # (12,T)
    min7_scores = min7 @ chroma_n  # (12,T)
    return np.vstack([maj_scores, min_scores, dom7_scores, maj7_scores, min7_scores])  # (60,T)

# --- Key estimation (Krumhansl-Schmuckler profiles) and naming helpers ---
_MAJOR_PROFILE = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
_MINOR_PROFILE = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

def _estimate_key_from_chroma(chroma):
    """Return (tonic_idx, mode) with mode in {"maj","min"}.
    Base: Krumhansl–Schmuckler profile correlation on time-averaged chroma.
    Enhancement: add a small bias toward MAJOR keys whose I/IV/V pitch-class energy is strong
    (computed from per-class maxima across time). This helps disambiguate A major vs C# minor, etc.
    """
    if chroma.ndim != 2 or chroma.shape[0] != 12:
        # Fallback to a safe default
        return 0, "maj"
    chroma_mean = chroma.mean(axis=1)
    if np.allclose(chroma_mean.sum(), 0):
        return 0, "maj"  # default C major if silence
    chroma_vec = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)

    # Base profile correlation for all 24 keys
    maj_scores = np.zeros(12, dtype=float)
    min_scores = np.zeros(12, dtype=float)
    profM = _MAJOR_PROFILE / np.linalg.norm(_MAJOR_PROFILE)
    profm = _MINOR_PROFILE / np.linalg.norm(_MINOR_PROFILE)
    for k in range(12):
        maj_scores[k] = float(np.dot(chroma_vec, np.roll(profM, k)))
        min_scores[k] = float(np.dot(chroma_vec, np.roll(profm, k)))

    # Cadence bias based on per-pitch-class maxima (robust to sparse textures)
    pc_max = chroma.max(axis=1)
    if pc_max.sum() > 1e-12:
        pc_max = pc_max / (np.linalg.norm(pc_max) + 1e-12)
        cadence = np.zeros(12, dtype=float)
        for k in range(12):
            I = k % 12; IV = (k + 5) % 12; V = (k + 7) % 12
            cadence[k] = pc_max[I] + pc_max[IV] + pc_max[V]
        # Normalize cadence to [0,1]
        cmin, cmax = float(cadence.min()), float(cadence.max())
        if cmax > cmin:
            cadence = (cadence - cmin) / (cmax - cmin)
        # Small weight keeps profile correlation dominant but nudges toward clear I/IV/V
        gamma = 0.15
        maj_scores = maj_scores + gamma * cadence

    # Choose best over 24 keys
    k_maj = int(np.argmax(maj_scores))
    k_min = int(np.argmax(min_scores))
    best_maj = float(maj_scores[k_maj])
    best_min = float(min_scores[k_min])

    if best_maj >= best_min:
        return k_maj, "maj"
    else:
        # If minor wins but its relative MAJOR has a much stronger cadence, flip to MAJOR
        rel_maj = (k_min + 3) % 12
        rel_maj_score = float(maj_scores[rel_maj])
        if rel_maj_score >= best_min + 0.05:
            return rel_maj, "maj"
        return k_min, "min"

# Major flat keys: F, Bb, Eb, Ab, Db, Gb, Cb  -> indices {5,10,3,8,1,6,11}
_FLAT_MAJOR_TONICS = {5, 10, 3, 8, 1, 6, 11}

def _use_flats_for_key(tonic_idx: int, mode: str) -> bool:
    """Return True if we should prefer flat spellings for this key."""
    if mode == "min":
        # Relative major is +3 semitones from minor tonic
        tonic_idx = (tonic_idx + 3) % 12
    return tonic_idx in _FLAT_MAJOR_TONICS

def _pc_name(pc: int, use_flats: bool) -> str:
    return FLAT_NAMES[pc] if use_flats else SHARP_NAMES[pc]

# --- Key hint parser ---
def _parse_key_hint(key_hint) -> tuple[int | None, str | None]:
    """Return (tonic_idx, mode) from a flexible key_hint dict.
    Accepts keys like: 'tonic_idx' (int), 'tonic_name'/'tonic' (str), and 'mode'/'scale' ('maj'/'major'/'min'/'minor').
    Returns (None, None) if unusable.
    """
    if not isinstance(key_hint, dict):
        return None, None
    tonic_idx = None
    mode = None
    try:
        if 'tonic_idx' in key_hint and key_hint['tonic_idx'] is not None:
            tonic_idx = int(key_hint['tonic_idx']) % 12
        else:
            name = key_hint.get('tonic_name') or key_hint.get('tonic')
            if isinstance(name, str):
                lookup = {
                    'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,
                    'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11
                }
                if name.strip() in lookup:
                    tonic_idx = lookup[name.strip()]
        m = key_hint.get('mode') or key_hint.get('scale')
        if isinstance(m, str):
            m = m.strip().lower()
            if m.startswith('maj'):
                mode = 'maj'
            elif m.startswith('min'):
                mode = 'min'
    except Exception:
        pass
    return tonic_idx, mode

def viterbi(log_probs, trans=0.995):
    """
    log_probs: (N_states, T) framewise log-likelihoods
    Simple self-biased transition matrix with uniform off-diagonal.
    """
    N, T = log_probs.shape
    stay = np.log(trans)
    switch = np.log((1.0-trans)/(N-1))
    dp = np.zeros_like(log_probs)
    back = np.zeros((N, T), dtype=np.int16)
    dp[:,0] = log_probs[:,0]
    for t in range(1, T):
        # For each state, max over prev states
        prev = dp[:,t-1][:,None] + np.full((N,N), switch)
        np.fill_diagonal(prev, dp[:,t-1] + stay)
        back[:,t] = np.argmax(prev, axis=0)
        dp[:,t] = log_probs[:,t] + np.max(prev, axis=0)
    path = np.zeros(T, dtype=np.int16)
    path[-1] = np.argmax(dp[:, -1])
    for t in range(T-2, -1, -1):
        path[t] = back[path[t+1], t+1]
    return path

# --- Beat & tempo estimation -------------------------------------------------

def estimate_beats(audio_path: str, sr=22050, hop=512, units_per_bar: int = 4, tightness: float = 100.0):
    """Estimate tempo, beats (sec), and downbeats (bar starts in sec).

    Returns dict:
      {
        "tempo": float,              # BPM
        "beats": List[float],        # seconds
        "downbeats": List[float],    # seconds, inferred bar starts (phase-aligned)
        "beat_strengths": List[float], # normalized [0,1] onset strengths at beat times
        "sr": int,
        "hop": int,
      }

    Method:
      1) Track beats from onset envelope (librosa).
      2) Infer bar phase (downbeat) by selecting the modulo-N offset whose
         beat subset aligns best with onset energy (simple heuristic), where
         N = units_per_bar (default 4 for 4/4).
    """
    from audio_engine import _load_audio_any
    y_stereo, sr_loaded = _load_audio_any(audio_path)
    if sr is None:
        sr = sr_loaded
    y = y_stereo.mean(axis=1)

    # Onset envelope & beat tracking
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=oenv, sr=sr, hop_length=hop, tightness=tightness, units='frames'
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)

    # If no beats, bail gracefully
    if beat_times.size == 0:
        return {"tempo": float(tempo), "beats": [], "downbeats": [], "beat_strengths": [], "sr": int(sr), "hop": int(hop)}

    # Heuristic downbeat phase: choose offset k in [0..N-1] maximizing average onset
    N = max(1, int(units_per_bar))
    # Onset energy at each beat frame
    oenv_at_beats = []
    for f in beat_frames:
        idx = int(min(len(oenv) - 1, max(0, f)))
        oenv_at_beats.append(float(oenv[idx]))
    oenv_at_beats = np.asarray(oenv_at_beats, dtype=float)

    # Normalize beat strengths to [0,1] for later use
    if oenv_at_beats.size:
        bs_min = float(oenv_at_beats.min())
        bs_max = float(oenv_at_beats.max())
        if bs_max > bs_min:
            beat_strengths = ((oenv_at_beats - bs_min) / (bs_max - bs_min)).tolist()
        else:
            beat_strengths = [1.0 for _ in oenv_at_beats]
    else:
        beat_strengths = []

    best_k = 0
    best_score = -1.0
    for k in range(N):
        sel = oenv_at_beats[k::N]
        score = float(sel.mean()) if sel.size else -1.0
        if score > best_score:
            best_score = score
            best_k = k
    downbeat_times = beat_times[best_k::N]

    return {
        "tempo": float(tempo),
        "beats": [float(t) for t in beat_times],
        "downbeats": [float(t) for t in downbeat_times],
        "beat_strengths": beat_strengths,
        "sr": int(sr),
        "hop": int(hop),
    }
def viterbi_timevarying(log_probs, stay_probs):
    """
    Time-varying self-biased Viterbi.

    Args:
        log_probs: (N, T) log-likelihoods.
        stay_probs: (T,) or (N,T) self-transition probability in [0,1] per time step (applies between t-1 -> t).
                    Off-diagonal mass is distributed uniformly.

    Returns:
        path: (T,) best state indices.
    """
    N, T = log_probs.shape
    if stay_probs.ndim == 1:
        stay_probs = np.broadcast_to(stay_probs[None, :], (N, T))
    dp = np.zeros_like(log_probs)
    back = np.zeros((N, T), dtype=np.int16)
    dp[:, 0] = log_probs[:, 0]
    for t in range(1, T):
        stay_t = np.clip(stay_probs[:, t], 1e-6, 1.0 - 1e-6)
        switch_t = (1.0 - stay_t) / max(1, N - 1)
        # Build (N,N) log transition for time t
        prev = np.full((N, N), -np.inf, dtype=np.float64)
        # Off-diagonal
        prev[:] = dp[:, t - 1][:, None] + np.log(switch_t)[:, None]
        # Diagonal (stay)
        diag = dp[:, t - 1] + np.log(stay_t)
        np.fill_diagonal(prev, diag)
        back[:, t] = np.argmax(prev, axis=0)
        dp[:, t] = log_probs[:, t] + np.max(prev, axis=0)
    path = np.zeros(T, dtype=np.int16)
    path[-1] = np.argmax(dp[:, -1])
    for t in range(T - 2, -1, -1):
        path[t] = back[path[t + 1], t + 1]
    return path
# --- Minor triad vs minor 7th refinement ------------------------------------


# --- Harmonic mix and beat-sync chroma helpers ------------------------------

def _mix_from_stems(stems: dict | None, sr_expected: int | None = None):
    """
    Combine available stems into a 'harmonic' mono mix for chord features.
    Excludes percussion. Accepts keys like: 'vocals','other','guitar','piano','bass','drums','accompaniment','harmonic'.
    Returns (y_harm, sr_or_None). If stems is None or empty, returns (None, None).
    """
    if not stems:
        return None, None
    # Priority 1: explicit harmonic/accompaniment provided
    for k in ('harmonic', 'accompaniment'):
        if k in stems and stems[k] is not None:
            y = stems[k]
            if y.ndim == 2:
                y = y.mean(axis=1)
            return y.astype(np.float32), sr_expected
    # Otherwise sum non-percussive sources
    prefer = ['vocals', 'guitar', 'piano', 'other', 'bass']
    acc = None
    for k in prefer:
        if k in stems and stems[k] is not None:
            yk = stems[k]
            if yk.ndim == 2:
                yk = yk.mean(axis=1)
            acc = yk.astype(np.float32) if acc is None else (acc + yk.astype(np.float32))
    if acc is None:
        return None, None
    # Gentle limiter to avoid clipping when summing
    m = np.max(np.abs(acc)) + 1e-9
    acc = acc / m
    return acc, sr_expected


def _beat_sync_chroma(chroma: np.ndarray, frame_times: np.ndarray, beats_sec: list[float], reduce='median'):
    """
    Aggregate framewise chroma (12,F) to beat-synchronous (12,B) using median/mean.
    Returns (C_beat, beat_idxs) where beat_idxs holds the frame indices used per beat start.
    """
    if chroma.size == 0 or not beats_sec:
        return chroma, np.arange(chroma.shape[1])
    beats = np.asarray(beats_sec, dtype=float)
    idxs = np.searchsorted(frame_times, beats, side='left')
    idxs = np.clip(idxs, 0, chroma.shape[1]-1)
    segs = []
    used = []
    for i in range(len(idxs)):
        a = idxs[i]
        b = idxs[i+1] if i+1 < len(idxs) else chroma.shape[1]
        if b <= a:
            b = min(a+1, chroma.shape[1])
        X = chroma[:, a:b]
        if X.size == 0:
            X = chroma[:, a:a+1]
        if reduce == 'mean':
            v = X.mean(axis=1)
        else:
            v = np.median(X, axis=1)
        # log + l2 normalize
        v = np.log1p(v)
        v = v / (np.linalg.norm(v) + 1e-12)
        segs.append(v)
        used.append(a)
    Cb = np.stack(segs, axis=1) if segs else chroma
    return Cb, np.asarray(used, dtype=int)

# --- Small resample helper ---
def _resample_mono(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Return mono signal at target_sr. Accepts mono or stereo arrays."""
    if target_sr == orig_sr:
        if y.ndim == 1:
            return y.astype(np.float32)
        return y.mean(axis=1).astype(np.float32)
    # ensure mono before resample for speed
    y_mono = y.mean(axis=1) if y.ndim == 2 else y
    return librosa.resample(y_mono.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)

# --- Tuning-aware chroma helper ---------------------------------------------

def _chroma_cqt_tuned(y: np.ndarray, sr: int, hop: int, tuning_bins: float | None = None) -> np.ndarray:
    """Compute chroma CQT with an optional global tuning correction (in bins)."""
    if tuning_bins is None:
        try:
            tuning_bins = float(librosa.estimate_tuning(y=y, sr=sr))
        except Exception:
            tuning_bins = 0.0
    return librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, tuning=tuning_bins)

# --- Bass root hint from bass stem -------------------------------------------------

def _bass_root_logits(bass_y: np.ndarray | None, sr: int, hop: int = 2048,
                      frame_times: np.ndarray | None = None,
                      beats_sec: list[float] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits_pc, times) where logits_pc is (12,Tb) over pitch classes.
    If `beats_sec` provided, aggregates to beats; otherwise framewise.
    """
    if bass_y is None or bass_y.size == 0:
        return np.zeros((12, 0), dtype=np.float32), np.asarray(beats_sec or [], dtype=float)
    # Focus on low range for bass root evidence
    C = librosa.feature.chroma_cqt(y=bass_y.astype(np.float32), sr=sr, hop_length=hop, tuning=None)
    times = librosa.frames_to_time(np.arange(C.shape[1]), sr=sr, hop_length=hop)
    if beats_sec:
        Cb, _ = _beat_sync_chroma(C, times, beats_sec, reduce='median')
    else:
        Cb = C
    # Softmax-ish logits per time over PCs
    Cb = Cb + 1e-6
    Cb = Cb / (np.linalg.norm(Cb, axis=0, keepdims=True) + 1e-12)
    Cb = Cb ** 2.5          # sharpen more (clearer bass roots)
    Cb = Cb / (np.sum(Cb, axis=0, keepdims=True) + 1e-12)
    logits = np.log(Cb + 1e-9)
    tvec = np.asarray(beats_sec, dtype=float) if beats_sec else times
    return logits.astype(np.float32), tvec

def snap_time_to_beats(t: float, beats: list[float]) -> float:
    """Return the beat time closest to t (in seconds)."""
    if not beats:
        return float(t)
    arr = np.asarray(beats, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(t))))
    return float(arr[idx])


def snap_interval_to_beats(a: float, b: float, beats: list[float]) -> tuple[float, float]:
    """Snap an interval [a,b] to nearest beats, preserving order and minimum width."""
    if not beats:
        return float(min(a, b)), float(max(a, b))
    a_s = snap_time_to_beats(a, beats)
    b_s = snap_time_to_beats(b, beats)
    if b_s < a_s:
        a_s, b_s = b_s, a_s
    if abs(b_s - a_s) < 1e-3:
        b_s = a_s + 1e-2  # ensure non-zero loop
    return float(a_s), float(b_s)

# Public helper: estimate musical key from an audio file
# Returns a dict with: {"tonic_idx": int, "mode": "maj"|"min", "tonic_name": str, "pretty": str}

def estimate_key(audio_path: str, sr=22050, hop=2048):
    from audio_engine import _load_audio_any
    y_stereo, sr_loaded = _load_audio_any(audio_path)
    if sr is None:
        sr = sr_loaded
    # resample mono to requested sr
    y = _resample_mono(y_stereo, sr_loaded, sr)
    # tuning-aware chroma
    try:
        tuning_bins = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        tuning_bins = 0.0
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, tuning=tuning_bins)
    tonic_idx, mode = _estimate_key_from_chroma(chroma)
    use_flats = _use_flats_for_key(tonic_idx, mode)
    tonic_name = _pc_name(tonic_idx, use_flats)
    pretty = f"{tonic_name} {'major' if mode=='maj' else 'minor'}"
    return {
        "tonic_idx": int(tonic_idx),
        "mode": mode,
        "tonic_name": tonic_name,
        "pretty": pretty,
    }

# --- Minor triad vs minor 7th refinement ------------------------------------

def _pc_index(semitone_name: str) -> int:
    names = {
        'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,'F#':6,'Gb':6,
        'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11
    }
    return names[semitone_name]


def _refine_minor_sevenths(
    segments: list,
    audio_path: str,
    sr: int = 22050,
    hop: int = 512,
    seventh_db_threshold: float = -26.0,  # dB relative to per-frame peak (gentler)
    seventh_rel_threshold: float = 0.40,  # fraction of triad peak in a frame (gentler)
    min_frames_ratio_for_upgrade: float = 0.60,  # >=60% frames show clear b7 to upgrade m→m7
    max_frames_ratio_for_keep_m7: float = 0.40,  # <40% frames show b7 → downgrade m7→m
    min_duration_for_upgrade: float = 0.50,      # keep short guard
) -> list:
    """Post-process labels to better separate m vs m7 (stricter, frame-wise voting).

    For each minor/minor7 segment:
      • Compute frame-wise chroma (CQT), normalize each frame to its max.
      • In each frame, check if b7 >= rel_threshold * triad_peak AND b7_dB >= dB_threshold.
      • Let ratio = (# frames satisfying) / (# frames). Decisions:
          - If label is m7 and ratio < max_frames_ratio_for_keep_m7 → downgrade to m.
          - If label is m and ratio >= min_frames_ratio_for_upgrade AND duration >= min_duration_for_upgrade → upgrade to m7.
    """
    from audio_engine import _load_audio_any

    if not segments:
        return segments

    y_stereo, sr_loaded = _load_audio_any(audio_path)
    if sr is None:
        sr = sr_loaded
    y = _resample_mono(y_stereo, sr_loaded, sr)
    # pre-compute tuning once for this track
    try:
        tuning_bins_all = float(librosa.estimate_tuning(y=y, sr=sr))
    except Exception:
        tuning_bins_all = 0.0

    def seg_chroma_frames(a: float, b: float):
        a_s = max(0, int(a * sr)); b_s = max(a_s + 1, int(b * sr))
        y_seg = y[a_s:b_s]
        if y_seg.size < hop:
            pad = np.zeros(min(hop, max(0, hop - y_seg.size)), dtype=y.dtype)
            y_seg = np.concatenate([y_seg, pad])
        C = librosa.feature.chroma_cqt(y=y_seg, sr=sr, hop_length=hop, tuning=tuning_bins_all)  # (12, F)
        if C.shape[1] == 0:
            return C
        # normalize each frame to its max so per-frame peak = 1
        m = np.max(C, axis=0, keepdims=True) + 1e-12
        return C / m

    b7_min_lin = 10 ** (seventh_db_threshold / 20.0)  # relative to per-frame peak (1.0)

    out = []
    for s in segments:
        lab = s.get('label', '')
        if len(lab) < 2:
            out.append(s); continue

        # Parse root with optional accidental
        root = lab[0]
        i = 1
        if i < len(lab) and lab[i] in ('#', 'b'):
            root += lab[i]; i += 1
        qual = lab[i:]

        if qual not in ('m', 'm7'):
            out.append(s); continue

        a_t = float(s['start']); b_t = float(s['end'])
        dur = max(0.0, b_t - a_t)
        C = seg_chroma_frames(a_t, b_t)
        if C.size == 0 or C.shape[1] == 0:
            out.append(s); continue

        r = _pc_index(root)
        triad_pcs = [(r + 0) % 12, (r + 3) % 12, (r + 7) % 12]
        b7_pc = (r + 10) % 12

        triad_peak_per_frame = np.max(C[triad_pcs, :], axis=0)  # (F,)
        b7_per_frame = C[b7_pc, :]  # (F,)

        good = (b7_per_frame >= b7_min_lin) & (b7_per_frame >= seventh_rel_threshold * (triad_peak_per_frame + 1e-12))
        ratio = float(np.mean(good.astype(np.float32)))

        new_lab = lab
        # Robustness: also check median b7 vs triad
        median_b7 = float(np.median(b7_per_frame))
        median_triad = float(np.median(triad_peak_per_frame + 1e-12))
        strong_median = (median_b7 >= (0.8 * seventh_rel_threshold) * median_triad) and (median_b7 >= 0.8 * b7_min_lin)

        if qual == 'm7':
            if ratio < max_frames_ratio_for_keep_m7:
                new_lab = root + 'm'
        else:  # qual == 'm'
            if ratio >= min_frames_ratio_for_upgrade and dur >= min_duration_for_upgrade and strong_median:
                new_lab = root + 'm7'

        if new_lab != lab:
            s = dict(s); s['label'] = new_lab
        out.append(s)

    return out


# --- Root sanity check: keep family, re-pick root by template fit -----------

def _parse_chord_label(lab: str):
    """Return (family_id, root_pc, root_text, qual_text).
    family_id: 0=maj,1=min,2=dom7,3=maj7,4=min7
    """
    fam = 0
    # detect suffixes (order matters: maj7 before 7, m7 before m)
    if lab.endswith('maj7'):
        fam = 3; core = lab[:-4]
    elif lab.endswith('m7'):
        fam = 4; core = lab[:-2]
    elif lab.endswith('m'):
        fam = 1; core = lab[:-1]
    elif lab.endswith('7'):
        fam = 2; core = lab[:-1]
    else:
        fam = 0; core = lab
    core = core.strip()
    if not core:
        return fam, 0, 'C', ''
    root = core[0]
    if len(core) > 1 and core[1] in ('#', 'b'):
        root += core[1]
    try:
        pc = _pc_index(root)
    except KeyError:
        pc = 0
    qual = lab[len(root):]
    return fam, pc, root, qual


def _family_name(fam: int) -> str:
    return {0:'', 1:'m', 2:'7', 3:'maj7', 4:'m7'}.get(fam, '')


def _root_sanity_pass(segments: list, audio_path: str, use_flats: bool | None = None,
                      sr: int = 22050, hop: int = 2048, min_improvement: float = 0.12) -> list:
    """Re-evaluate root inside each segment for its chord *family* using template fit.
    Keeps the family (maj/min/7th type) but allows the root to change if its
    family-template cosine score improves by at least `min_improvement`.
    This helps fix cases like labeling Db when F is a much better fit for the same family.
    """
    from audio_engine import _load_audio_any

    if not segments:
        return segments

    y_stereo, sr_loaded = _load_audio_any(audio_path)
    if sr is None:
        sr = sr_loaded
    y = _resample_mono(y_stereo, sr_loaded, sr)

    # Key-based naming unless provided (tuning-aware)
    if use_flats is None:
        try:
            try:
                tuning_bins = float(librosa.estimate_tuning(y=y, sr=sr))
            except Exception:
                tuning_bins = 0.0
            chroma_all = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop, tuning=tuning_bins)
            tonic_idx, mode = _estimate_key_from_chroma(chroma_all)
            use_flats = _use_flats_for_key(tonic_idx, mode)
        except Exception:
            use_flats = True

    maj, mino, dom7, maj7, min7 = chord_templates()
    fam_to_M = {0: maj, 1: mino, 2: dom7, 3: maj7, 4: min7}

    out = []
    for s in segments:
        lab = s.get('label', '')
        fam, root_pc, root_txt, _ = _parse_chord_label(lab)
        M = fam_to_M.get(fam)
        if M is None:
            out.append(s); continue
        a = max(0, int(float(s['start']) * sr))
        b = max(a + 1, int(float(s['end']) * sr))
        y_seg = y[a:b]
        if y_seg.size < hop:
            pad = np.zeros(min(hop, max(0, hop - y_seg.size)), dtype=y.dtype)
            y_seg = np.concatenate([y_seg, pad])
        # tuning-aware chroma for segment
        try:
            tuning_bins = float(librosa.estimate_tuning(y=y_seg, sr=sr))
        except Exception:
            tuning_bins = 0.0
        C = librosa.feature.chroma_cqt(y=y_seg, sr=sr, hop_length=hop, tuning=tuning_bins)
        v = C.mean(axis=1)
        nrm = np.linalg.norm(v) + 1e-12
        v = v / nrm
        # Current vs best
        cur_score = float(M[root_pc, :] @ v)
        scores = M @ v  # 12 roots
        best_pc = int(np.argmax(scores))
        best_score = float(scores[best_pc])
        if best_score - cur_score >= min_improvement:
            # Re-label with best root
            new_root_txt = _pc_name(best_pc, use_flats)
            new_lab = new_root_txt + _family_name(fam)
            ns = dict(s); ns['label'] = new_lab
            out.append(ns)
        else:
            out.append(s)
    return out

def estimate_chords_stem_aware(
    audio_path: str,
    sr=22050,
    hop=2048,
    beats: list[float] | None = None,
    downbeats: list[float] | None = None,
    beat_strengths: list[float] | None = None,
    stems: dict | None = None,
    use_key_prior: bool = True,
    alpha_harm: float = 0.40,
    beta_bass: float = 0.70,
    weights: dict | None = None,
    log_fn=None,
    style: str = "default",
    key_hint: dict | None = None,
):
    """Beat-synchronous, stem-aware chord estimator.
    Combines harmonic chroma (guitar/piano/other/vocals) with bass-root evidence.
    Returns list of segments [{start,end,label,conf,beat_sync=True}].
    """
    from audio_engine import _load_audio_any

    # Normalize style names (robust to non-string inputs)
    style_map = {
        'default': 'rock_pop',
        'rock': 'rock_pop', 'rock/pop': 'rock_pop', 'pop': 'rock_pop',
        'blues': 'blues', 'reggae': 'reggae', 'jazz': 'jazz',
        'rock_pop': 'rock_pop'
    }
    _style_key = style
    try:
        if _style_key is None:
            _style_key = 'rock_pop'
        # If someone passed a QAction, function, etc., coerce to string safely
        if not isinstance(_style_key, str):
            _style_key = str(_style_key)
        _style_key = _style_key.lower()
    except Exception:
        _style_key = 'rock_pop'
    style = style_map.get(_style_key, 'rock_pop')
    _use_fl = False
    y_stereo, sr_loaded = _load_audio_any(audio_path)
    sr = sr_loaded if sr is None else int(sr)
    # Build mono reference at target sr
    y_mono = _resample_mono(y_stereo, sr_loaded, sr)
    if callable(log_fn):
        try:
            log_fn(f"stem_aware: sr_loaded={sr_loaded}, sr_used={sr}")
        except Exception:
            pass

    # Build harmonic mix from stems (exclude vocals/drums); fallback to mono
    y_harm = None
    used_stems = []
    if stems:
        w = {'guitar': 1.0, 'piano': 1.0, 'other': 0.5, 'bass': 0.0, 'vocals': 0.0, 'drums': 0.0}
        if isinstance(weights, dict):
            w.update(weights)
        acc = None
        for k, arr in stems.items():
            try:
                # resample each stem to sr
                mono_in = arr.mean(axis=1).astype(np.float32) if arr.ndim == 2 else arr.astype(np.float32)
                # resample to match target sr
                if 'sr' in getattr(arr, '__dict__', {}):
                    mono = _resample_mono(arr, getattr(arr, 'sr'), sr)
                else:
                    # stems are same sr as source audio; use sr_loaded
                    mono = librosa.resample(mono_in, orig_sr=sr_loaded, target_sr=sr) if sr_loaded != sr else mono_in
                gain = float(w.get(k.lower(), w.get(k, 0.0)))
                if gain <= 0.0:
                    continue
                used_stems.append(f"{k}:{gain:.2f}")
                if acc is None:
                    acc = gain * mono
                else:
                    L = max(acc.shape[0], mono.shape[0])
                    if acc.shape[0] < L: acc = np.pad(acc, (0, L - acc.shape[0]))
                    if mono.shape[0] < L: mono = np.pad(mono, (0, L - mono.shape[0]))
                    acc = acc + gain * mono
            except Exception:
                continue
        if acc is not None:
            y_harm = acc
    if y_harm is None:
        y_harm = y_mono

    # Framewise chroma and times
    chroma_frames = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop)
    frame_times = librosa.frames_to_time(np.arange(chroma_frames.shape[1]), sr=sr, hop_length=hop)

    # Beats (if not given)
    if beats is None:
        bd = estimate_beats(audio_path, sr=sr, hop=512)
        beats = bd.get("beats", [])
        downbeats = bd.get("downbeats", []) if downbeats is None else downbeats
        beat_strengths = bd.get("beat_strengths", []) if beat_strengths is None else beat_strengths

    # Beat-sync chroma
    if beats:
        chroma, _ = _beat_sync_chroma(chroma_frames, frame_times, beats, reduce='median')
        times = np.asarray(beats, dtype=float)
    else:
        chroma = chroma_frames
        times = frame_times

    # Normalize chroma columns
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-12)

    # Harmonic template log-likelihoods (60,T)
    harm_scores = chord_likelihoods(chroma) + 1e-6
    log_harm = np.log(harm_scores)

    # Bass root evidence from bass stem (12,Tb) → expand to (60,Tb)
    bass_y = None
    if stems and isinstance(stems.get('bass', None), np.ndarray):
        b = stems['bass']
        bass_y = _resample_mono(b, sr_loaded, sr)
    bass_logits, _ = _bass_root_logits(bass_y, sr, hop=hop,
                                       frame_times=frame_times,
                                       beats_sec=beats if beats else None)

    # --- Bass-root-first decoding path ---
    use_bass_first = bass_logits.shape[1] > 0
    if use_bass_first:
        # Align lengths with harmonic emissions
        T_align = min(log_harm.shape[1], bass_logits.shape[1])
        log_h = log_harm[:, :T_align]
        log_b = bass_logits[:, :T_align]
        times = times[:T_align]
        if callable(log_fn):
            try:
                log_fn(f"DBG[bass_first]: T_align={T_align}, beats={len(beats or [])}, log_h={log_h.shape}, log_b={log_b.shape}, times={len(times)}")
            except Exception:
                pass

        # Augment bass logits to account for bass playing the fifth instead of the root (common in reggae/rock)
        # Mix a portion of probability mass from ±7 semitones (perfect fifth up/down) into each class.
        try:
            with np.errstate(over='ignore'):
                pb_b = np.exp(log_b - np.max(log_b, axis=0, keepdims=True))
                pb_b /= (np.sum(pb_b, axis=0, keepdims=True) + 1e-12)
            def _roll(x, k):
                return np.roll(x, k, axis=0)
            eps = 0.25  # share from fifths
            pb_aug = (1.0 - eps) * pb_b + (eps/2.0) * (_roll(pb_b, +7) + _roll(pb_b, -7))
            log_b = np.log(pb_aug + 1e-9)
        except Exception:
            pass

        # Build a surrogate "bass" from the harmonic mix (guitar/piano/other) for early bars without bass
        try:
            # Reuse the already computed y_harm (mono acc mix) and create logits the same way as bass
            harm_logits_full, _ = _bass_root_logits(y_harm, sr, hop=hop,
                                                    frame_times=frame_times,
                                                    beats_sec=beats if beats else None)
            log_b_harm = harm_logits_full[:, :T_align] if harm_logits_full.shape[1] >= T_align else harm_logits_full
            if log_b_harm.shape[1] != T_align:
                # pad with minimal uniform evidence if needed
                pad_T = T_align - log_b_harm.shape[1]
                if pad_T > 0:
                    log_b_harm = np.pad(log_b_harm, ((0,0),(0,pad_T)), mode='edge')
        except Exception:
            log_b_harm = None

        # Confidence gating: if bass is weak, fall back to harmonic-root evidence for those beats
        TH_CONF = 0.40
        with np.errstate(over='ignore'):
            pb = np.exp(log_b - np.max(log_b, axis=0, keepdims=True)); pb /= (np.sum(pb, axis=0, keepdims=True) + 1e-12)
        maxpb = np.max(pb, axis=0)
        if log_b_harm is not None and log_b_harm.shape[1] == T_align:
            with np.errstate(over='ignore'):
                ph = np.exp(log_b_harm - np.max(log_b_harm, axis=0, keepdims=True)); ph /= (np.sum(ph, axis=0, keepdims=True) + 1e-12)
            maxph = np.max(ph, axis=0)
            # Use harmonic surrogate where bass is not yet confident; otherwise keep bass
            use_harm_mask = (maxpb < TH_CONF) & (maxph >= (TH_CONF - 0.05))
            if np.any(use_harm_mask):
                log_b[:, use_harm_mask] = log_b_harm[:, use_harm_mask]
                pb[:, use_harm_mask] = ph[:, use_harm_mask]
                maxpb[use_harm_mask] = maxph[use_harm_mask]

        # Compute current top bass root per beat (after augmentation/merge)
        top_pc = np.argmax(pb, axis=0)            # (T_align,)
        top_conf = maxpb.copy()                    # (T_align,)

        # Add debug on bass confidence stats
        if callable(log_fn):
            try:
                q = np.quantile(top_conf, [0.25, 0.50, 0.75]).tolist()
                log_fn(f"DBG[bass_first]: top_conf quartiles={q}")
            except Exception:
                pass

        # Detect late bass entrance: first confident beat
        t0 = int(np.argmax(top_conf >= TH_CONF)) if np.any(top_conf >= TH_CONF) else 0
        # Snap t0 forward to the next downbeat if available to avoid splitting inside a bar
        if beats and downbeats:
            beat_arr = np.asarray(beats)[:T_align]
            down_arr = np.asarray(downbeats)
            try:
                t0_time = times[t0]
                dn_after = down_arr[down_arr >= t0_time]
                if dn_after.size:
                    t0_time = float(dn_after[0])
                    idx = np.searchsorted(beat_arr, t0_time, side='left')
                    t0 = int(np.clip(idx, 0, T_align-1))
            except Exception:
                pass

        # 1) Root Viterbi (12 states) with key/style prior on roots only
        # If bass enters late (t0>0), run Viterbi on t0..end, then backfill 0..t0-1 with first root.
        root_log_emissions = log_b.copy()
        # (A) Blend in harmonic root evidence when bass confidence is low
        try:
            # Build per-root harmonic emissions by taking max over families at each root
            # log_h has shape (60, T_align): 5 families × 12 roots
            harm_root = np.empty((12, T_align), dtype=np.float64)
            for r in range(12):
                harm_root[r, :] = np.max(log_h[r::12, :], axis=0)
            # Confidence-dependent blend: w_bass in [0.30, 1.0]
            w_bass = 0.30 + 1.40 * np.clip(top_conf, 0.0, 1.0)
            w_bass = np.clip(w_bass, 0.30, 1.0)  # shape (T_align,)
            # Expand to (12,T)
            root_log_emissions = (w_bass[None, :] * root_log_emissions) + ((1.0 - w_bass)[None, :] * harm_root)
            if callable(log_fn):
                try:
                    log_fn(f"DBG[bass_first]: harm-blend median_w_bass={float(np.median(w_bass)):.3f} low_pct={(float(np.mean(w_bass<0.6))*100):.1f}%")
                except Exception:
                    pass
        except Exception:
            pass
        if use_key_prior:
            try:
                hk_tonic, hk_mode = _parse_key_hint(key_hint)
                if hk_tonic is not None and hk_mode in ('maj','min'):
                    tonic_idx_r, mode_r = hk_tonic, hk_mode
                else:
                    tonic_idx_r, mode_r = _estimate_key_from_chroma(chroma[:, :T_align])
                scale = {0,2,4,5,7,9,11} if mode_r == 'maj' else {0,2,3,5,7,8,10}
                rel = np.fromiter((((r - tonic_idx_r) % 12) for r in range(12)), dtype=int)
                in_scale_mask = np.isin(rel, list(scale)).astype(np.float32)
                base_in  = 0.06 if style != 'jazz' else 0.03
                base_out = 0.08 if style != 'jazz' else 0.00
                root_log_emissions += in_scale_mask[:, None] * base_in
                root_log_emissions -= (1.0 - in_scale_mask)[:, None] * base_out
                # Cadence push I/IV/V for reggae/rock_pop and suppress III
                if mode_r == 'maj' and style in ('rock_pop','reggae'):
                    I  = int(tonic_idx_r % 12)
                    IV = int((tonic_idx_r + 5) % 12)
                    V  = int((tonic_idx_r + 7) % 12)
                    root_log_emissions[I,  :] += (0.35 if style=='rock_pop' else 0.40)
                    root_log_emissions[IV, :] += (0.25 if style=='rock_pop' else 0.30)
                    root_log_emissions[V,  :] += (0.33 if style=='rock_pop' else 0.40)
                    # Additional suppression of mediant (III) in major for rock/reggae
                    III = int((tonic_idx_r + 4) % 12)
                    root_log_emissions[III, :] -= 0.06
                    # (B) Suppress leading tone (VII) in major for rock/reggae
                    VII = int((tonic_idx_r + 11) % 12)
                    root_log_emissions[VII, :] -= 0.12
            except Exception:
                pass

        # Cadence heuristic: in 4/4, encourage V on the last beat of a bar and I on the downbeat
        try:
            if beats and downbeats and mode_r == 'maj' and style in ('rock_pop','reggae'):
                I  = int(tonic_idx_r % 12)
                V  = int((tonic_idx_r + 7) % 12)
                beat_arr = np.asarray(beats)[:T_align]
                down_arr = np.asarray(downbeats)
                # Map each beat index to its bar position (0..3) by counting beats since last downbeat
                # Assume units_per_bar = 4
                db_idx = np.searchsorted(beat_arr, down_arr, side='left')
                db_set = set(int(i) for i in db_idx if 0 <= i < T_align)
                last_in_bar = set(int(i+3) for i in db_idx if 0 <= i+3 < T_align)
                for i in range(T_align):
                    if i in db_set:
                        root_log_emissions[I, i] += 0.10  # push I on bar start
                    if i in last_in_bar:
                        root_log_emissions[V, i] += 0.12  # push V on beat 4
        except Exception:
            pass

        # (C) Debug: show first few blended root emissions
        if callable(log_fn):
            try:
                rr0 = ",".join(f"{float(x):.2f}" for x in root_log_emissions[:,0][:5])
                log_fn(f"DBG[bass_first]: root_emit_col0(first5)={rr0}")
            except Exception:
                pass

        # Bass-driven nudge: when bass top root is confident, favor that root explicitly
        try:
            BASS_STRONG = 0.40
            NUDGE = 0.28 if style in ('rock_pop','reggae') else 0.16
            for i in range(T_align):
                if top_conf[i] >= BASS_STRONG:
                    r = int(top_pc[i]) % 12
                    root_log_emissions[r, i] += NUDGE
        except Exception:
            pass

        # Track bass-change points to relax stay on clear changes
        bass_change = np.zeros(T_align, dtype=np.float32)
        try:
            # Require either both strong OR a clear margin flip on current frame
            MARGIN = 0.08  # prob advantage of new top over previous top
            for i in range(1, T_align):
                changed = (int(top_pc[i]) % 12 != int(top_pc[i-1]) % 12)
                if not changed:
                    continue
                strong_flip = (top_conf[i] >= BASS_STRONG and top_conf[i-1] >= BASS_STRONG)
                try:
                    prev_pc = int(top_pc[i-1]) % 12
                    margin_flip = (pb[int(top_pc[i]) % 12, i] - pb[prev_pc, i]) >= MARGIN
                except Exception:
                    margin_flip = False
                if strong_flip or margin_flip:
                    bass_change[i] = 1.0
        except Exception:
            pass
        # Root stay-prob by beats: allow changes more on downbeats/strong beats
        stay_r = np.full(T_align, 0.996, dtype=np.float64)
        if beats:
            down_set = set(np.round(np.asarray(downbeats or []) * 1000).astype(int).tolist())
            beat_ms = np.round(np.asarray(beats)[:T_align] * 1000).astype(int).tolist()
            strengths = (beat_strengths or [1.0]*len(beat_ms))[:T_align]
            for i, (bm, s) in enumerate(zip(beat_ms, strengths)):
                is_down = bm in down_set
                base = 0.985 if is_down else 0.996
                stay_i = base - 0.10 * float(s)
                # If bass clearly changed roots here, relax stay more so we can switch
                if 'bass_change' in locals() and bass_change[i] > 0:
                    stay_i -= 0.10  # stronger relaxation on change beats
                stay_r[i] = float(np.clip(stay_i, 0.90, 0.998))
        if t0 <= 0:
            root_path = viterbi_timevarying(root_log_emissions, stay_r)
        else:
            # Run Viterbi only on the confident tail and backfill the head with the first root
            root_tail = viterbi_timevarying(root_log_emissions[:, t0:], stay_r[t0:])
            root_path = np.empty(T_align, dtype=np.int16)
            root_path[t0:] = root_tail
            root_path[:t0] = root_tail[0]
        if callable(log_fn):
            try:
                rp_head = ",".join(str(int(x)) for x in root_path[:8])
                log_fn(f"DBG[bass_first]: root_path len={len(root_path)} head={rp_head}")
            except Exception:
                pass
        # Debug: show bass_change indices
        if callable(log_fn):
            try:
                ch_idx = ",".join(str(i) for i in np.nonzero(bass_change)[0][:12].tolist())
                log_fn(f"DBG[bass_first]: bass_change idx=({ch_idx}) strong_thresh=0.52")
            except Exception:
                pass

        # 2) Family Viterbi (5 states) with emissions taken at chosen root per time
        fam_emissions = np.zeros((5, T_align), dtype=np.float64)
        for t in range(T_align):
            r = int(root_path[t]) % 12
            for fam in range(5):
                fam_emissions[fam, t] = log_h[fam*12 + r, t]
        # Add style/key family priors (subset of key_bonus that depends on family, at chosen roots)
        try:
            fam_bonus = np.zeros_like(fam_emissions)
            # Reuse previously computed style/key context
            hk_tonic, hk_mode = _parse_key_hint(key_hint)
            if hk_tonic is not None and hk_mode in ('maj','min'):
                tonic_idx_f, mode_f = hk_tonic, hk_mode
            else:
                tonic_idx_f, mode_f = _estimate_key_from_chroma(chroma[:, :T_align])
            I  = int(tonic_idx_f % 12); IV = (I + 5) % 12; V = (I + 7) % 12
            II = (I + 2) % 12; VI = (I + 9) % 12
            if style == 'blues':
                for t in range(T_align):
                    r = int(root_path[t])
                    if r in (I, IV, V):
                        fam_bonus[2, t] += 0.20  # dom7 on I/IV/V
                fam_bonus[3, :] -= 0.10
            elif style == 'reggae' and mode_f == 'maj':
                for t in range(T_align):
                    r = int(root_path[t])
                    if r == I:  fam_bonus[0, t] += 0.38
                    if r == IV: fam_bonus[0, t] += 0.28
                    if r == V:  fam_bonus[0, t] += 0.28
                    if r == II: fam_bonus[4, t] += 0.10
                    if r == VI: fam_bonus[4, t] += 0.10
                    if r == V:
                        fam_bonus[2, t] += 0.15  # prefer V7 a bit more on V
                        fam_bonus[0, t] -= 0.04  # discourage plain V triad slightly
                fam_bonus[2, :] -= 0.05; fam_bonus[3, :] -= 0.06
            elif style == 'jazz':
                if mode_f == 'maj':
                    fam_bonus[4, :] += 0.10  # general m7 favor
                    fam_bonus[3, :] += 0.08  # maj7 favor
                else:
                    fam_bonus[4, :] += 0.06
                    fam_bonus[2, :] += 0.10
            else:  # rock_pop
                if mode_f == 'maj':
                    for t in range(T_align):
                        r = int(root_path[t])
                        if r == I:  fam_bonus[0, t] += 0.35
                        if r == IV: fam_bonus[0, t] += 0.25
                        if r == V:  fam_bonus[0, t] += 0.28
                        if r == V:  fam_bonus[2, t] += 0.08
                    fam_bonus[3, :] -= 0.08
        except Exception:
            fam_bonus = 0.0
        fam_emissions = fam_emissions + fam_bonus

        # Family stay-prob (favor staying within a bar, changes on downbeats/strong beats)
        stay_f = np.full(T_align, 0.997, dtype=np.float64)
        if beats:
            down_set = set(np.round(np.asarray(downbeats or []) * 1000).astype(int).tolist())
            beat_ms = np.round(np.asarray(beats)[:T_align] * 1000).astype(int).tolist()
            strengths = (beat_strengths or [1.0]*len(beat_ms))[:T_align]
            for i, (bm, s) in enumerate(zip(beat_ms, strengths)):
                is_down = bm in down_set
                base = 0.988 if is_down else 0.997
                stay_i = np.clip(base - 0.12 * float(s), 0.93, 0.999)
                stay_f[i] = stay_i
        fam_path = viterbi_timevarying(fam_emissions, stay_f)
        if callable(log_fn):
            try:
                fp_head = ",".join(str(int(x)) for x in fam_path[:8])
                log_fn(f"DBG[bass_first]: fam_path len={len(fam_path)} head={fp_head}")
            except Exception:
                pass

        # Map (root, family) → label (compute local enharmonic preference)
        try:
            tonic_idx2, mode2 = _parse_key_hint(key_hint)
            if tonic_idx2 is None or mode2 not in ('maj','min'):
                tonic_idx2, mode2 = _estimate_key_from_chroma(chroma[:, :T_align])
            _use_fl = _use_flats_for_key(int(tonic_idx2), mode2)
        except Exception:
            _use_fl = False

        def _fam_label(root_pc: int, fam: int, _use_flats_bound=_use_fl) -> str:
            root = _pc_name(root_pc, _use_flats_bound)
            return (
                root if fam == 0 else
                f"{root}m" if fam == 1 else
                f"{root}7" if fam == 2 else
                f"{root}maj7" if fam == 3 else
                f"{root}m7"
            )

        labels = [_fam_label(int(r), int(f)) for r, f in zip(root_path, fam_path)]

        # Confidence = margin vs. second-best family at chosen root
        best = np.max(fam_emissions, axis=0)
        srt = np.sort(fam_emissions, axis=0)
        margin = best - srt[-2]
        avg_margin = float(np.median(margin)) if margin.size else 0.0

        # Collapse runs into segments
        segments = []
        start_idx = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != labels[start_idx]:
                segments.append({
                    'start': float(times[start_idx]),
                    'end': float(times[min(i, len(times)-1)]),
                    'label': labels[start_idx],
                    'conf': float(np.median(margin[start_idx:i])) if i > start_idx else float(margin[start_idx]),
                })
                start_idx = i
        if callable(log_fn):
            try:
                t0 = times[0] if len(times) else 0.0
                t1 = times[-1] if len(times) else 0.0
                log_fn(f"DBG[bass_first]: labels={len(labels)}, segments={len(segments)}, time_span=[{t0:.2f},{t1:.2f}]")
            except Exception:
                pass

        # Post passes
        segments = _refine_minor_sevenths(segments, audio_path)
        segments = _root_sanity_pass(segments, audio_path, use_flats=_use_fl)
        if beats:
            segments = [dict(s, beat_sync=True) for s in segments]
        # Log debug summary: roots-only and full labels
        try:
            from collections import Counter
            def _root_of(lbl: str) -> str:
                r = lbl[:1]
                if len(lbl) >= 2 and lbl[1] in ('#','b'):
                    r = lbl[:2]
                return r
            roots = [_root_of(s['label']) for s in segments]
            fams  = [s['label'][len(_root_of(s['label'])):] or 'maj' for s in segments]
            cnt_roots = Counter(roots)
            cnt_labels = Counter([s['label'] for s in segments])
            top_roots = ", ".join(f"{k}:{v}" for k, v in cnt_roots.most_common(5))
            top_labels = ", ".join(f"{k}:{v}" for k, v in cnt_labels.most_common(5))
            msg = f"bass_first: roots → {top_roots} | labels → {top_labels} | segs={len(segments)}"
            if callable(log_fn):
                log_fn(msg)
            else:
                print(msg)
        except Exception:
            pass
        return segments
    # --- Bass-root emphasis: favor top-2 bass pitch classes per beat ---
    # (rest of original code follows as fallback)
    # --- Bass-root emphasis: favor top-2 bass pitch classes per beat ---
    bass_boost = np.zeros_like(log_harm)
    if bass_logits.shape[1] == log_harm.shape[1]:
        # For each time step, find top-2 bass roots and boost those roots (all families)
        top2 = np.argpartition(bass_logits, -2, axis=0)[-2:,:]
        for t in range(bass_logits.shape[1]):
            roots = set(int(r) for r in top2[:, t].tolist())
            if not roots:
                continue
            for fam in range(5):  # 0 maj,1 min,2 7,3 maj7,4 m7
                for r in roots:
                    bass_boost[fam*12 + r, t] += 0.7  # stronger push
        # Stronger penalty to non-top-2 roots
        bass_boost -= 0.4
    # else: leave bass_boost zeros if we couldn't align times

    # Optional key prior – genre aware
    key_bonus = np.zeros_like(log_harm)
    if use_key_prior:
        try:
            # Prefer UI-provided key_hint if available
            hk_tonic, hk_mode = _parse_key_hint(key_hint)
            if hk_tonic is not None and hk_mode in ('maj','min'):
                tonic_idx, mode = hk_tonic, hk_mode
                if callable(log_fn):
                    try: log_fn(f"key_prior: using key_hint tonic={tonic_idx} mode={mode}")
                    except Exception: pass
            else:
                tonic_idx, mode = _estimate_key_from_chroma(chroma)
                if callable(log_fn):
                    try: log_fn(f"key_prior: estimated tonic={tonic_idx} mode={mode}")
                    except Exception: pass
            # Slightly softer key lock for jazz (allows modulations)
            base_scale_w = 0.03 if style == 'jazz' else 0.06
            base_out_w   = 0.00 if style == 'jazz' else 0.08
            scale = {0,2,4,5,7,9,11} if mode == 'maj' else {0,2,3,5,7,8,10}
            rel = np.fromiter((((r - tonic_idx) % 12) for r in range(12)), dtype=int)
            in_mask = np.isin(rel, list(scale))
            out_mask = ~in_mask
            for fam in range(5):
                row = fam*12
                key_bonus[row:row+12, :] += in_mask[:, None] * base_scale_w
                if base_out_w > 0.0:
                    key_bonus[row:row+12, :] -= out_mask[:, None] * base_out_w
            # Extra: suppress mediant (III) in major for rock/reggae to avoid false iii
            if mode == 'maj' and style in ('rock_pop','reggae'):
                III = int((tonic_idx + 4) % 12)
                key_bonus[0*12 + III, :] -= 0.06
                key_bonus[1*12 + III, :] -= 0.06
                key_bonus[2*12 + III, :] -= 0.06
                key_bonus[3*12 + III, :] -= 0.06
                key_bonus[4*12 + III, :] -= 0.06
            # --- rest of style/genre specific pushes ---
            I  = int(tonic_idx % 12)
            II = int((tonic_idx + 2) % 12)
            III= int((tonic_idx + 4) % 12)
            IV = int((tonic_idx + 5) % 12)
            V  = int((tonic_idx + 7) % 12)
            VI = int((tonic_idx + 9) % 12)

            if style == 'blues':
                # Favor dominant 7 on I/IV/V; de-emphasize maj7 in general
                for t in range(log_harm.shape[1]):
                    for root in (I, IV, V):
                        key_bonus[2*12 + root, t] += 0.20   # dom7 boost
                key_bonus[3*12:(3*12)+12, :] -= 0.10  # maj7 downweight
            elif style == 'reggae':
                # Reggae: prefer clean I–IV–V major triads; allow V7 slightly; boost ii/vi minors a bit
                if mode == 'maj':
                    for t in range(log_harm.shape[1]):
                        key_bonus[0*12 + I,  t] += 0.35  # I
                        key_bonus[0*12 + IV, t] += 0.25  # IV
                        key_bonus[0*12 + V,  t] += 0.25  # V
                        key_bonus[4*12 + II, t] += 0.10  # ii m7 subtle
                        key_bonus[4*12 + VI, t] += 0.10  # vi m7 subtle
                        key_bonus[2*12 + V,  t] += 0.08  # V7 small allowance
                    # De-emphasize blanket dom7 and maj7 away from cadence
                    key_bonus[2*12:(2*12)+12, :] -= 0.05
                    key_bonus[3*12:(3*12)+12, :] -= 0.05
                    # In reggae, lean even more on bass root evidence
                    beta_bass *= 1.2
                    alpha_harm *= 0.9
            elif style == 'jazz':
                # Jazz: prefer 7th chords; emphasize ii–V–I and soften plain triads
                if mode == 'maj':
                    for t in range(log_harm.shape[1]):
                        key_bonus[4*12 + II, t] += 0.22  # ii m7
                        key_bonus[2*12 + V,  t] += 0.28  # V7
                        key_bonus[3*12 + I,  t] += 0.24  # I maj7
                        key_bonus[4*12 + VI, t] += 0.10  # vi m7 (common)
                        key_bonus[4*12 + III,t] += 0.06  # iii m7 (softer)
                    # Downweight plain triads to reflect 7th-heavy vocabulary
                    key_bonus[0*12:(0*12)+12, :] -= 0.05
                    key_bonus[1*12:(1*12)+12, :] -= 0.05
                else:  # minor key jazz: iiø–V7–i (approximate iiø as m7 here)
                    i_pc = int(tonic_idx % 12)
                    ii_pc = int((tonic_idx + 2) % 12)
                    V_pc  = int((tonic_idx + 7) % 12)
                    for t in range(log_harm.shape[1]):
                        key_bonus[4*12 + ii_pc, t] += 0.20  # ii m7 (proxy for iiø)
                        key_bonus[2*12 + V_pc,  t] += 0.28  # V7
                        key_bonus[4*12 + i_pc,  t] += 0.16  # i m7 (proxy)
                    key_bonus[0*12:(0*12)+12, :] -= 0.04
                    key_bonus[1*12:(1*12)+12, :] -= 0.02
            else:  # rock/pop (default)
                if mode == 'maj':
                    for t in range(log_harm.shape[1]):
                        key_bonus[0*12 + I,  t] += 0.25
                        key_bonus[0*12 + IV, t] += 0.18
                        key_bonus[0*12 + V,  t] += 0.20
                        key_bonus[2*12 + V,  t] += 0.06  # light V7 boost
                    # Rock/pop: bass is still strong, but balance with harmony
                    beta_bass *= 1.1
                    alpha_harm *= 0.95
        except Exception:
            pass
    if callable(log_fn):
        try:
            try:
                kh_pretty = (key_hint or {}).get('pretty', 'unknown')
            except Exception:
                kh_pretty = 'unknown'
            log_fn(f"key: {kh_pretty}, style: {style}")
        except Exception:
            pass

    # Combined emissions
    log_probs = alpha_harm * log_harm + beta_bass * log_bass + key_bonus + bass_boost

    # Time-varying stay prob from beat context
    T = log_probs.shape[1]
    stay = np.full(T, 0.997, dtype=np.float64)
    if beats:
        down_set = set(np.round(np.asarray(downbeats or []) * 1000).astype(int).tolist())
        beat_ms = np.round(np.asarray(beats) * 1000).astype(int).tolist()
        strengths = beat_strengths or [1.0]*len(beat_ms)
        for i, (bm, s) in enumerate(zip(beat_ms, strengths)):
            is_down = bm in down_set
            base = 0.988 if is_down else 0.997
            stay_i = np.clip(base - 0.12 * float(s), 0.93, 0.999)
            if i < T:
                stay[i] = stay_i

    states = viterbi_timevarying(log_probs, stay)

    # Enharmonic naming by key, prefer key_hint if available
    tonic_idx2, mode2 = _parse_key_hint(key_hint)
    if tonic_idx2 is None or mode2 not in ('maj','min'):
        tonic_idx2, mode2 = _estimate_key_from_chroma(chroma)
    use_flats = _use_fl

    def _state_to_label(s: int) -> str:
        fam = s // 12
        root_pc = int(s % 12)
        root = _pc_name(root_pc, use_flats)
        return (
            root if fam == 0 else
            f"{root}m" if fam == 1 else
            f"{root}7" if fam == 2 else
            f"{root}maj7" if fam == 3 else
            f"{root}m7"
        )

    labels = [_state_to_label(int(s)) for s in states]

    # Confidence = margin(best vs 2nd)
    best = np.max(log_probs, axis=0)
    srt = np.sort(log_probs, axis=0)
    margin = best - srt[-2]
    avg_margin = float(np.median(margin)) if margin.size else 0.0
    try:
        stem_keys = sorted(list(stems.keys())) if isinstance(stems, dict) else []
        msg = f"stem_aware: beats={len(beats or [])}, stems={stem_keys}, used={used_stems}, avg_margin={avg_margin:.3f}"
        if callable(log_fn):
            log_fn(msg)
        else:
            print(msg)
    except Exception:
        pass
    try:
        from collections import Counter
        cnt = Counter(labels)
        top5 = ", ".join(f"{k}:{v}" for k, v in cnt.most_common(5))
        if callable(log_fn):
            log_fn(f"stem_aware: top labels → {top5}")
        else:
            print(f"stem_aware: top labels → {top5}")
    except Exception:
        pass

    segments = []
    start = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            conf = float(np.median(margin[start:i])) if i > start else float(margin[start])
            segments.append({
                "start": float(times[start]),
                "end": float(times[min(i, len(times)-1)]),
                "label": labels[start],
                "conf": conf,
            })
            start = i

    # Post passes
    segments = _refine_minor_sevenths(segments, audio_path)
    segments = _root_sanity_pass(segments, audio_path, use_flats=use_flats)

    if beats:
        segments = [dict(s, beat_sync=True) for s in segments]
    return segments

def estimate_chords_chordino_vamp(audio_path: str, beats=None, downbeats=None, **kwargs):
    """
    Use Vamp host to run nnls-chroma:chordino → segments [{start,end,label}].
    If beats are provided, snap to beats; split at downbeats if provided.
    """
    import numpy as np

    y, sr = sf.read(audio_path, dtype='float32', always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # mono

    # Collect simple chord labels (time-stamped)
    res = vamp.collect(y, sr, "nnls-chroma:chordino", output="simplechord")
    evs = res.get("list", [])
    if not evs:
        return []

    # Convert events → segments (end = next start; last gets small tail)
    segs = []
    for i, ev in enumerate(evs):
        t0 = float(ev["timestamp"])
        t1 = float(evs[i+1]["timestamp"]) if i+1 < len(evs) else (t0 + 1.0)
        lab = str(ev["label"])
        if t1 > t0:
            segs.append({"start": t0, "end": t1, "label": lab})

    # Optional: snap to closest beats
    if beats:
        bt = np.asarray(beats, dtype=float)
        def _snap(t):
            j = int(np.clip(np.argmin(np.abs(bt - t)), 0, len(bt)-1))
            return float(bt[j])
        for s in segs:
            s["start"], s["end"] = _snap(s["start"]), _snap(s["end"])
            if s["end"] <= s["start"] and len(bt) > 1:
                s["end"] = s["start"] + (bt[1]-bt[0])

    # Optional: split at bar boundaries (downbeats)
    if downbeats and beats:
        split = []
        for s in segs:
            a, b, lab = float(s["start"]), float(s["end"]), s["label"]
            cuts = [x for x in downbeats if (x > a) and (x < b)]
            if not cuts:
                split.append(s)
            else:
                pts = [a] + list(cuts) + [b]
                for u, v in zip(pts[:-1], pts[1:]):
                    if v - u > 1e-3:
                        split.append({"start": float(u), "end": float(v), "label": lab})
        segs = split

    # Filter zero-length
    return [s for s in segs if (s["end"] - s["start"]) > 1e-3]

def estimate_chords(
    audio_path: str,
    sr=22050,
    hop=2048,
    beats: list[float] | None = None,
    downbeats: list[float] | None = None,
    beat_strengths: list[float] | None = None,
    stems: dict | None = None,
    use_hpss: bool = True,
    key_hint: dict | None = None,
):
    from audio_engine import _load_audio_any
    y_stereo, sr_loaded = _load_audio_any(audio_path)
    sr = sr_loaded if sr is None else int(sr)

    # Build harmonic mix: prefer stems, else HPSS on the mono mix (resampled)
    y_mono = _resample_mono(y_stereo, sr_loaded, sr)

    y_harm = None
    if stems:
        y_harm, _ = _mix_from_stems(stems, sr)
        if y_harm is not None and sr_loaded != sr:
            y_harm = librosa.resample(y_harm.astype(np.float32), orig_sr=sr_loaded, target_sr=sr)
    if y_harm is None:
        if use_hpss:
            try:
                y_h, _ = librosa.effects.hpss(y_mono)
                y_harm = y_h
            except Exception:
                y_harm = y_mono
        else:
            y_harm = y_mono

    # Framewise chroma for fallback/feature use
    chroma_frames = librosa.feature.chroma_cqt(y=y_harm, sr=sr, hop_length=hop)
    frame_times = librosa.frames_to_time(np.arange(chroma_frames.shape[1]), sr=sr, hop_length=hop)

    # If beats are provided (or can be estimated), aggregate beat-synchronously
    if beats is None:
        bd = estimate_beats(audio_path, sr=sr, hop=512)
        beats = bd.get("beats", [])
        downbeats = bd.get("downbeats", []) if downbeats is None else downbeats
        beat_strengths = bd.get("beat_strengths", []) if beat_strengths is None else beat_strengths

    if beats:
        chroma, beat_frame_idxs = _beat_sync_chroma(chroma_frames, frame_times, beats, reduce='median')
        times = np.asarray(beats, dtype=float)
    else:
        chroma = chroma_frames
        times = frame_times

    # Normalize chroma columns
    chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-12)

    scores = chord_likelihoods(chroma) + 1e-6
    log_probs = np.log(scores)

    # Build stay probability per time using beat context:
    T = log_probs.shape[1]
    stay = np.full(T, 0.997, dtype=np.float64)
    if beats:
        # Encourage changes on downbeats and strong beats
        down_set = set(np.round(np.asarray(downbeats or []) * 1000).astype(int).tolist())
        beat_ms = np.round(np.asarray(beats) * 1000).astype(int).tolist()
        strengths = beat_strengths or [1.0]*len(beat_ms)
        for i, (bm, s) in enumerate(zip(beat_ms, strengths)):
            # If this beat is a downbeat, allow more switching
            is_down = bm in down_set
            base = 0.990 if is_down else 0.997
            # Stronger beat -> lower stay (more likely to change)
            stay_i = np.clip(base - 0.08 * float(s), 0.92, 0.999)
            # Map to time index i (beat-synchronous)
            if i < T:
                stay[i] = stay_i
    states = viterbi_timevarying(log_probs, stay)

    # Choose enharmonics based on key_hint if available, else estimated key
    tonic_idx2, mode2 = _parse_key_hint(key_hint)
    if tonic_idx2 is None or mode2 not in ('maj','min'):
        tonic_idx2, mode2 = _estimate_key_from_chroma(chroma)
    use_flats = _use_flats_for_key(int(tonic_idx2), mode2)

    # State layout: 0..11 maj, 12..23 min, 24..35 dom7, 36..47 maj7, 48..59 min7
    def _state_to_label(s: int) -> str:
        fam = s // 12
        root_pc = int(s % 12)
        root = _pc_name(root_pc, use_flats)
        if fam == 0:   # maj triad
            return root
        if fam == 1:   # min triad
            return f"{root}m"
        if fam == 2:   # dom7
            return f"{root}7"
        if fam == 3:   # maj7
            return f"{root}maj7"
        # fam == 4: min7
        return f"{root}m7"

    labels = [_state_to_label(int(s)) for s in states]

    # Collapse runs into segments
    segments = []
    start = 0
    for i in range(1, len(labels)+1):
        if i==len(labels) or labels[i]!=labels[start]:
            segments.append({
                "start": float(times[start]),
                "end": float(times[min(i, len(times)-1)]),
                "label": labels[start],  # e.g., "Db" or "Ebm"
            })
            start = i

    # Stricter minor vs minor7 post-processing
    segments = _refine_minor_sevenths(segments, audio_path)
    # Root sanity check (keep family, re-pick root if clearly better)
    segments = _root_sanity_pass(segments, audio_path, use_flats=use_flats)

    beat_sync_used = bool(beats)
    if beat_sync_used:
        segments = [dict(s, beat_sync=True) for s in segments]
    return segments
