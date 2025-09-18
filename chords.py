# chords.py
import numpy as np
import librosa

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
    """Return (tonic_idx, mode) with mode in {"maj","min"} using profile correlation."""
    chroma_mean = chroma.mean(axis=1)
    if np.allclose(chroma_mean.sum(), 0):
        return 0, "maj"  # default C major if silence
    chroma_vec = chroma_mean / (np.linalg.norm(chroma_mean) + 1e-9)
    best_score = -1e9
    best = (0, "maj")
    for mode, prof in (("maj", _MAJOR_PROFILE), ("min", _MINOR_PROFILE)):
        prof_n = prof / np.linalg.norm(prof)
        for k in range(12):
            score = np.dot(chroma_vec, np.roll(prof_n, k))
            if score > best_score:
                best_score = score
                best = (k, mode)
    return best

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
    # mono mixdown for chroma
    y = y_stereo.mean(axis=1)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
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
    y = y_stereo.mean(axis=1)

    def seg_chroma_frames(a: float, b: float):
        a_s = max(0, int(a * sr)); b_s = max(a_s + 1, int(b * sr))
        y_seg = y[a_s:b_s]
        if y_seg.size < hop:
            pad = np.zeros(min(hop, max(0, hop - y_seg.size)), dtype=y.dtype)
            y_seg = np.concatenate([y_seg, pad])
        C = librosa.feature.chroma_cqt(y=y_seg, sr=sr, hop_length=hop)  # (12, F)
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
    y = y_stereo.mean(axis=1)

    # Key-based naming unless provided
    if use_flats is None:
        try:
            chroma_all = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
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
        C = librosa.feature.chroma_cqt(y=y_seg, sr=sr, hop_length=hop)
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

def estimate_chords(
    audio_path: str,
    sr=22050,
    hop=2048,
    beats: list[float] | None = None,
    downbeats: list[float] | None = None,
    beat_strengths: list[float] | None = None,
    stems: dict | None = None,
    use_hpss: bool = True,
):
    from audio_engine import _load_audio_any
    y_stereo, sr_loaded = _load_audio_any(audio_path)
    if sr is None:
        sr = sr_loaded

    # Build harmonic mix: prefer stems, else HPSS on the mono mix
    y_mono = y_stereo.mean(axis=1).astype(np.float32)

    y_harm = None
    if stems:
        y_harm, _ = _mix_from_stems(stems, sr)
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
            base = 0.994 if is_down else 0.997
            # Stronger beat -> lower stay (more likely to change)
            stay_i = np.clip(base - 0.08 * float(s), 0.94, 0.999)
            # Map to time index i (beat-synchronous)
            if i < T:
                stay[i] = stay_i
    states = viterbi_timevarying(log_probs, stay)

    # Choose enharmonics based on estimated key
    tonic_idx, mode = _estimate_key_from_chroma(chroma)
    use_flats = _use_flats_for_key(tonic_idx, mode)

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
