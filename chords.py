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

def estimate_chords(audio_path: str, sr=22050, hop=2048):
    from audio_engine import _load_audio_any
    y_stereo, sr_loaded = _load_audio_any(audio_path)
    if sr is None:
        sr = sr_loaded
    y = y_stereo.mean(axis=1)  # mono mixdown
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
    # Likelihoods and log
    scores = chord_likelihoods(chroma) + 1e-6
    log_probs = np.log(scores)
    states = viterbi(log_probs, trans=0.997)
    # Map to labels & times
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop)

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
    return segments
