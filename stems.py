
from pathlib import Path
import tempfile
import soundfile as sf
import numpy as np
from utils import get_output_root_for_track

try:
    from demucs.separate import main as demucs_separate
except Exception:
    demucs_separate = None

STEM_ORDER = ["vocals", "drums", "bass", "other"]


def separate_stems(audio_path: str, out_dir: str, model: str = "htdemucs_6s", two_stems: bool = False) -> Path:
    """Run Demucs and return the **final directory** containing rendered stem WAVs.

    Demucs' layout is: <demucs_root>/<model>/<track_basename>/*.wav
    where <demucs_root> is the directory passed via `-o`.

    This function is tolerant of callers accidentally passing a path like
    ".../stems/<track_slug>" as `out_dir`. In that case, it will normalize
    the Demucs output root to the parent ".../stems" so the final path
    becomes ".../stems/<model>/<track_basename>" — which matches the
    downstream code that looks under stems/<model>/<track>.
    """
    from pathlib import Path as _P

    if demucs_separate is None:
        raise RuntimeError("Demucs not available. Install demucs to enable stem separation.")

    # Normalize and ensure destination exists
    _out = _P(out_dir)
    _out.mkdir(parents=True, exist_ok=True)

    # Demucs expects an output *root*; callers sometimes pass
    #   .../stems/<track_slug>  (too deep)
    # Normalize to .../stems if that pattern is detected.
    demucs_root = _out
    if demucs_root.name != "stems" and demucs_root.parent.name == "stems":
        demucs_root = demucs_root.parent

    # Build args
    args = ["-n", model, "-o", str(demucs_root), str(audio_path)]
    if two_stems:
        args = ["--two-stems", "vocals"] + args

    # Run Demucs
    demucs_separate(args)

    # Compute and return the **final** directory that contains the WAVs
    track_name = _P(audio_path).stem
    final_dir = demucs_root / model / track_name
    return final_dir


def load_stem_arrays(stem_dir: Path) -> dict[str, np.ndarray]:
    """Load all .wav stems in the given directory into arrays shaped (N, C)."""
    arrays: dict[str, np.ndarray] = {}
    for wav in sorted(stem_dir.glob("*.wav")):
        name = wav.stem.lower()
        y, sr = sf.read(str(wav), always_2d=True)
        # Expect all stems at same sr; convert to float32
        y = y.astype(np.float32, copy=False)
        arrays[name] = y
    return arrays


def order_stem_names(names: list[str]) -> list[str]:
    # prefer known order then append unknowns
    known = [n for n in STEM_ORDER if n in names]
    rest = [n for n in names if n not in known]
    return known + sorted(rest)

def stems_dir_for(audio_path: Path) -> Path:
    root = get_output_root_for_track(Path(audio_path))
    d = root / "stems"
    d.mkdir(parents=True, exist_ok=True)
    return d
