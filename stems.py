
from pathlib import Path
import tempfile
import soundfile as sf
import numpy as np

try:
    from demucs.separate import main as demucs_separate
except Exception:
    demucs_separate = None

STEM_ORDER = ["vocals", "drums", "bass", "other"]


def separate_stems(audio_path: str, out_dir: str, model: str = "htdemucs_6s", two_stems: bool = False) -> Path:
    """Run Demucs and return the folder containing rendered stems.
    Output layout: out_dir/model/<track_name>/<stem>.wav
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if demucs_separate is None:
        raise RuntimeError("Demucs not available. Install demucs to enable stem separation.")
    args = ["-n", model, "-o", out_dir, audio_path]
    if two_stems:
        args = ["--two-stems", "vocals"] + args
    demucs_separate(args)
    return Path(out_dir)


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
