# stems.py
from demucs.separate import main as demucs_separate
from pathlib import Path
import tempfile

def separate_stems(audio_path: str, out_dir: str, model: str = "htdemucs"):
    """
    Runs Demucs to produce stems in out_dir/model/<track>/{vocals,drums,bass,other}.wav
    """
    tmp = tempfile.mkdtemp()
    # demucs CLI under the hood; faster to shell out but this keeps it in-Python
    demucs_separate([
        "--two-stems", "vocals",   # or remove for full set
        "--mp3", "--mp3-bitrate", "320",  # or keep WAVs; WAV is better for analysis
        "-n", model,
        "-o", out_dir,
        audio_path
    ])
    # Return the folder where stems were written
    out = Path(out_dir) / model
    # The track subfolder has the stems
    # Find the newest subfolder
    latest = max(out.glob("*"), key=lambda p: p.stat().st_mtime)
    return latest
