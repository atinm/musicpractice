from pathlib import Path
import tempfile

def temp_wav_path(prefix: str = "musicpractice_") -> str:
    tmp = Path(tempfile.gettempdir()) / f"{prefix}.wav"
    return str(tmp)
