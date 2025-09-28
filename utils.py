from pathlib import Path
import tempfile
import json
import re

APP_SETTINGS_PATH = Path.home() / ".musicpractice_settings.json"

def _slugify(name: str) -> str:
    # filesystem-friendly slug
    name = name.strip().lower()
    name = re.sub(r"[^\w\-\.]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-") or "untitled"

def load_settings() -> dict:
    if APP_SETTINGS_PATH.exists():
        try:
            return json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_settings(d: dict) -> None:
    try:
        APP_SETTINGS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass

def get_output_root_for_track(audio_path: Path) -> Path:
    """
    Returns the per-track root directory where we should store sessions,
    stems, pitch cache, etc., honoring a user-chosen 'output_root' if set.

    If settings['output_root'] is set:
        output_root/<slug(basename-without-ext)>
    else (default/legacy behavior):
        <audio_dir>/.musicpractice
    """
    audio_path = Path(audio_path)
    settings = load_settings()
    user_root = settings.get("output_root")
    if user_root:
        track_dir = _slugify(audio_path.stem)
        return Path(user_root) / track_dir
    else:
        return audio_path.parent / ".musicpractice"

def temp_wav_path(prefix: str = "musicpractice_") -> str:
    tmp = Path(tempfile.gettempdir()) / f"{prefix}.wav"
    return str(tmp)
