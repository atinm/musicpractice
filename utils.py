from pathlib import Path
import tempfile
import json
import re
import hashlib

APP_SETTINGS_PATH = Path.home() / ".musicpractice_settings.json"

def _slugify(name: str) -> str:
    # filesystem-friendly slug
    name = name.strip().lower()
    name = re.sub(r"[^\w\-\.]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-") or "untitled"

def get_app_data_dir() -> Path:
    """
    Returns the application private data directory for caching stems, etc.
    On macOS: ~/Library/Application Support/MusicPractice
    On Linux: ~/.local/share/musicpractice
    On Windows: ~/AppData/Local/MusicPractice
    """
    import platform
    system = platform.system()
    if system == "Darwin":  # macOS
        base = Path.home() / "Library" / "Application Support"
    elif system == "Windows":
        base = Path.home() / "AppData" / "Local"
    else:  # Linux and others
        base = Path.home() / ".local" / "share"

    app_dir = base / "MusicPractice"
    app_dir.mkdir(parents=True, exist_ok=True)
    return app_dir

def get_stems_cache_dir(audio_path: Path) -> Path:
    """
    Returns a unique directory for storing stems for a given audio file,
    in the application's private data directory.

    The directory is keyed by the audio file's absolute path and modification time,
    so if the same file is opened from different locations or modified, new stems
    will be generated.

    Structure: <app_data>/stems/<hash>/<model>/<track_name>/
    """
    audio_path = Path(audio_path).resolve()

    # Create a unique key based on file path, size, and mtime
    try:
        st = audio_path.stat()
        key_str = f"{audio_path}_{st.st_size}_{int(st.st_mtime)}"
    except Exception:
        key_str = str(audio_path)

    # Hash the key to create a filesystem-safe directory name
    key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]

    # Store in app data under stems/<hash>/
    stems_root = get_app_data_dir() / "stems" / key_hash
    stems_root.mkdir(parents=True, exist_ok=True)

    return stems_root

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
