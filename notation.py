from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os
import re
import shutil
import subprocess
import traceback
from typing import Iterable

LILYPOND_BIN = shutil.which("lilypond")
HELPER_SCRIPT = Path(__file__).with_name("basic_pitch_helper.py")
NOTATION_FORMAT_VERSION = 4

_NOTE_NAMES_SHARP = ["c", "cis", "d", "dis", "e", "f", "fis", "g", "gis", "a", "ais", "b"]
_BASS_STRING_OPEN_MIDI = {
    1: 43,  # G string
    2: 38,  # D string
    3: 33,  # A string
    4: 28,  # E string
}


@dataclass(frozen=True)
class QuantizedNote:
    start_step: int
    end_step: int
    midi_pitch: int
    velocity: int = 80


@dataclass(frozen=True)
class NotationArtifacts:
    raw_midi: Path
    raw_notes_json: Path
    clean_midi: Path
    clean_notes_json: Path
    quantized_midi: Path
    quantized_notes_json: Path
    lilypond: Path
    pdf: Path


def _slugify(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^\w\-\.]+", "-", name)
    name = re.sub(r"-+", "-", name)
    return name.strip("-") or "untitled"


def _helper_python_candidates() -> list[Path]:
    env_override = os.environ.get("BASIC_PITCH_PYTHON")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.append(Path.cwd() / ".venv-basic-pitch" / "bin" / "python")
    candidates.append(Path(__file__).resolve().parent / ".venv-basic-pitch" / "bin" / "python")
    return candidates


def basic_pitch_python() -> Path | None:
    for candidate in _helper_python_candidates():
        if candidate.exists():
            return candidate
    return None


def basic_pitch_available() -> bool:
    return basic_pitch_python() is not None and HELPER_SCRIPT.exists()


def lilypond_available() -> bool:
    return LILYPOND_BIN is not None


def dependency_error_message() -> str | None:
    missing = []
    if not basic_pitch_available():
        missing.append("Basic Pitch helper env (.venv-basic-pitch or BASIC_PITCH_PYTHON)")
    if not lilypond_available():
        missing.append("LilyPond")
    if not missing:
        return None
    return "Missing dependency: " + ", ".join(missing)


def _notes_json_path_for_midi(midi_path: Path) -> Path:
    return midi_path.with_suffix(".notes.json")


def _artifact_paths(output_dir: Path, stem_name: str) -> NotationArtifacts:
    stem_slug = _slugify(stem_name)
    raw_midi = output_dir / f"{stem_slug}_raw.mid"
    legacy_raw_midi = output_dir / f"{stem_slug}.mid"
    if not raw_midi.exists() and legacy_raw_midi.exists():
        raw_midi = legacy_raw_midi
    clean_midi = output_dir / f"{stem_slug}_clean.mid"
    quantized_midi = output_dir / f"{stem_slug}_quantized.mid"
    lilypond = output_dir / f"{stem_slug}.ly"
    pdf = output_dir / f"{stem_slug}.pdf"
    return NotationArtifacts(
        raw_midi=raw_midi,
        raw_notes_json=_notes_json_path_for_midi(raw_midi),
        clean_midi=clean_midi,
        clean_notes_json=_notes_json_path_for_midi(clean_midi),
        quantized_midi=quantized_midi,
        quantized_notes_json=_notes_json_path_for_midi(quantized_midi),
        lilypond=lilypond,
        pdf=pdf,
    )


def _mtime(path: Path | None) -> float:
    if path is None or not path.exists():
        return float("-inf")
    return path.stat().st_mtime


def _is_newer(path: Path, other: Path) -> bool:
    return path.exists() and (not other.exists() or _mtime(path) > _mtime(other))


def _lilypond_source_is_current(lilypond_path: Path) -> bool:
    if not lilypond_path.exists():
        return False
    try:
        text = lilypond_path.read_text(encoding="utf-8")
    except OSError:
        return False
    marker = f"% notation-format-version: {NOTATION_FORMAT_VERSION}"
    return marker in text


def _run_helper_command(args: list[str], log_fn=None) -> None:
    helper_python = basic_pitch_python()
    if helper_python is None or not HELPER_SCRIPT.exists():
        raise RuntimeError(
            "Basic Pitch helper environment is not configured. Install .venv-basic-pitch and set BASIC_PITCH_PYTHON if needed."
        )
    cmd = [str(helper_python), str(HELPER_SCRIPT), *args]
    if log_fn:
        log_fn(f"[notation] Helper command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if log_fn and proc.stdout.strip():
        log_fn(f"[notation] Helper stdout:\n{proc.stdout.rstrip()}")
    if log_fn and proc.stderr.strip():
        log_fn(f"[notation] Helper stderr:\n{proc.stderr.rstrip()}")
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown helper error"
        raise RuntimeError(f"Notation helper failed: {detail}")


def _write_note_events_json(note_events: Iterable[dict], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = []
    for note in note_events:
        start = max(0.0, float(note.get("start", 0.0)))
        end = max(start + 1e-3, float(note.get("end", start + 1e-3)))
        normalized.append(
            {
                "start": start,
                "end": end,
                "pitch": int(note.get("pitch", 60)),
                "velocity": max(1, min(127, int(note.get("velocity", 80)))),
            }
        )
    json_path.write_text(json.dumps(normalized, indent=2), encoding="utf-8")


def load_note_events_from_midi(midi_path: Path, log_fn=None) -> list[dict]:
    if not midi_path.exists():
        raise RuntimeError(f"MIDI file does not exist: {midi_path}")
    json_path = _notes_json_path_for_midi(midi_path)
    if not json_path.exists() or _is_newer(midi_path, json_path):
        _run_helper_command(["midi-to-json", str(midi_path), str(json_path)], log_fn=log_fn)
    return json.loads(json_path.read_text(encoding="utf-8"))


def write_note_events_to_midi(note_events: Iterable[dict], midi_path: Path, instrument: str, log_fn=None) -> None:
    json_path = _notes_json_path_for_midi(midi_path)
    _write_note_events_json(note_events, json_path)
    _run_helper_command(["json-to-midi", str(json_path), str(midi_path), instrument], log_fn=log_fn)


def transcribe_audio_to_midi(audio_path: Path, midi_path: Path) -> list[dict]:
    return transcribe_audio_to_midi_with_logs(audio_path, midi_path, log_fn=None)


def frequency_bounds_for_stem(stem_name: str, instrument: str = "guitar") -> tuple[float | None, float | None]:
    name = stem_name.strip().lower()
    inst = instrument.strip().lower()

    if "bass" in name or inst == "bass":
        return 40.0, 400.0
    if "guitar" in name or inst == "guitar":
        return 80.0, 1400.0
    if "vocal" in name or "voice" in name:
        return 80.0, 1200.0
    if "piano" in name or "keys" in name or "keyboard" in name:
        return 27.5, 4200.0
    if "drum" in name:
        return 30.0, 200.0
    return None, None


def transcribe_audio_to_midi_with_logs(
    audio_path: Path,
    midi_path: Path,
    log_fn=None,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
) -> list[dict]:
    helper_python = basic_pitch_python()
    if helper_python is None or not HELPER_SCRIPT.exists():
        raise RuntimeError(
            "Basic Pitch helper environment is not configured. Install .venv-basic-pitch and set BASIC_PITCH_PYTHON if needed."
        )
    json_path = midi_path.with_suffix(".notes.json")
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(helper_python),
        str(HELPER_SCRIPT),
        str(audio_path),
        str(midi_path),
        str(json_path),
        "none" if minimum_frequency is None else str(float(minimum_frequency)),
        "none" if maximum_frequency is None else str(float(maximum_frequency)),
    ]
    if log_fn:
        log_fn(f"[notation] Basic Pitch helper: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if log_fn and proc.stdout.strip():
        log_fn(f"[notation] Basic Pitch stdout:\n{proc.stdout.rstrip()}")
    if log_fn and proc.stderr.strip():
        log_fn(f"[notation] Basic Pitch stderr:\n{proc.stderr.rstrip()}")
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "unknown Basic Pitch error"
        raise RuntimeError(f"Basic Pitch helper failed: {detail}")
    if not json_path.exists():
        raise RuntimeError("Basic Pitch helper completed but did not produce note events.")
    if log_fn:
        log_fn(f"[notation] Wrote MIDI: {midi_path}")
        log_fn(f"[notation] Wrote note events: {json_path}")
    return json.loads(json_path.read_text(encoding="utf-8"))


def build_quantized_score(
    note_events: Iterable[dict],
    beats: list[float] | None = None,
    steps_per_beat: int = 4,
) -> tuple[list[QuantizedNote], int, list[float]]:
    notes = [dict(n) for n in note_events]
    if not notes:
        raise RuntimeError("No note events were transcribed from this stem.")

    grid = _build_time_grid(beats or [], notes, steps_per_beat=steps_per_beat)
    quantized: list[QuantizedNote] = []

    for note in notes:
        start = max(0.0, float(note["start"]))
        end = max(start + 1e-3, float(note["end"]))
        start_step = _nearest_grid_index(grid, start)
        end_step = max(start_step + 1, _nearest_grid_index(grid, end))
        quantized.append(
            QuantizedNote(
                start_step=start_step,
                end_step=end_step,
                midi_pitch=int(note["pitch"]),
                velocity=int(note.get("velocity", 80)),
            )
        )

    quantized.sort(key=lambda n: (n.start_step, n.midi_pitch, n.end_step))
    total_steps = max(n.end_step for n in quantized)
    return quantized, total_steps, grid


def preprocess_note_events(
    note_events: Iterable[dict],
    stem_name: str,
    instrument: str,
    log_fn=None,
) -> list[dict]:
    notes = [dict(n) for n in note_events]
    if not notes:
        return []

    if instrument.strip().lower() == "bass" or "bass" in stem_name.strip().lower():
        return _filter_bass_note_events(notes, log_fn=log_fn)

    return notes


def _filter_bass_note_events(notes: list[dict], log_fn=None) -> list[dict]:
    original_count = len(notes)

    # Bass notation works much better if we suppress very short, weak artifacts.
    cleaned: list[dict] = []
    for note in notes:
        start = float(note.get("start", 0.0))
        end = max(start + 1e-3, float(note.get("end", start + 1e-3)))
        pitch = int(note.get("pitch", 0))
        velocity = int(note.get("velocity", 80))
        duration = end - start

        if pitch < 28 or pitch > 60:
            continue
        if duration < 0.05 and velocity < 45:
            continue
        if duration < 0.03:
            continue

        normalized = dict(note)
        normalized["start"] = start
        normalized["end"] = end
        normalized["pitch"] = pitch
        normalized["velocity"] = velocity
        cleaned.append(normalized)

    if not cleaned:
        if log_fn:
            log_fn(f"[notation] Bass cleanup removed all {original_count} raw notes; falling back to unfiltered notes")
        return notes

    cleaned.sort(key=lambda n: (float(n["start"]), int(n["pitch"])))

    # Bass is usually monophonic. For notes starting at nearly the same instant,
    # keep the strongest low note and drop the rest.
    clusters: list[list[dict]] = []
    current: list[dict] = []
    cluster_gap_s = 0.08
    for note in cleaned:
        if not current:
            current = [note]
            continue
        if float(note["start"]) - float(current[-1]["start"]) <= cluster_gap_s:
            current.append(note)
        else:
            clusters.append(current)
            current = [note]
    if current:
        clusters.append(current)

    collapsed: list[dict] = []
    for cluster in clusters:
        best = min(
            cluster,
            key=lambda n: (
                int(n["pitch"]),
                -(float(n["end"]) - float(n["start"])),
                -int(n.get("velocity", 80)),
            ),
        )
        collapsed.append(best)

    if not collapsed:
        return cleaned

    monophonic: list[dict] = []
    for note in collapsed:
        if not monophonic:
            monophonic.append(note)
            continue

        prev = monophonic[-1]
        prev_start = float(prev["start"])
        prev_end = float(prev["end"])
        start = float(note["start"])
        end = float(note["end"])
        overlap = prev_end - start
        if overlap <= 0.03:
            monophonic.append(note)
            continue

        prev_score = (
            (prev_end - prev_start) * 120.0
            + int(prev.get("velocity", 80))
            - abs(int(prev["pitch"]) - int(note["pitch"])) * 4.0
        )
        score = (
            (end - start) * 120.0
            + int(note.get("velocity", 80))
            - abs(int(note["pitch"]) - int(prev["pitch"])) * 4.0
        )
        if score >= prev_score:
            monophonic[-1] = note
        else:
            prev_copy = dict(prev)
            prev_copy["end"] = max(prev_start + 1e-3, start)
            monophonic[-1] = prev_copy

    # Estimate the main register and reject far-away outliers.
    weighted_pitches: list[int] = []
    pitch_class_weights = {i: 0.0 for i in range(12)}
    for note in monophonic:
        dur = max(0.05, float(note["end"]) - float(note["start"]))
        vel = max(1, int(note.get("velocity", 80)))
        weight = max(1, int(round(dur * vel / 12.0)))
        weighted_pitches.extend([int(note["pitch"])] * weight)
        pitch_class_weights[int(note["pitch"]) % 12] += dur * vel

    weighted_pitches.sort()
    lo_idx = int(round((len(weighted_pitches) - 1) * 0.15))
    hi_idx = int(round((len(weighted_pitches) - 1) * 0.85))
    median_idx = len(weighted_pitches) // 2
    center = weighted_pitches[median_idx]
    lo_pitch = max(28, weighted_pitches[lo_idx] - 5)
    hi_pitch = min(60, weighted_pitches[hi_idx] + 5)

    strong_pitch_classes = {
        pitch_class
        for pitch_class, weight in pitch_class_weights.items()
        if weight >= max(pitch_class_weights.values()) * 0.3
    }
    filtered = [n for n in monophonic if lo_pitch <= int(n["pitch"]) <= hi_pitch]
    if not filtered:
        filtered = monophonic

    # Drop isolated jumps that are far from their local neighborhood and look ghostly.
    final_notes: list[dict] = []
    for i, note in enumerate(filtered):
        prev_note = filtered[i - 1] if i > 0 else None
        next_note = filtered[i + 1] if i + 1 < len(filtered) else None
        pitch = int(note["pitch"])
        duration = float(note["end"]) - float(note["start"])
        velocity = int(note.get("velocity", 80))

        isolated = True
        for neighbor in (prev_note, next_note):
            if neighbor is None:
                continue
            near_in_time = abs(float(neighbor["start"]) - float(note["start"])) <= 0.35
            near_in_pitch = abs(int(neighbor["pitch"]) - pitch) <= 7
            if near_in_time and near_in_pitch:
                isolated = False
                break

        local_neighbors = [n for n in (prev_note, next_note) if n is not None]
        local_center = center
        if local_neighbors:
            neighbor_pitches = sorted(int(n["pitch"]) for n in local_neighbors)
            local_center = neighbor_pitches[len(neighbor_pitches) // 2]

        off_scale = strong_pitch_classes and (pitch % 12) not in strong_pitch_classes
        big_jump = abs(pitch - local_center) > 7
        weak_ghost = duration < 0.14 and velocity < 75

        if isolated and weak_ghost and (abs(pitch - center) > 7 or off_scale or big_jump):
            continue
        final_notes.append(note)

    if log_fn:
        log_fn(
            "[notation] Bass cleanup: "
            f"{original_count} raw -> {len(cleaned)} range/length -> {len(collapsed)} clustered -> {len(monophonic)} monophonic -> {len(final_notes)} final "
            f"(register {lo_pitch}-{hi_pitch}, center {center})"
        )

    return final_notes or filtered or monophonic or cleaned or notes


def _build_time_grid(beats: list[float], notes: list[dict], steps_per_beat: int) -> list[float]:
    if len(beats) >= 2:
        beat_times = sorted(float(b) for b in beats)
        grid: list[float] = []
        for i, beat in enumerate(beat_times[:-1]):
            next_beat = beat_times[i + 1]
            beat_len = max(1e-3, next_beat - beat)
            for sub in range(steps_per_beat):
                grid.append(beat + (beat_len * sub / steps_per_beat))

        final_beat = beat_times[-1]
        if len(beat_times) >= 2:
            beat_len = max(1e-3, beat_times[-1] - beat_times[-2])
        else:
            beat_len = 0.5
        note_end = max(float(n["end"]) for n in notes)
        safety_end = max(note_end, final_beat + beat_len)
        t = final_beat
        while t <= safety_end + 1e-6:
            for sub in range(steps_per_beat):
                val = t + (beat_len * sub / steps_per_beat)
                if val <= safety_end + 1e-6:
                    grid.append(val)
            t += beat_len
        if not grid:
            grid = [0.0]
        if grid[-1] < safety_end:
            grid.append(safety_end)
        return sorted(set(round(v, 6) for v in grid))

    max_end = max(float(n["end"]) for n in notes)
    step_s = 0.125
    count = max(2, int(math.ceil(max_end / step_s)) + 4)
    return [i * step_s for i in range(count)]


def _nearest_grid_index(grid: list[float], t: float) -> int:
    best_i = 0
    best_dist = float("inf")
    for i, g in enumerate(grid):
        dist = abs(g - t)
        if dist < best_dist:
            best_i = i
            best_dist = dist
    return best_i


def _grid_time_at(grid: list[float], index: int) -> float:
    if not grid:
        return 0.0
    if index < len(grid):
        return float(grid[index])
    if len(grid) == 1:
        step = 0.125
    else:
        step = max(1e-3, float(grid[-1]) - float(grid[-2]))
    return float(grid[-1]) + step * (index - len(grid) + 1)


def _quantized_notes_to_note_events(quantized_notes: list[QuantizedNote], grid: list[float]) -> list[dict]:
    events: list[dict] = []
    for note in quantized_notes:
        start = _grid_time_at(grid, note.start_step)
        end = max(start + 1e-3, _grid_time_at(grid, note.end_step))
        events.append(
            {
                "start": start,
                "end": end,
                "pitch": int(note.midi_pitch),
                "velocity": max(1, min(127, int(note.velocity))),
            }
        )
    events.sort(key=lambda event: (float(event["start"]), int(event["pitch"]), float(event["end"])))
    return events


def _bass_string_candidates(midi_pitch: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for string_no, open_pitch in _BASS_STRING_OPEN_MIDI.items():
        fret = int(midi_pitch) - open_pitch
        if 0 <= fret <= 24:
            candidates.append((string_no, fret))
    return candidates


def _bass_fingering_transition_cost(
    prev_note: QuantizedNote,
    prev_choice: tuple[int, int],
    note: QuantizedNote,
    choice: tuple[int, int],
) -> float:
    prev_string, prev_fret = prev_choice
    string_no, fret = choice
    gap_steps = max(0, note.start_step - prev_note.end_step)

    cost = 0.0
    fret_delta = abs(fret - prev_fret)
    string_delta = abs(string_no - prev_string)
    cost += fret_delta * 0.55
    cost += string_delta * 1.1
    if string_no != prev_string:
        cost += 0.8
    if gap_steps <= 2:
        cost += fret_delta * 0.3
    if gap_steps <= 1 and string_no != prev_string:
        cost += 1.5
    if gap_steps <= 1 and string_no < prev_string and fret < prev_fret:
        cost += 1.0
    if string_no != prev_string and fret_delta <= 5:
        cost -= 1.25
    if string_no != prev_string and fret_delta >= 7:
        cost += 1.5
    if string_no == prev_string and fret_delta >= 5:
        cost += 1.1
    return cost


def _estimate_bass_phrase_target_fret(candidates_by_note: list[list[tuple[int, int]]]) -> float:
    representative_frets: list[int] = []
    for candidates in candidates_by_note:
        if not candidates:
            continue
        frets = sorted(fret for _string_no, fret in candidates)
        representative_frets.append(frets[len(frets) // 2])
    representative_frets.sort()
    if not representative_frets:
        return 5.0
    return float(representative_frets[len(representative_frets) // 2])


def _bass_fingering_base_cost(note: QuantizedNote, choice: tuple[int, int], target_fret: float) -> float:
    string_no, fret = choice
    pitch = int(note.midi_pitch)

    cost = 0.0
    cost += fret * 0.08
    cost += abs(fret - target_fret) * 0.35
    cost += {1: 3.0, 2: 0.9, 3: 0.35, 4: 0.0}[string_no]

    if pitch <= 38 and string_no <= 2:
        cost += 6.0
    elif pitch <= 43 and string_no == 1:
        cost += 3.0

    if string_no == 4 and fret >= 9:
        cost += (fret - 8) * 0.9
    if string_no == 3 and fret >= 10:
        cost += (fret - 9) * 0.5

    if fret > 12:
        cost += (fret - 12) * 0.8
    return cost


def _choose_bass_string_numbers(notes: list[QuantizedNote]) -> dict[QuantizedNote, int]:
    monophonic_notes = list(notes)
    if not monophonic_notes:
        return {}

    candidates_by_note: list[list[tuple[int, int]]] = []
    for note in monophonic_notes:
        candidates = _bass_string_candidates(note.midi_pitch)
        if not candidates:
            return {}
        candidates_by_note.append(candidates)
    target_fret = _estimate_bass_phrase_target_fret(candidates_by_note)

    scores: list[dict[tuple[int, int], float]] = []
    backpointers: list[dict[tuple[int, int], tuple[int, int] | None]] = []

    first_scores: dict[tuple[int, int], float] = {}
    first_prev: dict[tuple[int, int], tuple[int, int] | None] = {}
    for choice in candidates_by_note[0]:
        first_scores[choice] = _bass_fingering_base_cost(monophonic_notes[0], choice, target_fret)
        first_prev[choice] = None
    scores.append(first_scores)
    backpointers.append(first_prev)

    for i in range(1, len(monophonic_notes)):
        note = monophonic_notes[i]
        prev_note = monophonic_notes[i - 1]
        current_scores: dict[tuple[int, int], float] = {}
        current_prev: dict[tuple[int, int], tuple[int, int] | None] = {}
        for choice in candidates_by_note[i]:
            base_cost = _bass_fingering_base_cost(note, choice, target_fret)
            best_score = float("inf")
            best_prev: tuple[int, int] | None = None
            for prev_choice, prev_score in scores[-1].items():
                score = prev_score + base_cost + _bass_fingering_transition_cost(prev_note, prev_choice, note, choice)
                if score < best_score:
                    best_score = score
                    best_prev = prev_choice
            current_scores[choice] = best_score
            current_prev[choice] = best_prev
        scores.append(current_scores)
        backpointers.append(current_prev)

    final_choice = min(scores[-1], key=scores[-1].get)
    chosen: list[tuple[int, int]] = [final_choice]
    for i in range(len(monophonic_notes) - 1, 0, -1):
        prev_choice = backpointers[i][chosen[-1]]
        if prev_choice is None:
            break
        chosen.append(prev_choice)
    chosen.reverse()
    if len(chosen) != len(monophonic_notes):
        return {}

    return {note: choice[0] for note, choice in zip(monophonic_notes, chosen)}


def render_notation_bundle(
    stem_audio_path: Path,
    output_dir: Path,
    title: str,
    stem_name: str,
    beats: list[float] | None = None,
    instrument: str = "guitar",
    log_fn=None,
    rebuild_from_audio: bool = False,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = _artifact_paths(output_dir, stem_name)
    min_hz, max_hz = frequency_bounds_for_stem(stem_name, instrument=instrument)

    if log_fn:
        log_fn(f"[notation] Starting render for stem '{stem_name}' from {stem_audio_path}")
        log_fn(f"[notation] Output directory: {output_dir}")
        log_fn(f"[notation] Instrument mode: {instrument}")
        log_fn(f"[notation] Frequency bounds: min={min_hz} Hz, max={max_hz} Hz")

    if not rebuild_from_audio and artifacts.quantized_midi.exists() and (
        not artifacts.clean_midi.exists() or _mtime(artifacts.quantized_midi) >= _mtime(artifacts.clean_midi)
    ):
        if log_fn:
            log_fn(f"[notation] Reusing quantized MIDI: {artifacts.quantized_midi}")
        quantized_events = load_note_events_from_midi(artifacts.quantized_midi, log_fn=log_fn)
        return _render_from_quantized_events(
            quantized_events=quantized_events,
            artifacts=artifacts,
            title=title,
            stem_name=stem_name,
            instrument=instrument,
            log_fn=log_fn,
        )

    if not rebuild_from_audio and artifacts.clean_midi.exists():
        if log_fn:
            log_fn(f"[notation] Reusing clean MIDI: {artifacts.clean_midi}")
        clean_events = load_note_events_from_midi(artifacts.clean_midi, log_fn=log_fn)
        return _render_from_clean_events(
            clean_events=clean_events,
            artifacts=artifacts,
            beats=beats or [],
            title=title,
            stem_name=stem_name,
            instrument=instrument,
            log_fn=log_fn,
        )

    if rebuild_from_audio or not artifacts.raw_midi.exists():
        if log_fn:
            reason = "forced rebuild from audio" if rebuild_from_audio else "no cached raw MIDI found"
            log_fn(f"[notation] Creating raw MIDI from audio ({reason})")
        transcribe_audio_to_midi_with_logs(
            stem_audio_path,
            artifacts.raw_midi,
            log_fn=log_fn,
            minimum_frequency=min_hz,
            maximum_frequency=max_hz,
        )
    elif log_fn:
        log_fn(f"[notation] Reusing raw MIDI: {artifacts.raw_midi}")

    raw_events = load_note_events_from_midi(artifacts.raw_midi, log_fn=log_fn)
    filtered_events = preprocess_note_events(
        raw_events,
        stem_name=stem_name,
        instrument=instrument,
        log_fn=log_fn,
    )
    if log_fn and len(filtered_events) != len(raw_events):
        log_fn(f"[notation] Filtered note events: {len(raw_events)} -> {len(filtered_events)}")
    write_note_events_to_midi(filtered_events, artifacts.clean_midi, instrument=instrument, log_fn=log_fn)
    if log_fn:
        log_fn(f"[notation] Wrote clean MIDI: {artifacts.clean_midi}")
    return _render_from_clean_events(
        clean_events=filtered_events,
        artifacts=artifacts,
        beats=beats or [],
        title=title,
        stem_name=stem_name,
        instrument=instrument,
        log_fn=log_fn,
    )


def _render_from_clean_events(
    clean_events: list[dict],
    artifacts: NotationArtifacts,
    beats: list[float],
    title: str,
    stem_name: str,
    instrument: str,
    log_fn=None,
) -> Path:
    quantized, total_steps, grid = build_quantized_score(clean_events, beats=beats)
    if log_fn:
        log_fn(f"[notation] Quantized {len(quantized)} notes over {total_steps} sixteenth-note steps")
    quantized_events = _quantized_notes_to_note_events(quantized, grid)
    write_note_events_to_midi(quantized_events, artifacts.quantized_midi, instrument=instrument, log_fn=log_fn)
    if log_fn:
        log_fn(f"[notation] Wrote quantized MIDI: {artifacts.quantized_midi}")
    return _render_from_quantized_events(
        quantized_events=quantized_events,
        artifacts=artifacts,
        title=title,
        stem_name=stem_name,
        instrument=instrument,
        log_fn=log_fn,
    )


def _render_from_quantized_events(
    quantized_events: list[dict],
    artifacts: NotationArtifacts,
    title: str,
    stem_name: str,
    instrument: str,
    log_fn=None,
) -> Path:
    lilypond_current = _lilypond_source_is_current(artifacts.lilypond)
    if (
        artifacts.pdf.exists()
        and artifacts.lilypond.exists()
        and lilypond_current
        and _mtime(artifacts.pdf) >= _mtime(artifacts.lilypond)
        and _mtime(artifacts.pdf) >= _mtime(artifacts.quantized_midi)
    ):
        if log_fn:
            log_fn(f"[notation] Reusing existing PDF: {artifacts.pdf}")
        return artifacts.pdf

    if log_fn and artifacts.lilypond.exists() and not lilypond_current:
        log_fn(f"[notation] Regenerating LilyPond source because format version changed: {artifacts.lilypond}")

    quantized, total_steps, _grid = build_quantized_score(quantized_events, beats=None)
    lilypond_text = build_lilypond_document(
        quantized_notes=quantized,
        total_steps=total_steps,
        title=title,
        stem_name=stem_name,
        instrument=instrument,
    )
    artifacts.lilypond.write_text(lilypond_text, encoding="utf-8")
    if log_fn:
        log_fn(f"[notation] Wrote LilyPond source: {artifacts.lilypond}")
    return render_lilypond_pdf(artifacts.lilypond, log_fn=log_fn)


def build_lilypond_document(
    quantized_notes: list[QuantizedNote],
    total_steps: int,
    title: str,
    stem_name: str,
    instrument: str,
) -> str:
    string_numbers: dict[QuantizedNote, int] = {}
    if instrument == "bass":
        string_numbers = _choose_bass_string_numbers(quantized_notes)
    staff_music = _notes_to_lily_music(quantized_notes, total_steps)
    tab_music = _notes_to_lily_music(quantized_notes, total_steps, string_numbers=string_numbers)
    title_escaped = title.replace('"', "'")
    stem_escaped = stem_name.replace('"', "'")

    if instrument == "bass":
        clef = '"bass_8"'
        tuning = "#bass-tuning"
    else:
        clef = '"treble_8"'
        tuning = "#guitar-tuning"

    return f'''% notation-format-version: {NOTATION_FORMAT_VERSION}
\\version "2.24.4"

\\header {{
  title = "{title_escaped}"
  subtitle = "{stem_escaped}"
  tagline = ""
}}

staffMusic = {{
  \\numericTimeSignature
  \\time 4/4
  \\clef {clef}
  {staff_music}
}}

tabMusic = {{
  \\numericTimeSignature
  \\time 4/4
  \\clef {clef}
  {tab_music}
}}

\\score {{
  <<
    \\new Staff \\with {{
      instrumentName = "{stem_escaped}"
    }} \\staffMusic
    \\new TabStaff \\with {{
      stringTunings = {tuning}
      restrainOpenStrings = ##t
    }} \\tabMusic
  >>
  \\layout {{ }}
}}
'''


def _notes_to_lily_music(
    notes: list[QuantizedNote],
    total_steps: int,
    string_numbers: dict[QuantizedNote, int] | None = None,
) -> str:
    by_step: dict[int, list[QuantizedNote]] = {}
    for note in notes:
        by_step.setdefault(note.start_step, []).append(note)

    cursor = 0
    tokens: list[str] = []
    measure_steps = 16
    while cursor < total_steps:
        bucket = by_step.get(cursor, [])
        if bucket:
            end_step = min(n.end_step for n in bucket)
            duration_steps = max(1, end_step - cursor)
            pitches = sorted({n.midi_pitch for n in bucket})
            event_string = None
            if len(bucket) == 1 and string_numbers:
                event_string = string_numbers.get(bucket[0])
            tokens.extend(_event_tokens(pitches, duration_steps, string_number=event_string))
            cursor = end_step
        else:
            next_start = min((s for s in by_step.keys() if s > cursor), default=total_steps)
            rest_steps = max(1, next_start - cursor)
            tokens.extend(_event_tokens([], rest_steps))
            cursor = next_start

        if cursor < total_steps and cursor % measure_steps == 0:
            tokens.append("|")

    return " ".join(tokens) if tokens else "r1"


def _event_tokens(pitches: list[int], steps: int, string_number: int | None = None) -> list[str]:
    chunks = _split_steps(steps)
    rendered: list[str] = []
    for i, chunk in enumerate(chunks):
        denom = _duration_denominator(chunk)
        if pitches:
            if len(pitches) == 1:
                token = f"{_midi_to_lily(pitches[0])}{denom}"
                if string_number is not None:
                    token += f"\\{int(string_number)}"
            else:
                chord = " ".join(_midi_to_lily(p) for p in pitches)
                token = f"<{chord}>{denom}"
        else:
            token = f"r{denom}"
        if i < len(chunks) - 1 and pitches:
            token += "~"
        rendered.append(token)
    return rendered


def _split_steps(steps: int) -> list[int]:
    remaining = max(1, int(steps))
    chunks: list[int] = []
    for size in (16, 8, 4, 2, 1):
        while remaining >= size:
            chunks.append(size)
            remaining -= size
    if not chunks:
        chunks = [1]
    return chunks


def _duration_denominator(steps: int) -> str:
    mapping = {
        16: "1",
        8: "2",
        4: "4",
        2: "8",
        1: "16",
    }
    return mapping.get(steps, "16")


def _midi_to_lily(midi_pitch: int) -> str:
    midi_pitch = int(midi_pitch)
    name = _NOTE_NAMES_SHARP[midi_pitch % 12]
    octave = midi_pitch // 12 - 1
    if octave >= 4:
        marks = "'" * (octave - 3)
    else:
        marks = "," * (3 - octave)
    return f"{name}{marks}"


def render_lilypond_pdf(lilypond_path: Path, log_fn=None) -> Path:
    if LILYPOND_BIN is None:
        raise RuntimeError("LilyPond is not installed or not on PATH.")

    out_dir = lilypond_path.parent
    cmd = [
        LILYPOND_BIN,
        "--pdf",
        "--output",
        str(out_dir / lilypond_path.stem),
        str(lilypond_path),
    ]
    if log_fn:
        log_fn(f"[notation] LilyPond command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if log_fn and proc.stdout.strip():
        log_fn(f"[notation] LilyPond stdout:\n{proc.stdout.rstrip()}")
    if log_fn and proc.stderr.strip():
        log_fn(f"[notation] LilyPond stderr:\n{proc.stderr.rstrip()}")
    if proc.returncode != 0:
        stderr = proc.stderr.strip() or proc.stdout.strip() or "unknown LilyPond error"
        raise RuntimeError(f"LilyPond render failed: {stderr}")
    pdf_path = out_dir / f"{lilypond_path.stem}.pdf"
    if not pdf_path.exists():
        raise RuntimeError("LilyPond completed but no PDF was created.")
    if log_fn:
        log_fn(f"[notation] Generated PDF: {pdf_path}")
    return pdf_path
