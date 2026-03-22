from __future__ import annotations

import json
import sys
from pathlib import Path

from basic_pitch.inference import predict
import pretty_midi


def _normalize_note_event(event) -> dict:
    if hasattr(event, "start_time_s"):
        start = float(event.start_time_s)
        end = float(event.end_time_s)
        pitch = int(round(event.pitch))
        velocity = int(round(getattr(event, "amplitude", 0.8) * 127))
        return {"start": start, "end": end, "pitch": pitch, "velocity": max(1, min(127, velocity))}

    if isinstance(event, dict):
        start = float(event.get("start", event.get("start_time_s", 0.0)))
        end = float(event.get("end", event.get("end_time_s", start)))
        pitch = int(round(event.get("pitch", event.get("midi", 60))))
        raw_velocity = event.get("velocity", event.get("amplitude", 0.8))
        velocity = int(round(float(raw_velocity) * 127)) if float(raw_velocity) <= 1.0 else int(round(float(raw_velocity)))
        return {"start": start, "end": end, "pitch": pitch, "velocity": max(1, min(127, velocity))}

    if isinstance(event, (list, tuple)) and len(event) >= 3:
        start = float(event[0])
        end = float(event[1])
        pitch = int(round(event[2]))
        raw_velocity = event[3] if len(event) > 3 else 0.8
        velocity = int(round(float(raw_velocity) * 127)) if float(raw_velocity) <= 1.0 else int(round(float(raw_velocity)))
        return {"start": start, "end": end, "pitch": pitch, "velocity": max(1, min(127, velocity))}

    raise TypeError(f"Unsupported Basic Pitch note event: {event!r}")


def _load_note_events_from_midi(midi_path: Path) -> list[dict]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    events: list[dict] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            events.append(
                {
                    "start": float(note.start),
                    "end": float(note.end),
                    "pitch": int(note.pitch),
                    "velocity": max(1, min(127, int(note.velocity))),
                }
            )
    events.sort(key=lambda event: (float(event["start"]), int(event["pitch"]), float(event["end"])))
    return events


def _program_for_instrument(name: str) -> int:
    instrument = name.strip().lower()
    if instrument == "bass":
        return int(pretty_midi.instrument_name_to_program("Electric Bass (finger)"))
    if instrument == "guitar":
        return int(pretty_midi.instrument_name_to_program("Electric Guitar (clean)"))
    return 0


def _write_note_events_to_midi(note_events: list[dict], midi_path: Path, instrument_name: str) -> None:
    midi = pretty_midi.PrettyMIDI()
    program = _program_for_instrument(instrument_name)
    instrument = pretty_midi.Instrument(program=program, is_drum=False, name=instrument_name)
    for event in note_events:
        start = max(0.0, float(event.get("start", 0.0)))
        end = max(start + 1e-3, float(event.get("end", start + 1e-3)))
        pitch = int(event.get("pitch", 60))
        velocity = max(1, min(127, int(event.get("velocity", 80))))
        instrument.notes.append(pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end))
    midi.instruments.append(instrument)
    midi.write(str(midi_path))


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "midi-to-json":
        if len(argv) != 4:
            print("usage: basic_pitch_helper.py midi-to-json <input-midi> <output-json>", file=sys.stderr)
            return 2
        midi_path = Path(argv[2])
        json_path = Path(argv[3])
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(_load_note_events_from_midi(midi_path), indent=2), encoding="utf-8")
        return 0

    if len(argv) >= 2 and argv[1] == "json-to-midi":
        if len(argv) not in (4, 5):
            print(
                "usage: basic_pitch_helper.py json-to-midi <input-json> <output-midi> [<instrument>]",
                file=sys.stderr,
            )
            return 2
        json_path = Path(argv[2])
        midi_path = Path(argv[3])
        instrument_name = argv[4] if len(argv) >= 5 else "guitar"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        note_events = json.loads(json_path.read_text(encoding="utf-8"))
        _write_note_events_to_midi([_normalize_note_event(event) for event in note_events], midi_path, instrument_name)
        return 0

    if len(argv) not in (4, 6):
        print(
            "usage: basic_pitch_helper.py <input-audio> <output-midi> <output-json> [<min-hz> <max-hz>]",
            file=sys.stderr,
        )
        return 2

    audio_path = Path(argv[1])
    midi_path = Path(argv[2])
    json_path = Path(argv[3])
    min_hz = float(argv[4]) if len(argv) >= 5 and argv[4] != "none" else None
    max_hz = float(argv[5]) if len(argv) >= 6 and argv[5] != "none" else None

    _, midi_data, note_events = predict(
        str(audio_path),
        minimum_frequency=min_hz,
        maximum_frequency=max_hz,
    )
    midi_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    midi_data.write(str(midi_path))
    json_path.write_text(
        json.dumps([_normalize_note_event(event) for event in note_events], indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
