# MusicPractice (Minimal)

## Setup

python -m venv .venv
source .venv/bin/activate # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# macOS (Homebrew) — required for sounddevice

brew install portaudio ffmpeg

# Ubuntu/Debian

sudo apt-get update && sudo apt-get install -y python3-pip python3-venv portaudio19-dev ffmpeg

## Run

python app.py

## Notes

- Time‑stretch renders to a temporary WAV and then plays it (pitch preserved).
- Chord detection is major/minor triads with simple Viterbi smoothing.
- Click a chord row to set the loop to that segment automatically.
