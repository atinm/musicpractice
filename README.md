# MusicPractice

A minimal music practice application for musicians to analyze, loop, and practice with audio tracks. Features real-time audio analysis, stem separation, and interactive waveform visualization.

## Features

- **Audio Analysis**: Automatic chord detection, key estimation, and beat tracking
- **Interactive Waveform**: Visual waveform with chord annotations and beat markers
- **Loop Management**: Create, save, and manage multiple loops with visual flags
- **Time Stretching**: Pitch-preserving tempo adjustment (0.5x - 1.5x)
- **Stem Separation**: AI-powered source separation using Demucs (vocals, drums, bass, etc.)
- **Session Management**: Save and restore analysis data and loop configurations
- **Cross-Platform**: Works on macOS, Windows, and Linux

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install System Dependencies

#### macOS (Homebrew)

```bash
brew install portaudio ffmpeg
```

#### Ubuntu/Debian

```bash
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv portaudio19-dev ffmpeg
```

#### Windows

- Install [PortAudio](https://www.portaudio.com/download.html)
- Install [FFmpeg](https://ffmpeg.org/download.html) and add to PATH

## Usage

```bash
python app.py
```

### Basic Controls

- **Load Audio**: File → Open or drag & drop audio files (WAV, MP3, FLAC, M4A)
- **Playback**: Space bar or toolbar buttons
- **Loop Creation**:
  - Drag on waveform to create new loops
  - Click "Set A" and "Set B" buttons to set loop points
  - Right-click loop flags to rename or delete
- **Navigation**:
  - Click waveform to seek
  - Mouse wheel to zoom in/out
  - Arrow keys to pan left/right
- **Time Stretching**: Adjust rate slider (0.5x - 1.5x) for pitch-preserving tempo changes

### Keyboard Shortcuts

- **Space**: Play/Pause
- **Left/Right Arrow**: Skip to previous/next bar
- **Home**: Go to start of track
- **Ctrl+O**: Open audio file
- **Ctrl+S**: Save session
- **Ctrl+Shift+O**: Load session
- **Ctrl+Shift+R**: Recompute analysis

### Advanced Features

#### Stem Separation

- View → Separate & Show Stems
- Automatically separates audio into vocals, drums, bass, guitar, piano, and other
- Individual volume and mute controls for each stem
- Uses Demucs AI model for high-quality separation

#### Waveform View Modes

- View → Show Stem Waveforms: Display individual waveforms for each separated stem
- View → Show Combined Waveform: Display the original full mix waveform
- Toggle between views to analyze individual parts or the complete mix

#### Session Management

- File → Save Session: Saves loops, analysis data, and settings
- File → Load Session: Restores previous session state
- Sessions are automatically saved alongside audio files

#### Analysis Management

- Analysis → Recompute analysis (Ctrl+Shift+R): Force re-analysis of current track
- Analysis → Always recompute on open: Automatically clear cache and re-analyze when opening files
- Clears cached chord detection, beat tracking, and stem separation data

#### Beat Snapping

- Toggle "Snap" checkbox to snap loop edges to detected beats
- Automatic beat and bar detection with visual markers

## Technical Details

### Audio Analysis

- **Chord Detection**: Template matching with Viterbi smoothing for major/minor triads and 7th chords
- **Key Estimation**: Krumhansl-Schmuckler key profiles with enharmonic spelling
- **Beat Tracking**: Librosa-based onset detection and tempo estimation
- **Time Stretching**: Phase vocoder implementation preserving pitch

### Supported Formats

- **Input**: WAV, MP3, FLAC, M4A, AAC
- **Output**: WAV (for rendered time-stretched audio)

### Dependencies

- **Core**: PySide6 (Qt), NumPy, SciPy, Librosa, SoundDevice
- **Audio Processing**: SoundFile, FFmpeg
- **AI Separation**: Demucs, PyTorch (optional, CPU works)
- **Utilities**: TQDM for progress bars

## File Structure

```
musicpractice/
├── app.py              # Main application and GUI
├── audio_engine.py     # Audio playback and time stretching
├── chords.py           # Chord detection and key estimation
├── stems.py            # Stem separation with Demucs
├── timestretch.py      # Time stretching utilities
├── utils.py            # Helper functions
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Notes

- Time-stretch renders to temporary WAV files for playback (pitch preserved)
- Chord detection uses major/minor triads and 7th chords with Viterbi smoothing
- Stem separation requires significant CPU/GPU resources and may take several minutes
- Session files are saved as `.musicpractice.json` alongside audio files
- All analysis is cached and restored when reopening files
- Use "Recompute analysis" if you need fresh analysis results or encounter issues
- Stem separation results are cached in `.musicpractice/stems/` directories
