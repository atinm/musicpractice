import librosa
import soundfile as sf

def render_time_stretch(input_path: str, rate: float, output_path: str) -> str:
    """
    Render a new audio file at the given rate (<1.0 slower, >1.0 faster) while preserving pitch.
    """
    y, sr = librosa.load(input_path, sr=None, mono=True)
    D = librosa.stft(y, n_fft=2048, hop_length=512, win_length=2048)
    D_stretch = librosa.phase_vocoder(D, rate=rate)
    y_out = librosa.istft(D_stretch, hop_length=512, win_length=2048)
    sf.write(output_path, y_out, sr)
    return output_path