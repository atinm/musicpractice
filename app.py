import sys
from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut
import numpy as np

from audio_engine import LoopPlayer
from chords import estimate_chords, estimate_key, estimate_beats
try:
    from timestretch import render_time_stretch
    HAS_STRETCH = True
except Exception:
    HAS_STRETCH = False
from utils import temp_wav_path


class WaveformView(QtWidgets.QWidget):
    """Scrolling waveform that moves right→left with a chord lane underneath."""

    requestSetLoop = QtCore.Signal(float, float)  # (A, B) in seconds (absolute timeline)
    requestSeek = QtCore.Signal(float)  # absolute seconds to seek playhead
    def __init__(self, parent=None, window_s: float = 15.0):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(160)
        self.window_s = window_s
        self.player = None  # type: LoopPlayer | None
        self.chords = []    # list of {start, end, label}
        self.beats = []   # list[float] in seconds (absolute timeline)
        self.bars = []    # list[float] in seconds (absolute timeline)
        self.origin = 0.0  # seconds; shift visual time so 0 = music start
        self.content_end = None  # seconds; last non-silent audio
        self.loopA = None  # visual loop start (seconds)
        self.loopB = None  # visual loop end (seconds)
        self._drag_mode = None  # 'set' | 'resizeA' | 'resizeB' | 'move' | None
        self.snap_enabled = True  # whether to snap to beats when dragging loop
        self._press_t = None
        self._press_loopA = None
        self._press_loopB = None
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)  # ~30 FPS
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def set_beats(self, beats: list | None, bars: list | None):
        self.beats = beats or []
        self.bars = bars or []
        self.update()

    def set_player(self, player: 'LoopPlayer'):
        self.player = player
        self.update()

    def set_chords(self, segments: list):
        self.chords = segments or []
        self.update()

    def set_origin_offset(self, seconds: float):
        self.origin = float(max(0.0, seconds))
        self.update()

    def set_music_span(self, origin_seconds: float, end_seconds: float):
        self.origin = float(max(0.0, origin_seconds))
        self.content_end = float(max(self.origin, end_seconds))
        self.update()

    def set_loop_visual(self, A: float, B: float):
        self.loopA = float(A)
        self.loopB = float(B)
        self.update()

    def set_snap_enabled(self, enabled: bool):
        self.snap_enabled = bool(enabled)
        self.update()

    def _time_at_x(self, x: int, t0: float, t1: float, w: int) -> float:
        x = max(0, min(w - 1, x))
        return t0 + (t1 - t0) * (x / max(1, w - 1))

    def _current_window(self):
        if not self.player:
            # center playhead at 0 with lookahead/behind
            half = self.window_s * 0.5
            return -half, half
        t = self.player.position_seconds()
        half = self.window_s * 0.5
        # Do NOT clamp to 0 here; negative start will render as empty padding
        start = t - half
        end = t + half
        return start, end

    def _mono_slice_minmax(self, start_s: float, end_s: float, width: int):
        if not self.player or width <= 2:
            return None, None
        y = self.player.y
        sr = self.player.sr
        total_s = self.player.n / float(sr)
        # Overlap of requested window with actual audio
        ov_start = max(0.0, start_s)
        ov_end = min(end_s, total_s)
        # Pre-allocate blank columns for full width
        mins = np.zeros(width, dtype=float)
        maxs = np.zeros(width, dtype=float)
        if ov_end <= ov_start:
            return mins, maxs  # entire window is outside audio → all blank
        # Map overlap to pixel columns
        x0 = int((ov_start - start_s) / (end_s - start_s) * width)
        x1 = int((ov_end   - start_s) / (end_s - start_s) * width)
        x0 = max(0, min(width, x0))
        x1 = max(x0 + 1, min(width, x1))
        # Slice audio
        a = int(ov_start * sr)
        b = int(ov_end   * sr)
        mono = y[a:b].mean(axis=1)
        N = mono.shape[0]
        if N <= 1:
            return mins, maxs
        # Downsample to the overlap width
        seg_w = x1 - x0
        step = max(1, N // seg_w)
        trimmed = mono[: (N // step) * step]
        if trimmed.size == 0:
            return mins, maxs
        reshaped = trimmed.reshape(-1, step)
        mins_seg = reshaped.min(axis=1)
        maxs_seg = reshaped.max(axis=1)
        L = min(seg_w, len(mins_seg))
        mins[x0:x0+L] = mins_seg[:L]
        maxs[x0:x0+L] = maxs_seg[:L]
        return mins, maxs

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self.player:
            return
        t0, t1 = self._current_window()
        t = self._time_at_x(int(e.position().x()), t0, t1, self.width())
        self._press_t = t
        # start a new loop at this position
        self.set_loop_visual(t, t)
        e.accept()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if not self.player or self._press_t is None:
            return
        t0, t1 = self._current_window()
        t = self._time_at_x(int(e.position().x()), t0, t1, self.width())
        a = min(self._press_t, t)
        b = max(self._press_t, t)
        if b - a < 0.01:
            b = a + 0.01
        self.set_loop_visual(a, b)
        e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if self._press_t is None or self.loopA is None or self.loopB is None:
            self._press_t = None
            return
        a = float(min(self.loopA, self.loopB))
        b = float(max(self.loopA, self.loopB))
        # Snap A/B to nearest beats if available
        if self.beats and self.snap_enabled:
            arr = np.asarray(self.beats, dtype=float)
            ia = int(np.argmin(np.abs(arr - a)))
            ib = int(np.argmin(np.abs(arr - b)))
            a_s = float(arr[min(ia, len(arr) - 1)])
            b_s = float(arr[min(ib, len(arr) - 1)])
            if b_s < a_s:
                a_s, b_s = b_s, a_s
            # ensure non-zero loop
            if abs(b_s - a_s) < 1e-3:
                b_s = a_s + 1e-2
            a, b = a_s, b_s
        self.requestSetLoop.emit(a, b)
        self._press_t = None
        e.accept()

    def wheelEvent(self, e: QtGui.QWheelEvent):
        if not self.player:
            e.ignore()
            return
        delta = e.angleDelta()  # QPoint: x = horizontal, y = vertical
        dx = int(delta.x())
        dy = int(delta.y())

        # If horizontal motion dominates, PAN left/right and keep playhead centered
        if abs(dx) > abs(dy) and dx != 0:
            # Each 120 units is one notch; pan by 20% of window per notch
            notches = dx / 120.0
            step = self.window_s * 0.2 * (1 if notches > 0 else -1)
            # Accumulate proportional to notches magnitude
            step *= abs(notches)
            cur = self.player.position_seconds()
            dur = self.player.duration_seconds() if hasattr(self.player, 'duration_seconds') else (self.player.n / float(self.player.sr))
            t_new = max(0.0, min(dur, cur + step))
            # Seek (Main will keep playhead centered in view)
            self.requestSeek.emit(float(t_new))
            e.accept()
            return

        # Otherwise, treat as vertical ZOOM (default/mouse wheel up/down)
        if dy == 0:
            e.ignore()
            return

        # Compute the natural maximum: full musical span (end - origin); fall back to full file duration.
        if self.content_end is not None:
            max_window = max(3.0, (self.content_end - self.origin))
        else:
            dur = self.player.duration_seconds() if hasattr(self.player, 'duration_seconds') else (self.player.n / float(self.player.sr))
            max_window = max(3.0, dur - self.origin)
        # Zoom factor (10% per notch)
        notches = dy / 120.0
        factor = (0.9 ** notches) if notches > 0 else (1 / (0.9 ** abs(notches)))
        new_window = self.window_s * factor
        # Clamp to [min_window, max_window]
        min_window = 3.0
        new_window = max(min_window, min(max_window, new_window))
        if abs(new_window - self.window_s) > 1e-6:
            self.window_s = new_window
            self.update()
        e.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if not self.player:
            return
        key = e.key()
        if key not in (Qt.Key_Left, Qt.Key_Right):
            return super().keyPressEvent(e)
        step = self.window_s * 0.2  # pan by 20% of window
        t = self.player.position_seconds()
        dur = self.player.duration_seconds() if hasattr(self.player, 'duration_seconds') else (self.player.n / float(self.player.sr))
        if key == Qt.Key_Left:
            t_new = max(0.0, t - step)
        else:
            t_new = min(dur, t + step)
        self.requestSeek.emit(float(t_new))
        e.accept()

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        w = self.width()
        h = self.height()
        if w < 10 or h < 10:
            return
        wf_h = int(h * 0.7)
        ch_h = h - wf_h
        t0, t1 = self._current_window()

        p.fillRect(0, 0, w, wf_h, QtGui.QColor(20, 20, 20))
        p.fillRect(0, wf_h, w, ch_h, QtGui.QColor(28, 28, 28))

        # time grid in *visual* seconds where 0 = music start (origin)
        p.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60)))
        rel_t0 = t0 - self.origin
        rel_t1 = t1 - self.origin
        first_s = int(np.floor(rel_t0))
        last_s = int(np.ceil(rel_t1))
        for s in range(first_s, last_s + 1):
            x = int((s - rel_t0) / (rel_t1 - rel_t0) * w)
            p.drawLine(x, 0, x, wf_h)
            if s >= 0:
                p.drawText(x + 2, 12, f"{s}s")

        # beat grid over waveform
        if self.beats:
            p.setPen(QtGui.QPen(QtGui.QColor(70, 90, 110)))
            for bt in self.beats:
                if bt < t0 or bt > t1:
                    continue
                x = int((bt - t0) / (t1 - t0) * w)
                p.drawLine(x, 0, x, wf_h)
        if self.bars:
            p.setPen(QtGui.QPen(QtGui.QColor(120, 170, 220), 2))
            for bar_t in self.bars:
                if bar_t < t0 or bar_t > t1:
                    continue
                x = int((bar_t - t0) / (t1 - t0) * w)
                p.drawLine(x, 0, x, wf_h)

        # loop overlay (if available)
        if self.loopA is not None and self.loopB is not None:
            a = max(t0, min(t1, float(self.loopA)))
            b = max(t0, min(t1, float(self.loopB)))
            if b < a:
                a, b = b, a
            x0 = int((a - t0) / (t1 - t0) * w)
            x1 = int((b - t0) / (t1 - t0) * w)
            p.fillRect(QtCore.QRect(x0, 0, max(1, x1 - x0), wf_h), QtGui.QColor(255, 255, 255, 28))
            p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))
            p.drawLine(x0, 0, x0, wf_h)
            p.drawLine(x1, 0, x1, wf_h)

        mins, maxs = self._mono_slice_minmax(t0, t1, w)
        if mins is not None:
            mid = wf_h // 2
            p.setPen(QtGui.QPen(QtGui.QColor(0, 180, 255)))
            for x, (mn, mx) in enumerate(zip(mins, maxs)):
                y1 = mid - int(mx * (wf_h * 0.45))
                y2 = mid - int(mn * (wf_h * 0.45))
                if y1 > y2:
                    y1, y2 = y2, y1
                p.drawLine(x, y1, x, y2)

        # center playhead for heads‑up on upcoming changes
        p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 0)))
        cx = w // 2
        p.drawLine(cx, 0, cx, wf_h)

        # chord lane
        if self.chords:
            font = p.font()
            font.setPointSizeF(max(9.0, self.font().pointSizeF()))
            p.setFont(font)
            for seg in self.chords:
                a = max(t0, float(seg['start']))
                b = min(t1, float(seg['end']))
                if b <= a:
                    continue
                # relative (visual) times, though x mapping uses absolute t0/t1
                a_rel = a - self.origin
                b_rel = b - self.origin
                x0 = int((a_rel - rel_t0) / (rel_t1 - rel_t0) * w)
                x1 = int((b_rel - rel_t0) / (rel_t1 - rel_t0) * w)
                rect = QtCore.QRect(x0, wf_h, max(1, x1 - x0), ch_h)
                p.fillRect(rect, QtGui.QColor(50, 80, 110))
                p.setPen(QtGui.QPen(QtGui.QColor(120, 170, 220)))
                p.drawRect(rect)
                p.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230)))
                p.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, seg['label'])


class ChordWorker(QThread):
    done = Signal(list)
    def __init__(self, path: str):
        super().__init__()
        self.path = path
    def run(self):
        try:
            segs = estimate_chords(self.path)
        except Exception:
            segs = []
        self.done.emit(segs)


class KeyWorker(QThread):
    done = Signal(dict)
    def __init__(self, path: str):
        super().__init__()
        self.path = path
    def run(self):
        try:
            info = estimate_key(self.path)
        except Exception:
            info = {"pretty": "unknown"}
        self.done.emit(info)


# --- BeatWorker for beat/bar estimation ---
class BeatWorker(QThread):
    done = Signal(dict)
    def __init__(self, path: str):
        super().__init__()
        self.path = path
    def run(self):
        try:
            info = estimate_beats(self.path)
        except Exception:
            info = {"tempo": 0.0, "beats": [], "bars": []}
        self.done.emit(info)


class Main(QtWidgets.QMainWindow):
    def populate_beats_async(self, path: str):
        self.wave.set_beats([], [])
        self.beat_worker = BeatWorker(path)
        self.beat_worker.done.connect(self._beats_ready)
        self.beat_worker.start()

    def _beats_ready(self, info: dict):
        beats = info.get("beats", [])
        bars = info.get("downbeats") or info.get("bars", [])
        self.wave.set_beats(beats, bars)
        if info.get("tempo"):
            self.statusBar().showMessage(f"Tempo: {info['tempo']:.1f} BPM · Beats: {len(beats)}")

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenPractice — Minimal")
        self.resize(960, 600)
        self.setAcceptDrops(True)
        self.settings = QSettings("OpenPractice", "OpenPractice")

        self.player: LoopPlayer | None = None
        self.current_path: str | None = None
        self.A = 0.0
        self.B = 10.0

        # === UI ===
        self.rate_spin = QtWidgets.QDoubleSpinBox()
        self.rate_spin.setRange(0.5, 1.5)
        self.rate_spin.setSingleStep(0.05)
        self.rate_spin.setValue(1.0)
        self.rate_label = QtWidgets.QLabel("Rate: 1.00x")

        # Toolbar
        self.toolbar = QtWidgets.QToolBar("Main", self)
        self.toolbar.setMovable(False)
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.addToolBar(self.toolbar)

        # Actions
        self.act_open = QAction("Load Audio…", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.load_audio)

        self.act_play = QAction("Play", self)
        self.act_play.triggered.connect(self.play)

        self.act_pause = QAction("Pause", self)
        self.act_pause.triggered.connect(self.pause)

        self.act_setA = QAction("Set A", self)
        self.act_setA.triggered.connect(self.set_A)

        self.act_setB = QAction("Set B", self)
        self.act_setB.triggered.connect(self.set_B)

        self.act_render = QAction("Render@Rate", self)
        self.act_render.setEnabled(HAS_STRETCH)
        if not HAS_STRETCH:
            self.act_render.setToolTip("timestretch.render_time_stretch not available")
        self.act_render.triggered.connect(self.render_rate)

        # Populate toolbar
        self.toolbar.addAction(self.act_open)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_play)
        self.toolbar.addAction(self.act_pause)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.act_setA)
        self.toolbar.addAction(self.act_setB)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.rate_label)
        self.toolbar.addWidget(self.rate_spin)
        self.toolbar.addAction(self.act_render)

        # Snap toggle
        self.snap_checkbox = QtWidgets.QCheckBox("Snap")
        self.snap_checkbox.setChecked(True)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.snap_checkbox)

        self.wave = WaveformView()
        self.wave.requestSetLoop.connect(self._apply_loop_from_wave)
        self.wave.requestSeek.connect(self._seek_from_wave)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        # Header row with song title (left) and key (right)
        self.title_label = QtWidgets.QLabel("No song loaded")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.key_header_label = QtWidgets.QLabel("Key: —")
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.key_header_label)
        layout.addLayout(header)

        layout.addWidget(self.wave, 1)
        self.setCentralWidget(central)

        # Menu (File→Open)
        open_act = QAction("Open…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.load_audio)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_act)

        # Signals
        self.rate_spin.valueChanged.connect(self._rate_changed)
        self.snap_checkbox.toggled.connect(lambda on: self.wave.set_snap_enabled(bool(on)))

        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_play)

        # Status
        self.statusBar().showMessage("Ready")

    def _seek_from_wave(self, t: float):
        if not self.player:
            return
        try:
            self.player.set_position_seconds(float(t), within_loop=False)
        except Exception:
            pass
        # ensure key focus remains on the waveform for continued arrow keys
        self.wave.setFocus()

    def _apply_loop_from_wave(self, a: float, b: float):
        if not self.player:
            return
        self.A, self.B = float(a), float(b)
        self.player.set_loop_seconds(self.A, self.B)
        self.wave.set_loop_visual(self.A, self.B)
        self.statusBar().showMessage(f"Loop: {self.A:.2f}s → {self.B:.2f}s")

    def _detect_leading_silence_seconds(self, y: np.ndarray, sr: int, frame: int = 2048, hop: int = 512, db_threshold: float = -40.0, min_frames: int = 5) -> float:
        """Return seconds until audio exceeds threshold for min_frames.
        Threshold in dBFS; -40 dB ≈ 0.01 linear. Uses mono mix RMS per frame."""
        mono = y.mean(axis=1) if y.ndim == 2 else y
        # frame-wise rms
        n = mono.shape[0]
        if n < frame:
            return 0.0
        strides = (mono.strides[0], mono.strides[0])
        shape = (max(1, (n - frame) // hop + 1), frame)
        # build frames safely
        idxs = np.arange(shape[0]) * hop
        rms = []
        for i in idxs:
            seg = mono[i:i+frame]
            if seg.shape[0] < frame:
                # zero-pad last frame
                pad = np.zeros(frame, dtype=mono.dtype)
                pad[:seg.shape[0]] = seg
                seg = pad
            rms.append(np.sqrt(np.mean(seg * seg) + 1e-12))
        rms = np.asarray(rms)
        thr = 10 ** (db_threshold / 20.0)
        above = rms > thr
        # find first index where we have min_frames consecutive Trues
        count = 0
        for i, v in enumerate(above):
            count = count + 1 if v else 0
            if count >= min_frames:
                first_frame = i - min_frames + 1
                secs = max(0.0, first_frame * hop / float(sr))
                # slight preroll so the transient centers better
                return max(0.0, secs - 0.05)
        return 0.0

    def _detect_trailing_silence_seconds(self, y: np.ndarray, sr: int, frame: int = 2048, hop: int = 512, db_threshold: float = -40.0, min_frames: int = 5) -> float:
        """Return time (seconds) of the last non‑silent audio before trailing silence begins.
        If nothing exceeds threshold, returns total duration. Uses mono mix RMS per frame.
        """
        mono = y.mean(axis=1) if y.ndim == 2 else y
        n = mono.shape[0]
        if n < frame:
            return n / float(sr)
        # frame-wise rms (forward, like the leading detector)
        idxs = np.arange(max(1, (n - frame) // hop + 1)) * hop
        rms = []
        for i in idxs:
            seg = mono[i:i+frame]
            if seg.shape[0] < frame:
                pad = np.zeros(frame, dtype=mono.dtype)
                pad[:seg.shape[0]] = seg
                seg = pad
            rms.append(np.sqrt(np.mean(seg * seg) + 1e-12))
        rms = np.asarray(rms)
        thr = 10 ** (db_threshold / 20.0)
        above = rms > thr
        # Scan from the end to find the last run of >= min_frames above‑threshold frames
        count = 0
        last_end_frame = None
        for i in range(len(above) - 1, -1, -1):
            if above[i]:
                count += 1
                if count >= min_frames:
                    last_end_frame = i + (min_frames - 1)
                    break
            else:
                count = 0
        if last_end_frame is None:
            return n / float(sr)
        # Convert frame index to seconds; include the frame length for a natural end
        end_samples = last_end_frame * hop + frame
        end_secs = min(n / float(sr), end_samples / float(sr))
        # small post‑roll for natural tail
        return min(n / float(sr), end_secs + 0.05)

    # === Actions ===
    def _rate_changed(self, val: float):
        self.rate_label.setText(f"Rate: {val:.2f}x")

    def load_audio(self):
        start_dir = self.settings.value("last_dir", str(Path.home()))
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Audio", start_dir, "Audio (*.wav *.mp3 *.flac *.m4a)")
        if not fn:
            return
        self.current_path = fn
        self.settings.setValue("last_dir", str(Path(fn).parent))
        name = Path(fn).name
        self.setWindowTitle(f"OpenPractice — {name}")
        self.title_label.setText(name)
        # (Re)create player
        if self.player:
            self.player.stop(); self.player.close()
        self.player = LoopPlayer(fn)
        self.wave.set_player(self.player)
        # Align visual time 0 to first non-silent audio
        lead = self._detect_leading_silence_seconds(self.player.y, self.player.sr)
        tail = self._detect_trailing_silence_seconds(self.player.y, self.player.sr)
        self.wave.set_music_span(lead, tail)
        # Seek playback to music start for immediate context
        try:
            self.player.set_position_seconds(lead, within_loop=False)
        except Exception:
            pass
        # Default loop: musical span (from music start to detected tail)
        total_s = self.player.n / self.player.sr
        self.A = lead
        self.B = max(lead + 0.1, min(total_s, tail))
        self.player.set_loop_seconds(self.A, self.B)
        self.wave.set_loop_visual(self.A, self.B)
        self.wave.setFocus()
        mus_len = max(0.0, self.B - self.A)
        self.statusBar().showMessage(f"Loaded: {Path(fn).name} [music {mus_len:.1f}s of {total_s:.1f}s]")
        # Kick off chord worker
        self.populate_chords_async(fn)
        self.populate_key_async(fn)
        self.populate_beats_async(fn)

    def populate_chords_async(self, path: str):
        self.wave.set_chords([])
        self.worker = ChordWorker(path)
        self.worker.done.connect(self._chords_ready)
        self.worker.start()

    def populate_key_async(self, path: str):
        self.key_header_label.setText("Key: …")
        self.key_worker = KeyWorker(path)
        self.key_worker.done.connect(self._key_ready)
        self.key_worker.start()

    def _key_ready(self, info: dict):
        pretty = info.get("pretty", "unknown")
        self.key_header_label.setText(f"Key: {pretty}")

    def _chords_ready(self, segs: list):
        origin = getattr(self.wave, 'origin', 0.0)
        adj = []
        for s in segs or []:
            try:
                st = float(s.get('start', 0.0))
                en = float(s.get('end', 0.0))
                lab = s.get('label', '')
            except Exception:
                continue
            if en <= origin:
                continue  # entirely before music start
            st = max(st, origin)  # clamp start to origin
            adj.append({'start': st, 'end': en, 'label': lab})
        self.wave.set_chords(adj)
        self.statusBar().showMessage(f"Chords: {len(adj)} segments")

    def toggle_play(self):
        if not self.player:
            return
        # crude toggle based on whether stream is active
        try:
            self.play() if not self.player._playing else self.pause()
        except Exception:
            pass

    def play(self):
        if self.player:
            self.player.start()

    def pause(self):
        if self.player:
            self.player.stop()

    def set_A(self):
        if self.player:
            self.A = self.player.position_seconds()
            if self.A >= self.B:
                self.B = self.A + 0.1
            self.player.set_loop_seconds(self.A, self.B)
            self.wave.set_loop_visual(self.A, self.B)
            self.statusBar().showMessage(f"A set to {self.A:.2f}s")

    def set_B(self):
        if self.player:
            self.B = self.player.position_seconds()
            if self.B <= self.A:
                self.B = self.A + 0.1
            self.player.set_loop_seconds(self.A, self.B)
            self.wave.set_loop_visual(self.A, self.B)
            self.statusBar().showMessage(f"B set to {self.B:.2f}s")

    def render_rate(self):
        if not (self.player and self.current_path and HAS_STRETCH):
            return
        rate = float(self.rate_spin.value())
        out = temp_wav_path("openpractice_render")
        self.statusBar().showMessage("Rendering…")
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            render_time_stretch(self.current_path, rate, out)
            # Reload rendered file but keep same A/B seconds
            self.player.reload(out)
            self.player.set_loop_seconds(self.A, self.B)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            self.statusBar().showMessage("Ready")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for url in e.mimeData().urls():
                if url.toLocalFile().lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):
        for url in e.mimeData().urls():
            fn = url.toLocalFile()
            if fn.lower().endswith((".wav", ".mp3", ".flac", ".m4a")):
                # mimic File→Open
                self.current_path = fn
                self.settings.setValue("last_dir", str(Path(fn).parent))
                name = Path(fn).name
                if self.player:
                    self.player.stop(); self.player.close()
                self.player = LoopPlayer(fn)
                self.wave.set_player(self.player)
                lead = self._detect_leading_silence_seconds(self.player.y, self.player.sr)
                tail = self._detect_trailing_silence_seconds(self.player.y, self.player.sr)
                self.wave.set_music_span(lead, tail)
                try:
                    self.player.set_position_seconds(lead, within_loop=False)
                except Exception:
                    pass
                total_s = self.player.n / self.player.sr
                self.A = lead
                self.B = max(lead + 0.1, min(total_s, tail))
                self.player.set_loop_seconds(self.A, self.B)
                self.wave.set_loop_visual(self.A, self.B)
                self.wave.setFocus()
                mus_len = max(0.0, self.B - self.A)
                self.statusBar().showMessage(f"Loaded: {name} [music {mus_len:.1f}s of {total_s:.1f}s]")
                self.setWindowTitle(f"OpenPractice — {name}")
                self.title_label.setText(name)
                self.populate_chords_async(fn)
                self.populate_key_async(fn)
                self.populate_beats_async(fn)
                break

    def closeEvent(self, e):
        try:
            if self.player:
                self.player.stop(); self.player.close()
        finally:
            super().closeEvent(e)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())