import sys
from pathlib import Path
from PySide6 import QtWidgets
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut

from audio_engine import LoopPlayer
from chords import estimate_chords, estimate_key
try:
    from timestretch import render_time_stretch
    HAS_STRETCH = True
except Exception:
    HAS_STRETCH = False
from utils import temp_wav_path


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


class Main(QtWidgets.QMainWindow):
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
        self.btn_load = QtWidgets.QPushButton("Load Audio…")
        self.btn_play = QtWidgets.QPushButton("Play")
        self.btn_pause = QtWidgets.QPushButton("Pause")
        self.btn_setA = QtWidgets.QPushButton("Set A")
        self.btn_setB = QtWidgets.QPushButton("Set B")
        self.rate_spin = QtWidgets.QDoubleSpinBox()
        self.rate_spin.setRange(0.5, 1.5)
        self.rate_spin.setSingleStep(0.05)
        self.rate_spin.setValue(1.0)
        self.rate_label = QtWidgets.QLabel("Rate: 1.00x")
        self.btn_render = QtWidgets.QPushButton("Render@Rate")
        self.btn_render.setEnabled(HAS_STRETCH)
        if not HAS_STRETCH:
            self.btn_render.setToolTip("timestretch.render_time_stretch not available")

        top = QtWidgets.QHBoxLayout()
        for w in (self.btn_load, self.btn_play, self.btn_pause, self.btn_setA, self.btn_setB, self.rate_label, self.rate_spin, self.btn_render):
            top.addWidget(w)
        top.addStretch(1)

        self.chord_table = QtWidgets.QTableWidget(0, 3)
        self.chord_table.setHorizontalHeaderLabels(["Start", "End", "Chord"])
        self.chord_table.horizontalHeader().setStretchLastSection(True)
        self.chord_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.chord_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.addLayout(top)

        # Header row with song title (left) and key (right)
        self.title_label = QtWidgets.QLabel("No song loaded")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.key_header_label = QtWidgets.QLabel("Key: —")
        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.title_label)
        header.addStretch(1)
        header.addWidget(self.key_header_label)
        layout.addLayout(header)

        layout.addWidget(QtWidgets.QLabel("Detected Chords (double‑click a row to loop that segment)"))
        layout.addWidget(self.chord_table)
        self.setCentralWidget(central)

        # Menu (File→Open)
        open_act = QAction("Open…", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self.load_audio)
        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(open_act)

        # Signals
        self.btn_load.clicked.connect(self.load_audio)
        self.btn_play.clicked.connect(self.play)
        self.btn_pause.clicked.connect(self.pause)
        self.btn_setA.clicked.connect(self.set_A)
        self.btn_setB.clicked.connect(self.set_B)
        self.rate_spin.valueChanged.connect(self._rate_changed)
        self.btn_render.clicked.connect(self.render_rate)
        self.chord_table.cellDoubleClicked.connect(self.loop_from_row)

        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_play)

        # Status
        self.statusBar().showMessage("Ready")

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
        # Default loop to first 10 seconds or 20% of duration
        total_s = self.player.n / self.player.sr
        self.A, self.B = 0.0, max(5.0, total_s * 0.2)
        self.player.set_loop_seconds(self.A, self.B)
        self.statusBar().showMessage(f"Loaded: {Path(fn).name} [{total_s:.1f}s]")
        # Kick off chord worker
        self.populate_chords_async(fn)
        self.populate_key_async(fn)

    def populate_chords_async(self, path: str):
        self.chord_table.setRowCount(0)
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
        self.chord_table.setRowCount(len(segs))
        for i, s in enumerate(segs):
            self.chord_table.setItem(i, 0, QtWidgets.QTableWidgetItem(f"{s['start']:.2f}"))
            self.chord_table.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{s['end']:.2f}"))
            self.chord_table.setItem(i, 2, QtWidgets.QTableWidgetItem(s['label']))
        self.statusBar().showMessage(f"Chords: {len(segs)} segments")

    def loop_from_row(self, row: int, _col: int):
        try:
            t0 = float(self.chord_table.item(row, 0).text())
            t1 = float(self.chord_table.item(row, 1).text())
        except Exception:
            return
        if self.player:
            self.player.set_loop_seconds(t0, t1)
            self.A, self.B = t0, t1

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
            self.statusBar().showMessage(f"A set to {self.A:.2f}s")

    def set_B(self):
        if self.player:
            self.B = self.player.position_seconds()
            if self.B <= self.A:
                self.B = self.A + 0.1
            self.player.set_loop_seconds(self.A, self.B)
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
                total_s = self.player.n / self.player.sr
                self.A, self.B = 0.0, max(5.0, total_s * 0.2)
                self.player.set_loop_seconds(self.A, self.B)
                self.statusBar().showMessage(f"Loaded: {name} [{total_s:.1f}s]")
                self.setWindowTitle(f"OpenPractice — {name}")
                self.title_label.setText(name)
                self.populate_chords_async(fn)
                self.populate_key_async(fn)
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