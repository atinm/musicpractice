import sys
import json
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
    requestAddLoop = QtCore.Signal(float, float)          # create a new saved loop [a,b]
    requestUpdateLoop = QtCore.Signal(int, float, float)  # update saved loop id → [a,b]
    requestSelectLoop = QtCore.Signal(int)                # select a saved loop by id
    requestDeleteSelected = QtCore.Signal()               # delete currently selected saved loop
    requestRenameLoop = QtCore.Signal(int)     # loop id to rename
    requestDeleteLoopId = QtCore.Signal(int)   # loop id to delete

    def set_saved_loops(self, loops: list[dict] | None):
        self.saved_loops = list(loops or [])
        # Clear stale selection if the id no longer exists
        if hasattr(self, 'selected_loop_id') and self.selected_loop_id is not None:
            existing_ids = {int(L.get('id')) for L in self.saved_loops if 'id' in L}
            if int(self.selected_loop_id) not in existing_ids:
                self.selected_loop_id = None
        self.update()

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
        # Interaction geometry & multi-loop scaffolding (safe if unused)
        self.HANDLE_PX = 6
        self.FLAG_W = 10
        self.FLAG_H = 10
        self.FLAG_STRIP = 16  # pixels reserved at the top of waveform for flags & loop names
        self.saved_loops = []            # optional: list of dicts {id,a,b,label}
        self.selected_loop_id = None
        # drag state
        self._press_kind = None          # 'new' | 'edgeA' | 'edgeB' | None
        self._press_loop_id = None
        self._press_dx = 0.0             # offset used when moving a loop via flag
        # click vs drag tracking
        self._press_x = None             # x at mouse press (px)
        self._drag_started = False       # becomes True once threshold passed
        self._click_thresh_px = 4        # pixels before treating as drag
        self._freeze_window = False
        self._manual_t0 = None
        self._manual_t1 = None

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

    def clear_loop_visual(self):
        self.loopA = None
        self.loopB = None
        self.update()

    def set_snap_enabled(self, enabled: bool):
        self.snap_enabled = bool(enabled)
        self.update()

    def _x_from_time(self, t: float, t0: float, t1: float, w: int) -> int:
        """Convert time to pixel column, clamped to [0, w-1]."""
        return int(max(0, min(w - 1, (t - t0) / (t1 - t0) * w)))

    def _hit_test(self, pos: QtCore.QPoint, t0: float, t1: float, w: int, wf_h: int):
        """Return (kind, loop_id) if hit on a flag or edge."""
        x = pos.x()
        y = pos.y()
        flag_top = 0
        flag_bottom = min(wf_h, self.FLAG_STRIP)
        if hasattr(self, 'saved_loops'):
            for L in self.saved_loops:
                a = float(L['a'])
                b = float(L['b'])
                fx = self._x_from_time(a, t0, t1, w)
                # START flag hit-test → resize A
                if abs(x - fx) <= self.FLAG_W and (0 <= y <= wf_h) and (flag_top <= y <= flag_bottom):
                    return ("edgeA", int(L['id']))
                # END flag hit-test (acts like grabbing edgeB for resize)
                fxb = self._x_from_time(b, t0, t1, w)
                if abs(x - fxb) <= self.FLAG_W and (0 <= y <= wf_h) and (flag_top <= y <= flag_bottom):
                    return ("edgeB", int(L['id']))
                # edge hit zones (within waveform)
                if 0 <= y <= wf_h:
                    xa = self._x_from_time(a, t0, t1, w)
                    xb = self._x_from_time(b, t0, t1, w)
                    if abs(x - xa) <= self.HANDLE_PX:
                        return ("edgeA", int(L['id']))
                    if abs(x - xb) <= self.HANDLE_PX:
                        return ("edgeB", int(L['id']))
        # Fallback: active loop handles/flags when no saved loop was hit
        if self.loopA is not None and self.loopB is not None:
            a = float(min(self.loopA, self.loopB)); b = float(max(self.loopA, self.loopB))
            fx = self._x_from_time(a, t0, t1, w)
            # START flag → resize A on active loop
            if abs(x - fx) <= self.FLAG_W and (0 <= y <= wf_h) and (flag_top <= y <= flag_bottom):
                return ("edgeA", -1)
            # END flag (treat like edgeB resize)
            fxb = self._x_from_time(b, t0, t1, w)
            if abs(x - fxb) <= self.FLAG_W and (0 <= y <= wf_h) and (flag_top <= y <= flag_bottom):
                return ("edgeB", -1)
            # edges within waveform
            if 0 <= y <= wf_h:
                xa = self._x_from_time(a, t0, t1, w)
                xb = self._x_from_time(b, t0, t1, w)
                if abs(x - xa) <= self.HANDLE_PX:
                    return ("edgeA", -1)
                if abs(x - xb) <= self.HANDLE_PX:
                    return ("edgeB", -1)
        return (None, None)

    def _time_at_x(self, x: int, t0: float, t1: float, w: int) -> float:
        x = max(0, min(w - 1, x))
        return t0 + (t1 - t0) * (x / max(1, w - 1))

    def _window_from_center(self, c: float):
        half = self.window_s * 0.5
        return c - half, c + half

    def freeze_window_now(self):
        """Capture the current visual window and hold it until unfreeze."""
        t0, t1 = self._current_window()
        self._manual_t0, self._manual_t1 = float(t0), float(t1)
        self._freeze_window = True
        self.update()

    def unfreeze_and_center(self):
        """Release the hold so the view recenters on the playhead."""
        self._freeze_window = False
        self._manual_t0 = None
        self._manual_t1 = None
        self.update()

    def _current_window(self):
        # If frozen (after a click seek), hold the captured window
        if self._freeze_window and self._manual_t0 is not None and self._manual_t1 is not None:
            return float(self._manual_t0), float(self._manual_t1)
        if not self.player:
            half = self.window_s * 0.5
            return -half, half
        t = self.player.position_seconds()
        return self._window_from_center(t)

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

    def contextMenuEvent(self, e: QtGui.QContextMenuEvent):
        # Right-click → offer a small menu if we're over a loop flag/edge
        if not hasattr(self, 'saved_loops') or not self.saved_loops:
            return
        t0, t1 = self._current_window()
        w = self.width(); h = self.height(); wf_h = int(h * 0.7)
        pos = e.pos()
        kind, lid = self._hit_test(pos, t0, t1, w, wf_h)
        if lid is None or lid == -1:
            return  # only support saved loops via menu
        # Select it for visual feedback
        self.selected_loop_id = int(lid)
        self.requestSelectLoop.emit(int(lid))
        # Build menu
        menu = QtWidgets.QMenu(self)
        actRename = menu.addAction("Rename loop…")
        actDelete = menu.addAction("Delete loop")
        chosen = menu.exec(e.globalPos())
        if chosen == actRename:
            self.requestRenameLoop.emit(int(lid))
        elif chosen == actDelete:
            self.requestDeleteLoopId.emit(int(lid))

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if not self.player:
            return
        t0, t1 = self._current_window()
        w = self.width(); h = self.height(); wf_h = int(h * 0.7)
        kind, lid = self._hit_test(e.position().toPoint(), t0, t1, w, wf_h)
        t = self._time_at_x(int(e.position().x()), t0, t1, w)
        self._press_t = t
        self._press_x = int(e.position().x())
        self._drag_started = False
        if kind is None:
            # empty space: may become seek (click) or new loop (drag)
            self._press_kind = 'empty'
            self._press_loop_id = None
        else:
            # grabbed an existing loop edge; prepare to resize
            self._press_kind = kind
            self._press_loop_id = lid
            # preview extents for feedback
            L = next((x for x in self.saved_loops if x.get('id') == lid), None)
            if L is None and lid == -1 and self.loopA is not None and self.loopB is not None:
                a = float(min(self.loopA, self.loopB)); b = float(max(self.loopA, self.loopB))
            elif L is not None:
                a = float(min(L['a'], L['b'])); b = float(max(L['a'], L['b']))
            else:
                a = b = t
            self.set_loop_visual(a, b)
            if lid is not None:
                self.selected_loop_id = lid if lid != -1 else None
                if lid != -1:
                    self.requestSelectLoop.emit(int(lid))
        e.accept()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if not self.player or self._press_t is None:
            return
        t0, t1 = self._current_window(); w = self.width(); h = self.height()
        t = self._time_at_x(int(e.position().x()), t0, t1, w)
        if self._press_kind == 'empty':
            dx = abs(int(e.position().x()) - int(self._press_x))
            if not self._drag_started and dx >= self._click_thresh_px:
                # begin creating a new loop
                self._drag_started = True
                a = min(self._press_t, t); b = max(self._press_t, t)
                if b - a < 0.01: b = a + 0.01
                self.set_loop_visual(a, b)
            elif self._drag_started:
                a = min(self._press_t, t); b = max(self._press_t, t)
                if b - a < 0.01: b = a + 0.01
                self.set_loop_visual(a, b)
        elif self._press_kind in ('edgeA', 'edgeB') and self._press_loop_id is not None:
            L = next((x for x in self.saved_loops if x.get('id') == self._press_loop_id), None)
            if L is None:
                if self.loopA is None or self.loopB is None:
                    return
                a = float(min(self.loopA, self.loopB)); b = float(max(self.loopA, self.loopB))
            else:
                a = float(min(L['a'], L['b'])); b = float(max(L['a'], L['b']))
            if self._press_kind == 'edgeA':
                a = min(t, b - 0.01)
            elif self._press_kind == 'edgeB':
                b = max(t, a + 0.01)
            self._drag_started = True
            self.set_loop_visual(a, b)
        e.accept()

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        # Click (no drag) in empty space → seek to that time
        if self._press_kind == 'empty' and not self._drag_started and self._press_t is not None:
            self.freeze_window_now()
            self.requestSeek.emit(float(self._press_t))
            # reset
            self._press_kind = None; self._press_loop_id = None; self._press_t = None; self._press_x = None; self._drag_started = False
            e.accept();
            return

        if self._press_t is None or self.loopA is None or self.loopB is None:
            # reset
            self._press_kind = None; self._press_loop_id = None; self._press_t = None; self._press_x = None; self._drag_started = False
            return
        a = float(min(self.loopA, self.loopB)); b = float(max(self.loopA, self.loopB))
        # Optional snap to beats
        if self.beats and self.snap_enabled:
            arr = np.asarray(self.beats, dtype=float)
            ia = int(np.argmin(np.abs(arr - a)))
            ib = int(np.argmin(np.abs(arr - b)))
            a_s = float(arr[min(ia, len(arr) - 1)])
            b_s = float(arr[min(ib, len(arr) - 1)])
            if b_s < a_s: a_s, b_s = b_s, a_s
            if abs(b_s - a_s) < 1e-3: b_s = a_s + 1e-2
            a, b = a_s, b_s
        # Commit
        if (self._press_kind in ('empty', 'new')) and self._drag_started:
            self.requestAddLoop.emit(a, b)
        elif self._press_kind in ('edgeA', 'edgeB') and self._press_loop_id is not None:
            self.requestUpdateLoop.emit(int(self._press_loop_id), a, b)
        else:
            self.requestSetLoop.emit(a, b)
        # reset
        self._press_kind = None; self._press_loop_id = None; self._press_t = None; self._press_x = None; self._drag_started = False
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
            if self._freeze_window and self._manual_t0 is not None and self._manual_t1 is not None:
                mid = 0.5 * (self._manual_t0 + self._manual_t1)
                self._manual_t0, self._manual_t1 = self._window_from_center(mid)
                self.update()
        e.accept()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if not self.player:
            return
        key = e.key()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self.requestDeleteSelected.emit()
            e.accept()
            return
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
        ch_h = 32
        wf_h = max(10, h - ch_h)
        t0, t1 = self._current_window()
        # Visual times relative to the music start (origin)
        rel_t0 = t0 - self.origin
        rel_t1 = t1 - self.origin

        p.fillRect(0, 0, w, wf_h, QtGui.QColor(20, 20, 20))
        p.fillRect(0, wf_h, w, ch_h, QtGui.QColor(28, 28, 28))
        # Constrain subsequent waveform drawings (grid, beats, flags, loop fills, waveform, playhead)
        p.save()
        p.setClipRect(0, 0, w, wf_h)

        # --- TOP RULER: Prefer BEAT ruler; back-fill beats prior to first downbeat. Fallback → time ruler.
        have_beats = bool(self.beats)
        have_bars = bool(self.bars)
        long_len = min(12, wf_h)
        short_len = min(6, wf_h)
        if have_beats:
            beat_times = np.asarray(self.beats, dtype=float)
            # Estimate average beat period for back-fill if needed
            if beat_times.size >= 2:
                diffs = np.diff(beat_times)
                good = diffs[(diffs > 0.1) & (diffs < 2.0)]
                period = float(np.median(good)) if good.size else float(np.median(diffs))
            else:
                period = 0.5
            # Map bars (downbeats) to nearest beat index (±30 ms) to infer phase / beats-per-bar
            down_idx = set()
            phase_i0 = None
            beats_per_bar = None
            bar_index_by_beat = {}
            if have_bars and beat_times.size:
                idxs = []
                for bar_i, bar_t in enumerate(self.bars, start=1):
                    i = int(np.argmin(np.abs(beat_times - bar_t)))
                    if abs(beat_times[i] - bar_t) <= 0.03:
                        idxs.append(i)
                        bar_index_by_beat[i] = bar_i  # 1-based numbering
                if idxs:
                    idxs = sorted(set(int(i) for i in idxs))
                    phase_i0 = idxs[0]
                    if len(idxs) >= 2:
                        gaps = np.diff(idxs)
                        if gaps.size:
                            beats_per_bar = int(np.round(np.median(gaps)))
                    for i in idxs:
                        down_idx.add(i)
            # Build beats covering window: prepend synthetic beats before first to show pickup
            beats_full = []
            base_first_idx = 0
            if beat_times.size:
                first_bt = float(beat_times[0])
                k = 1
                # back-fill slightly beyond window to ensure coverage at left edge
                while period > 0 and (first_bt - k * period) > (t0 - 2 * period):
                    beats_full.append(first_bt - k * period)
                    k += 1
                beats_full = list(reversed(beats_full)) + beat_times.tolist()
                base_first_idx = k - 1
            # Pens
            pen_long = QtGui.QPen(QtGui.QColor(150, 170, 200)); pen_long.setWidth(1)
            pen_short = QtGui.QPen(QtGui.QColor(90, 90, 90))
            text_pen = QtGui.QPen(QtGui.QColor(200, 210, 220))
            # Draw ticks (+ bar numbers on downbeats, including a backfilled bar 0)
            for j, bt in enumerate(beats_full):
                if bt < t0 or bt > t1:
                    continue
                x = int((bt - t0) / (t1 - t0) * w)

                # Decide downbeat and compute a global beat index to look up/derive bar number
                i_global = j - base_first_idx
                is_down = False
                if beats_per_bar and phase_i0 is not None:
                    is_down = ((i_global - phase_i0) % max(1, beats_per_bar) == 0)
                elif have_bars and down_idx:
                    is_down = (i_global in down_idx)
                else:
                    is_down = (j == 0)

                if is_down:
                    p.setPen(pen_long)
                    p.drawLine(x, 0, x, long_len)

                    # Determine bar number to draw, if any.
                    bar_no = None
                    # Prefer explicit mapping from detected downbeats (1-based numbers)
                    if i_global in bar_index_by_beat:
                        bar_no = int(bar_index_by_beat[i_global])
                    # Otherwise, if meter is known, infer the bar index relative to the first downbeat
                    elif beats_per_bar and phase_i0 is not None:
                        # Bar 1 occurs at phase_i0. Previous downbeat is bar 0, etc.
                        bar_no = 1 + (i_global - phase_i0) // max(1, beats_per_bar)
                    # If we still don't know, leave unlabeled.

                    # Draw label only for non-negative bars (0, 1, 2, ...); skip negatives
                    if bar_no is not None and bar_no >= 0:
                        p.setPen(text_pen)
                        text_y = int(min(long_len + 10, wf_h - 2))
                        p.drawText(x + 2, text_y, str(bar_no))
                else:
                    p.setPen(pen_short)
                    p.drawLine(x, 0, x, short_len)
        else:
            # Fallback: time ruler with long ticks at seconds + labels, short at half-seconds
            first_s = int(np.floor(rel_t0))
            last_s = int(np.ceil(rel_t1))
            p.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90)))
            for s in range(first_s, last_s + 1):
                x = int((s - rel_t0) / (rel_t1 - rel_t0) * w)
                p.drawLine(x, 0, x, long_len)
                if s >= 0:
                    p.setPen(QtGui.QPen(QtGui.QColor(180, 180, 180)))
                    text_y = int(min(long_len + 10, wf_h - 2))
                    p.drawText(x + 2, text_y, f"{s}s")
                    p.setPen(QtGui.QPen(QtGui.QColor(90, 90, 90)))
            p.setPen(QtGui.QPen(QtGui.QColor(70, 70, 70)))
            half_start = np.ceil(rel_t0 * 2.0) / 2.0
            hmark = half_start
            while hmark <= rel_t1 + 1e-9:
                if abs(hmark - round(hmark)) > 1e-6:
                    xh = int((hmark - rel_t0) / (rel_t1 - rel_t0) * w)
                    p.drawLine(xh, 0, xh, short_len)
                hmark += 0.5

        # loop overlay (if available)
        if self.loopA is not None and self.loopB is not None:
            a = max(t0, min(t1, float(self.loopA)))
            b = max(t0, min(t1, float(self.loopB)))
            if b < a:
                a, b = b, a
            x0 = int((a - t0) / (t1 - t0) * w)
            x1 = int((b - t0) / (t1 - t0) * w)
            # Overlay: greenish-blue fill, clipped to waveform area, more opacity
            p.fillRect(QtCore.QRect(x0, 0, max(1, x1 - x0), wf_h), QtGui.QColor(0, 200, 180, 48))
            p.setPen(QtGui.QPen(QtGui.QColor(210, 210, 210)))
            p.drawLine(x0, 0, x0, wf_h)
            p.drawLine(x1, 0, x1, wf_h)
            # flags at top strip for the active loop
            tip_y = 2
            base_y = min(self.FLAG_STRIP - 2, tip_y + self.FLAG_H)
            flag_poly_a = QtGui.QPolygon([
                QtCore.QPoint(x0, tip_y),
                QtCore.QPoint(x0 - self.FLAG_W//2, base_y),
                QtCore.QPoint(x0 + self.FLAG_W//2, base_y),
            ])
            flag_poly_b = QtGui.QPolygon([
                QtCore.QPoint(x1, tip_y),
                QtCore.QPoint(x1 - self.FLAG_W//2, base_y),
                QtCore.QPoint(x1 + self.FLAG_W//2, base_y),
            ])
            # Yellow flags for active loop
            p.setPen(QtGui.QPen(QtGui.QColor(255, 215, 0)))
            p.setBrush(QtGui.QColor(255, 215, 0))  # yellow
            p.drawPolygon(flag_poly_a)
            p.drawPolygon(flag_poly_b)

        # Draw saved loops' flags & labels in the top strip
        if hasattr(self, 'saved_loops') and self.saved_loops:
            tip_y = 2
            base_y = min(self.FLAG_STRIP - 2, tip_y + self.FLAG_H)
            # Yellow flags for saved loops
            p.setPen(QtGui.QPen(QtGui.QColor(255, 215, 0)))
            p.setBrush(QtGui.QColor(255, 215, 0))  # yellow
            for L in self.saved_loops:
                a = float(L['a']); b = float(L['b'])
                x0 = self._x_from_time(a, t0, t1, w)
                x1 = self._x_from_time(b, t0, t1, w)
                # start flag
                flag_poly = QtGui.QPolygon([
                    QtCore.QPoint(x0, tip_y),
                    QtCore.QPoint(x0 - self.FLAG_W//2, base_y),
                    QtCore.QPoint(x0 + self.FLAG_W//2, base_y),
                ])
                p.drawPolygon(flag_poly)
                # end flag
                flag_poly_b = QtGui.QPolygon([
                    QtCore.QPoint(x1, tip_y),
                    QtCore.QPoint(x1 - self.FLAG_W//2, base_y),
                    QtCore.QPoint(x1 + self.FLAG_W//2, base_y),
                ])
                p.drawPolygon(flag_poly_b)
                # label (if any): near start flag; if start flag is offscreen but the loop overlaps, pin to left margin
                label = str(L.get('label', '') or '')
                if label:
                    x_label = x0 + self.FLAG_W + 4
                    # If start flag is left of view but the loop overlaps the window, pin to left margin
                    if x0 < 0 and x1 > 0:
                        x_label = 4
                    # Only draw label if any part of the loop is visible
                    if x0 <= w and x1 >= 0:
                        p.setPen(QtGui.QPen(QtGui.QColor(210, 220, 230)))
                        p.drawText(int(max(0, min(w - 40, x_label))), int(base_y - 2), label)
                        p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))

        # Draw waveform min/max and playhead below the top strip
        mins, maxs = self._mono_slice_minmax(t0, t1, w)
        if mins is not None:
            body_top = min(wf_h, self.FLAG_STRIP)
            body_h = max(1, wf_h - body_top)
            mid = body_top + body_h // 2
            p.setPen(QtGui.QPen(QtGui.QColor(0, 180, 255)))
            for x, (mn, mx) in enumerate(zip(mins, maxs)):
                y1 = mid - int(mx * (body_h * 0.45))
                y2 = mid - int(mn * (body_h * 0.45))
                if y1 > y2:
                    y1, y2 = y2, y1
                p.drawLine(x, y1, x, y2)

        # playhead: center when following, absolute position when frozen
        p.setPen(QtGui.QPen(QtGui.QColor(255, 200, 0)))
        if self._freeze_window and self.player is not None:
            try:
                tpos = float(self.player.position_seconds())
            except Exception:
                tpos = (t0 + t1) * 0.5
            px = int(max(0, min(w - 1, (tpos - t0) / (t1 - t0) * w)))
        else:
            px = w // 2
        p.drawLine(px, min(wf_h, self.FLAG_STRIP), px, wf_h)
        # End waveform layer; ensure no brush/pen bleed into chord lane
        p.restore()
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))

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
            if self.isInterruptionRequested():
                return
            segs = estimate_chords(self.path)
        except Exception:
            segs = []
        if not self.isInterruptionRequested():
            self.done.emit(segs)

class KeyWorker(QThread):
    done = Signal(dict)
    def __init__(self, path: str):
        super().__init__()
        self.path = path
    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            info = estimate_key(self.path)
        except Exception:
            info = {"pretty": "unknown"}
        if not self.isInterruptionRequested():
            self.done.emit(info)


# --- BeatWorker for beat/bar estimation ---
class BeatWorker(QThread):
    done = Signal(dict)
    def __init__(self, path: str):
        super().__init__()
        self.path = path
    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            info = estimate_beats(self.path)
        except Exception:
            info = {"tempo": 0.0, "beats": [], "bars": []}
        if not self.isInterruptionRequested():
            self.done.emit(info)


class Main(QtWidgets.QMainWindow):
    @staticmethod
    def _feq(a: float | None, b: float | None, eps: float = 1e-3) -> bool:
        if a is None or b is None:
            return False
        return abs(float(a) - float(b)) <= eps

    def populate_beats_async(self, path: str):
        self.wave.set_beats([], [])
        # Stop previous beat worker if still running
        if self.beat_worker and self.beat_worker.isRunning():
            self.beat_worker.requestInterruption()
            self.beat_worker.quit()
            self.beat_worker.wait(1000)
        bw = BeatWorker(path)
        bw.setParent(self)
        bw.done.connect(self._beats_ready)
        bw.finished.connect(lambda: setattr(self, "beat_worker", None))
        self.beat_worker = bw
        bw.start()

    def _beats_ready(self, info: dict):
        beats = info.get("beats", [])
        bars = info.get("downbeats") or info.get("bars", [])
        self.wave.set_beats(beats, bars)
        if info.get("tempo"):
            self.statusBar().showMessage(f"Tempo: {info['tempo']:.1f} BPM · Beats: {len(beats)}")
        # persist
        try:
            self.last_tempo = float(info.get("tempo") or 0.0) or None
        except Exception:
            self.last_tempo = None
        self.last_beats = list(beats)
        self.last_bars = list(bars)
        self.save_session()

    def _cleanup_on_quit(self):
        # Stop audio stream if present
        try:
            if self.player:
                if hasattr(self.player, "stop"):
                    self.player.stop()
                if hasattr(self.player, "close"):
                    self.player.close()
        except Exception:
            pass
        # Stop workers
        for w in (self.beat_worker, self.chord_worker, self.key_worker):
            try:
                if w and w.isRunning():
                    w.requestInterruption()
                    w.quit()
                    w.wait(1500)
            except Exception:
                pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenPractice — Minimal")
        self.resize(960, 600)
        self.setAcceptDrops(True)
        self.settings = QSettings("OpenPractice", "OpenPractice")

        self.player: LoopPlayer | None = None
        self.current_path: str | None = None
        # Saved analysis/session state
        self.last_tempo: float | None = None
        self.last_beats: list[float] = []
        self.last_bars: list[float] = []
        self.last_key: dict | None = None
        self.last_chords: list[dict] = []
        self.A = 0.0
        self.B = 10.0
        # Saved loops model
        self.saved_loops: list[dict] = []   # {id:int, a:float, b:float, label:str}
        self._loop_id_seq = 1
        self._active_saved_loop_id: int | None = None
        self.beat_worker = None
        self.chord_worker = None
        self.key_worker = None

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
        self.wave.requestAddLoop.connect(self._add_loop_from_wave)
        self.wave.requestUpdateLoop.connect(self._update_loop_from_wave)
        self.wave.requestSelectLoop.connect(self._select_loop_from_wave)
        self.wave.requestDeleteSelected.connect(self._delete_selected_loop)
        self.wave.requestRenameLoop.connect(self._rename_loop_id)
        self.wave.requestDeleteLoopId.connect(self._delete_loop_id)

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
        save_act = QAction("Save Session", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self.save_session)

        load_act = QAction("Load Session", self)
        load_act.setShortcut(QKeySequence("Ctrl+Shift+O"))
        load_act.triggered.connect(self.load_session)

        file_menu.addSeparator()
        file_menu.addAction(save_act)
        file_menu.addAction(load_act)

        # Signals
        self.rate_spin.valueChanged.connect(self._rate_changed)
        self.snap_checkbox.toggled.connect(lambda on: self.wave.set_snap_enabled(bool(on)))

        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self)
        self.space_shortcut.activated.connect(self.toggle_play)
        QtWidgets.QApplication.instance().aboutToQuit.connect(self._cleanup_on_quit)

        # Status
        self.statusBar().showMessage("Ready")
        self.loop_poll = QtCore.QTimer(self)
        self.loop_poll.setInterval(80)  # ~12.5 Hz is plenty
        self.loop_poll.timeout.connect(self._auto_loop_tick)
        self.loop_poll.start()

    def _session_sidecar_path(self, audio_path: str) -> Path:
        p = Path(audio_path)
        folder = p.parent / ".openpractice"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{p.stem}.openpractice.json"

    def load_session(self):
        # Use current file’s sidecar if a song is loaded; else prompt for JSON
        if self.current_path:
            sidecar = self._session_sidecar_path(self.current_path)
        else:
            dlg = QtWidgets.QFileDialog(self, "Open Session JSON")
            dlg.setNameFilter("OpenPractice Session (*.openpractice.json);;JSON (*.json)")
            if dlg.exec() != QtWidgets.QDialog.Accepted:
                return
            sidecar = Path(dlg.selectedFiles()[0])

        if not Path(sidecar).exists():
            self.statusBar().showMessage("No session file found for this audio")
            return

        try:
            data = json.loads(Path(sidecar).read_text())
        except Exception as e:
            self.statusBar().showMessage(f"Load failed: {e}")
            return

        # View / snap
        self.wave.set_origin_offset(float(data.get("origin", 0.0)))
        ce = float(data.get("content_end", 0.0))
        if ce > 0:
            self.wave.set_music_span(float(data.get("origin", 0.0)), ce)
        self.wave.set_snap_enabled(bool(data.get("snap_enabled", True)))

        # Loops
        self.saved_loops = list(data.get("saved_loops", []))
        self._active_saved_loop_id = data.get("active_loop_id")
        self._sync_saved_loops_to_view()
        # Activate the stored active loop; if none, default to first saved loop
        L = None
        if self._active_saved_loop_id is not None:
            L = next((x for x in self.saved_loops if x.get('id') == self._active_saved_loop_id), None)
        if L is None and self.saved_loops:
            L = self.saved_loops[0]
            self._active_saved_loop_id = int(L.get('id')) if 'id' in L else None
        if L is not None:
            a = float(min(L['a'], L['b']))
            b = float(max(L['a'], L['b']))
            self.wave.set_loop_visual(a, b)
            if self.player:
                self.player.set_loop_seconds(a, b)
        else:
            # No loops saved → clear any previous visual
            self.wave.clear_loop_visual()

        # Beats / bars / tempo
        self.last_tempo = float(data.get("tempo", 0.0)) or None
        self.last_beats = list(data.get("beats", []))
        self.last_bars = list(data.get("bars", []))
        self.wave.set_beats(self.last_beats, self.last_bars)

        # Key
        self.last_key = data.get("key") or {}
        if self.last_key:
            self.key_header_label.setText(f"Key: {self.last_key.get('pretty','—')}")

        # Chords
        self.last_chords = list(data.get("chords", []))
        if self.last_chords:
            self.wave.set_chords(self.last_chords)

        # Rate
        if "rate" in data:
            try:
                self.rate_spin.setValue(float(data.get("rate", 1.0)))
            except Exception:
                pass

        self.statusBar().showMessage(f"Session loaded from {Path(sidecar).name}")

    def save_session(self):
        if not self.current_path:
            self.statusBar().showMessage("No audio loaded – nothing to save")
            return
        sidecar = self._session_sidecar_path(self.current_path)
        payload = {
            "audio_path": str(self.current_path),
            "origin": float(self.wave.origin or 0.0),
            "content_end": float(self.wave.content_end or 0.0),
            "snap_enabled": bool(self.wave.snap_enabled),
            "saved_loops": self.saved_loops,
            "active_loop_id": self._active_saved_loop_id,
            "tempo": float(self.last_tempo or 0.0),
            "beats": list(self.last_beats or []),
            "bars": list(self.last_bars or []),
            "key": self.last_key or {},
            "chords": list(self.last_chords or []),
            "rate": float(self.rate_spin.value()),
        }
        try:
            Path(sidecar).write_text(json.dumps(payload, indent=2))
            self.statusBar().showMessage(f"Session saved → {sidecar.name}")
        except Exception as e:
            self.statusBar().showMessage(f"Save failed: {e}")

    def _sync_saved_loops_to_view(self):
        if hasattr(self, 'wave'):
            self.wave.set_saved_loops(self.saved_loops)
            self.wave.update()

    def _next_loop_label(self) -> str:
        used = {L.get('label') for L in self.saved_loops if L.get('label')}
        # First try single letters A..Z
        for i in range(26):
            lab = chr(ord('A') + i)
            if lab not in used:
                return lab
        # Then A1..Z1, A2..Z2, etc.
        suffix = 1
        while True:
            for i in range(26):
                lab = f"{chr(ord('A') + i)}{suffix}"
                if lab not in used:
                    return lab
            suffix += 1

    def _add_loop_from_wave(self, a: float, b: float):
        L = {"id": self._loop_id_seq, "a": float(a), "b": float(b), "label": self._next_loop_label()}
        self._loop_id_seq += 1
        self.saved_loops.append(L)
        self.saved_loops.sort(key=lambda x: min(x['a'], x['b']))
        self._active_saved_loop_id = L['id']
        # Set playback to this loop immediately
        if self.player:
            self.player.set_loop_seconds(min(a,b), max(a,b))
        self.wave.set_loop_visual(min(a,b), max(a,b))
        self._sync_saved_loops_to_view()
        self.statusBar().showMessage(f"Added loop {L['label']}: {min(a,b):.2f}s → {max(a,b):.2f}s")
        self.save_session()

    def _update_loop_from_wave(self, lid: int, a: float, b: float):
        for L in self.saved_loops:
            if L['id'] == lid:
                L['a'], L['b'] = float(min(a,b)), float(max(a,b))
                break
        self.saved_loops.sort(key=lambda x: min(x['a'], x['b']))
        self._active_saved_loop_id = lid
        if self.player:
            self.player.set_loop_seconds(min(a,b), max(a,b))
        self.wave.set_loop_visual(min(a,b), max(a,b))
        if self._active_saved_loop_id is None or int(self._active_saved_loop_id) == int(lid):
            self._active_saved_loop_id = int(lid)
            self.wave.set_loop_visual(float(min(a,b)), float(max(a,b)))
            if self.player:
                self.player.set_loop_seconds(float(min(a,b)), float(max(a,b)))
        self._sync_saved_loops_to_view()
        self.save_session()

    def _select_loop_from_wave(self, lid: int):
        L = next((x for x in self.saved_loops if int(x.get('id')) == int(lid)), None)
        if not L:
            return
        a = float(min(L['a'], L['b']))
        b = float(max(L['a'], L['b']))
        self._active_saved_loop_id = int(lid)
        self.wave.selected_loop_id = int(lid)
        self.wave.set_loop_visual(a, b)
        if self.player:
            try:
                self.player.set_loop_seconds(a, b)
            except Exception:
                pass
        self._sync_saved_loops_to_view()

    def _delete_selected_loop(self):
        lid = getattr(self.wave, 'selected_loop_id', None)
        if lid is None:
            lid = self._active_saved_loop_id
        if lid is None:
            self.statusBar().showMessage("No loop selected to delete")
            return
        self._delete_loop_id(int(lid))

    def _auto_loop_tick(self):
        if not self.player or not self.saved_loops:
            return
        try:
            t = float(self.player.position_seconds())
        except Exception:
            return
        # Find loop that contains t (inclusive of start, exclusive of end)
        cur = next((L for L in self.saved_loops if min(L['a'], L['b']) <= t < max(L['a'], L['b'])), None)
        if cur:
            lid = int(cur['id'])
            if self._active_saved_loop_id != lid:
                # Entered a loop → set player loop to it
                a = float(min(cur['a'], cur['b'])); b = float(max(cur['a'], cur['b']))
                self._active_saved_loop_id = lid
                try:
                    self.player.set_loop_seconds(a, b)
                except Exception:
                    pass
                self.wave.set_loop_visual(a, b)
        else:
            # Outside all loops → disable looping so playback runs to the natural end
            if self._active_saved_loop_id is not None:
                self._active_saved_loop_id = None
                try:
                    # Preferred: a clear method if LoopPlayer provides it
                    self.player.clear_loop()
                except Exception:
                    # Fallback: set a degenerate loop (interpreted as no-loop by most players)
                    try:
                        self.player.set_loop_seconds(0.0, 0.0)
                    except Exception:
                        pass

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

    def _rename_loop_id(self, lid: int):
        L = next((x for x in self.saved_loops if x['id'] == lid), None)
        if not L:
            return
        text, ok = QtWidgets.QInputDialog.getText(self, "Rename Loop", "Name:", text=L.get('label',''))
        if ok:
            L['label'] = str(text)
            self._sync_saved_loops_to_view()
            self.statusBar().showMessage(f"Renamed loop to '{L['label']}'")
            self.save_session()

    def _delete_loop_id(self, lid: int):
        # Find the loop being deleted so we know its span before removal
        Ldel = next((x for x in self.saved_loops if int(x.get('id')) == int(lid)), None)
        before = len(self.saved_loops)

        # Remove from model
        self.saved_loops = [L for L in self.saved_loops if int(L.get('id')) != int(lid)]

        # If we removed the active loop, clear overlay & player loop
        cleared_overlay = False
        if self._active_saved_loop_id is not None and int(self._active_saved_loop_id) == int(lid):
            self._active_saved_loop_id = None
            if hasattr(self.wave, 'clear_loop_visual'):
                self.wave.clear_loop_visual()
                cleared_overlay = True
            if self.player:
                try:
                    dur = self.player.duration_seconds() if hasattr(self.player, 'duration_seconds') else (self.player.n / float(self.player.sr))
                    self.player.set_loop_seconds(0.0, float(dur))
                except Exception:
                    pass

        # If it WASN'T the active loop, but the overlay matches the deleted span, clear it too
        if not cleared_overlay and Ldel is not None and self.wave is not None:
            a = float(min(Ldel['a'], Ldel['b']))
            b = float(max(Ldel['a'], Ldel['b']))
            if self._feq(self.wave.loopA, a) and self._feq(self.wave.loopB, b):
                if hasattr(self.wave, 'clear_loop_visual'):
                    self.wave.clear_loop_visual()
                if self.player:
                    try:
                        dur = self.player.duration_seconds() if hasattr(self.player, 'duration_seconds') else (self.player.n / float(self.player.sr))
                        self.player.set_loop_seconds(0.0, float(dur))
                    except Exception:
                        pass

        # Clear view selection if it pointed to the deleted loop
        if hasattr(self.wave, 'selected_loop_id') and self.wave.selected_loop_id is not None:
            try:
                if int(self.wave.selected_loop_id) == int(lid):
                    self.wave.selected_loop_id = None
            except Exception:
                self.wave.selected_loop_id = None

        # Push updates to the view and repaint
        self.wave.set_saved_loops(self.saved_loops)
        self.wave.update()

        self.statusBar().showMessage("Loop deleted" if len(self.saved_loops) < before else "No loop deleted")
        self.save_session()

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

        # Align visual time 0 to first non-silent audio and position the transport
        lead = self._detect_leading_silence_seconds(self.player.y, self.player.sr)
        tail = self._detect_trailing_silence_seconds(self.player.y, self.player.sr)
        self.wave.set_music_span(lead, tail)
        try:
            self.player.set_position_seconds(lead, within_loop=False)
        except Exception:
            pass

        # Try to restore prior session (loops, beats, chords, key, snap, rate)
        try:
            self.load_session()
        except Exception:
            pass

        # Compute only what is missing
        if not self.last_beats:
            self.populate_beats_async(self.current_path)
        if not self.last_chords:
            self.populate_chords_async(self.current_path)
        if not (self.last_key and self.last_key.get("pretty")):
            self.populate_key_async(self.current_path)

        # Status
        total_s = self.player.n / self.player.sr
        mus_len = max(0.0, (tail - lead))
        self.statusBar().showMessage(f"Loaded: {Path(fn).name} [music {mus_len:.1f}s of {total_s:.1f}s]")
        # Focus waveform for immediate key control
        self.wave.setFocus()

    def populate_chords_async(self, path: str):
        if self.chord_worker and self.chord_worker.isRunning():
            self.chord_worker.requestInterruption(); self.chord_worker.quit(); self.chord_worker.wait(1000)
        cw = ChordWorker(path)
        cw.setParent(self)
        cw.done.connect(self._chords_ready)
        cw.finished.connect(lambda: setattr(self, "chord_worker", None))
        self.chord_worker = cw
        cw.start()

    def populate_key_async(self, path: str):
        if self.key_worker and self.key_worker.isRunning():
            self.key_worker.requestInterruption(); self.key_worker.quit(); self.key_worker.wait(1000)
        kw = KeyWorker(path)
        kw.setParent(self)
        kw.done.connect(self._key_ready)
        kw.finished.connect(lambda: setattr(self, "key_worker", None))
        self.key_worker = kw
        kw.start()

    def _chords_ready(self, segs: list):
        self.last_chords = list(segs or [])
        self.wave.set_chords(self.last_chords)
        self.statusBar().showMessage(f"Chords: {len(self.last_chords)} segments")
        self.save_session()

    def _key_ready(self, info: dict):
        self.last_key = info or {}
        self.key_header_label.setText(f"Key: {self.last_key.get('pretty','—')}")
        self.save_session()

    def toggle_play(self):
        if not self.player:
            return
        # crude toggle based on whether stream is active
        try:
            self.play() if not self.player._playing else self.pause()
        except Exception:
            pass

    def play(self):
        if not self.player:
            return
        # When starting playback, re-center on the playhead
        try:
            self.wave.unfreeze_and_center()
        except Exception:
            pass
        try:
            self.player.play()
        except Exception:
            pass

    def pause(self):
        if self.player:
            self.player.pause()

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
                # Reset saved loops for this file; we will add new ones as the user creates them
                self.saved_loops.clear()
                self._loop_id_seq = 1
                self._active_saved_loop_id = None
                self._sync_saved_loops_to_view()
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

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self._cleanup_on_quit()
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())