import sys
import json
import shutil
from pathlib import Path
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut
import numpy as np
import traceback
import logging
from audio_engine import LoopPlayer
from chords import estimate_chords as _estimate_chords_base, estimate_key, estimate_beats
try:
    from timestretch import render_time_stretch
    HAS_STRETCH = True
except Exception:
    HAS_STRETCH = False
from utils import temp_wav_path
from stems import separate_stems, load_stem_arrays, order_stem_names

import os
from pathlib import Path
import tempfile

class WaveformView(QtWidgets.QWidget):
    def _coalesce_adjacent_same_label(self, segments: list[dict], eps: float = 1e-6) -> list[dict]:
        """Merge adjacent segments with the same label when end≈start.
        Keeps time order; does not cross bar boundaries (assumes caller split by bars first).
        """
        if not segments:
            return []
        segs = sorted(({'start': float(s['start']), 'end': float(s['end']), 'label': s.get('label')}
                       for s in segments if 'start' in s and 'end' in s),
                      key=lambda d: (d['start'], d['end']))
        out: list[dict] = []
        for s in segs:
            if not out:
                out.append(dict(s));
                continue
            last = out[-1]
            if s.get('label') == last.get('label') and abs(float(last['end']) - float(s['start'])) <= eps:
                # extend
                last['end'] = max(float(last['end']), float(s['end']))
            else:
                out.append(dict(s))
        return out
    def _snap_segments_to_beats_within_bars(self, segments: list[dict], beats: list[float], bars: list[float], eps: float = 1e-3) -> list[dict]:
        """Snap each segment boundary to the beat grid *inside its bar* without creating new splits.
        Assumes segments do not cross bars (call _split_segments_at_bars first).
        - Start snaps to the nearest beat *at/after* the current start (ceil to beat within the bar)
        - End snaps to the nearest beat *at/before* the current end (floor to beat within the bar)
        - If snapping would invert/collapse the segment, keep the original boundary (or nudge minimally)
        """
        if not segments:
            return []
        if not beats:
            return list(segments)
        # Precompute sorted beats & bars
        try:
            beat_arr = np.asarray(list(beats), dtype=float)
            beat_arr.sort()
        except Exception:
            return list(segments)
        bars_sorted = sorted(float(b) for b in (bars or []))

        def enclosing_bar(s: float, e: float) -> tuple[float, float] | None:
            if not bars_sorted:
                return (float('-inf'), float('inf'))
            prev = float(self.origin or 0.0)
            for b in bars_sorted:
                if s < b:
                    return (prev, float(b))
                prev = float(b)
            return (prev, float('inf'))

        out: list[dict] = []
        for seg in segments:
            try:
                s = float(seg.get('start')); e = float(seg.get('end'))
                lab = seg.get('label')
            except Exception:
                continue
            if e <= s + 1e-12:
                continue
            bs, be = enclosing_bar(s, e)
            # Candidate beats inside this bar span (inclusive of edges)
            inside = beat_arr[(beat_arr >= bs - 1e-9) & (beat_arr <= be + 1e-9)]
            if inside.size == 0:
                out.append({'start': s, 'end': e, 'label': lab})
                continue

            # --- Snap start: ceil to nearest beat at/after s (within bar) ---
            # searchsorted returns insertion index to keep order
            i_s = int(np.searchsorted(inside, s, side='left'))
            if i_s >= inside.size:
                s_snap = float(inside[-1])  # clamp to last beat in bar
            else:
                s_snap = float(inside[i_s])

            # --- Snap end: floor to nearest beat at/before e (within bar) ---
            i_e = int(np.searchsorted(inside, e, side='right')) - 1
            if i_e < 0:
                e_snap = float(inside[0])  # clamp to first beat in bar
            else:
                e_snap = float(inside[i_e])

            # Keep within bar & maintain order; if snapping collapses, try to minimally widen
            s_snap = max(bs, min(be, s_snap))
            e_snap = max(bs, min(be, e_snap))
            if e_snap <= s_snap + eps:
                # Try nudging end to the next beat strictly after s_snap
                i_after = int(np.searchsorted(inside, s_snap + eps, side='left'))
                if i_after < inside.size:
                    e_snap = float(inside[i_after])
                else:
                    # Fall back to original edges if still collapsed
                    s_snap, e_snap = s, e

            out.append({'start': s_snap, 'end': e_snap, 'label': lab})

        # Keep stable chronological order
        out.sort(key=lambda d: (d['start'], d['end']))
        return out
    def _bar_index_at(self, t: float) -> int | None:
        if not self.bars:
            return None
        try:
            bars = sorted(float(b) for b in self.bars)
        except Exception:
            return None
        if not bars:
            return None
        for i, b in enumerate(bars):
            if t < b:
                return max(0, i - 1)
        return len(bars) - 1
    """Scrolling waveform that moves right→left with a chord lane underneath."""

    requestSetLoop = QtCore.Signal(float, float)  # (A, B) in seconds (absolute timeline)
    requestSeek = QtCore.Signal(float)  # absolute seconds to seek playhead
    requestAddLoop = QtCore.Signal(float, float)          # create a new saved loop [a,b]
    requestUpdateLoop = QtCore.Signal(int, float, float)  # update saved loop id → [a,b]
    requestSelectLoop = QtCore.Signal(int)                # select a saved loop by id
    requestDeleteSelected = QtCore.Signal()               # delete currently selected saved loop
    requestRenameLoop = QtCore.Signal(int)     # loop id to rename
    requestDeleteLoopId = QtCore.Signal(int)   # loop id to delete
    requestEditChord = QtCore.Signal(float)         # time (sec) under cursor to edit chord label
    requestSplitChordAt = QtCore.Signal(float)      # time (sec) to split chord at that point
    requestJoinChordForward = QtCore.Signal(float)  # time (sec) to join this chord with the next (same bar)

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
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
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
        # Context menu suppression flag for right-click handling (unused now; we rely on contextMenuEvent only)
        self._suppress_next_ctx = False  # (unused now; we rely on contextMenuEvent only)

    def set_beats(self, beats: list | None, bars: list | None):
        self.beats = beats or []
        self.bars = bars or []
        self.update()

    def set_player(self, player: 'LoopPlayer'):
        self.player = player
        self.update()

    def set_chords(self, segments: list[dict]):
        """Replace chord segments and trigger a repaint."""
        import copy
        self.chords = copy.deepcopy(list(segments or []))
        # If you have any chord drawing cache, clear it
        if hasattr(self, "_chords_viz_cache"):
            try:
                self._chords_viz_cache.clear()
            except Exception:
                self._chords_viz_cache = None
        print(f"[DBG] set_chords: {len(self.chords)} segments; first={self.chords[0] if self.chords else None}")
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

    def _wf_geom(self):
        """Return (wf_h, ch_h) matching paintEvent's layout."""
        h = self.height()
        CHORD_LANE_H = 32
        ch_h = CHORD_LANE_H
        wf_h = max(10, h - ch_h)
        return wf_h, ch_h

    def _time_in_chord_lane(self, pos: QtCore.QPoint, t0: float, t1: float, w: int, wf_h: int):
        """Return absolute time if pos is in the chord lane; else None."""
        x = int(pos.x()); y = int(pos.y())
        lane_top = wf_h
        lane_bottom = wf_h + 32
        in_lane = not (y < (lane_top - 4) or y >= (lane_bottom + 4))
        print(f"[DBG] _time_in_chord_lane: pos=({x},{y}) lane=({lane_top},{lane_bottom}) in_lane={in_lane}")
        if not in_lane:
            return None
        t = self._time_at_x(x, t0, t1, w)
        print(f"[DBG] _time_in_chord_lane: time={t:.3f}s")
        return t

    def _open_chord_context_at(self, pos: QtCore.QPoint, global_pos: QtCore.QPoint):
        """Try to open the chord context menu for a widget-relative pos; return True if handled."""
        t0, t1 = self._current_window()
        w = self.width(); h = self.height(); wf_h, _ = self._wf_geom()
        t_ch = self._time_in_chord_lane(pos, t0, t1, w, wf_h)
        print(f"[DBG] _open_chord_context_at: pos={pos}, t_ch={t_ch}")
        if t_ch is None:
            print("[DBG] _open_chord_context_at: not in chord lane → False")
            return False

        # Determine if join-with-next is allowed: must be contiguous and within the same bar
        allow_join = False
        try:
            segs = sorted((dict(s) for s in (self.chords or [])), key=lambda s: float(s.get('start', 0.0)))
            idx = None
            for i, s in enumerate(segs):
                try:
                    a = float(s.get('start')); b = float(s.get('end'))
                except Exception:
                    continue
                if a <= float(t_ch) < b or (i == len(segs) - 1 and abs(float(t_ch) - b) < 1e-6):
                    idx = i; break
            if idx is not None and (idx + 1) < len(segs):
                cur = segs[idx]; nxt = segs[idx + 1]
                a1 = float(cur['start']); b1 = float(cur['end'])
                a2 = float(nxt['start']); b2 = float(nxt['end'])
                contiguous = abs(b1 - a2) <= 1e-3
                # compute bar indices using midpoints
                m1 = 0.5 * (a1 + b1); m2 = 0.5 * (a2 + b2)
                bi1 = self._bar_index_at(m1); bi2 = self._bar_index_at(m2)
                same_bar = (bi1 is not None and bi1 == bi2)
                allow_join = bool(contiguous and same_bar)
                print(f"[DBG] chord-join: idx={idx} contiguous={contiguous} same_bar={same_bar} (bi1={bi1}, bi2={bi2})")
        except Exception as ex:
            print(f"[DBG] chord-join check failed: {ex}")
            allow_join = False

        menu = QtWidgets.QMenu(self)
        actEdit = menu.addAction("Change chord…")
        actSplit = menu.addAction("Split chord here")
        actJoin = None
        if allow_join:
            actJoin = menu.addAction("Join with next (same bar)")
        chosen = menu.exec(global_pos)
        if chosen == actEdit:
            self.requestEditChord.emit(float(t_ch))
        elif chosen == actSplit:
            self.requestSplitChordAt.emit(float(t_ch))
        elif actJoin is not None and chosen == actJoin:
            self.requestJoinChordForward.emit(float(t_ch))
        print("[DBG] _open_chord_context_at: menu handled = True")
        return True

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
        # Chord box context menu when right-clicking in the chord lane
        pos_local = e.pos()
        try:
            pos_global = e.globalPos()
        except Exception:
            pos_global = QtGui.QCursor.pos()
        print(f"[DBG] contextMenuEvent: local={pos_local}, global={pos_global}")
        if self._open_chord_context_at(pos_local, pos_global):
            e.accept()
            return
        # Right-click → offer a small menu if we're over a loop flag/edge
        if not hasattr(self, 'saved_loops') or not self.saved_loops:
            e.ignore()
            return
        t0, t1 = self._current_window()
        w = self.width(); h = self.height(); wf_h, _ = self._wf_geom()
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
        # Right-click: let the standard contextMenuEvent handle it (no action on press)
        if e.button() == Qt.RightButton:
            print("[DBG] mousePressEvent RightButton: delegating to contextMenuEvent")
            e.ignore()
            return
        # For left/middle clicks, we require a player to interact
        if not self.player:
            e.ignore()
            return
        # Only handle left-button presses here
        if e.button() != Qt.LeftButton:
            e.ignore()
            return
        t0, t1 = self._current_window()
        w = self.width(); h = self.height(); wf_h, _ = self._wf_geom()
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
        # Only handle left-button releases for seeking/loop commits
        if e.button() != Qt.LeftButton:
            e.ignore()
            return
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
            # Ensure the view is not frozen so it will re-center on the new playhead
            if self._freeze_window:
                self.unfreeze_and_center()
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

    def _snap_segments_to_bars(self, segments: list[dict], bars: list[float]) -> list[dict]:
        """Snap chord segment boundaries to bar boundaries.
        Start snaps to the nearest bar at or before the start; end snaps to the nearest bar at or after the end.
        Bar list is in absolute seconds.
        """
        if not segments:
            return []
        # Always treat the visual origin as a bar boundary (bar 0)
        bars_aug = list(bars or [])
        try:
            bars_aug.append(float(self.origin or 0.0))
        except Exception:
            bars_aug.append(0.0)
        bars_sorted = sorted(float(b) for b in bars_aug)
        if not bars_sorted:
            return list(segments)
        out: list[dict] = []
        for seg in segments:
            try:
                s = float(seg.get('start'))
                e = float(seg.get('end'))
                lab = seg.get('label')
            except Exception:
                continue
            if e <= s:
                continue
            # find bar at/before start
            bs = max((b for b in bars_sorted if b <= s), default=s)
            # find bar at/after end
            be = min((b for b in bars_sorted if b >= e), default=e)
            # ensure strictly increasing
            if be <= bs:
                # if collapsed due to missing bars, keep original
                bs, be = s, e
            out.append({'start': bs, 'end': be, 'label': lab})
        return out

    def _split_segments_at_bars(self, segments: list[dict], bars: list[float]) -> list[dict]:
        """Return segments split so none spans across a bar boundary.
        Each output segment keeps the original 'label'.
        """
        if not segments:
            return []
        bars_aug = set(float(b) for b in (bars or []))
        try:
            bars_aug.add(float(self.origin or 0.0))
        except Exception:
            bars_aug.add(0.0)
        bars_sorted = sorted(bars_aug)
        if not bars_sorted:
            return list(segments)
        out: list[dict] = []
        for seg in segments:
            try:
                start = float(seg.get('start'))
                end = float(seg.get('end'))
                label = seg.get('label')
            except Exception:
                continue
            if end <= start:
                continue
            t0 = start
            # collect internal bar cuts strictly inside (start, end)
            cuts = [b for b in bars_sorted if start < b < end]
            prev = t0
            for cut in cuts:
                out.append({'start': prev, 'end': cut, 'label': label})
                prev = cut
            out.append({'start': prev, 'end': end, 'label': label})
        return out

    def _ensure_leading_bar(self, segments: list[dict], bars: list[float]) -> list[dict]:
        """If the first visible bar (origin→first bar) has no chord segment,
        create one using the first chord's label. Does not modify input list in-place."""
        segs = list(segments or [])
        if not segs:
            return segs
        try:
            origin = float(self.origin or 0.0)
        except Exception:
            origin = 0.0
        # Determine the first bar boundary at or after origin
        first_bar = None
        if bars:
            try:
                first_bar = min(b for b in bars if float(b) >= origin)
            except ValueError:
                first_bar = None
        segs_sorted = sorted(segs, key=lambda s: float(s.get('start', 0.0)))
        first_seg = segs_sorted[0]
        try:
            s0 = float(first_seg.get('start'))
            lab0 = first_seg.get('label')
        except Exception:
            return segs
        # If the first segment already starts at or before origin (within epsilon), nothing to do
        if s0 <= origin + 1e-6:
            return segs
        # Insert a leading segment covering up to the first bar (if present) or to s0
        stop = min(first_bar, s0) if first_bar is not None else s0
        lead = {'start': origin, 'end': float(stop), 'label': lab0}
        return [lead] + segs_sorted

    def _unique_by_span(self, segments: list[dict], eps: float = 1e-6) -> list[dict]:
        """Return segments with unique (start,end) spans. Later items win."""
        uniq: dict[tuple[float, float], dict] = {}
        def _q(x: float) -> float:
            return round(float(x) / eps) * eps
        for s in segments or []:
            try:
                a = _q(s.get('start'))
                b = _q(s.get('end'))
            except Exception:
                continue
            if b <= a:
                continue
            # later items overwrite earlier ones for the same span
            uniq[(a, b)] = {'start': a, 'end': b, 'label': s.get('label')}
        # stable order by time
        return sorted(uniq.values(), key=lambda x: (x['start'], x['end']))

    def paintEvent(self, e: QtGui.QPaintEvent):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        w = self.width()
        h = self.height()
        if w < 10 or h < 10:
            return
        wf_h, ch_h = self._wf_geom()
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

        # Draw dashed bar grid lines across the waveform area
        if have_bars and self.bars:
            pen_bar = QtGui.QPen(QtGui.QColor(120, 130, 150, 110))  # light, subtle
            pen_bar.setWidth(1)
            pen_bar.setStyle(QtCore.Qt.DashLine)
            p.setPen(pen_bar)
            for bar_t in self.bars:
                bt = float(bar_t)
                if bt < t0 or bt > t1:
                    continue
                x = int((bt - t0) / (t1 - t0) * w)
                p.drawLine(x, 0, x, wf_h)  # full height of waveform area
            # Also draw a dashed line at the origin as bar 0
            try:
                x0 = int(((float(self.origin or 0.0)) - t0) / (t1 - t0) * w)
                if 0 <= x0 <= w - 1:
                    p.drawLine(x0, 0, x0, wf_h)
            except Exception:
                pass

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
            # Revert: only split at bar boundaries; no beat snapping in paint path.
            base = list(self.chords)
            split_bars = self._split_segments_at_bars(base, self.bars)
            segs_to_draw = self._ensure_leading_bar(split_bars, self.bars)
            segs_to_draw = self._unique_by_span(segs_to_draw)

            # DEBUG: confirm what we will draw
            try:
                first_lab = segs_to_draw[0].get('label') if segs_to_draw else None
                # Determine first *visible* segment in the current window
                segs_visible = []
                for s in segs_to_draw:
                    try:
                        sa = float(s.get('start')); sb = float(s.get('end'))
                    except Exception:
                        continue
                    if sb <= t0 or sa >= t1:
                        continue
                    segs_visible.append(s)
                vis_lab = segs_visible[0].get('label') if segs_visible else None
                vis_a = segs_visible[0].get('start') if segs_visible else None
                vis_b = segs_visible[0].get('end') if segs_visible else None
            except Exception:
                pass

            # Ensure segs_visible exists even if DEBUG block above failed
            if 'segs_visible' not in locals():
                segs_visible = []
                for s in segs_to_draw:
                    try:
                        sa = float(s.get('start')); sb = float(s.get('end'))
                    except Exception:
                        continue
                    if sb <= t0 or sa >= t1:
                        continue
                    segs_visible.append(s)
                segs_visible.sort(key=lambda s: float(s.get('start', 0.0)))

            font = p.font()
            font.setPointSizeF(max(9.0, self.font().pointSizeF()))
            p.setFont(font)
            for seg in segs_to_draw:
                try:
                    a = max(t0, float(seg['start']))
                    b = min(t1, float(seg['end']))
                except Exception:
                    continue
                if b <= a:
                    continue
                # relative (visual) times, though x mapping uses absolute t0/t1
                a_rel = a - self.origin
                b_rel = b - self.origin
                x0 = int((a_rel - rel_t0) / (rel_t1 - rel_t0) * w)
                x1 = int((b_rel - rel_t0) / (rel_t1 - rel_t0) * w)
                rect = QtCore.QRect(x0, wf_h, max(1, x1 - x0), ch_h)
                p.fillRect(rect, QtGui.QColor(50, 80, 110))
                # Brighter outline for the chord box, with increased width
                bright_pen = QtGui.QPen(QtGui.QColor(0, 0, 0))
                bright_pen.setWidth(2)
                p.setPen(bright_pen)
                p.drawRect(rect)
                # Chord label
                p.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230)))
                p.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, seg['label'])

            # Draw subtle beat boundaries inside each chord box (only if chord spans >1 beat)
            # Use beats with backfill so bar 0 (origin→first downbeat) gets interior ticks.
            try:
                beat_arr = None
                if self.beats:
                    bt = np.asarray(self.beats, dtype=float)
                    bt.sort()
                    if bt.size >= 2:
                        diffs = np.diff(bt)
                        good = diffs[(diffs > 0.1) & (diffs < 2.0)]
                        period = float(np.median(good)) if good.size else float(np.median(diffs))
                    else:
                        period = 0.5
                    # Backfill a few beats before the first to cover the first bar window
                    beats_full = []
                    if bt.size:
                        first_bt = float(bt[0])
                        k = 1
                        # back-fill slightly beyond window to ensure left-edge coverage
                        while period > 0 and (first_bt - k * period) > (t0 - 2 * period):
                            beats_full.append(first_bt - k * period)
                            k += 1
                        beats_full = list(reversed(beats_full)) + bt.tolist()
                    else:
                        beats_full = bt.tolist()
                    beat_arr = np.asarray(beats_full, dtype=float)
                else:
                    beat_arr = None
            except Exception:
                beat_arr = None

            try:
                if beat_arr is not None:
                    pen_beat = QtGui.QPen(QtGui.QColor(200, 210, 230, 110))
                    pen_beat.setWidth(1)
                    pen_beat.setStyle(QtCore.Qt.DotLine)
                    for seg in segs_to_draw:
                        try:
                            sa = float(seg['start']); sb = float(seg['end'])
                        except Exception:
                            continue
                        # Beats strictly inside the chord interval
                        if beat_arr is None:
                            continue
                        inside = beat_arr[(beat_arr > sa + 1e-9) & (beat_arr < sb - 1e-9)] if beat_arr.size else beat_arr
                        if inside is None or (hasattr(inside, 'size') and inside.size == 0):
                            continue
                        # Clip to the chord rect so beat lines don't bleed outside
                        a_clamped = max(t0, sa); b_clamped = min(t1, sb)
                        if b_clamped <= a_clamped:
                            continue
                        x0_clip = int((a_clamped - t0) / (t1 - t0) * w)
                        x1_clip = int((b_clamped - t0) / (t1 - t0) * w)
                        rect_clip = QtCore.QRect(x0_clip, wf_h, max(1, x1_clip - x0_clip), ch_h)
                        p.save()
                        p.setClipRect(rect_clip)
                        p.setPen(pen_beat)
                        for bt in inside:
                            if bt <= t0 or bt >= t1:
                                continue
                            x = int((float(bt) - t0) / (t1 - t0) * w)
                            p.drawLine(x, wf_h, x, wf_h + ch_h - 1)
                        p.restore()
            except Exception:
                pass

            # Draw rounded bar boundaries as translucent boxes across the chord lane
            try:
                if have_bars and self.bars:
                    bars_sorted = [float(b) for b in self.bars]
                    bars_sorted = sorted(bars_sorted)
                    # Build (start, end) pairs for bars that intersect the visible window
                    prev = float(self.origin or 0.0)
                    intervals = []
                    for b in bars_sorted:
                        intervals.append((prev, float(b)))
                        prev = float(b)
                    # Add a trailing interval to cover view past the last known bar (optional)
                    intervals.append((prev, float('inf')))

                    # Use the same brighter/thicker line as chord boxes so the bar outline
                    # acts as the visual separator between waveform and chord lane.
                    pen_round = QtGui.QPen(QtGui.QColor(0, 0, 0))
                    pen_round.setWidth(2)
                    p.setPen(pen_round)
                    p.setBrush(QtCore.Qt.NoBrush)
                    radius = 6
                    for (bs, be) in intervals:
                        # Clip to the visible window
                        a = max(t0, bs)
                        b = min(t1, be)
                        if not (b > a):
                            continue
                        x0 = int((a - t0) / (t1 - t0) * w)
                        x1 = int((b - t0) / (t1 - t0) * w)
                        rect_bar = QtCore.QRect(x0, wf_h, max(1, x1 - x0), ch_h)
                        # Rounded outline to hint the bar container
                        p.drawRoundedRect(rect_bar, radius, radius)
            except Exception:
                pass



class ChordWorker(QThread):
    done = Signal(list)
    status = Signal(str)
    demucs_line = Signal(str)  # forward Demucs logs to LogDock (like _toggle_stems)
    stems_ready = Signal(str)  # emits the stem **leaf** directory path
    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def _log(self, msg: str):
        s = str(msg)
        try:
            self.demucs_line.emit(s.rstrip("\n") + "\n")
        except Exception:
            pass
        try:
            self.status.emit(s)
        except Exception:
            pass

    def _log_exc(self, where: str, exc: Exception):
        tb = traceback.format_exc()
        try:
            self.demucs_line.emit(f"[ERROR] {where}: {exc}\n{tb}\n")
        except Exception:
            pass
        try:
            self.status.emit(f"Error in {where}: {exc}")
        except Exception:
            pass
        try:
            logging.exception("%s", where)
        except Exception:
            pass

    def _song_cache_dir(self, audio_path: str) -> Path:
        """Match Main._song_cache_dir layout so Demucs cache location is identical."""
        p = Path(audio_path)
        try:
            st = p.stat()
            meta = f"{st.st_size}_{int(st.st_mtime)}"
        except Exception:
            meta = "0_0"
        safe = p.stem.replace(os.sep, "_")
        root = p.parent / ".musicpractice" / "stems" / f"{safe}__{meta}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _stem_leaf_dir(self, audio_path: str, out_dir: Path, model: str) -> Path:
        src = Path(audio_path)
        return out_dir / model / src.stem

    def _wait_for_stems(self, audio_path: str, out_dir: Path, model: str = "htdemucs_6s", timeout_s: int = 900):
        """
        Block until htdemucs_6s outputs exist and are size-stable under
        out_dir/<model>/<track>/ for the given audio file.
        Returns dict from load_stem_arrays(leaf_dir) or {} on timeout.
        """
        from PySide6 import QtCore
        # htdemucs_6s fixed set
        expected = {"bass", "drums", "guitar", "piano", "other", "vocals"}
        stem_leaf = self._stem_leaf_dir(audio_path, out_dir, model)
        self._log(f"Waiting for stems in: {stem_leaf}")
        start_ms = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        last_sizes = {}

        def all_present_and_stable():
            files = {}
            for name in expected:
                candidates = list(stem_leaf.glob(f"{name}.wav"))
                if not candidates:
                    return False, {}
                f = max(candidates, key=lambda p: p.stat().st_mtime_ns)
                files[name] = f
            sizes_now = {k: files[k].stat().st_size for k in files}
            stable = (sizes_now == last_sizes) and all(sz > 0 for sz in sizes_now.values())
            return stable, files

        # init last_sizes
        for name in expected:
            candidates = list(stem_leaf.glob(f"{name}.wav"))
            last_sizes[name] = max((p.stat().st_size for p in candidates), default=0)

        elapsed_ms = 0
        while elapsed_ms < timeout_s * 1000:
            if self.isInterruptionRequested():
                return {}
            QtCore.QThread.msleep(500)
            try:
                stable, _ = all_present_and_stable()
            except Exception:
                stable = False
            if stable:
                try: self.status.emit("Stems ready. Loading…")
                except Exception: pass
                try:
                    self._log("Stems appear stable; loading arrays…")
                    loaded = load_stem_arrays(stem_leaf)
                except Exception:
                    loaded = None
                return loaded or {}
            # refresh sizes
            for k in list(last_sizes.keys()):
                candidates = list(stem_leaf.glob(f"{k}.wav"))
                if candidates:
                    last_sizes[k] = max(p.stat().st_size for p in candidates)
            now_ms = QtCore.QTime.currentTime().msecsSinceStartOfDay()
            elapsed_ms = now_ms - start_ms
            if (elapsed_ms // 1000) % 5 == 0:
                try: self.status.emit("Waiting for stems to finish writing…")
                except Exception: pass
        self._log("Timed out waiting for stems; proceeding without stems.")
        return {}

    def estimate_chords(self, audio_path: str, sr=22050, hop=2048, **kwargs):
        """
        Wrapper: always run beat detection first, and attach stems if available,
        then call chords.estimate_chords with these as inputs.
        """
        # If caller didn't provide beat info, compute it now.
        if not kwargs.get("beats"):
            bd = estimate_beats(audio_path, sr=sr, hop=512)
            kwargs["beats"] = bd.get("beats", [])
            kwargs["downbeats"] = bd.get("downbeats", [])
            kwargs["beat_strengths"] = bd.get("beat_strengths", [])

        # Always-on stem extraction (force run) using the same cache layout as _toggle_stems
        model = "htdemucs_6s"
        out_dir = self._song_cache_dir(audio_path)
        leaf = self._stem_leaf_dir(audio_path, out_dir, model)
        maybe_pre = None
        # Only force a fresh Demucs render if Main requested recompute
        try:
            force = bool(getattr(self.parent(), "_force_stems_recompute", False))
        except Exception:
            force = False
        try:
            if leaf.exists() and force:
                self._log(f"Removing existing stems leaf to force render: {leaf}")
                shutil.rmtree(leaf)
        except Exception as e:
            self._log_exc("Pre-clean leaf dir", e)
        self._log(f"[ENTRY] estimate_chords: Demucs run (force={force}) for {audio_path}")        # Start Demucs and stream logs; always wait afterwards
        self._log(f"Demucs starting: {model} → {out_dir}")
        try:
            worker = DemucsWorker(audio_path, str(out_dir), model)
            # Forward every line to the main LogDock through ChordWorker.demucs_line
            worker.line.connect(lambda s: self.demucs_line.emit(s))
            # Block this worker thread until Demucs finishes
            loop = QtCore.QEventLoop()
            worker.done.connect(lambda _p: loop.quit())
            worker.failed.connect(lambda _e: loop.quit())
            worker.start()
            loop.exec()
        except Exception as e:
            self._log_exc("Demucs run", e)

        self._log("Demucs finished rendering. Verifying files…")
        try:
            maybe_pre = self._wait_for_stems(audio_path, out_dir, model=model, timeout_s=900)
            if isinstance(maybe_pre, dict) and maybe_pre:
                leaf_dir = self._stem_leaf_dir(audio_path, out_dir, model)
                self.stems_ready.emit(str(leaf_dir))
        except Exception as e:
            self._log_exc("Wait for stems", e)
            maybe_pre = None

        if not maybe_pre:
            # Last resort: attempt direct load and log outcome
            try:
                maybe_pre = load_stem_arrays(leaf)
                if maybe_pre:
                    self._log("Loaded stems via direct scan of leaf directory.")
                    self.stems_ready.emit(str(leaf))
                else:
                    self._log("Direct leaf load returned empty dict.")
            except Exception as e:
                self._log_exc("Direct leaf load", e)
                maybe_pre = None

        if not kwargs.get("stems") and isinstance(maybe_pre, dict) and maybe_pre:
            kwargs["stems"] = maybe_pre
        if not kwargs.get("stems"):
            try:
                self.status.emit("Proceeding without stems (timeout or missing files).")
            except Exception:
                pass
        return _estimate_chords_base(audio_path, sr=sr, hop=hop, **kwargs)

    def run(self):
        try:
            if self.isInterruptionRequested():
                return
            try:
                self.status.emit("Analyzing: beats + stems + chords…")
            except Exception:
                pass
            self.demucs_line.emit("[ENTRY] ChordWorker.run → estimate_chords\n")
            segs = self.estimate_chords(self.path)
        except Exception as e:
            self._log_exc("ChordWorker.run", e)
            segs = []
        if not self.isInterruptionRequested():
            try:
                self.status.emit("Analysis complete.")
            except Exception:
                pass
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

class DemucsWorker(QtCore.QThread):
    line = QtCore.Signal(str)
    done = QtCore.Signal(str)   # emits the stem root directory as string
    failed = QtCore.Signal(str)

    def __init__(self, audio_path: str, outdir: str, model: str):
        super().__init__()
        self.audio_path = audio_path
        self.outdir = outdir
        self.model = model

    # Make the worker behave like a text stream for stdout/stderr redirection
    def write(self, msg: str):
        try:
            s = str(msg)
        except Exception:
            s = ""
        if s:
            self.line.emit(s)

    def flush(self):
        pass

    def run(self):
        import sys
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self
        try:
            self.line.emit(f"Separating with model: {self.model}\n")
            self.line.emit(f"Output: {self.outdir}\n")
            stem_dir = separate_stems(self.audio_path, self.outdir, model=self.model)
            self.done.emit(str(stem_dir))
        except Exception as e:
            self.failed.emit(str(e))
        finally:
            sys.stdout, sys.stderr = old_out, old_err

class LogDock(QtWidgets.QDockWidget):
    """A dock that behaves like a file-like stream (write/flush) to show Demucs logs."""
    def __init__(self, title="Demucs Log", parent=None):
        super().__init__(title, parent)
        self.setObjectName("DemucsLogDock")
        self.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.TopDockWidgetArea)
        w = QtWidgets.QWidget(self)
        v = QtWidgets.QVBoxLayout(w)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)
        self.view = QtWidgets.QPlainTextEdit(w)
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        # Buffer for partial lines; we only print complete lines
        self._buffer = ""
        v.addWidget(self.view, 1)
        w.setLayout(v)
        self.setWidget(w)

    def clear(self):
        self.view.clear()

    def write(self, msg: str):
        # Coerce to string
        if not isinstance(msg, str):
            try:
                msg = str(msg)
            except Exception:
                msg = ""
        if not msg:
            return
        # Normalize carriage returns (tqdm progress) and buffer until full lines
        msg = msg.replace("\r", "\n")
        self._buffer += msg
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line:
                continue
            QtCore.QMetaObject.invokeMethod(
                self.view,
                "appendPlainText",
                Qt.QueuedConnection,
                QtCore.Q_ARG(str, line)
            )

    def flush(self):
        if getattr(self, "_buffer", ""):
            line = self._buffer
            self._buffer = ""
            QtCore.QMetaObject.invokeMethod(
                self.view,
                "appendPlainText",
                Qt.QueuedConnection,
                QtCore.Q_ARG(str, line.rstrip("\n"))
            )

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

    def _demucs_log(self, text: str):
        if hasattr(self, 'log_dock') and self.log_dock is not None:
            self.log_dock.write(text)

    def _on_stems_ready(self, stem_dir_str: str):
        """Always wire stems into the player and show the mixer dock when stems are ready."""
        try:
            self._load_stems_from_dir(Path(stem_dir_str))
            self.statusBar().showMessage("Stems loaded into mixer.", 1500)
            if self.stems_dock:
                self.stems_dock.show()
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load stems: {e}")
            if hasattr(self, 'log_dock') and self.log_dock is not None:
                self.log_dock.write(f"Failed to load stems: {e}\n")

    def _find_existing_stem_leaf(self, audio_path: str, model: str = "htdemucs_6s") -> Path | None:
        """Return a Demucs leaf directory containing wavs if present, else None."""
        root = self._song_cache_dir(audio_path)
        leaf = self._stem_leaf_dir(audio_path, root, model)
        try:
            if leaf.is_dir() and list(leaf.glob("*.wav")):
                return leaf
            # fall back: scan subdirs for the most recent leaf with wavs
            candidates = [d for d in root.rglob("*") if d.is_dir() and list(d.glob("*.wav"))]
            if candidates:
                return max(candidates, key=lambda p: p.stat().st_mtime)
        except Exception:
            pass
        return None

    def _maybe_load_cached_stems(self, audio_path: str):
        """If stems already exist on disk, load them into the mixer and show the dock."""
        leaf = self._find_existing_stem_leaf(audio_path)
        if leaf is None:
            return
        try:
            self._load_stems_from_dir(leaf)
            self.statusBar().showMessage("Loaded existing stems from cache.", 1500)
            if self.stems_dock:
                self.stems_dock.show()
        except Exception as e:
            self.statusBar().showMessage(f"Cached stems present but failed to load: {e}")
            if hasattr(self, 'log_dock') and self.log_dock is not None:
                self.log_dock.write(f"Cached stems load error: {e}\n")

    def _clear_stem_rows(self):
        while self.stems_layout.count():
            item = self.stems_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

    def _load_stems_from_dir(self, stem_dir: Path):
        # Demucs usually writes outdir/<model>/<track_name>/*.wav → descend to leaf containing wavs
        if stem_dir.is_dir():
            leaf = stem_dir
            try:
                candidates = [d for d in stem_dir.rglob("*") if d.is_dir() and list(d.glob("*.wav"))]
                if candidates:
                    leaf = max(candidates, key=lambda p: p.stat().st_mtime)
            except Exception:
                pass
            stem_dir = leaf
        arrays = load_stem_arrays(stem_dir)
        if not arrays:
            raise RuntimeError("No stem WAVs found")
        if self.player and hasattr(self.player, 'set_stems_arrays'):
            self.player.set_stems_arrays(arrays)
            if hasattr(self.player, 'use_stems_only'):
                self.player.use_stems_only(True)
        if hasattr(self, '_clear_stem_rows'):
            self._clear_stem_rows()
        # Display stems in fixed preferred order (case-insensitive), then any extras
        preferred_order = ["vocals", "drums", "bass", "guitar", "piano", "other"]

        # Build a normalization map from canonical key -> original key in arrays
        def _norm(s: str) -> str:
            return s.strip().lower().replace(" ", "").replace("_", "-")

        by_norm: dict[str, str] = {}
        for original in arrays.keys():
            n = _norm(original)
            # keep first occurrence; subsequent duplicates won't override
            by_norm.setdefault(n, original)

        used = set()
        for want in preferred_order:
            # accept common aliases and plural/singular variants
            aliases = [want]
            if want == "vocals":
                aliases += ["vocal", "voice", "voices"]
            if want == "drums":
                aliases += ["drum"]
            if want == "bass":
                aliases += ["bassguitar", "bass-guitar", "bass_guitar"]
            if want == "guitar":
                aliases += ["guitars"]
            if want == "piano":
                aliases += ["keys", "keyboard"]
            if want == "other":
                aliases += ["others", "misc", "accompaniment", "backing"]

            for al in aliases:
                match = by_norm.get(_norm(al))
                if match and match not in used:
                    self._add_stem_row(match, arrays[match])
                    used.add(match)
                    break

        # Add any extras not covered above, in stable order from arrays
        for k in arrays.keys():
            if k not in used:
                self._add_stem_row(k, arrays[k])
        if self.stems_dock:
            self.stems_dock.show()

    def _demucs_done(self, stem_dir_str: str):
        if hasattr(self, 'log_dock') and self.log_dock is not None:
            self.log_dock.write("\nSeparation complete.\n")
        try:
            self._load_stems_from_dir(Path(stem_dir_str))
        except Exception as e:
            self.statusBar().showMessage(f"Load stems failed: {e}")
            if self.log_dock:
                self.log_dock.write(f"Load stems failed: {e}\n")
        finally:
            setattr(self, 'demucs_worker', None)

    def _demucs_failed(self, err: str):
        if hasattr(self, 'log_dock') and self.log_dock is not None:
            self.log_dock.write(f"\nERROR: {err}\n")
        self.statusBar().showMessage(f"Stems failed: {err}")
        setattr(self, 'demucs_worker', None)

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
                    w.wait(-1)
            except Exception:
                pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MusicPractice — Minimal")
        self.resize(960, 600)
        self.setAcceptDrops(True)
        self.settings = QSettings("MusicPractice", "MusicPractice")

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
        self.log_dock = None  # created on demand when separating stems
        self._force_stems_recompute = False

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

        self.actRecompute = QAction("Recompute analysis", self)
        self.actRecompute.setShortcut(QKeySequence("Ctrl+Shift+R"))
        self.actRecompute.triggered.connect(self._recompute_current)

        self.actAlwaysRecompute = QAction("Always recompute on open", self)
        self.actAlwaysRecompute.setCheckable(True)
        # Restore persisted value
        try:
            s = QSettings("musicpractice", "musicpractice")
            always = bool(s.value("analysis/always_recompute_on_open", False, type=bool))
            self.actAlwaysRecompute.setChecked(always)
        except Exception:
            pass
        self.actAlwaysRecompute.toggled.connect(self._toggle_always_recompute)

        # Menu
        try:
            mb = self.menuBar()
            analysisMenu = mb.addMenu("Analysis")
            analysisMenu.addAction(self.actRecompute)
            analysisMenu.addSeparator()
            analysisMenu.addAction(self.actAlwaysRecompute)
        except Exception:
            pass

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
        self.toolbar.addSeparator()
        try:
            self.actRecompute.setToolTip("Delete cached analysis and re-run stems + chords (Ctrl+Shift+R)")
        except Exception:
            pass
        self.toolbar.addAction(self.actRecompute)

        # === Stems UI (dock hidden by default) ===
        self.stems_dock = QtWidgets.QDockWidget("Stems", self)
        self.stems_dock.setObjectName("StemsDock")
        self.stems_dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.stems_panel = QtWidgets.QWidget(self.stems_dock)
        self.stems_layout = QtWidgets.QVBoxLayout(self.stems_panel)
        self.stems_layout.setContentsMargins(6, 6, 6, 6)
        self.stems_layout.setSpacing(8)
        self.stems_panel.setLayout(self.stems_layout)
        self.stems_dock.setWidget(self.stems_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stems_dock)
        self.stems_dock.hide()

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
        self.wave.requestEditChord.connect(self._edit_chord_at_time)
        self.wave.requestSplitChordAt.connect(self._split_chord_at_time)
        self.wave.requestJoinChordForward.connect(self._join_chord_forward_at_time)

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

        # View menu
        view_menu = self.menuBar().addMenu("View")
        self.act_toggle_stems = QAction("Separate && Show Stems", self)
        self.act_toggle_stems.setCheckable(True)
        self.act_toggle_stems.toggled.connect(self._toggle_stems)
        view_menu.addAction(self.act_toggle_stems)

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

    def _toggle_always_recompute(self, checked: bool):
        try:
            s = QSettings("musicpractice", "musicpractice")
            s.setValue("analysis/always_recompute_on_open", bool(checked))
        except Exception:
            pass

    def _recompute_current(self):
        self._force_stems_recompute = True
        try:
            path = getattr(self, "current_path", None)
            if not path:
                return
            self._clear_cached_analysis(path, remove_stems=True)
            self._clear_stem_rows()
            self.start_chord_analysis(path)  # runs Demucs + waits + chords
        except Exception:
            pass

    def _session_sidecar_path(self, audio_path: str) -> Path:
        p = Path(audio_path)
        folder = p.parent / ".musicpractice"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{p.stem}.musicpractice.json"

    def load_session(self):
        # Use current file’s sidecar if a song is loaded; else prompt for JSON
        if self.current_path:
            sidecar = self._session_sidecar_path(self.current_path)
        else:
            dlg = QtWidgets.QFileDialog(self, "Open Session JSON")
            dlg.setNameFilter("MusicPractice Session (*.musicpractice.json);;JSON (*.json)")
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

        # Chords (normalize to bar-aligned & split form on load)
        self.last_chords = list(data.get("chords", []))
        if self.last_chords:
            try:
                snapped = self.wave._snap_segments_to_bars(self.last_chords, self.last_bars or [])
                split = self.wave._split_segments_at_bars(snapped, self.last_bars or [])
                filled = self.wave._ensure_leading_bar(split, self.last_bars or [])
                self.last_chords = self.wave._unique_by_span(filled)
            except Exception:
                pass
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

    def _song_cache_dir(self, audio_path: str) -> Path:
        p = Path(audio_path)
        try:
            st = p.stat()
            meta = f"{st.st_size}_{int(st.st_mtime)}"
        except Exception:
            meta = "0_0"
        safe = p.stem.replace(os.sep, "_")
        root = p.parent / ".musicpractice" / "stems" / f"{safe}__{meta}"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _bar_index_at_time(self, t: float) -> int | None:
        if not self.last_bars:
            return None
        bars = sorted(float(b) for b in self.last_bars)
        if not bars:
            return None
        for i, b in enumerate(bars):
            if t < b:
                return max(0, i - 1)
        return len(bars) - 1

    def _find_chord_index_at_time(self, t: float) -> int | None:
        """Return index in self.last_chords that contains absolute time t."""
        try:
            seq = list(self.last_chords or [])
        except Exception:
            return None
        for i, seg in enumerate(seq):
            try:
                a = float(seg.get('start'))
                b = float(seg.get('end'))
            except Exception:
                continue
            if a <= t < b or (i == len(seq) - 1 and abs(t - b) < 1e-6):
                return i
        return None

    def _edit_chord_at_time(self, t: float):
        """Rename the chord segment whose interval contains time t; persist and redraw."""
        print(f"[DBG] _edit_chord_at_time(t={t:.3f})")
        idx = self._find_chord_index_at_time(float(t))
        if idx is None:
            print("[DBG] _edit_chord_at_time: no chord at time")
            return
        cur = dict(self.last_chords[idx]) if (0 <= idx < len(self.last_chords)) else {}
        cur_label = str(cur.get('label', ''))
        text, ok = QtWidgets.QInputDialog.getText(self, "Change chord", "Chord label:", text=cur_label)
        if not ok:
            print("[DBG] _edit_chord_at_time: cancelled")
            return
        cur['label'] = str(text)
        self.last_chords[idx] = cur
        # Reflect in UI and save
        self.wave.set_chords(self.last_chords)
        try:
            self.wave.update(); self.wave.repaint()
        except Exception:
            pass
        self.save_session()
        self.statusBar().showMessage(f"Chord updated → {text}", 1500)
        print(f"[DBG] _edit_chord_at_time: updated idx={idx} → '{text}'")

    def _split_chord_at_time(self, t: float):
        """Split the chord containing time t so the two parts have equal beat counts;
        if the total beat count is odd, the first (left) part has one more beat.
        The split is snapped to a beat strictly inside the chord span when possible.
        """
        t = float(t)
        print(f"[DBG] _split_chord_at_time(request t={t:.3f})")

        # Find chord index at click time
        idx = self._find_chord_index_at_time(t)
        if idx is None:
            print("[DBG] _split_chord_at_time: no chord at click time")
            return
        try:
            seg = dict(self.last_chords[idx])
            a = float(seg.get('start')); b = float(seg.get('end'))
            lab = seg.get('label')
        except Exception:
            print("[DBG] _split_chord_at_time: bad segment data")
            return
        if not (b > a + 1e-6):
            print("[DBG] _split_chord_at_time: zero/negative length segment; abort")
            return

        # Collect beats strictly inside (a, b)
        eps_in = 1e-6
        inside = []
        try:
            if self.last_beats:
                arr = np.asarray(self.last_beats, dtype=float)
                inside = arr[(arr > a + eps_in) & (arr < b - eps_in)].tolist()
        except Exception:
            inside = []

        # Choose split point
        if inside:
            m = len(inside)  # number of interior beats
            # Choose interior beat index so left intervals = ceil((m+1)/2)
            k = int(np.ceil((m + 1) / 2.0))  # 1-based index
            k = max(1, min(m, k))
            split_t = float(inside[k - 1])
            print(f"[DBG] _split_chord_at_time: interior beats m={m} → k={k} → split_t={split_t:.3f}")
        else:
            # Fallback: use the temporal midpoint
            split_t = 0.5 * (a + b)
            print(f"[DBG] _split_chord_at_time: no interior beats → midpoint split_t={split_t:.3f}")

        # Keep strictly inside the span
        eps = 1e-3
        if split_t <= a + eps:
            split_t = min(a + eps, b - eps)
        if split_t >= b - eps:
            split_t = max(b - eps, a + eps)

        left = {'start': a, 'end': split_t, 'label': lab}
        right = {'start': split_t, 'end': b, 'label': lab}
        self.last_chords = self.last_chords[:idx] + [left, right] + self.last_chords[idx+1:]

        # Minimal normalization: enforce bar boundaries, keep user beat split
        try:
            split2 = self.wave._split_segments_at_bars(self.last_chords, self.last_bars or [])
            filled = self.wave._ensure_leading_bar(split2, self.last_bars or [])
            self.last_chords = self.wave._unique_by_span(filled)
        except Exception:
            pass

        # Update UI and persist
        self.wave.set_chords(self.last_chords)
        self.save_session()
        self.statusBar().showMessage(f"Chord split at {split_t:.2f}s", 1200)
        print(f"[DBG] _split_chord_at_time: idx={idx} split at {split_t:.3f}s → two segments")

    def _join_chord_forward_at_time(self, t: float):
        """Join the chord at time t with the next chord if contiguous and within same bar."""
        t = float(t)
        print(f"[DBG] _join_chord_forward_at_time(t={t:.3f})")
        idx = self._find_chord_index_at_time(t)
        if idx is None or idx + 1 >= len(self.last_chords):
            print("[DBG] _join_chord_forward_at_time: no join candidate")
            return
        cur = self.last_chords[idx]
        nxt = self.last_chords[idx + 1]
        try:
            cmid = 0.5 * (float(cur['start']) + float(cur['end']))
            nmid = 0.5 * (float(nxt['start']) + float(nxt['end']))
        except Exception:
            print("[DBG] _join_chord_forward_at_time: bad segments")
            return
        bar_cur = self._bar_index_at_time(cmid)
        bar_nxt = self._bar_index_at_time(nmid)
        if bar_cur is None or bar_nxt is None or bar_cur != bar_nxt:
            print("[DBG] _join_chord_forward_at_time: different bars")
            return
        if abs(float(cur['end']) - float(nxt['start'])) > 1e-3:
            print("[DBG] _join_chord_forward_at_time: not contiguous")
            return
        merged = {
            'start': float(cur['start']),
            'end': float(nxt['end']),
            'label': cur.get('label') or nxt.get('label')
        }
        self.last_chords = self.last_chords[:idx] + [merged] + self.last_chords[idx+2:]
        self.wave.set_chords(self.last_chords)
        self.save_session()
        print(f"[DBG] _join_chord_forward_at_time: merged idx={idx} & idx={idx+1}")

    def _toggle_stems(self, on: bool):
        """Run stem separation (Demucs) and show a simple mixer UI with per-stem volume & mute."""
        # Turning OFF: hide and clear
        if not on:
            self.stems_dock.hide()
            # delete UI rows
            while self.stems_layout.count():
                item = self.stems_layout.takeAt(0)
                w = item.widget()
                if w:
                    w.deleteLater()
            # clear player stems
            if self.player and hasattr(self.player, 'clear_stems'):
                self.player.clear_stems()
            # also hide the log dock if present
            if hasattr(self, "log_dock") and self.log_dock is not None:
                self.log_dock.hide()
            return

        # Turning ON requires an audio file
        if not self.current_path:
            self.statusBar().showMessage("Open a track first")
            self.act_toggle_stems.setChecked(False)
            return

        # Run Demucs into a per-song cache folder
        outdir = self._song_cache_dir(self.current_path)
        preexisting = sorted(outdir.rglob("*.wav"))
        if preexisting:
            stem_dir = outdir
            # Use unified loader so ordering is guaranteed
            try:
                self._load_stems_from_dir(Path(stem_dir))
                if self.stems_dock:
                    self.stems_dock.show()
            except Exception as e:
                self.statusBar().showMessage(f"Load stems failed: {e}")
                if getattr(self, "log_dock", None):
                    self.log_dock.write(f"Load stems failed: {e}\n")
                if hasattr(self, 'act_toggle_stems'):
                    self.act_toggle_stems.setChecked(False)
                return
            return
        else:
            # Ensure a log dock exists and is visible
            if not hasattr(self, "log_dock") or self.log_dock is None:
                self.log_dock = LogDock(parent=self)
                self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
            self.log_dock.setWindowTitle("Demucs Log")
            self.log_dock.show()
            self.log_dock.raise_()
            self.log_dock.clear()

            # Launch Demucs in background and stream logs
            model = 'htdemucs_6s'
            self.demucs_worker = DemucsWorker(self.current_path, str(outdir), model)
            self.demucs_worker.line.connect(self._demucs_log)
            self.demucs_worker.done.connect(self._demucs_done)
            self.demucs_worker.failed.connect(self._demucs_failed)
            self.demucs_worker.start()
            return

        leaf = stem_dir
        candidates = [d for d in stem_dir.rglob("*") if d.is_dir() and list(d.glob("*.wav"))]
        if candidates:
            leaf = max(candidates, key=lambda p: p.stat().st_mtime)
        stem_dir = leaf

        # Load arrays
        try:
            arrays = load_stem_arrays(stem_dir)
        except Exception as e:
            self.statusBar().showMessage(f"Load stems failed: {e}")
            self.act_toggle_stems.setChecked(False)
            return
        if not arrays:
            self.statusBar().showMessage("No stems produced")
            self.act_toggle_stems.setChecked(False)
            return

        # Feed to player
        try:
            if self.player and hasattr(self.player, 'set_stems_arrays'):
                self.player.set_stems_arrays(arrays)
                if hasattr(self.player, 'use_stems_only'):
                    self.player.use_stems_only(True)
        except Exception as e:
            self.statusBar().showMessage(f"Player stems error: {e}")
            self.act_toggle_stems.setChecked(False)
            return

        # Build UI rows
        for name in order_stem_names(list(arrays.keys())):
            self._add_stem_row(name, arrays[name])

        self.stems_dock.show()

    def _add_stem_row(self, name: str, arr: np.ndarray):
        row = QtWidgets.QWidget(self.stems_panel)
        outer = QtWidgets.QVBoxLayout(row)
        outer.setContentsMargins(4, 6, 4, 6)
        outer.setSpacing(6)

        # Header: label (left) + mute (right)
        header = QtWidgets.QWidget(row)
        hl = QtWidgets.QHBoxLayout(header)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(6)

        lab = QtWidgets.QLabel(name.capitalize(), header)
        lab.setMinimumWidth(60)
        lab.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        mute = QtWidgets.QToolButton(header)
        mute.setText("M")
        mute.setCheckable(True)
        mute.setToolTip(f"Mute {name}")

        hl.addWidget(lab, 1)
        hl.addStretch(1)
        hl.addWidget(mute)
        header.setLayout(hl)

        # Volume slider (horizontal) under the header
        vol = QtWidgets.QSlider(Qt.Horizontal, row)
        vol.setRange(0, 100)
        vol.setValue(100)
        vol.setSingleStep(1)
        vol.setPageStep(5)
        vol.setTracking(True)
        vol.setToolTip(f"{name} level")

        # Wire up callbacks
        def _vol_changed(v, nm=name):
            if self.player and hasattr(self.player, 'set_stem_gain'):
                self.player.set_stem_gain(nm, float(v) / 100.0)

        def _mute_toggled(m, nm=name, slider=vol):
            if self.player and hasattr(self.player, 'set_stem_mute'):
                self.player.set_stem_mute(nm, bool(m))
            slider.setEnabled(not m)

        vol.valueChanged.connect(_vol_changed)
        mute.toggled.connect(_mute_toggled)

        outer.addWidget(header)
        outer.addWidget(vol)
        row.setLayout(outer)
        self.stems_layout.addWidget(row)

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

    def _find_chord_index_at_time(self, t: float) -> int | None:
        if not self.last_chords:
            return None
        for i, seg in enumerate(self.last_chords):
            try:
                a = float(seg.get('start')); b = float(seg.get('end'))
            except Exception:
                continue
            if a <= t < b or (i == len(self.last_chords) - 1 and abs(t - b) < 1e-6):
                return i
        return None

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

    def _sidecar_path(self, audio_path: str) -> Path:
        p = Path(audio_path)
        return p.parent / ".musicpractice" / f"{p.stem}.musicpractice.json"

    def _stem_leaf_dir(self, audio_path: str, out_dir_or_model=None, model: str = "htdemucs_6s") -> Path:
        """
        Backward-compatible helper that accepts either:
        - (audio_path) → uses cache dir + default model
        - (audio_path, model_str) → uses cache dir + given model
        - (audio_path, out_dir: PathLike, model_str) → explicit out_dir and model
        - Keyword args out_dir=..., model=...
        """
        p = Path(audio_path)
        # Resolve out_dir based on the 2nd arg's type/shape
        if out_dir_or_model is None:
            out_dir = self._song_cache_dir(audio_path)
        elif isinstance(out_dir_or_model, (Path, os.PathLike)):
            out_dir = Path(out_dir_or_model)
        elif isinstance(out_dir_or_model, str):
            # If it's clearly a path-like string (has separator or exists), treat as out_dir; else it's a model
            looks_path = (os.sep in out_dir_or_model) or out_dir_or_model.startswith(".") \
                        or out_dir_or_model.startswith("/") or Path(out_dir_or_model).exists()
            if looks_path:
                out_dir = Path(out_dir_or_model)
            else:
                out_dir = self._song_cache_dir(audio_path)
                model = out_dir_or_model
        else:
            out_dir = self._song_cache_dir(audio_path)
        return Path(out_dir) / model / p.stem

    def _clear_cached_analysis(self, audio_path: str, remove_stems: bool = True):
        # Delete sidecar JSON
        try:
            sidecar = self._sidecar_path(audio_path)
            if sidecar.exists():
                sidecar.unlink()
        except Exception:
            pass
        # Optionally delete stems leaf
        if remove_stems:
            try:
                leaf = self._stem_leaf_dir(audio_path, "htdemucs_6s")
                if leaf.exists():
                    shutil.rmtree(leaf)
            except Exception:
                pass
        # Clear in-memory chords and the view
        try:
            self.last_chords = []
        except Exception:
            pass
        try:
            if hasattr(self, "wave"):
                self.wave.set_chords([])
        except Exception:
            pass

    # === Actions ===
    def _rate_changed(self, val: float):
        """Update playback rate on the player and keep the UI/transport coherent."""
        try:
            r = float(val)
        except Exception:
            r = 1.0
        # Clamp to supported range
        r = max(0.5, min(1.5, r))
        # Update label
        if hasattr(self, 'rate_label') and self.rate_label is not None:
            self.rate_label.setText(f"Rate: {r:.2f}x")
        # Apply to player
        if getattr(self, 'player', None) and hasattr(self.player, 'set_rate'):
            ok = self.player.set_rate(r)
            if ok is False:
                msg = getattr(self.player, 'last_rate_error', None) or "Time-stretch unavailable — using normal speed."
                self.statusBar().showMessage(msg, 6000)
                # Nudge transport so UI ↔ audio mapping stays stable
            try:
                pos = float(self.player.position_seconds())
                self.player.set_position_seconds(pos, within_loop=True)
            except Exception:
                pass

    def render_rate(self):
        """Render a pitch-preserving stretched WAV using the current rate (if available)."""
        if not getattr(self, 'current_path', None) or not getattr(self, 'player', None):
            self.statusBar().showMessage("No audio loaded")
            return
        try:
            rate = float(self.rate_spin.value())
        except Exception:
            rate = 1.0
        if abs(rate - 1.0) < 1e-6:
            self.statusBar().showMessage("Rate is 1.0x — nothing to render")
            return
        if not HAS_STRETCH:
            QtWidgets.QMessageBox.information(self, "Time-stretch", "Time-stretch engine not available.")
            return
        try:
            out_wav = temp_wav_path(prefix="render_", suffix=".wav")
            render_time_stretch(self.current_path, out_wav, rate)
            self.statusBar().showMessage(f"Rendered stretched audio → {Path(out_wav).name}")
        except Exception as e:
            self.statusBar().showMessage(f"Render failed: {e}")

    def load_audio(self):
        start_dir = self.settings.value("last_dir", str(Path.home()))
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Audio", start_dir, "Audio (*.wav *.mp3 *.flac *.m4a)")
        if not fn:
            return
        self.current_path = fn
        self.settings.setValue("last_dir", str(Path(fn).parent))
        name = Path(fn).name
        self.setWindowTitle(f"MusicPractice — {name}")
        self.title_label.setText(name)

        # (Re)create player
        if self.player:
            self.player.stop(); self.player.close()
        self.player = LoopPlayer(fn)
        self.wave.set_player(self.player)
        # Apply current UI rate to fresh player (keeps UI and audio consistent)
        self._rate_changed(self.rate_spin.value())

        self._maybe_load_cached_stems(self.current_path)

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
        try:
            if hasattr(self, "actAlwaysRecompute") and self.actAlwaysRecompute.isChecked():
                self._clear_cached_analysis(self.current_path, remove_stems=True)
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
        # Delegate to the unified chord analysis entrypoint which runs Demucs
        # with live logs and waits for stems before estimating chords.
        return self.start_chord_analysis(path)

    def populate_key_async(self, path: str):
        if self.key_worker and self.key_worker.isRunning():
            self.key_worker.requestInterruption(); self.key_worker.quit(); self.key_worker.wait(1000)
        kw = KeyWorker(path)
        kw.setParent(self)
        kw.done.connect(self._key_ready)
        kw.finished.connect(lambda: setattr(self, "key_worker", None))
        self.key_worker = kw
        kw.start()

    def start_chord_analysis(self, path: str):
        # Stop any existing chord worker
        if hasattr(self, "chord_worker") and self.chord_worker and self.chord_worker.isRunning():
            self.chord_worker.requestInterruption()
            self.chord_worker.quit()
            self.chord_worker.wait(-1)
        cw = ChordWorker(path)
        cw.setParent(self)
        # Ensure Demucs logs are visible and wire stem loading
        if not hasattr(self, "log_dock") or self.log_dock is None:
            self.log_dock = LogDock(parent=self)
            self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setWindowTitle("Demucs Log")
        self.log_dock.show(); self.log_dock.raise_(); self.log_dock.clear()

        cw.demucs_line.connect(self._demucs_log)
        cw.stems_ready.connect(self._on_stems_ready)

        cw.status.connect(lambda s: self.statusBar().showMessage(str(s)))
        cw.done.connect(self._chords_ready)
        cw.finished.connect(lambda: (setattr(self, "chord_worker", None), setattr(self, "_force_stems_recompute", False)))
        self.chord_worker = cw
        self.statusBar().showMessage("Analyzing: beats + stems + chords…")
        cw.start()

    def _chords_ready(self, segments: list[dict]):
        # Normalize chords to bar-aligned & split form before persisting
        try:
            snapped = self.wave._snap_segments_to_bars(segments or [], self.last_bars or [])
            split = self.wave._split_segments_at_bars(snapped, self.last_bars or [])
            filled = self.wave._ensure_leading_bar(split, self.last_bars or [])
            dedup = self.wave._unique_by_span(filled)
            segments = dedup
        except Exception:
            pass
        self.last_chords = list(segments or [])
        self.wave.set_chords(self.last_chords)
        self.save_session()
        # Clear the waiting message after a short delay
        self.statusBar().showMessage(f"Chords: {len(self.last_chords)} segments")
        QtCore.QTimer.singleShot(1500, lambda: self.statusBar().clearMessage())

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

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.isLocalFile():
                    suf = Path(u.toLocalFile()).suffix.lower()
                    if suf in {'.wav', '.mp3', '.flac', '.m4a'}:
                        e.acceptProposedAction()
                        return
        e.ignore()

    def dropEvent(self, e: QtGui.QDropEvent):
        urls = [u for u in e.mimeData().urls() if u.isLocalFile()]
        if not urls:
            e.ignore(); return
        path = urls[0].toLocalFile()
        # Attach player and waveform
        if self.player:
            try:
                self.player.stop(); self.player.close()
            except Exception:
                pass
        self.current_path = path
        self.settings.setValue("last_dir", str(Path(path).parent))
        name = Path(path).name
        self.setWindowTitle(f"MusicPractice — {name}")
        self.title_label.setText(name)
        self.player = LoopPlayer(path)
        self.wave.set_player(self.player)
        # Apply current UI rate so playback speed matches the control
        self._rate_changed(self.rate_spin.value())
        # Align visual time 0 to first non‑silent audio
        lead = self._detect_leading_silence_seconds(self.player.y, self.player.sr)
        tail = self._detect_trailing_silence_seconds(self.player.y, self.player.sr)
        self.wave.set_music_span(lead, tail)
        try:
            self.player.set_position_seconds(lead, within_loop=False)
        except Exception:
            pass
        # Restore session and compute only missing analyses
        try:
            self.load_session()
        except Exception:
            pass
        try:
            if hasattr(self, "actAlwaysRecompute") and self.actAlwaysRecompute.isChecked():
                self._clear_cached_analysis(self.current_path, remove_stems=True)
        except Exception:
            pass
        if not self.last_beats:
            self.populate_beats_async(self.current_path)
        if not self.last_chords:
            self.populate_chords_async(self.current_path)
        if not (self.last_key and self.last_key.get('pretty')):
            self.populate_key_async(self.current_path)
        total_s = self.player.n / self.player.sr
        mus_len = max(0.0, (tail - lead))
        self.statusBar().showMessage(f"Loaded: {Path(path).name} [music {mus_len:.1f}s of {total_s:.1f}s]")
        e.acceptProposedAction()

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self._cleanup_on_quit()
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = Main(); w.show()
    sys.exit(app.exec())
