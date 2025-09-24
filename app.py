import sys
import json
import shutil
import os
from pathlib import Path

# Ensure the Vamp plugin path is visible even when launched from IDE / different shells
os.environ.setdefault(
    "VAMP_PATH",
    f"{Path.home()}/Library/Audio/Plug-Ins/Vamp:/Library/Audio/Plug-Ins/Vamp"
)
from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import Qt, QThread, Signal, QSettings
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QActionGroup
import numpy as np
import traceback
import logging
from audio_engine import LoopPlayer
from chords import estimate_chords_stem_aware as _stem_aware, estimate_chords as _fast_est, estimate_chords_chordino_vamp as _chordino, estimate_key, estimate_beats
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


# Import piano widget and note detection
from piano_widget import PianoRollWidget
from note_detection import compute_note_confidence

class WaveformView(QtWidgets.QWidget):
    def __init__(self, parent=None, window_s: float = 15.0):
        super().__init__(parent)

        # ---- UI basics ----
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(160)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

        # Visual time window (seconds)
        self.window_s = float(window_s)

        # ---- Caches / performance ----
        self._waveform_cache = {}            # (stem, t0, t1, w) -> (mins, maxs, x0, x1)
        self._cache_max_entries = 1000
        self._last_paint_time = 0
        self._paint_throttle_ms = 16         # ~60fps cap

        # Multi-resolution pyramid for ultra-fast scrolling
        self._pyramid_cache = {}             # stem_name -> {level: (mins, maxs, N_total)}
        self._pyramid_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        # ---- Player & analysis state ----
        self.player = None                   # type: LoopPlayer | None
        self.chords = []                     # list[{start,end,label}]
        self.beats = []                      # beat times (s)
        self.bars = []                       # downbeat times (s)
        self.origin = 0.0                    # visual zero aligned to music start
        self.content_end = None              # last non-silent sample time (s)

        # ---- Loop state ----
        self.loopA = None
        self.loopB = None

        # ---- View mode ----
        self.show_stems = True               # stacked stems vs mixed view

        # ---- Solo / Focus (visual vs audio) ----
        self.soloed_stem = None              # audio solo target (or None)
        self.focus_stem = None               # VISUAL-ONLY focus (solo-style visuals without soloing audio)

        # ---- Spectrum / Piano band for focus/solo visuals ----
        self.spectrum_data = None
        self.spectrum_band_height = 60
        from piano_widget import PianoRollWidget
        self.piano_roll_widget = PianoRollWidget(self)
        self.piano_roll_widget.hide()

        # ---- Interaction / dragging ----
        self._drag_mode = None               # 'set' | 'resizeA' | 'resizeB' | 'move' | None
        self.snap_enabled = True             # snap loops to beats when dragging
        self._press_t = None
        self._press_loopA = None
        self._press_loopB = None

        # Loop flag geometry
        self.HANDLE_PX = 6
        self.FLAG_W = 10
        self.FLAG_H = 10
        self.FLAG_STRIP = 16                 # px reserved at the top for flags/labels

        # Saved loops UI state
        self.saved_loops = []                # optional: list of dicts {id,a,b,label}
        self.selected_loop_id = None

        # Mouse tracking state
        self._press_kind = None              # 'new' | 'edgeA' | 'edgeB' | None
        self._press_loop_id = None
        self._press_dx = 0.0
        self._press_x = None
        self._drag_started = False
        self._click_thresh_px = 4

        # Smooth pan accumulator for wheel scrolling
        self._pan_accum_s = 0.0

        # Freeze window (after seek) support
        self._freeze_window = False
        self._manual_t0 = None
        self._manual_t1 = None

        # ---- Repaint timer ----
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(33)           # ~30 FPS
        self.timer.timeout.connect(self.update)
        self.timer.start()

        # ---- Ensure child layout is correct on construct ----
        try:
            self._layout_children()
        except Exception:
            pass

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


    def set_beats(self, beats: list | None, bars: list | None):
        self.beats = beats or []
        self.bars = bars or []
        self.update()

    def set_player(self, player: 'LoopPlayer'):
        self.player = player
        try:
            _ = int(getattr(self.player, 'sr', 0) or 0)
            _y = getattr(self.player, 'y', None)
            if _ is None or _ <= 0 or _y is None:
                return
        except Exception:
            pass
        self.update()

    def set_show_stems(self, show: bool):
        self.show_stems = bool(show)
        self.update()

    def set_soloed_stem(self, stem_name: str | None):
        """Set which stem is soloed for display. None means no solo."""
        self.soloed_stem = stem_name

        # Show/hide piano roll widget based on focus mode
        if stem_name is None:
            # Not in focus mode - hide piano widget
            if self.piano_roll_widget is not None:
                self.piano_roll_widget.hide()
        else:
            # In focus mode - ensure piano widget is created and shown
            if self.piano_roll_widget is None:
                # Create piano widget if it doesn't exist
                self.piano_roll_widget = PianoRollWidget(self)

            # Always show the piano widget in focus mode; geometry handled by layout pass
            self.piano_roll_widget.show()
            self.piano_roll_widget.raise_()
            try:
                self._layout_children()
            except Exception:
                pass

        # Re-layout child widgets when solo state changes
        try:
            self._layout_children()
        except Exception:
            pass
        self.update()

    def _visual_focus_stem(self) -> str | None:
        """
        Decide which stem (if any) should drive solo-style VISUALS.
        Preference order: explicit Focus (visual-only) → Solo (audio).
        """
        return getattr(self, 'focus_stem', None) or getattr(self, 'soloed_stem', None)

    def get_focus_stem(self) -> str | None:
        return self._visual_focus_stem()

    def set_focus_stem(self, stem_name: str | None):
        """
        Set/clear a VISUAL-ONLY focus stem. Does not change audio routing/mutes.
        When set, the UI shows single wide waveform + piano keyboard for this stem.
        """
        self.focus_stem = stem_name or None
        vis_focus = self._visual_focus_stem()

        # Manage embedded piano widget like in solo visuals
        if vis_focus is None:
            if getattr(self, 'piano_roll_widget', None) is not None:
                self.piano_roll_widget.hide()
        else:
            if getattr(self, 'piano_roll_widget', None) is None:
                from piano_widget import PianoRollWidget
                self.piano_roll_widget = PianoRollWidget(self)
            self.piano_roll_widget.show()
            self.piano_roll_widget.raise_()

        try:
            self._layout_children()
        except Exception:
            pass
        self.update()

    def showEvent(self, ev):
        try:
            self._layout_children()
        except Exception:
            pass
        super().showEvent(ev)

    def resizeEvent(self, ev):
        try:
            self._layout_children()
        except Exception:
            pass
        super().resizeEvent(ev)

    def _compute_spectrum_at_playhead(self):
        """Compute frequency spectrum at current playhead position for soloed stem using librosa CQT."""
        if not self._visual_focus_stem() or not self.player:
            self.spectrum_data = None
            return

        try:
            # Get current playhead position (prefer explicit seconds method)
            playhead_time = None
            try:
                if hasattr(self.player, 'position_seconds') and callable(self.player.position_seconds):
                    playhead_time = float(self.player.position_seconds())
                else:
                    playhead_time = float(getattr(self.player, 'position', 0.0))
            except Exception:
                playhead_time = 0.0
            # Allow t=0.0; only bail if playhead_time is negative
            if playhead_time is None or playhead_time < 0.0:
                self.spectrum_data = None
                return

            # Get the soloed stem data
            stems = self._get_stems_for_display()
            solo_stem_data = None
            focus_name = self._visual_focus_stem()
            for stem_name, arr in stems:
                if stem_name == focus_name:
                    solo_stem_data = arr
                    break

            if solo_stem_data is None:
                self.spectrum_data = None
                return

            # Get sample rate
            sr = getattr(self.player, 'sr', 22050)
            hop_length = 512

            # Convert time to sample index
            sample_idx = int(playhead_time * sr)
            if sample_idx >= len(solo_stem_data):
                self.spectrum_data = None
                return

            # Extract a larger window for better CQT analysis (e.g., 4096 samples)
            window_size = 4096
            start_idx = max(0, sample_idx - window_size // 2)
            end_idx = min(len(solo_stem_data), start_idx + window_size)

            if end_idx - start_idx < window_size // 2:
                self.spectrum_data = None
                return

            # Get the audio window
            audio_window = solo_stem_data[start_idx:end_idx]

            # --- Silence gate: if the window is very quiet, treat as no notes ---
            try:
                import numpy as _np
                aw = _np.asarray(audio_window, dtype=_np.float32)
                # short-circuit if empty
                if aw.size == 0:
                    self.spectrum_data = None
                    return
                # Root-mean-square and dBFS
                rms = float(_np.sqrt(_np.mean(aw * aw)) + 1e-12)
                rms_db = 20.0 * _np.log10(rms)
            except Exception:
                rms_db = -120.0

            SILENCE_DB = -60.0  # gate threshold; tweak if needed
            if rms_db <= SILENCE_DB:
                # Produce a flat zero-confidence vector so UI shows a flat baseline
                zero_vec = np.zeros((1, 88), dtype=np.float32)
                zero_full = np.zeros((max(1, (end_idx - start_idx) // hop_length), 88), dtype=np.float32)
                self.spectrum_data = {
                    'note_conf': zero_vec,
                    'time': playhead_time,
                    'sample_pos': sample_idx,
                    'window_start_idx': start_idx,
                    'local_sample_pos': sample_idx - start_idx,
                    'hop_length': hop_length,
                    'full_note_conf': zero_full,
                    'rms_db': rms_db,
                    'silent': True,
                }
                return

            # Use librosa CQT for better note detection
            note_conf = compute_note_confidence(
                audio_window,
                sr=sr,
                hop_length=hop_length,
                bins_per_octave=36,
                n_octaves=8,
                fmin=27.5,  # A0
                ema_alpha=0.4,
                medfilt_width=3
            )

            # Get the frame corresponding to the playhead position within the window
            frame_idx = min(note_conf.shape[0] - 1, (sample_idx - start_idx) // hop_length)
            current_frame = note_conf[frame_idx:frame_idx+1]  # Shape (1, 88)

            self.spectrum_data = {
                'note_conf': current_frame,           # shape (1, 88) snapshot
                'time': playhead_time,                 # seconds
                'sample_pos': sample_idx,              # absolute sample at playhead
                'window_start_idx': start_idx,         # absolute sample index where the CQT window starts
                'local_sample_pos': sample_idx - start_idx,  # sample offset into the local window
                'hop_length': hop_length,              # hop used for CQT/note_conf
                'full_note_conf': note_conf,           # (frames, 88) for interpolation
                'rms_db': rms_db,
                'silent': False,
            }

        except Exception as e:
            print(f"Error computing spectrum: {e}")
            self.spectrum_data = None

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
        self.update()

    def export_chords(self) -> list[dict]:
        """Return a plain list of chord segments suitable for JSON serialization."""
        try:
            return [
                {"start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "label": str(s.get("label", ""))}
                for s in (self.chords or [])
                if ("start" in s and "end" in s)
            ]
        except Exception:
            return []

    def import_chords(self, segments: list[dict] | None):
        """Load chord segments from JSON and trigger redraw."""
        if not segments:
            return
        try:
            self.set_chords([
                {"start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "label": str(s.get("label", ""))}
                for s in segments if ("start" in s and "end" in s)
            ])
        except Exception:
            # be defensive; ignore malformed entries
            pass

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

        # In visual focus (Focus or Solo), reserve space for spectrum/piano band
        if self._visual_focus_stem():
            ph = 0
            try:
                if hasattr(self, 'piano_roll_widget') and self.piano_roll_widget is not None and self.piano_roll_widget.isVisible():
                    ph = int(self.piano_roll_widget.height())
            except Exception:
                ph = 0
            if ph <= 0:
                ph = int(getattr(self, 'spectrum_band_height', 60))
            wf_h = max(10, h - ch_h - ph)
        else:
            wf_h = max(10, h - ch_h)
        return wf_h, ch_h

    def _layout_children(self):
        """Place child widgets (e.g., piano_roll_widget) based on current size/state.
        Never call this from paintEvent to avoid re-entrant paints.
        """
        try:
            if getattr(self, 'piano_roll_widget', None) is None:
                return

            # Use same layout math as paintEvent
            wf_h, ch_h = self._wf_geom()
            # Pick a practical height: use configured height or minHeight, whichever is larger
            desired_h = getattr(self, "spectrum_band_height", 90)
            min_h = int(self.piano_roll_widget.minimumHeight()) if hasattr(self.piano_roll_widget, "minimumHeight") else 0
            spectrum_h = max(int(desired_h), int(min_h)) if self._visual_focus_stem() else 0

            if self._visual_focus_stem():
                from PySide6.QtCore import QRect
                new_rect = QRect(0, wf_h, self.width(), spectrum_h)
                if self.piano_roll_widget.geometry() != new_rect:
                    self.piano_roll_widget.setGeometry(new_rect)
                    # Keep spectrum_band_height in sync with the real child height
                    try:
                        self.spectrum_band_height = int(self.piano_roll_widget.height())
                    except Exception:
                        pass
                if not self.piano_roll_widget.isVisible():
                    self.piano_roll_widget.show()
                self.piano_roll_widget.raise_()
            else:
                if self.piano_roll_widget.isVisible():
                    self.piano_roll_widget.hide()
        except Exception:
            # Never let layout crash painting/rendering
            pass

    def _time_in_chord_lane(self, pos: QtCore.QPoint, t0: float, t1: float, w: int, wf_h: int):
        """Return absolute time if pos is in the chord lane; else None."""
        x = int(pos.x())
        y = int(pos.y())
        lane_top = wf_h
        # Account for spectrum band in focus mode
        if self._visual_focus_stem():
            try:
                ph = int(self.piano_roll_widget.height()) if hasattr(self, 'piano_roll_widget') and self.piano_roll_widget.isVisible() else int(getattr(self, 'spectrum_band_height', 60))
            except Exception:
                ph = int(getattr(self, 'spectrum_band_height', 60))
            lane_top = wf_h + ph
        # Chord lane is a fixed 32 px tall; set bottom regardless of solo state
        CHORD_LANE_H = 32
        lane_bottom = lane_top + CHORD_LANE_H
        in_lane = not (y < (lane_top - 4) or y >= (lane_bottom + 4))
        if not in_lane:
            return None
        t = self._time_at_x(x, t0, t1, w)
        return t

    def _open_chord_context_at(self, pos: QtCore.QPoint, global_pos: QtCore.QPoint):
        """Try to open the chord context menu for a widget-relative pos; return True if handled."""
        t0, t1 = self._current_window()
        w = self.width(); h = self.height(); wf_h, _ = self._wf_geom()
        t_ch = self._time_in_chord_lane(pos, t0, t1, w, wf_h)
        if t_ch is None:
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
        except Exception as ex:
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
        try:
            if not self.player or width <= 2:
                return None, None
            y = getattr(self.player, 'y', None)
            sr = int(getattr(self.player, 'sr', 0) or 0)
            if y is None or sr <= 0:
                return None, None
            # Ensure ndarray and contiguous float32
            y = np.asarray(y)
            if y.ndim == 2:
                y = y.mean(axis=1)
            y = np.asarray(y, dtype=np.float32, order='C')
            N_total = int(y.shape[0])
            if N_total <= 1:
                return np.zeros(width), np.zeros(width)
            total_s = N_total / float(sr)
            # Overlap of requested window with actual audio
            ov_start = max(0.0, float(start_s))
            ov_end = min(float(end_s), total_s)
            mins = np.zeros(width, dtype=np.float32)
            maxs = np.zeros(width, dtype=np.float32)
            if ov_end <= ov_start:
                return mins, maxs
            # Map overlap to pixel columns
            span = float(end_s - start_s) if float(end_s - start_s) != 0.0 else 1.0
            x0 = int((ov_start - start_s) / span * width)
            x1 = int((ov_end   - start_s) / span * width)
            x0 = max(0, min(width, x0))
            x1 = max(x0 + 1, min(width, x1))
            # Slice audio safely
            a = max(0, min(N_total, int(ov_start * sr)))
            b = max(a + 1, min(N_total, int(ov_end   * sr)))
            seg = y[a:b]
            if seg.size <= 1:
                return mins, maxs
            # Downsample to the overlap width
            seg_w = x1 - x0
            if seg_w <= 0:
                return mins, maxs
            step = max(1, seg.size // seg_w)
            trimmed_len = (seg.size // step) * step
            if trimmed_len <= 0:
                return mins, maxs
            seg = np.ascontiguousarray(seg[:trimmed_len])
            reshaped = seg.reshape(-1, step)
            mins_seg = reshaped.min(axis=1)
            maxs_seg = reshaped.max(axis=1)
            L = min(seg_w, len(mins_seg))
            if L > 0:
                mins[x0:x0+L] = mins_seg[:L]
                maxs[x0:x0+L] = maxs_seg[:L]
            return mins, maxs
        except Exception:
            # Fail safe: draw nothing rather than crash
            return None, None

    def _get_stems_for_display(self):
        """Return a list of (name, array) stems to draw if available, else [].
        Accepts multiple player APIs: get_stems_arrays(), stems_arrays, stem_arrays, stems.
        """
        if not self.player:
            return []
        arrays = None
        # 1) Preferred accessor
        getter = getattr(self.player, 'get_stems_arrays', None)
        if callable(getter):
            try:
                arrays = getter()
            except Exception:
                arrays = None
        # 2) Common attribute names
        if arrays is None or not isinstance(arrays, dict) or not arrays:
            for attr in ('stems_arrays', 'stem_arrays', 'stems'):
                a = getattr(self.player, attr, None)
                if isinstance(a, dict) and a:
                    arrays = a
                    break
        if not isinstance(arrays, dict) or not arrays:
            return []
        # First, prefer explicit order provided by the player/Main (matches mixer rows)
        explicit = getattr(self.player, 'stem_order', None)
        if isinstance(explicit, (list, tuple)) and explicit:
            seen = set()
            ordered = [(k, arrays[k]) for k in explicit if k in arrays and not (k in seen or seen.add(k))]
            # Append any remaining keys in original dict order (preserves load order)
            for k in arrays.keys():
                if k not in seen:
                    ordered.append((k, arrays[k]))
                    seen.add(k)
            return ordered

        # Otherwise, fall back to Demucs canonical order
        try:
            demucs_order = order_stem_names("htdemucs_6s")
            demucs_ordered = [(k, arrays[k]) for k in demucs_order if k in arrays]
            remaining = [k for k in arrays.keys() if k not in {k for k, _ in demucs_ordered}]
            demucs_ordered += [(k, arrays[k]) for k in remaining]
            return demucs_ordered
        except Exception:
            return list(arrays.items())

    def _is_stem_muted(self, name: str) -> bool:
        if not self.player:
            return False
        # If player exposes a query method, use it
        q = getattr(self.player, 'is_stem_muted', None)
        if callable(q):
            try:
                return bool(q(name))
            except Exception:
                pass
        # Otherwise, check a common dict
        state = getattr(self.player, 'stem_mute', None)
        if isinstance(state, dict):
            # Try exact, lowercase, and common aliases
            if name in state:
                return bool(state[name])
            low = name.lower()
            if low in state:
                return bool(state[low])
        return False

    def _slice_minmax_for_array_cached(self, stem_name: str, y: np.ndarray, start_s: float, end_s: float, width: int, sr: int):
        """Cached version of _slice_minmax_for_array to avoid recomputation on scroll/zoom."""
        # Initialize cache if it doesn't exist (for existing instances)
        if not hasattr(self, '_waveform_cache'):
            self._waveform_cache = {}
        if not hasattr(self, '_cache_max_entries'):
            self._cache_max_entries = 10000  # Large cache for ultra-fine precision scrolling

        # Create cache key with very fine time precision for smooth scrolling
        # Use 0.001 second precision (1ms) for ultra-smooth scrolling
        cache_key = (stem_name, round(start_s, 3), round(end_s, 3), width)

        # Check cache first
        if cache_key in self._waveform_cache:
            return self._waveform_cache[cache_key]

        # Compute result
        result = self._slice_minmax_for_array(y, start_s, end_s, width, sr)

        # Cache the result (with size limit)
        if len(self._waveform_cache) >= self._cache_max_entries:
            # Remove oldest entries (simple FIFO) - remove fewer entries to reduce churn
            oldest_keys = list(self._waveform_cache.keys())[:100]
            for key in oldest_keys:
                del self._waveform_cache[key]

        self._waveform_cache[cache_key] = result
        return result

    def clear_waveform_cache(self):
        """Clear the waveform cache. Call this when stems change or audio is reloaded."""
        if not hasattr(self, '_waveform_cache'):
            self._waveform_cache = {}
        self._waveform_cache.clear()

        # Also clear pyramid cache
        if not hasattr(self, '_pyramid_cache'):
            self._pyramid_cache = {}
        self._pyramid_cache.clear()

    def _build_pyramid_for_stem(self, stem_name: str, arr: np.ndarray, sr: int):
        """Build multi-resolution pyramid for a stem. Expensive operation done once per stem."""
        if not hasattr(self, '_pyramid_cache'):
            self._pyramid_cache = {}
        if not hasattr(self, '_pyramid_levels'):
            self._pyramid_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

        # Convert to mono if needed
        y = np.asarray(arr)
        if y.ndim == 2:
            y = y.mean(axis=1)
        y = np.asarray(y, dtype=np.float32, order='C')

        N_total = int(y.shape[0])
        if N_total <= 1:
            return

        # Build pyramid at multiple resolution levels
        pyramid = {}
        for level in self._pyramid_levels:
            if level > N_total:
                break

            # Downsample by this factor
            step = level
            trimmed_len = (N_total // step) * step
            if trimmed_len <= 0:
                continue

            seg = np.ascontiguousarray(y[:trimmed_len])
            reshaped = seg.reshape(-1, step)
            mins_array = reshaped.min(axis=1)
            maxs_array = reshaped.max(axis=1)

            pyramid[level] = (mins_array, maxs_array, N_total)

        self._pyramid_cache[stem_name] = pyramid

    def _get_pyramid_slice(self, stem_name: str, start_s: float, end_s: float, width: int, sr: int):
        """Get waveform slice from pyramid - ultra-fast indexing operation."""
        if stem_name not in self._pyramid_cache:
            return None, None, 0, 0

        pyramid = self._pyramid_cache[stem_name]
        if not pyramid:
            return None, None, 0, 0

        # Choose appropriate pyramid level based on target resolution
        target_samples = width * 2  # Aim for 2x display width for good quality
        best_level = 1
        for level in self._pyramid_levels:
            if level <= target_samples:
                best_level = level
            else:
                break

        if best_level not in pyramid:
            return None, None, 0, 0

        mins_array, maxs_array, N_total = pyramid[best_level]

        # Calculate time range
        total_s = N_total / float(sr)
        ov_start = max(0.0, float(start_s))
        ov_end = min(float(end_s), total_s)

        if ov_end <= ov_start:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0, 0

        # Map to pyramid array indices
        span = float(end_s - start_s) if float(end_s - start_s) != 0.0 else 1.0
        x0 = int((ov_start - start_s) / span * width)
        x1 = int((ov_end - start_s) / span * width)
        x0 = max(0, min(width, x0))
        x1 = max(x0 + 1, min(width, x1))

        # Calculate which pyramid samples to use
        a = max(0, min(N_total, int(ov_start * sr)))
        b = max(a + 1, min(N_total, int(ov_end * sr)))

        # Map to pyramid array indices
        start_idx = a // best_level
        end_idx = min(len(mins_array), (b + best_level - 1) // best_level)

        if start_idx >= end_idx or start_idx >= len(mins_array):
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), x0, x0

        # Extract the slice - this is just array indexing, very fast!
        mins_slice = mins_array[start_idx:end_idx]
        maxs_slice = maxs_array[start_idx:end_idx]

        # Ensure we have enough samples to fill the display width
        # If we have fewer samples than needed, we need to stretch/interpolate
        target_length = x1 - x0
        if len(mins_slice) < target_length and len(mins_slice) > 0:
            # Simple linear interpolation to fill the gap
            indices = np.linspace(0, len(mins_slice) - 1, target_length)
            mins_slice = np.interp(indices, np.arange(len(mins_slice)), mins_slice)
            maxs_slice = np.interp(indices, np.arange(len(maxs_slice)), maxs_slice)
        elif len(mins_slice) > target_length:
            # Downsample if we have too many samples
            indices = np.linspace(0, len(mins_slice) - 1, target_length)
            mins_slice = np.interp(indices, np.arange(len(mins_slice)), mins_slice)
            maxs_slice = np.interp(indices, np.arange(len(maxs_slice)), maxs_slice)

        return mins_slice, maxs_slice, x0, x1

    def _slice_minmax_for_array(self, y: np.ndarray, start_s: float, end_s: float, width: int, sr: int):
        try:
            if width <= 2 or y is None or sr is None:
                return None, None, 0, 0
            sr = int(sr)
            if sr <= 0:
                return None, None, 0, 0
            y = np.asarray(y)
            if y.ndim == 2:
                y = y.mean(axis=1)
            y = np.asarray(y, dtype=np.float32, order='C')
            N_total = int(y.shape[0])
            if N_total <= 1:
                return None, None, 0, 0
            total_s = N_total / float(sr)
            ov_start = max(0.0, float(start_s))
            ov_end = min(float(end_s), total_s)
            if ov_end <= ov_start:
                return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0, 0
            span = float(end_s - start_s) if float(end_s - start_s) != 0.0 else 1.0
            x0 = int((ov_start - start_s) / span * width)
            x1 = int((ov_end   - start_s) / span * width)
            x0 = max(0, min(width, x0))
            x1 = max(x0 + 1, min(width, x1))
            a = max(0, min(N_total, int(ov_start * sr)))
            b = max(a + 1, min(N_total, int(ov_end   * sr)))
            seg = y[a:b]
            if seg.size <= 1:
                return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), x0, x0
            seg_w = x1 - x0
            if seg_w <= 0:
                return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), x0, x0
            step = max(1, seg.size // seg_w)
            trimmed_len = (seg.size // step) * step
            if trimmed_len <= 0:
                return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), x0, x0
            seg = np.ascontiguousarray(seg[:trimmed_len])
            reshaped = seg.reshape(-1, step)
            mins_seg = reshaped.min(axis=1)
            maxs_seg = reshaped.max(axis=1)
            return mins_seg, maxs_seg, x0, x1
        except Exception:
            return None, None, 0, 0

    def contextMenuEvent(self, e: QtGui.QContextMenuEvent):
        # Chord box context menu when right-clicking in the chord lane
        pos_local = e.pos()
        try:
            pos_global = e.globalPos()
        except Exception:
            pos_global = QtGui.QCursor.pos()
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

        # If horizontal motion dominates, PAN left/right smoothly and keep playhead centered
        if abs(dx) > abs(dy) and dx != 0:
            # Treat 120 units as one notch, but allow fine-grained trackpad deltas.
            notches = dx / 120.0
            # Smaller per-notch movement for smoother pan (5% of window per notch)
            step_per_notch = self.window_s * 0.05
            delta_s = step_per_notch * notches

            # Accumulate sub-threshold deltas to avoid stutter, then apply
            self._pan_accum_s += float(delta_s)
            apply_s = self._pan_accum_s
            # Keep residual tiny fraction to prevent drift due to rounding
            self._pan_accum_s -= apply_s

            if apply_s != 0.0:
                cur = self.player.position_seconds()
                dur = self.player.duration_seconds() if hasattr(self.player, 'duration_seconds') else (self.player.n / float(self.player.sr))
                t_new = max(0.0, min(dur, cur + apply_s))
                if self._freeze_window:
                    self.unfreeze_and_center()
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
            less_eq = [b for b in bars_sorted if b <= s]
            bs = max(less_eq) if less_eq else bars_sorted[0]
            # find bar at/after end
            great_eq = [b for b in bars_sorted if b >= e]
            be = min(great_eq) if great_eq else bars_sorted[-1]
            # ensure strictly increasing
            if be <= bs:
                # if collapsed due to missing bars, keep original
                bs, be = s, e
            out.append({'start': bs, 'end': be, 'label': lab})
        return out

    def _snap_and_split_to_bars(self, chords, beats, downbeats):
        """
        chords: list of {start, end, label}
        beats: list[float] beat times (ascending)
        downbeats: list[float] bar-start times (subset of beats)

        Returns list of {start, end, label}:
        - starts/ends snapped to nearest beat
        - segments split at every bar boundary
        - keeps multiple blocks if chords change within a bar
        """
        import numpy as np
        if not chords:
            return []
        # De-duplicate & sort grids
        bt = np.asarray(sorted(set(float(x) for x in (beats or []))), dtype=float)
        db = np.asarray(sorted(set(float(x) for x in (downbeats or []))), dtype=float)

        def _snap(t):
            if bt.size == 0:
                return float(t)
            j = int(np.clip(np.argmin(np.abs(bt - t)), 0, bt.size - 1))
            return float(bt[j])

        # 1) snap to beats
        snapped = []
        for d in chords:
            try:
                a_raw = float(d["start"]); b_raw = float(d["end"]); lab = d.get("label")
            except Exception:
                continue
            a = _snap(a_raw); b = _snap(b_raw)
            if bt.size > 1 and b <= a:
                # minimal nonzero span = one beat
                b = a + float(bt[1] - bt[0])
            if b > a:
                snapped.append({"start": a, "end": b, "label": lab})

        if not snapped:
            return []

        if db.size == 0:
            return snapped

        # 2) split at every bar boundary
        out = []
        for seg in snapped:
            a, b, lab = float(seg["start"]), float(seg["end"]), seg.get("label")
            # bar cuts strictly inside [a, b)
            cuts = [float(t) for t in db if a < float(t) < b]
            pts = [a] + cuts + [b]
            for u, v in zip(pts[:-1], pts[1:]):
                if (v - u) > 1e-6:
                    out.append({"start": u, "end": v, "label": lab})
        # Stable by time
        out.sort(key=lambda s: (s["start"], s["end"]))
        return out

    def _unique_by_span_no_crossbar(self, segs, bars):
        """Merge only truly identical neighbors inside the SAME bar."""
        if not segs:
            return []
        import bisect
        # Sort bars once (float) for stable bar index computations
        bars_sorted = sorted(float(b) for b in (bars or []))
        def bar_idx(t):
            return max(0, bisect.bisect_right(bars_sorted, float(t)) - 1) if bars_sorted else 0

        out = [dict(segs[0])]
        for s in segs[1:]:
            prev = out[-1]
            same_label = (s.get("label") == prev.get("label"))
            same_bar   = (bar_idx(s.get("start")) == bar_idx(prev.get("end", 0.0) - 1e-6))
            contiguous = abs(float(s.get("start", 0.0)) - float(prev.get("end", 0.0))) < 1e-6
            if same_label and same_bar and contiguous:
                prev["end"] = float(s.get("end", prev["end"]))
            else:
                out.append(dict(s))
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
            cuts = []
            last_cut = None
            for b in bars_sorted:
                if start < b < end:
                    if last_cut is None or abs(b - last_cut) > 1e-9:
                        cuts.append(b)
                        last_cut = b
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

    def _initialize_paint_attributes(self):
        """Initialize paint-related attributes if they don't exist."""
        if not hasattr(self, '_last_paint_time'):
            self._last_paint_time = 0
        if not hasattr(self, '_paint_throttle_ms'):
            self._paint_throttle_ms = 4  # Reduce to ~240fps for very smooth scrolling
        if not hasattr(self, '_waveform_cache'):
            self._waveform_cache = {}
        if not hasattr(self, '_cache_max_entries'):
            self._cache_max_entries = 10000  # Large cache for ultra-fine precision scrolling
        if not hasattr(self, '_pyramid_cache'):
            self._pyramid_cache = {}
        if not hasattr(self, '_pyramid_levels'):
            self._pyramid_levels = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    def _handle_piano_roll_widget(self):
        """Handle piano roll widget creation and data updates in focus mode."""
        if self._visual_focus_stem():
            # Refresh spectrum/note data right before drawing
            try:
                self._compute_spectrum_at_playhead()
            except Exception:
                logging.exception("_compute_spectrum_at_playhead failed")
            # Ensure child exists even if constructed in a different init path
            if not hasattr(self, 'piano_roll_widget') or self.piano_roll_widget is None:
                from piano_widget import PianoRollWidget
                self.piano_roll_widget = PianoRollWidget(self)
                self.piano_roll_widget.hide()
            # Defer child geometry to the layout pass (never mutate geometry in paint)
            try:
                self._layout_children()
            except Exception:
                pass

            sd = getattr(self, 'spectrum_data', None)
            if sd and 'full_note_conf' in sd and 'sample_pos' in sd:
                note = sd['full_note_conf']
                hop = int(sd.get('hop_length', 512))
                sr = getattr(self.player, 'sr', 44100)
                self.piano_roll_widget.set_data(note, hop, sr)
                local_pos = int(sd.get('local_sample_pos', sd['sample_pos'] - sd.get('window_start_idx', 0)))
                self.piano_roll_widget.update_playhead(local_pos)
            elif sd and 'note_conf' in sd:
                self.piano_roll_widget.set_data(sd['note_conf'], 1, getattr(self.player, 'sr', 44100))
                self.piano_roll_widget.update_playhead(0)
            else:
                self.piano_roll_widget.clear_data()
        else:
            # Let the layout pass hide/show; do not change visibility in paint
            try:
                self._layout_children()
            except Exception:
                pass

    def _get_chord_area_top(self, wf_h: int) -> int:
        """Calculate the top position of the chord area, accounting for spectrum band in focus mode."""
        chord_top = wf_h
        if self._visual_focus_stem():
            ph = 0
            try:
                if hasattr(self, 'piano_roll_widget') and self.piano_roll_widget is not None and self.piano_roll_widget.isVisible():
                    ph = int(self.piano_roll_widget.height())
            except Exception:
                ph = 0
            if ph <= 0:
                ph = int(getattr(self, 'spectrum_band_height', 60))
            chord_top = wf_h + ph
        return chord_top

    def _draw_waveforms(self, p: QtGui.QPainter, w: int, wf_h: int, t0: float, t1: float, rel_t0: float, rel_t1: float):
        """Draw waveforms, grids, loops, and playhead in the waveform area."""
        # Constrain subsequent waveform drawings (grid, beats, flags, loop fills, waveform, playhead)
        p.save()
        p.setClipRect(0, 0, w, wf_h)

        try:
            # Draw waveform(s) and playhead below the top strip
            body_top = min(wf_h, self.FLAG_STRIP)
            body_h = max(1, wf_h - body_top)
            have_beats = bool(self.beats)
            have_bars = bool(self.bars)
            long_len = min(12, wf_h)
            short_len = min(6, wf_h)

            focus_name = self._visual_focus_stem()
            in_focus_mode = False
            if focus_name:
                # Solo mode: show only the soloed stem as a full-width waveform
                stems = self._get_stems_for_display()
                solo_stem_data = None
                for stem_name, arr in stems:
                    if stem_name == focus_name:
                        solo_stem_data = (stem_name, arr)
                        break

                if solo_stem_data:
                    in_focus_mode = True
                    # Draw soloed stem as full-width waveform (like combined waveform)
                    stem_name, arr = solo_stem_data
                    sr = getattr(self.player, 'sr', None)
                    if sr is not None:
                        if stem_name not in self._pyramid_cache:
                            self._build_pyramid_for_stem(stem_name, arr, sr)

                        # Use pyramid for ultra-fast rendering
                        if (hasattr(self, '_pyramid_cache') and
                            stem_name in self._pyramid_cache and
                            self._pyramid_cache[stem_name]):
                            mins, maxs, x0_seg, x1_seg = self._get_pyramid_slice(stem_name, t0, t1, w, sr)
                            if mins is None or len(mins) == 0:
                                mins, maxs, x0_seg, x1_seg = self._slice_minmax_for_array_cached(stem_name, arr, t0, t1, w, sr)
                        else:
                            mins, maxs, x0_seg, x1_seg = self._slice_minmax_for_array_cached(stem_name, arr, t0, t1, w, sr)

                        if mins is not None and maxs is not None:
                            mid = body_top + body_h // 2
                            amp = max(1, int(body_h * 0.45))
                            # Draw as filled waveform like combined view
                            p.setPen(QtGui.QPen(QtGui.QColor(100, 150, 255)))
                            for i in range(len(mins)):
                                x = x0_seg + i
                                if 0 <= x < w:
                                    y1 = mid - int(mins[i] * amp)
                                    y2 = mid - int(maxs[i] * amp)
                                    p.drawLine(x, y1, x, y2)
                    # Skip the rest of the waveform drawing logic in focus mode
                    stems = []
                else:
                    stems = self._get_stems_for_display() if getattr(self, "show_stems", True) else []

                # Draw spectrum band in focus mode (moved to after chord background)
                if in_focus_mode:
                    self._compute_spectrum_at_playhead()
                    spectrum_h = getattr(self, 'spectrum_band_height', 80)
                    # Note: piano roll will be drawn later after chord background
            else:
                # Normal mode: show stems or combined waveform
                stems = self._get_stems_for_display() if getattr(self, "show_stems", True) else []

            # Build pyramids for stems if not already built
            sr = getattr(self.player, 'sr', None)
            if stems and sr is not None:
                for stem_name, arr in stems:
                    if stem_name not in self._pyramid_cache:
                        self._build_pyramid_for_stem(stem_name, arr, sr)

            if stems:
                self._draw_stem_waveforms(p, stems, w, body_top, body_h, t0, t1, sr)
            elif not in_focus_mode:
                self._draw_combined_waveform(p, w, body_top, body_h, t0, t1)

            # Draw grids, loops, and other overlays
            self._draw_waveform_overlays(p, w, wf_h, t0, t1, have_bars, have_beats, long_len, short_len, rel_t0, rel_t1)

            # Draw playhead
            self._draw_playhead(p, w, wf_h, t0, t1)

        except Exception:
            pass

        # End waveform layer; ensure no brush/pen bleed into chord lane
        p.restore()
        p.setBrush(QtCore.Qt.NoBrush)
        p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 220)))

    def _draw_stem_waveforms(self, p: QtGui.QPainter, stems: list, w: int, body_top: int, body_h: int, t0: float, t1: float, sr: int):
        """Draw individual stem waveforms in rows."""
        # Divide body area into rows for each stem
        rows = len(stems)
        row_gap = 2
        row_h = max(14, int((body_h - (rows-1)*row_gap) / max(1, rows)))
        y_cursor = body_top
        for (stem_name, arr) in stems:
            # Background (dim if muted)
            muted = self._is_stem_muted(stem_name)
            bg = QtGui.QColor(40, 40, 40) if muted else QtGui.QColor(28, 28, 28)
            p.fillRect(0, y_cursor, w, row_h, bg)
            # Border
            p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0)))
            p.drawRect(0, y_cursor, max(1, w-1), max(1, row_h-1))
            # Envelope - use pyramid for ultra-fast rendering
            if (hasattr(self, '_pyramid_cache') and
                stem_name in self._pyramid_cache and
                self._pyramid_cache[stem_name]):
                # Use pyramid for ultra-fast scrolling
                mins, maxs, x0_seg, x1_seg = self._get_pyramid_slice(stem_name, t0, t1, w, sr)
                # Fallback to cached computation if pyramid fails
                if mins is None or len(mins) == 0:
                    mins, maxs, x0_seg, x1_seg = self._slice_minmax_for_array_cached(stem_name, arr, t0, t1, w, sr)
            else:
                # Fallback to cached computation
                mins, maxs, x0_seg, x1_seg = self._slice_minmax_for_array_cached(stem_name, arr, t0, t1, w, sr)
            if mins is not None and maxs is not None:
                mid = y_cursor + row_h // 2
                amp = max(1, int(row_h * 0.45))
                # Envelope color: lighter when muted
                if muted:
                    color = QtGui.QColor(170, 190, 210)  # light grey-blue for muted
                else:
                    color = QtGui.QColor(0, 180, 255)   # bright cyan for active

                L = min(len(mins), max(0, x1_seg - x0_seg), w - x0_seg)
                if L > 0:
                    # Create filled polygon for smooth waveform display
                    points = []
                    # Top envelope (maxs)
                    for i in range(L):
                        x = x0_seg + i
                        y = mid - int(maxs[i] * amp)
                        points.append(QtCore.QPoint(x, y))
                    # Bottom envelope (mins) in reverse
                    for i in range(L-1, -1, -1):
                        x = x0_seg + i
                        y = mid - int(mins[i] * amp)
                        points.append(QtCore.QPoint(x, y))

                    if len(points) > 2:
                        # Set brush and pen for this specific stem
                        p.setBrush(QtGui.QBrush(color))
                        p.setPen(QtGui.QPen(color))
                        polygon = QtGui.QPolygon(points)
                        p.drawPolygon(polygon)  # This will now be filled due to the brush

                        # Reset brush to transparent to avoid affecting other drawing
                        p.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            # Stem label at top-left inside the row
            p.setPen(QtGui.QPen(QtGui.QColor(210, 220, 230)))
            fm = p.fontMetrics()
            text_y = y_cursor + fm.ascent() + 2  # small top padding
            p.drawText(6, text_y, str(stem_name))
            y_cursor += row_h + row_gap

    def _draw_combined_waveform(self, p: QtGui.QPainter, w: int, body_top: int, body_h: int, t0: float, t1: float):
        """Draw single mixed waveform from the player's audio buffer."""
        mins, maxs = self._mono_slice_minmax(t0, t1, w)
        if mins is not None:
            mid = body_top + body_h // 2
            p.setPen(QtGui.QPen(QtGui.QColor(0, 180, 255)))
            for x, (mn, mx) in enumerate(zip(mins, maxs)):
                y1 = mid - int(mx * (body_h * 0.45))
                y2 = mid - int(mn * (body_h * 0.45))
                if y1 > y2:
                    y1, y2 = y2, y1
                p.drawLine(x, y1, x, y2)

    def _draw_waveform_overlays(self, p: QtGui.QPainter, w: int, wf_h: int, t0: float, t1: float,
                               have_bars: bool, have_beats: bool, long_len: int, short_len: int,
                               rel_t0: float, rel_t1: float):
        """Draw grids, loops, and other overlays on the waveform."""
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

        # Loop overlay (if available)
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
            # Flags at top strip for the active loop
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
                # Start flag
                flag_poly = QtGui.QPolygon([
                    QtCore.QPoint(x0, tip_y),
                    QtCore.QPoint(x0 - self.FLAG_W//2, base_y),
                    QtCore.QPoint(x0 + self.FLAG_W//2, base_y),
                ])
                p.drawPolygon(flag_poly)
                # End flag
                flag_poly_b = QtGui.QPolygon([
                    QtCore.QPoint(x1, tip_y),
                    QtCore.QPoint(x1 - self.FLAG_W//2, base_y),
                    QtCore.QPoint(x1 + self.FLAG_W//2, base_y),
                ])
                p.drawPolygon(flag_poly_b)
                # Label (if any): near start flag; if start flag is offscreen but the loop overlaps, pin to left margin
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

        # Draw top ruler (beats or time)
        self._draw_top_ruler(p, w, wf_h, t0, t1, rel_t0, rel_t1, have_beats, have_bars, long_len, short_len)

    def _draw_top_ruler(self, p: QtGui.QPainter, w: int, wf_h: int, t0: float, t1: float,
                       rel_t0: float, rel_t1: float, have_beats: bool, have_bars: bool,
                       long_len: int, short_len: int):
        """Draw the top ruler (beats or time)."""
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
                # Back-fill slightly beyond window to ensure coverage at left edge
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

    def _draw_playhead(self, p: QtGui.QPainter, w: int, wf_h: int, t0: float, t1: float):
        """Draw the playhead."""
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

    def _draw_chord_lane(self, p: QtGui.QPainter, w: int, wf_h: int, ch_h: int, t0: float, t1: float, rel_t0: float, rel_t1: float):
        """Draw the chord lane with chord rectangles and labels."""
        if self.chords:
            # Revert: only split at bar boundaries; no beat snapping in paint path.
            base = list(self.chords)
            split_bars = self._split_segments_at_bars(base, self.bars)
            segs_to_draw = self._ensure_leading_bar(split_bars, self.bars)
            segs_to_draw = self._unique_by_span(segs_to_draw)

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
                # Position chord rectangles below spectrum band in focus mode (use real child height)
                chord_top = wf_h
                if self._visual_focus_stem():
                    ph = 0
                    try:
                        if hasattr(self, 'piano_roll_widget') and self.piano_roll_widget is not None and self.piano_roll_widget.isVisible():
                            ph = int(self.piano_roll_widget.height())
                    except Exception:
                        ph = 0
                    if ph <= 0:
                        ph = int(getattr(self, 'spectrum_band_height', 60))
                    chord_top = wf_h + ph
                rect = QtCore.QRect(x0, chord_top, max(1, x1 - x0), ch_h)
                p.setBrush(QtGui.QColor(50, 80, 110))   # fill color
                p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20)))  # outline
                p.drawRoundedRect(rect, 6, 6)
                # Draw a dashed beat grid inside the chord box if beats are available
                if self.beats:
                    try:
                        # Only draw lines for beats strictly within [a, b]
                        beat_arr = [float(bt) for bt in self.beats]
                        for beat in beat_arr:
                            if a < beat < b:
                                # Map beat time to x in current rect
                                beat_rel = beat - self.origin
                                x_beat = int((beat_rel - rel_t0) / (rel_t1 - rel_t0) * w)
                                # Clamp to inside the chord rect
                                if x0 < x_beat < x1:
                                    pen_beat = QtGui.QPen(QtGui.QColor(200, 200, 200, 120))
                                    pen_beat.setStyle(QtCore.Qt.DashLine)
                                    p.setPen(pen_beat)
                                    p.drawLine(x_beat, chord_top, x_beat, chord_top + ch_h)
                    except Exception:
                        pass
                # Chord label (drawn last, overlays above grid)
                p.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230)))
                p.drawText(rect.adjusted(4, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, seg['label'])

    def paintEvent(self, e: QtGui.QPaintEvent):
        """Main paint event handler - coordinates all drawing operations."""
        self._initialize_paint_attributes()

        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing, False)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, False)  # Disable for better performance

        w = self.width()
        h = self.height()
        if w < 10 or h < 10:
            return

        wf_h, ch_h = self._wf_geom()
        t0, t1 = self._current_window()
        rel_t0 = t0 - self.origin
        rel_t1 = t1 - self.origin

        # Fill background
        p.fillRect(0, 0, w, wf_h, QtGui.QColor(20, 20, 20))

        # Handle piano roll widget in focus mode
        self._handle_piano_roll_widget()

        # Fill chord area background
        chord_top = self._get_chord_area_top(wf_h)
        p.fillRect(0, chord_top, w, ch_h, QtGui.QColor(28, 28, 28))

        # Draw waveforms and related elements
        self._draw_waveforms(p, w, wf_h, t0, t1, rel_t0, rel_t1)

        # Draw chord lane
        self._draw_chord_lane(p, w, wf_h, ch_h, t0, t1, rel_t0, rel_t1)



class ChordWorker(QThread):
    done = Signal(list)
    status = Signal(str)
    demucs_line = Signal(str)  # forward Demucs logs to LogDock
    stems_ready = Signal(str)  # emits the stem **leaf** directory path
    def __init__(self, path: str, style: str = 'rock_pop'):
        super().__init__()
        self.path = path
        self.style = style
        self.key_hint = None
        self.backend = 'internal'

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
        # If Main gave us a key_hint, propagate
        if getattr(self, "key_hint", None) and not kwargs.get("key_hint"):
            kwargs["key_hint"] = dict(self.key_hint)

        # If caller didn't provide beat info, compute it now.
        if not kwargs.get("beats"):
            bd = estimate_beats(audio_path, sr=sr, hop=512)
            kwargs["beats"] = bd.get("beats", [])
            kwargs["downbeats"] = bd.get("downbeats", [])
            kwargs["beat_strengths"] = bd.get("beat_strengths", [])

        # Always-on stems: prefer cached; run Demucs only if missing or forced
        model = "htdemucs_6s"
        out_dir = self._song_cache_dir(audio_path)
        leaf = self._stem_leaf_dir(audio_path, out_dir, model)
        maybe_pre = None

        # Only force a fresh Demucs render if Main requested recompute
        try:
            force = bool(getattr(self.parent(), "_force_stems_recompute", False))
        except Exception:
            force = False

        # Determine if all six expected stems already exist in the leaf
        expected = {"bass", "drums", "guitar", "piano", "other", "vocals"}
        have_all = leaf.is_dir() and all((leaf / f"{n}.wav").exists() for n in expected)

        # If cached stems are present and not forcing, load them and skip Demucs
        if have_all and not force:
            try:
                self._log(f"Reusing cached stems: {leaf}")
                maybe_pre = load_stem_arrays(leaf)
                if isinstance(maybe_pre, dict) and maybe_pre:
                    self.stems_ready.emit(str(leaf))
            except Exception as e:
                self._log_exc("Load cached stems", e)
                maybe_pre = None
        else:
            # Optional pre-clean if forcing
            if force and leaf.exists():
                try:
                    self._log(f"Removing existing stems leaf to force render: {leaf}")
                    shutil.rmtree(leaf)
                except Exception as e:
                    self._log_exc("Pre-clean leaf dir", e)
            # Run Demucs and stream logs; then wait for files
            self._log(f"[ENTRY] estimate_chords: Demucs run (force={force}) for {audio_path}")
            self._log(f"Demucs starting: {model} → {out_dir}")
            try:
                worker = DemucsWorker(audio_path, str(out_dir), model)
                worker.line.connect(lambda s: self.demucs_line.emit(s))
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

        if not kwargs.get("stems") and isinstance(maybe_pre, dict) and maybe_pre:
            kwargs["stems"] = maybe_pre
        if not kwargs.get("stems"):
            try:
                self.status.emit("Proceeding without stems (timeout or missing files).")
            except Exception:
                pass

        beats_kw = kwargs.get("beats")
        down_kw = kwargs.get("downbeats")
        bs_kw = kwargs.get("beat_strengths")
        stems_kw = kwargs.get("stems")
        backend = getattr(self, "backend", getattr(self, "chord_backend", "internal"))
        backend = (backend or "internal").lower()
        self._log(f"Chord backend: {backend}")

        if backend == "chordino":
            return _chordino(
                audio_path,
                beats=kwargs.get("beats"),
                downbeats=kwargs.get("downbeats"),
            )

        if stems_kw and beats_kw:
            self._log("Using stem-aware chord detector")
            return _stem_aware(audio_path, sr=sr, hop=hop,
                            beats=beats_kw, downbeats=down_kw, beat_strengths=bs_kw,
                            stems=stems_kw,
                            key_hint=kwargs.get("key_hint"),
                            log_fn=lambda m: self._log(m),
                            style=getattr(self, "style", "rock_pop"))
        else:
            self._log("Using fast chord detector (no stems)")
            return _fast_est(audio_path, sr=sr, hop=hop,
                            beats=beats_kw, downbeats=down_kw, beat_strengths=bs_kw,
                            stems=stems_kw,
                            key_hint=kwargs.get("key_hint"),
                            log_fn=lambda m: self._log(m),
                            style=getattr(self, "style", "rock_pop"))

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

        # Track whether we have content to determine visibility
        self._has_content = False
        # Start hidden - will show when content is added
        self.hide()

    def clear(self):
        self.view.clear()
        self._has_content = False
        self.hide()

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
            # Show the dock when we have content to display
            if not self._has_content:
                self._has_content = True
                self.show()
                self.raise_()
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
            # Show the dock when we have content to display
            if not self._has_content:
                self._has_content = True
                self.show()
                self.raise_()
            QtCore.QMetaObject.invokeMethod(
                self.view,
                "appendPlainText",
                Qt.QueuedConnection,
                QtCore.Q_ARG(str, line.rstrip("\n"))
            )

def _probe_chordino() -> bool:
    try:
        import vamp, numpy as np
        # Quick check via plugin listing
        ids = set(vamp.list_plugins())  # e.g. {'nnls-chroma:chordino', 'qm-vamp-plugins:qm-keydetector', ...}
        if "nnls-chroma:chordino" not in ids:
            return False
        # Sanity-run a tiny collect so we know it actually loads
        sr = 44100
        y = np.zeros(sr, dtype=np.float32)  # 1 second of silence
        _ = vamp.collect(y, sr, "nnls-chroma:chordino")
        return True
    except Exception:
        return False

def _probe_qm_key() -> bool:
    try:
        import vamp, numpy as np
        if "qm-vamp-plugins:qm-keydetector" not in set(vamp.list_plugins()):
            return False
        sr = 44100
        y = np.zeros(sr, dtype=np.float32)
        _ = vamp.collect(y, sr, "qm-vamp-plugins:qm-keydetector")
        return True
    except Exception:
        return False

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


    def _action_reextract_stems(self):
        """Force Demucs re-extraction, then recompute beats+chords using new stems."""
        if not getattr(self, 'current_path', None):
            self.statusBar().showMessage("No audio loaded.", 1500)
            return
        # Cancel any running chord worker cleanly
        try:
            if getattr(self, 'chord_worker', None) and self.chord_worker.isRunning():
                self.chord_worker.requestInterruption(); self.chord_worker.quit(); self.chord_worker.wait()
        except Exception:
            pass
        # Force re-separation for this run
        try:
            self._force_stems_recompute = True
        except Exception:
            pass
        # Make sure Demucs log dock is visible
        try:
            if not hasattr(self, 'log_dock') or self.log_dock is None:
                self.log_dock = LogDock(parent=self)
                self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
            self.log_dock.setWindowTitle('Demucs Log')
            self.log_dock.clear()  # clear() will hide the dock if empty
        except Exception:
            pass
        # Explicit recompute: unlock chords so auto analysis can overwrite
        self.chords_locked = False
        # Clear previous analysis; force fresh beats+key+chords and stems reload
        self._clear_cached_analysis(keep_stems=False)
        self.statusBar().showMessage("Recomputing: beats + key + chords…")
        # Recompute visible beats/bars for the ruler in parallel
        try:
            self.populate_beats_async(self.current_path)
            self.populate_key_async(self.current_path)
            self.populate_chords_async(self.current_path, force=True)
        except Exception:
            pass
        # Kick the integrated pipeline (ChordWorker waits for stems, then estimates chords)
        self.start_chord_analysis(self.current_path, force=False)

    def _action_recompute_analysis(self):
        """Recompute beats+chords using existing stems (do not re-run Demucs)."""
        if not getattr(self, 'current_path', None):
            self.statusBar().showMessage("No audio loaded.", 1500)
            return
        # Cancel any running chord worker cleanly
        try:
            if getattr(self, 'chord_worker', None) and self.chord_worker.isRunning():
                self.chord_worker.requestInterruption(); self.chord_worker.quit(); self.chord_worker.wait()
        except Exception:
            pass
        # Do NOT force stems; reuse cached stems
        try:
            self._force_stems_recompute = False
        except Exception:
            pass
        # Explicit recompute: unlock chords so auto analysis can overwrite
        self.chords_locked = False
        # Clear previous analysis but keep current stems
        self._clear_cached_analysis(keep_stems=True)
        self.statusBar().showMessage("Recomputing: beats + key + chords…")
        # Recompute visible beats/bars for the ruler in parallel
        try:
            self.populate_beats_async(self.current_path)
            self.populate_key_async(self.current_path)
            self.populate_chords_async(self.current_path, force=True)
        except Exception:
            pass

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
        if self.player:
            try:
                if hasattr(self.player, 'set_stems_arrays'):
                    self.player.set_stems_arrays(arrays, stem_dir)
                # Also expose a public attribute for the view to read
                setattr(self.player, 'stems_arrays', arrays)
                try:
                    mute_map = getattr(self.player, 'stem_mute', None)
                    if not isinstance(mute_map, dict):
                        mute_map = {}
                    for k in arrays.keys():
                        mute_map.setdefault(k, False)  # default not muted
                    setattr(self.player, 'stem_mute', mute_map)
                except Exception:
                    pass

                if hasattr(self.player, 'use_stems_only'):
                    self.player.use_stems_only(True)
            except Exception:
                # Fallback: still expose dict for the view
                try:
                    setattr(self.player, 'stems_arrays', arrays)
                except Exception:
                    pass
        # Clear waveform cache and nudge the view to repaint using stems
        try:
            if hasattr(self, 'wave') and self.wave is not None:
                self.wave.clear_waveform_cache()
                self.wave.update()
        except Exception:
            pass
        if hasattr(self, '_clear_stem_rows'):
            self._clear_stem_rows()
        # Display stems in fixed preferred order (case-insensitive), then any extras
        preferred_order = ["vocals", "drums", "bass", "guitar", "piano", "other"]
        order_list: list[str] = []
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
                    order_list.append(match)
                    used.add(match)
                    break

        # Add any extras not covered above, in stable order from arrays
        for k in arrays.keys():
            if k not in used:
                self._add_stem_row(k, arrays[k])
                order_list.append(k)
        try:
            self.stem_order = list(order_list)
            if self.player is not None:
                setattr(self.player, 'stem_order', list(order_list))
        except Exception:
            pass
        try:
            if hasattr(self, 'wave') and self.wave is not None:
                self.wave.update()
        except Exception:
            pass

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
                    w.wait()
            except Exception:
                pass

    def __init__(self):
        super().__init__()
        self.setWindowTitle("MusicPractice — Minimal")
        self.resize(960, 600)
        self.setAcceptDrops(True)
        self.settings = QSettings("MusicPractice", "MusicPractice")

        # Initialize note index for transposition
        self.NOTE_INDEX.update({n:i for i,n in enumerate(self.NOTE_RING_FLATS)})

        self.player: LoopPlayer | None = None
        self.current_path: str | None = None
        # Saved analysis/session state
        self.last_tempo: float | None = None
        self.last_beats: list[float] = []
        self.last_bars: list[float] = []
        self.last_key: dict | None = None
        self.last_chords: list[dict] = []
        self.chords_locked: bool = False
        self.A = None  # no active loop by default
        self.B = None
        # Saved loops model
        self.saved_loops: list[dict] = []   # {id:int, a:float, b:float, label:str}
        self._loop_id_seq = 1
        self._active_saved_loop_id: int | None = None
        self.beat_worker = None
        self.chord_worker = None
        self.key_worker = None
        self.log_dock = None  # created on demand when separating stems
        self._force_stems_recompute = False
        self.stem_order = []  # ordered list of stem names as shown in the mixer

        # === UI ===
        self.rate_spin = QtWidgets.QDoubleSpinBox()
        self.rate_spin.setRange(0.5, 1.5)
        self.rate_spin.setSingleStep(0.05)
        self.rate_spin.setValue(1.0)
        self.rate_label = QtWidgets.QLabel("Rate: 1.00x")

        # Pitch controls
        self.pitch_spin = QtWidgets.QSpinBox()
        self.pitch_spin.setRange(-12, 12)
        self.pitch_spin.setValue(0)
        self.pitch_spin.setSuffix(" st")
        self.pitch_spin.setToolTip("Pitch shift in semitones (-12 to +12)")
        self.pitch_spin.setMinimumWidth(80)
        self.pitch_label = QtWidgets.QLabel("Pitch:")

        # Pitch reset button
        self.pitch_reset_btn = QtWidgets.QPushButton("Reset")
        self.pitch_reset_btn.setToolTip("Reset pitch to original (0 semitones)")
        self.pitch_reset_btn.setMaximumWidth(60)
        self.pitch_reset_btn.clicked.connect(self._reset_pitch)

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
            # Create File menu first, if not already present
            fileMenu = mb.addMenu("File")
            fileMenu.addAction(self.act_open)
            fileMenu.addSeparator()
            fileMenu.addAction(QAction("Save Session", self, shortcut=QKeySequence.StandardKey.Save, triggered=self.save_session))
            fileMenu.addAction(QAction("Load Session", self, shortcut=QKeySequence("Ctrl+Shift+O"), triggered=self.load_session))
            # ---- View menu (after File) ----
            view_menu = self.menuBar().addMenu("View")
            view_group = QtGui.QActionGroup(self)
            view_group.setExclusive(True)

            self.act_view_stems = QtGui.QAction("Show Stem Waveforms", self)
            self.act_view_stems.setCheckable(True)
            self.act_view_combined = QtGui.QAction("Show Combined Waveform", self)
            self.act_view_combined.setCheckable(True)
            # Default to combined waveform on startup
            current_show_stems = False if getattr(self, 'wave', None) is None else bool(self.wave.show_stems)
            # Force default to combined = not show_stems
            if getattr(self, 'wave', None) is not None:
                self.wave.show_stems = False
                current_show_stems = False
            self.act_view_stems.setChecked(current_show_stems)
            self.act_view_combined.setChecked(not current_show_stems)
            self.act_view_stems.triggered.connect(lambda: self._set_waveform_view_mode(True))
            view_group.addAction(self.act_view_stems)

            self.act_view_combined.triggered.connect(lambda: self._set_waveform_view_mode(False))
            view_group.addAction(self.act_view_combined)

            view_menu.addAction(self.act_view_stems)
            view_menu.addAction(self.act_view_combined)

            # ---- Analysis menu (after File) ----
            analysis_menu = None
            for a in self.menuBar().actions():
                if a.text() == "Analysis":
                    analysis_menu = a.menu()
                    break
            if analysis_menu is None:
                analysis_menu = self.menuBar().addMenu("Analysis")
            backend_menu = analysis_menu.addMenu("Chord Backend")
            ag = QActionGroup(self); ag.setExclusive(True)

            # --- Backend (Internal vs Chordino) ---
            self.act_backend_internal = QAction("Internal", self, checkable=True)
            self.act_backend_chordino = QAction("Chordino", self, checkable=True)

            ag = QActionGroup(self)
            ag.setExclusive(True)
            ag.addAction(self.act_backend_internal)
            ag.addAction(self.act_backend_chordino)
            backend_menu.addAction(self.act_backend_internal)
            backend_menu.addAction(self.act_backend_chordino)

            def _have_chordino():
                try:
                    return bool(_probe_chordino())
                except Exception:
                    return False

            def _set_chord_backend(name: str):
                name = (name or "internal").lower()
                if name not in ("internal", "chordino"):
                    name = "internal"
                # One place of truth
                self.chord_backend = name
                self.backend = name  # keep legacy attr aligned
                # reflect in menu
                try:
                    self.act_backend_internal.setChecked(name == "internal")
                    self.act_backend_chordino.setChecked(name == "chordino")
                except Exception:
                    pass
                # show/hide style menu
                try:
                    self._update_style_menu_visibility()
                except Exception:
                    pass

            self._set_chord_backend = _set_chord_backend  # bind for reuse

            have = _have_chordino()
            self._set_chord_backend('chordino' if have else 'internal')

            self.act_backend_internal.triggered.connect(lambda _=False: self._on_backend_changed("internal"))
            self.act_backend_chordino.triggered.connect(lambda _=False: self._on_backend_changed("chordino"))

            self.actReextractStems = QAction("Re-extract Stems", self)
            self.actReextractStems.setStatusTip("Run Demucs again and refresh stems before chord detection")
            self.actReextractStems.triggered.connect(self._action_reextract_stems)

            self.actRecomputeAnalysis = QAction("Recompute Analysis", self)
            self.actRecomputeAnalysis.setStatusTip("Recompute beats & chords using current stems")
            self.actRecomputeAnalysis.triggered.connect(self._action_recompute_analysis)

            analysis_menu.addAction(self.actReextractStems)
            analysis_menu.addAction(self.actRecomputeAnalysis)

            # Default analysis style
            if not hasattr(self, 'analysis_style'):
                self.analysis_style = 'rock_pop'

            # Ensure an Analysis menu exists
            try:
                analysis_menu = None
                for w in self.menuBar().findChildren(QtWidgets.QMenu):
                    if w.title() == 'Analysis':
                        analysis_menu = w
                        break
                if analysis_menu is None:
                    analysis_menu = self.menuBar().addMenu('Analysis')

                # Add Style submenu
                self.style_menu = None
                for w in analysis_menu.findChildren(QtWidgets.QMenu):
                    if w.title() == 'Style':
                        self.style_menu = w
                        break
                if self.style_menu is None:
                    self.style_menu = analysis_menu.addMenu('Style')

                # Exclusive action group
                self._style_group = QtGui.QActionGroup(self)
                self._style_group.setExclusive(True)

                styles = [
                    ('Rock/Pop', 'rock_pop'),
                    ('Blues', 'blues'),
                    ('Reggae', 'reggae'),
                    ('Jazz', 'jazz'),
                ]
                self._style_actions = {}
                for text, key in styles:
                    act = QtGui.QAction(text, self, checkable=True)
                    act.setChecked(key == self.analysis_style)
                    act.triggered.connect(lambda checked, k=key: self._set_analysis_style(k))
                    self._style_group.addAction(act)
                    self.style_menu.addAction(act)
                    self._style_actions[key] = act

                if self.analysis_style not in self._style_actions:
                    self.analysis_style = 'rock_pop'
                    self._style_actions['rock_pop'].setChecked(True)
                self._on_backend_changed(self.chord_backend)
            except Exception:
                pass
        except Exception:
            pass

        # Populate toolbar
        self.toolbar.addAction(self.act_open)
        self.toolbar.addSeparator()
        # --- Transport toolbar (universal media symbols) ---
        self.transport_tb = self.addToolBar("Transport")
        self.transport_tb.setObjectName("TransportToolbar")
        self.transport_tb.setMovable(False)
        self.transport_tb.setFloatable(False)
        self.transport_tb.setIconSize(QtCore.QSize(18, 18))
        self.transport_tb.setToolButtonStyle(Qt.ToolButtonIconOnly)

        style = self.style()
        ico_start = style.standardIcon(QtWidgets.QStyle.SP_MediaSkipBackward)
        ico_play  = style.standardIcon(QtWidgets.QStyle.SP_MediaPlay)
        ico_skip_back = self.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward)
        ico_skip_fwd  = self.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekForward)

        self.act_skip_back = QtGui.QAction(ico_skip_back, "", self)
        self.act_skip_back.setToolTip("Skip backward 1 bar")
        self.act_skip_back.setShortcut(Qt.Key_Left)
        self.act_skip_back.triggered.connect(self._skip_prev_bar)

        self.act_skip_fwd = QtGui.QAction(ico_skip_fwd, "", self)
        self.act_skip_fwd.setToolTip("Skip forward 1 bar")
        self.act_skip_fwd.setShortcut(Qt.Key_Right)
        self.act_skip_fwd.triggered.connect(self._skip_next_bar)
        # Go to start
        self.act_goto_start = QtGui.QAction(ico_start, "", self)
        self.act_goto_start.setToolTip("Go to start")
        self.act_goto_start.setShortcut(Qt.Key_Home)
        self.act_goto_start.triggered.connect(self._goto_start)

        # Play/Pause toggle (icon updates dynamically)
        self.act_playpause = QtGui.QAction(ico_play, "", self)
        self.act_playpause.setToolTip("Play/Pause (Space)")
        self.act_playpause.setShortcut(Qt.Key_Space)
        self.act_playpause.triggered.connect(self._toggle_playpause)

        self.transport_tb.addAction(self.act_goto_start)
        self.transport_tb.addAction(self.act_skip_back)
        self.transport_tb.addAction(self.act_playpause)
        self.transport_tb.addAction(self.act_skip_fwd)
        try:
            self._dock_transport_next_to_load()
        except Exception:
            pass
        self.toolbar.addWidget(self.rate_label)
        self.toolbar.addWidget(self.rate_spin)
        self.toolbar.addAction(self.act_render)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(self.pitch_label)
        self.toolbar.addWidget(self.pitch_spin)
        self.toolbar.addWidget(self.pitch_reset_btn)
        self.toolbar.addSeparator()
        self.analysis_toolbar = getattr(self, 'analysis_toolbar', None)
        if self.analysis_toolbar is None:
            self.analysis_toolbar = self.addToolBar("Analysis")
            self.analysis_toolbar.setObjectName("AnalysisToolbar")
        self.analysis_toolbar.addAction(self.actReextractStems)
        self.analysis_toolbar.addAction(self.actRecomputeAnalysis)
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
        try:
            self._set_waveform_view_mode(False)
        except Exception:
            pass
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

        # Signals
        self.rate_spin.valueChanged.connect(self._rate_changed)
        self.pitch_spin.valueChanged.connect(self._pitch_changed)
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

    def _on_backend_changed(self, backend: str):
        """Update chosen backend and show/hide Style menu accordingly."""
        self.chord_backend = backend
        # Update checkmarks if the actions exist
        try:
            if getattr(self, "act_backend_internal", None):
                self.act_backend_internal.setChecked(backend == "internal")
            if getattr(self, "act_backend_chordino", None):
                self.act_backend_chordino.setChecked(backend == "chordino")
        except Exception:
            pass
        # Hide Style menu for chordino, show for internal
        try:
            if getattr(self, "style_menu", None):
                self.style_menu.menuAction().setVisible(backend != "chordino")
        except Exception:
            pass

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
            self._clear_cached_analysis(keep_stems=False)
            self._clear_stem_rows()
            self.start_chord_analysis(path, force=True)  # runs Demucs + waits + chords
        except Exception:
            pass

    def _reset_state_for_new_track(self):
        """Reset playback, workers, analysis caches, stems UI, and visuals so a new
        audio load behaves like first run and the pipeline (beats→stems→chords) is fresh."""
        # Stop playback
        try:
            if getattr(self, 'player', None) and hasattr(self.player, 'stop'):
                self.player.stop()
        except Exception:
            pass

        # Cancel background workers
        for wname in ('beat_worker', 'chord_worker', 'key_worker'):
            try:
                w = getattr(self, wname, None)
                if w and w.isRunning():
                    w.requestInterruption(); w.quit(); w.wait(-1)
                setattr(self, wname, None)
            except Exception:
                pass

        # Clear in-memory analysis/session state
        for attr, val in (
            ('last_tempo', None),
            ('last_beats', []),
            ('last_bars', []),
            ('last_key', None),
            ('last_chords', []),
            ('A', None), ('B', None)
        ):
            try:
                setattr(self, attr, val)
            except Exception:
                pass

        # Clear stems from engine and UI dock
        try:
            if getattr(self, 'player', None):
                if hasattr(self.player, 'set_stems_arrays'):
                    self.player.set_stems_arrays({}, None)
                setattr(self.player, 'stems_arrays', {})
                setattr(self.player, 'stem_mute', {})
            # Clear waveform cache when stems are cleared
            if hasattr(self, 'wave') and self.wave is not None:
                self.wave.clear_waveform_cache()
        except Exception:
            pass
        try:
            if hasattr(self, 'stems_layout') and self.stems_layout is not None:
                while self.stems_layout.count():
                    item = self.stems_layout.takeAt(0)
                    w = item.widget()
                    if w is not None:
                        w.setParent(None); w.deleteLater()
        except Exception:
            pass

        # Clear waveform overlays and chords
        try:
            if hasattr(self, 'wave') and self.wave is not None:
                self.wave.set_beats([], [])
                self.wave.set_chords([])
                if hasattr(self.wave, 'clear_loop_visual'):
                    self.wave.clear_loop_visual()
                self.wave.selected_loop_id = None
                self.wave.update()
        except Exception:
            pass

        # Clear status and Demucs log
        try:
            self.statusBar().clearMessage()
        except Exception:
            pass
        try:
            if hasattr(self, 'log_dock') and self.log_dock:
                self.log_dock.clear()
        except Exception:
            pass
        self.last_key = None
        self.populate_key_async(self.current_path)

    def _session_sidecar_path(self, audio_path: str) -> Path:
        p = Path(audio_path)
        folder = p.parent / ".musicpractice"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{p.stem}.musicpractice.json"

    def _dock_transport_next_to_load(self):
        """Place transport controls (icon-only buttons) to the RIGHT of the Load/Open action
        on an existing toolbar. Ensure order: [Go to Start] then [Play/Pause].
        Falls back to the dedicated transport toolbar if no suitable toolbar is found.
        """
        try:
            if not hasattr(self, "act_goto_start") or not hasattr(self, "act_playpause"):
                return  # actions not created yet

            # Find a toolbar that already has a Load/Open action
            target_tb = None
            load_act = None
            for tb in self.findChildren(QtWidgets.QToolBar):
                for act in tb.actions():
                    txt = (act.text() or "").lower().replace("&", "")
                    if "load" in txt or "open" in txt:
                        target_tb = tb
                        load_act = act
                        break
                if target_tb:
                    break

            if not target_tb or load_act is None:
                return  # keep dedicated transport toolbar

            # Remove the standalone transport toolbar if present; we'll host icon buttons instead.
            try:
                if hasattr(self, "transport_tb") and self.transport_tb is not None:
                    self.removeToolBar(self.transport_tb)
            except Exception:
                pass

            # Build icon-only buttons if not created yet
            if not hasattr(self, "_transport_btn_start") or self._transport_btn_start is None:
                self._transport_btn_start = QtWidgets.QToolButton(target_tb)
                self._transport_btn_start.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaSkipBackward))
                self._transport_btn_start.setToolTip("Go to start")
                self._transport_btn_start.setAutoRaise(True)
                self._transport_btn_start.clicked.connect(self._goto_start)

            if not hasattr(self, "_transport_btn_skip_back") or self._transport_btn_skip_back is None:
                self._transport_btn_skip_back = QtWidgets.QToolButton(target_tb)
                self._transport_btn_skip_back.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekBackward))
                self._transport_btn_skip_back.setToolTip("Skip backward 1 bar")
                self._transport_btn_skip_back.setAutoRaise(True)
                self._transport_btn_skip_back.clicked.connect(self._skip_prev_bar)

            if not hasattr(self, "_transport_btn_play") or self._transport_btn_play is None:
                self._transport_btn_play = QtWidgets.QToolButton(target_tb)
                self._transport_btn_play.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaPlay))
                self._transport_btn_play.setToolTip("Play/Pause (Space)")
                self._transport_btn_play.setAutoRaise(True)
                self._transport_btn_play.clicked.connect(self._toggle_playpause)

            if not hasattr(self, "_transport_btn_skip_fwd") or self._transport_btn_skip_fwd is None:
                self._transport_btn_skip_fwd = QtWidgets.QToolButton(target_tb)
                self._transport_btn_skip_fwd.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekForward))
                self._transport_btn_skip_fwd.setToolTip("Skip forward 1 bar")
                self._transport_btn_skip_fwd.setAutoRaise(True)
                self._transport_btn_skip_fwd.clicked.connect(self._skip_next_bar)

            # Insert immediately AFTER Load/Open; order: Start, SkipBack, Play, SkipFwd
            acts = list(target_tb.actions())
            try:
                idx = acts.index(load_act)
            except ValueError:
                idx = -1
            insert_after = acts[idx + 1] if (idx >= 0 and (idx + 1) < len(acts)) else None

            if insert_after is None:
                target_tb.addWidget(self._transport_btn_start)
                target_tb.addWidget(self._transport_btn_skip_back)
                target_tb.addWidget(self._transport_btn_play)
                target_tb.addWidget(self._transport_btn_skip_fwd)
            else:
                target_tb.insertWidget(insert_after, self._transport_btn_start)
                target_tb.insertWidget(insert_after, self._transport_btn_skip_back)
                target_tb.insertWidget(insert_after, self._transport_btn_play)
                target_tb.insertWidget(insert_after, self._transport_btn_skip_fwd)

            # Keep play/pause icon synced
            self._refresh_transport_icon()
        except Exception:
            pass

    def _refresh_transport_icon(self):
        try:
            style = self.style()
            if self._is_playing():
                icon = style.standardIcon(QtWidgets.QStyle.SP_MediaPause)
            else:
                icon = style.standardIcon(QtWidgets.QStyle.SP_MediaPlay)
            # Update the action (for menus/shortcuts)
            if hasattr(self, 'act_playpause') and self.act_playpause is not None:
                self.act_playpause.setIcon(icon)
            # Update the toolbar button if present
            if hasattr(self, '_transport_btn_play') and self._transport_btn_play is not None:
                self._transport_btn_play.setIcon(icon)
        except Exception:
            pass

    def _goto_start(self):
        # Seek to musical origin if known; otherwise 0.0
        try:
            t0 = float(getattr(self.wave, "origin", 0.0) or 0.0)
        except Exception:
            t0 = 0.0
        try:
            if self.player and hasattr(self.player, "seek_seconds"):
                self.player.seek_seconds(t0)
            elif self.player and hasattr(self.player, "set_position_seconds"):
                self.player.set_position_seconds(t0)
            else:
                self.wave.requestSeek.emit(float(t0))
        except Exception:
            self.wave.requestSeek.emit(float(t0))
        # recentre the view if it was frozen by a click
        try:
            if hasattr(self, "wave") and self.wave is not None:
                self.wave.unfreeze_and_center()
        except Exception:
            pass

    def _is_playing(self) -> bool:
        p = getattr(self, 'player', None)
        if not p:
            return False
        try:
            # Prefer explicit paused/playing queries
            if hasattr(p, 'is_paused') and callable(p.is_paused):
                return not bool(p.is_paused())
            if hasattr(p, 'paused'):
                return not bool(getattr(p, 'paused'))
            if hasattr(p, 'state'):
                st = getattr(p, 'state')
                if isinstance(st, str):
                    return st.lower() in ('play', 'playing', 'running')
                try:
                    return int(st) == 1  # commonly 1 == playing
                except Exception:
                    pass
            if hasattr(p, 'is_playing') and callable(p.is_playing):
                return bool(p.is_playing())
            if hasattr(p, 'playing'):
                return bool(getattr(p, 'playing'))
        except Exception:
            pass
        return False

    def _toggle_playpause(self):
        p = getattr(self, 'player', None)
        if not p:
            return
        try:
            if self._is_playing():
                # --- PAUSE fallbacks ---
                if hasattr(p, 'pause') and callable(p.pause):
                    p.pause()
                elif hasattr(p, 'set_paused') and callable(p.set_paused):
                    p.set_paused(True)
                elif hasattr(p, 'toggle_pause') and callable(p.toggle_pause):
                    p.toggle_pause()
                elif hasattr(p, 'play') and callable(p.play):
                    # Some players accept a boolean
                    try:
                        p.play(False)
                    except Exception:
                        pass
            else:
                # --- PLAY/RESUME fallbacks ---
                if hasattr(p, 'play') and callable(p.play):
                    try:
                        p.play()
                    except TypeError:
                        # In case play(bool) exists
                        try:
                            p.play(True)
                        except Exception:
                            pass
                elif hasattr(p, 'set_paused') and callable(p.set_paused):
                    p.set_paused(False)
                elif hasattr(p, 'resume') and callable(p.resume):
                    p.resume()
        finally:
            # Keep the icon in sync regardless of outcome
            try:
                self._refresh_transport_icon()
            except Exception:
                pass

    def _current_position_seconds(self) -> float:
        try:
            if self.player and hasattr(self.player, 'position_seconds'):
                return float(self.player.position_seconds())
        except Exception:
            pass
        return 0.0

    def _bars_list(self) -> list[float]:
        # Prefer WaveformView's bars; fall back to cached last_bars
        try:
            if hasattr(self, 'wave') and self.wave and getattr(self.wave, 'bars', None):
                return [float(b) for b in self.wave.bars]
        except Exception:
            pass
        try:
            if getattr(self, 'last_bars', None):
                return [float(b) for b in self.last_bars]
        except Exception:
            pass
        return []

    def _seek_seconds(self, t: float):
        try:
            if self.player and hasattr(self.player, 'seek_seconds'):
                self.player.seek_seconds(float(t))
            elif self.player and hasattr(self.player, 'set_position_seconds'):
                self.player.set_position_seconds(float(t))
            else:
                self.wave.requestSeek.emit(float(t))
        except Exception:
            self.wave.requestSeek.emit(float(t))

    def _skip_prev_bar(self):
        bars = sorted(self._bars_list())
        pos = self._current_position_seconds()
        if bars:
            try:
                origin = float(getattr(self.wave, 'origin', 0.0) or 0.0)
            except Exception:
                origin = 0.0
            if all(abs(origin - b) > 1e-6 for b in bars):
                bars = [origin] + bars
            prevs = [b for b in bars if b < pos - 1e-6]
            t_new = prevs[-1] if prevs else bars[0]
        else:
            step = getattr(self.wave, 'window_s', 2.0) if hasattr(self, 'wave') and self.wave else 2.0
            t_new = max(0.0, pos - step)
        self._seek_seconds(t_new)
        try:
            if hasattr(self, 'wave') and self.wave: self.wave.unfreeze_and_center()
        except Exception:
            pass

    def _skip_next_bar(self):
        bars = sorted(self._bars_list())
        pos = self._current_position_seconds()
        if bars:
            try:
                origin = float(getattr(self.wave, 'origin', 0.0) or 0.0)
            except Exception:
                origin = 0.0
            if all(abs(origin - b) > 1e-6 for b in bars):
                bars = [origin] + bars
            nexts = [b for b in bars if b > pos + 1e-6]
            t_new = nexts[0] if nexts else bars[-1]
        else:
            step = getattr(self.wave, 'window_s', 2.0) if hasattr(self, 'wave') and self.wave else 2.0
            t_new = pos + step
        self._seek_seconds(t_new)
        try:
            if hasattr(self, 'wave') and self.wave: self.wave.unfreeze_and_center()
        except Exception:
            pass

    def _set_waveform_view_mode(self, show_stems: bool):
        try:
            if hasattr(self, 'wave') and self.wave is not None:
                self.wave.set_show_stems(bool(show_stems))
        except Exception:
            pass
        # keep menu states in sync
        try:
            if hasattr(self, 'act_view_stems') and self.act_view_stems:
                self.act_view_stems.setChecked(bool(show_stems))
            if hasattr(self, 'act_view_combined') and self.act_view_combined:
                self.act_view_combined.setChecked(not bool(show_stems))
        except Exception:
            pass

    def _set_backend(self, name: str):
        """Set chord detection backend and keep menu checks in sync."""
        try:
            name = (name or "internal").lower()
        except Exception:
            name = "internal"
        self.backend = "chordino" if name == "chordino" else "internal"
        try:
            if hasattr(self, "act_backend_internal"):
                self.act_backend_internal.setChecked(self.backend == "internal")
            if hasattr(self, "act_backend_chordino"):
                self.act_backend_chordino.setChecked(self.backend == "chordino")
        except Exception:
            pass
        try:
            self._update_style_menu_visibility()
        except Exception:
            pass
        try:
            self.statusBar().showMessage(f"Chord backend: {self.backend}", 1500)
        except Exception:
            pass

    def _set_analysis_style(self, style_key: str):
        self.analysis_style = style_key
        try:
            self.statusBar().showMessage(f"Analysis style: {style_key}", 1500)
        except Exception:
            pass

    def _update_style_menu_visibility(self):
        """
        Hide the chord style chooser when using Chordino (unused there),
        show it for internal detectors.
        """
        active = getattr(self, "backend", getattr(self, "chord_backend", "internal"))
        use_style = (active != "chordino")

        # If you keep a reference to the style menu, update that (adjust attr name if needed)
        for attr in ("menu_style", "style_menu", "menuStyle", "styleMenu"):
            m = getattr(self, attr, None)
            if isinstance(m, QtWidgets.QMenu):
                m.menuAction().setVisible(use_style)

        # Fallback: hide/show individual actions if you don't have the menu object
        for attr in ("act_style_rock_pop", "act_style_jazz",
                    "act_style_blues", "act_style_reggae", "act_style_default"):
            a = getattr(self, attr, None)
            if isinstance(a, QtGui.QAction):
                a.setVisible(use_style)

    def populate_key_async(self, path: str):
        if getattr(self, 'key_worker', None) and self.key_worker.isRunning():
            self.key_worker.requestInterruption(); self.key_worker.quit(); self.key_worker.wait(1000)
        kw = KeyWorker(path)
        kw.setParent(self)
        kw.done.connect(self._key_ready)
        kw.finished.connect(lambda: setattr(self, "key_worker", None))
        self.key_worker = kw
        kw.start()

    def _key_ready(self, info: dict):
        try:
            self.last_key = dict(info or {})
        except Exception:
            self.last_key = {"pretty": "unknown"}
        label = self.last_key.get("pretty") or "unknown"
        self.statusBar().showMessage(f"Key: {label}", 1500)
        if getattr(self, "_pending_chords_waiting_for_key", False) and getattr(self, "current_path", None):
            self._pending_chords_waiting_for_key = False
            self.start_chord_analysis(self.current_path, force=False)

    def _normalize_chords_for_view(self, segments: list[dict]) -> list[dict]:
        """Normalize raw chord segments for display without writing the session.
        Pipeline: snap to bars → split at bars → ensure leading bar → unique-by-span.
        Falls back to the input on error.
        """
        try:
            bars = list(self.last_bars or [])
            snapped = self.wave._snap_segments_to_bars(segments or [], bars)
            split = self.wave._split_segments_at_bars(snapped, bars)
            filled = self.wave._ensure_leading_bar(split, bars)
            dedup = self.wave._unique_by_span(filled)
            return dedup
        except Exception as ex:
            logging.error(f"_normalize_chords_for_view failed: {ex}")
            return list(segments or [])

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
        self.chords_locked = bool(data.get("chords_locked", False))

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
        # Don't set chords on waveform yet - wait until after pitch is applied
        if self.last_chords:
            try:
                if getattr(self, "chords_locked", False):
                    # Preserve exactly as saved
                    pass  # Will be set after pitch application
                else:
                    # Only normalize when not locked
                    normalized = self._normalize_chords_for_view(self.last_chords)
                    self.last_chords = list(normalized)
            except Exception as ex:
                logging.error(f"load_session chords normalize failed: {ex}")
        # Rate
        if "rate" in data:
            try:
                self.rate_spin.setValue(float(data.get("rate", 1.0)))
            except Exception:
                pass

        # Pitch - apply after all other session data is loaded
        if "pitch" in data:
            try:
                pitch_value = int(data.get("pitch", 0))
                # Temporarily disconnect the signal to avoid triggering _pitch_changed during session load
                self.pitch_spin.valueChanged.disconnect()
                self.pitch_spin.setValue(pitch_value)
                # Reconnect the signal
                self.pitch_spin.valueChanged.connect(self._pitch_changed)
                # Apply pitch shift to audio and update display after a short delay
                # to ensure all session data is fully loaded
                QtCore.QTimer.singleShot(50, lambda: self._apply_pitch_from_session(pitch_value))
            except Exception as e:
                print(f"Error loading pitch from session: {e}")
                # Make sure to reconnect the signal even if there's an error
                try:
                    self.pitch_spin.valueChanged.connect(self._pitch_changed)
                except Exception:
                    pass

        self.statusBar().showMessage(f"Session loaded from {Path(sidecar).name}")

    def _apply_pitch_from_session(self, semitones: int):
        """Apply pitch shift from session load - updates audio, key, and chords."""
        try:
            # Apply to audio player if available
            if getattr(self, 'player', None) and hasattr(self.player, 'set_pitch_shift'):
                # Create progress callback for status bar updates
                def progress_callback(message):
                    # Force immediate update and use a longer timeout
                    self.statusBar().showMessage(message, 5000)  # 5 seconds
                    # Force the UI to update immediately
                    QtCore.QCoreApplication.processEvents()

                ok = self.player.set_pitch_shift(semitones, progress_callback)
                if ok:
                    print(f"Applied pitch shift {semitones:+d} semitones from session")
                    self.statusBar().showMessage(f"Pitch shift {semitones:+d} semitones applied", 2000)
                else:
                    print(f"Failed to apply pitch shift {semitones:+d} semitones from session")
                    self.statusBar().showMessage("Pitch shifting failed", 3000)

            # Update key and chord display with transposed values
            self._update_transposed_display(semitones)

            # If no pitch shift, set the original chords on the waveform display
            if abs(semitones) < 0.01 and hasattr(self, 'last_chords') and self.last_chords:
                if hasattr(self, 'wave') and self.wave:
                    self.wave.set_chords(self.last_chords)

        except Exception as e:
            print(f"Error applying pitch from session: {e}")

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
            "chords_locked": bool(getattr(self, "chords_locked", False)),
            "rate": float(self.rate_spin.value()),
            "pitch": int(self.pitch_spin.value()),
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
        idx = self._find_chord_index_at_time(float(t))
        if idx is None:
            return
        cur = dict(self.last_chords[idx]) if (0 <= idx < len(self.last_chords)) else {}
        cur_label = str(cur.get('label', ''))
        text, ok = QtWidgets.QInputDialog.getText(self, "Change chord", "Chord label:", text=cur_label)
        if not ok:
            return
        cur['label'] = str(text)
        self.last_chords[idx] = cur
        # Reflect in UI and save
        self.wave.set_chords(self.last_chords)
        try:
            self.wave.update(); self.wave.repaint()
        except Exception:
            pass
        self.chords_locked = True
        self.save_session()
        self.statusBar().showMessage(f"Chord updated → {text}", 1500)

    def _split_chord_at_time(self, t: float):
        """Split the chord containing time t so the two parts have equal beat counts;
        if the total beat count is odd, the first (left) part has one more beat.
        The split is snapped to a beat strictly inside the chord span when possible.
        """
        t = float(t)

        # Find chord index at click time
        idx = self._find_chord_index_at_time(t)
        if idx is None:
            return
        try:
            seg = dict(self.last_chords[idx])
            a = float(seg.get('start')); b = float(seg.get('end'))
            lab = seg.get('label')
        except Exception:
            return
        if not (b > a + 1e-6):
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
        else:
            # Fallback: use the temporal midpoint
            split_t = 0.5 * (a + b)

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
        self.chords_locked = True
        self.save_session()
        self.statusBar().showMessage(f"Chord split at {split_t:.2f}s", 1200)

    def _join_chord_forward_at_time(self, t: float):
        """Join the chord at time t with the next chord if contiguous and within same bar."""
        t = float(t)
        idx = self._find_chord_index_at_time(t)
        if idx is None or idx + 1 >= len(self.last_chords):
            return
        cur = self.last_chords[idx]
        nxt = self.last_chords[idx + 1]
        try:
            cmid = 0.5 * (float(cur['start']) + float(cur['end']))
            nmid = 0.5 * (float(nxt['start']) + float(nxt['end']))
        except Exception:
            return
        bar_cur = self._bar_index_at_time(cmid)
        bar_nxt = self._bar_index_at_time(nmid)
        if bar_cur is None or bar_nxt is None or bar_cur != bar_nxt:
            return
        if abs(float(cur['end']) - float(nxt['start'])) > 1e-3:
            return
        merged = {
            'start': float(cur['start']),
            'end': float(nxt['end']),
            'label': cur.get('label') or nxt.get('label')
        }
        self.last_chords = self.last_chords[:idx] + [merged] + self.last_chords[idx+2:]
        self.wave.set_chords(self.last_chords)
        self.chords_locked = True
        self.save_session()

    def _add_stem_row(self, name: str, array: np.ndarray):
        row = QtWidgets.QWidget(self)
        vlay = QtWidgets.QVBoxLayout(row)
        vlay.setContentsMargins(6, 4, 6, 6)
        vlay.setSpacing(4)

        # --- Top line: Label (left) + Focus + Solo + Mute (right) ---
        top = QtWidgets.QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        lbl = QtWidgets.QLabel(str(name), row)
        f = lbl.font()
        f.setBold(True)
        lbl.setFont(f)
        top.addWidget(lbl, 0, Qt.AlignVCenter | Qt.AlignLeft)
        top.addStretch(1)

        # Visual-only Focus (solo-style visuals without soloing audio)
        focus_btn = QtWidgets.QCheckBox("Focus", row)
        focus_btn.setChecked(False)
        top.addWidget(focus_btn, 0, Qt.AlignVCenter | Qt.AlignRight)

        # Solo radio button
        solo_btn = QtWidgets.QRadioButton("Solo", row)
        solo_btn.setChecked(False)
        top.addWidget(solo_btn, 0, Qt.AlignVCenter | Qt.AlignRight)

        chk = QtWidgets.QCheckBox("Mute", row)
        chk.setChecked(False)
        top.addWidget(chk, 0, Qt.AlignVCenter | Qt.AlignRight)
        vlay.addLayout(top)

        # --- Second line: 0  [==== slider with ticks ====]  100 ---
        slider_line = QtWidgets.QHBoxLayout()
        slider_line.setContentsMargins(0, 0, 0, 0)
        slider_line.setSpacing(6)

        lbl0 = QtWidgets.QLabel("0", row)
        lbl100 = QtWidgets.QLabel("100", row)

        sld = QtWidgets.QSlider(Qt.Horizontal, row)
        sld.setRange(0, 100)
        sld.setValue(100)
        sld.setSingleStep(1)
        sld.setPageStep(5)
        sld.setTracking(True)
        sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        sld.setTickInterval(10)

        slider_line.addWidget(lbl0, 0, Qt.AlignVCenter)
        slider_line.addWidget(sld, 1)
        slider_line.addWidget(lbl100, 0, Qt.AlignVCenter)
        vlay.addLayout(slider_line)

        stem_key = str(name)

        def _apply_mute(m: bool):
            m = bool(m)
            # Update engine (if supported)
            try:
                if self.player and hasattr(self.player, 'set_stem_mute'):
                    self.player.set_stem_mute(stem_key, m)
            except Exception:
                pass
            # Update public mute map so WaveformView can dim the background
            try:
                if self.player is not None:
                    mute_map = getattr(self.player, 'stem_mute', None)
                    if not isinstance(mute_map, dict):
                        mute_map = {}
                    mute_map[stem_key] = m
                    setattr(self.player, 'stem_mute', mute_map)
            except Exception:
                pass
            # Repaint the main waveform immediately
            try:
                if hasattr(self, 'wave') and self.wave is not None:
                    self.wave.update()
                    QtCore.QTimer.singleShot(0, self.wave.update)
            except Exception:
                pass

        def _apply_focus(on: bool):
            on = bool(on)
            try:
                # Prefer a visual-focus API if WaveformView provides it
                wave = getattr(self, 'wave', None)
                if wave is None:
                    return

                # Determine current focused stem (if API exists)
                current_focus = None
                if hasattr(wave, 'get_focus_stem') and callable(wave.get_focus_stem):
                    current_focus = wave.get_focus_stem()
                elif hasattr(wave, 'focus_stem'):
                    current_focus = getattr(wave, 'focus_stem')

                if on:
                    # Set this stem as the visual focus
                    if hasattr(wave, 'set_focus_stem') and callable(wave.set_focus_stem):
                        wave.set_focus_stem(stem_key)
                    else:
                        # Fallback: set attribute + update
                        try:
                            wave.focus_stem = stem_key
                        except Exception:
                            pass
                        wave.update()

                    # Uncheck other Focus checkboxes in the dock to make it exclusive
                    try:
                        for widget in self.stems_layout.parent().findChildren(QtWidgets.QCheckBox):
                            if widget is not focus_btn and widget.text() == "Focus":
                                widget.setChecked(False)
                    except Exception:
                        pass
                else:
                    # If this stem was focused, clear focus
                    if current_focus == stem_key:
                        if hasattr(wave, 'set_focus_stem') and callable(wave.set_focus_stem):
                            wave.set_focus_stem(None)
                        else:
                            try:
                                wave.focus_stem = None
                            except Exception:
                                pass
                        wave.update()
            except Exception:
                pass

        def _apply_solo(soloed: bool):
            soloed = bool(soloed)

            # Check if this was the currently soloed stem BEFORE updating the engine
            was_soloed_stem = False
            try:
                if self.player and hasattr(self.player, 'get_soloed_stem'):
                    current_solo = self.player.get_soloed_stem()
                    was_soloed_stem = (current_solo == stem_key)
            except Exception:
                pass

            # Update engine
            try:
                if self.player and hasattr(self.player, 'set_stem_solo'):
                    self.player.set_stem_solo(stem_key, soloed)
            except Exception:
                pass

            # Update waveform view to show focus mode
            try:
                if hasattr(self, 'wave') and self.wave is not None:
                    if soloed:
                        # Solo mode: show only this stem
                        self.wave.set_soloed_stem(stem_key)
                    else:
                        # Unsolo: if this was the soloed stem, return to current view mode
                        if was_soloed_stem:
                            self.wave.set_soloed_stem(None)
                    self.wave.update()
                    QtCore.QTimer.singleShot(0, self.wave.update)
            except Exception:
                pass

            # Update all other solo buttons to be unchecked
            if soloed:
                try:
                    # Find all other solo buttons and uncheck them
                    for widget in self.stems_layout.parent().findChildren(QtWidgets.QRadioButton):
                        if widget != solo_btn and widget.text() == "Solo":
                            widget.setChecked(False)
                except Exception:
                    pass

        def _apply_gain(val: int):
            g = float(val) / 100.0
            try:
                if self.player and hasattr(self.player, 'set_stem_gain'):
                    self.player.set_stem_gain(stem_key, g)
            except Exception:
                pass

        chk.toggled.connect(_apply_mute)
        focus_btn.toggled.connect(_apply_focus)
        solo_btn.toggled.connect(_apply_solo)
        sld.valueChanged.connect(_apply_gain)

        # Init visual/engine state
        _apply_gain(100)

        # Add row to the dock's layout
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

    def _clear_cached_analysis(self, keep_stems: bool = True):
        """Clear computed analysis so recompute truly recalculates beats, key, chords.
        If keep_stems is False, also forget any cached stems metadata (not files)."""
        # Clear in-memory analysis caches/fields if present
        for attr in (
            'last_beats', 'last_downbeats', 'last_beat_strengths',
            'last_chords', 'last_key', 'beat_grid', 'last_bars', 'last_tempo',
        ):
            if hasattr(self, attr):
                try:
                    setattr(self, attr, None)
                except Exception:
                    pass
        # Clear any session dict entries if used
        try:
            if hasattr(self, 'session') and isinstance(self.session, dict):
                for k in ('beats', 'downbeats', 'beat_strengths', 'chords', 'key', 'tempo', 'bars'):
                    self.session.pop(k, None)
        except Exception:
            pass
        # Reset visual caches
        try:
            if getattr(self, 'wave', None):
                self.wave.set_beats([], [])
                self.wave.set_chords([])
        except Exception:
            pass

        # Remove on-disk analysis sidecars for this track (do not touch audio or stems)
        try:
            cur = getattr(self, 'current_path', None)
            if cur:
                p = Path(cur)
                candidates = []
                # Common patterns used by earlier versions
                candidates.append(p.with_suffix(p.suffix + ".json"))   # e.g., song.mp3.json
                candidates.append(p.with_suffix(".json"))               # e.g., song.json
                for base in ("chords", "beats", "analysis", "session"):
                    candidates.append(p.parent / f"{p.stem}.{base}.json")
                # Also look under .musicpractice next to the audio
                dot = p.parent / ".musicpractice"
                for name in ("chords.json", "beats.json", "analysis.json", f"{p.stem}.json"):
                    candidates.append(dot / name)
                # Delete any that exist
                for f in candidates:
                    try:
                        if f and f.is_file():
                            f.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

        # Optionally clear stems metadata so UI will reload them (files remain on disk)
        if not keep_stems:
            for attr in ('_loaded_stems', '_loaded_stems_leaf'):
                if hasattr(self, attr):
                    try:
                        setattr(self, attr, None)
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

    def _pitch_changed(self, val: int):
        """Update pitch shift on the player and transpose key/chords."""
        try:
            semitones = int(val)
        except Exception:
            semitones = 0
        # Clamp to supported range
        semitones = max(-12, min(12, semitones))

        # Apply to player if it has pitch shifting capability
        if getattr(self, 'player', None) and hasattr(self.player, 'set_pitch_shift'):
            try:
                # Create progress callback for status bar updates
                def progress_callback(message):
                    # Force immediate update and use a longer timeout
                    self.statusBar().showMessage(message, 5000)  # 5 seconds
                    # Force the UI to update immediately
                    QtCore.QCoreApplication.processEvents()

                ok = self.player.set_pitch_shift(semitones, progress_callback)
                if ok:
                    self.statusBar().showMessage(f"Pitch shifted by {semitones:+d} semitones", 2000)
                    # Update key and chord display
                    self._update_transposed_display(semitones)
                    # Save session with new pitch value
                    self.save_session()
                else:
                    self.statusBar().showMessage("Pitch shifting failed", 3000)
            except Exception as e:
                self.statusBar().showMessage(f"Pitch shifting error: {e}", 3000)
        else:
            # Player doesn't support pitch shifting yet
            self.statusBar().showMessage("Pitch shifting not yet implemented in audio engine", 3000)

    def _reset_pitch(self):
        """Reset pitch to original (0 semitones)."""
        self.pitch_spin.setValue(0)

    # --- Pitch/key/chord transposition helpers (display-only) ---
    NOTE_RING_SHARPS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    NOTE_RING_FLATS  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
    NOTE_INDEX = {n:i for i,n in enumerate(NOTE_RING_SHARPS)}

    def _parse_root(self, token: str) -> tuple[str, str]:
        if not token: return "", ""
        t = token.strip()
        if len(t) >= 2 and t[1] in ("#", "b"): return t[:2], t[2:]
        return t[:1], t[1:]

    def _transpose_note_name(self, name: str, semis: int, prefer_flats: bool | None = None) -> str:
        if not name: return name
        i = self.NOTE_INDEX.get(name) or self.NOTE_INDEX.get(name[:1].upper()+name[1:])
        if i is None: return name
        j = (i + int(semis)) % 12
        ring = self.NOTE_RING_FLATS if prefer_flats else self.NOTE_RING_SHARPS
        return ring[j]

    def transpose_chord_label(self, label: str, semis: int, prefer_flats: bool | None = None) -> str:
        if not label: return label
        lab = str(label).strip()
        if "/" in lab: main, bass = lab.split("/", 1)
        else: main, bass = lab, None
        root, qual = self._parse_root(main)
        if not root: return label
        new_root = self._transpose_note_name(root, semis, prefer_flats)
        if bass:
            b_root, b_rest = self._parse_root(bass.strip())
            bass = self._transpose_note_name(b_root, semis, prefer_flats) + b_rest
        return new_root + qual + ("/"+bass if bass else "")

    def transpose_chords_list(self, chords_list: list[dict], semis: int, prefer_flats: bool | None = None) -> list[dict]:
        out = []
        for s in chords_list or []:
            try:
                a = float(s.get("start")); b = float(s.get("end"))
                lab = str(s.get("label",""))
            except Exception:
                continue
            if b <= a: continue
            out.append({"start": a, "end": b, "label": self.transpose_chord_label(lab, semis, prefer_flats)})
        return out

    def transpose_key_dict(self, key_info: dict, semis: int, prefer_flats: bool | None = None) -> dict:
        if not isinstance(key_info, dict): return key_info or {}
        name = key_info.get("pretty") or key_info.get("name") or ""
        tok = name.split()
        if not tok: return dict(key_info)
        root, rest = tok[0], " ".join(tok[1:])
        new_root = self._transpose_note_name(root, semis, prefer_flats)
        new_name = (new_root + (" "+rest if rest else "")).strip()
        out = dict(key_info); out["name"] = new_name; out["pretty"] = new_name
        return out

    def _update_transposed_display(self, semitones: int):
        """Update key label and chord display with transposed values."""
        try:
            # If no pitch shift, show original key and chords
            if abs(semitones) < 0.01:
                # Show original key
                if hasattr(self, 'last_key') and self.last_key:
                    original_key_name = self.last_key.get('pretty') or self.last_key.get('name') or ""
                    if original_key_name and hasattr(self, 'key_header_label') and self.key_header_label:
                        self.key_header_label.setText(f"Key: {original_key_name}")

                # Show original chords
                if hasattr(self, 'last_chords') and self.last_chords:
                    if hasattr(self, 'wave') and self.wave:
                        # Use QTimer to delay the chord setting to ensure it happens after other operations
                        QtCore.QTimer.singleShot(200, lambda: self._delayed_set_transposed_chords(self.last_chords))
                return

            # Update key display using the comprehensive transposition function
            if hasattr(self, 'last_key') and self.last_key:
                transposed_key_dict = self.transpose_key_dict(self.last_key, semitones)
                transposed_key_name = transposed_key_dict.get('pretty') or transposed_key_dict.get('name') or ""
                if transposed_key_name and hasattr(self, 'key_header_label') and self.key_header_label:
                    self.key_header_label.setText(f"Key: {transposed_key_name}")

            # Update chord display using the comprehensive transposition function
            if hasattr(self, 'last_chords') and self.last_chords:
                transposed_chords = self.transpose_chords_list(self.last_chords, semitones)

                # Update the waveform display with a delay to ensure other operations complete first
                if hasattr(self, 'wave') and self.wave:
                    # Use QTimer to delay the chord setting to ensure it happens after other operations
                    QtCore.QTimer.singleShot(200, lambda: self._delayed_set_transposed_chords(transposed_chords))

        except Exception as e:
            # Don't show error to user for display updates, just log it
            try:
                print(f"Error updating transposed display: {e}")
            except Exception:
                pass

    def _delayed_set_transposed_chords(self, transposed_chords):
        """Delayed method to set transposed chords on waveform display after other operations complete."""
        try:
            if hasattr(self, 'wave') and self.wave:
                self.wave.set_chords(transposed_chords)
                # Force a repaint to ensure the display is updated
                self.wave.repaint()
        except Exception as e:
            print(f"Error in delayed chord display update: {e}")

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
        # Fresh start for new track
        self._reset_state_for_new_track()

        # Update current path ASAP
        self.current_path = fn

        # Respect "Always recompute on open" toggle by forcing stems/chords recompute
        try:
            self._force_stems_recompute = bool(self.actAlwaysRecompute.isChecked())
        except Exception:
            pass

        # Ensure Demucs Log dock is visible for this run
        try:
            if not hasattr(self, 'log_dock') or self.log_dock is None:
                self.log_dock = LogDock(parent=self)
                self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
            self.log_dock.setWindowTitle('Demucs Log')
            self.log_dock.clear()  # clear() will hide the dock if empty
        except Exception:
            pass

        # Kick the integrated pipeline: ChordWorker will do beats → Demucs (wait) → chords
        self.start_chord_analysis(self.current_path, force=False)

        # Clear any active loop visuals when opening a file
        if hasattr(self, "wave") and self.wave:
            self.wave.clear_loop_visual()
            self.wave.selected_loop_id = None
        self.settings.setValue("last_dir", str(Path(fn).parent))
        name = Path(fn).name
        self.setWindowTitle(f"MusicPractice — {name}")
        self.title_label.setText(name)

        # (Re)create player
        if self.player:
            self.player.stop(); self.player.close()
        self.player = LoopPlayer(fn)
        self.wave.set_player(self.player)
        try:
            self._refresh_transport_icon()
        except Exception:
            pass
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
                self._clear_cached_analysis(keep_stems=False)
        except Exception:
            pass
        # Compute only what is missing
        if not self.last_beats:
            self.populate_beats_async(self.current_path)
        if not self.last_chords:
            self.populate_chords_async(self.current_path, force=False)
        if not (self.last_key and self.last_key.get("pretty")):
            self.populate_key_async(self.current_path)

        # Status
        total_s = self.player.n / self.player.sr
        mus_len = max(0.0, (tail - lead))
        self.statusBar().showMessage(f"Loaded: {Path(fn).name} [music {mus_len:.1f}s of {total_s:.1f}s]")
        # Focus waveform for immediate key control
        self.wave.setFocus()

    def populate_chords_async(self, path: str, force: bool = False):
        """
        Start (or skip) chord analysis. If we already have chords loaded from the
        session and `force` is False, do nothing. Use `force=True` for explicit
        recomputation from the UI.
        """
        try:
            have_cached = bool(self.last_chords) and len(self.last_chords) > 0
        except Exception:
            have_cached = False

        if have_cached and not force:
            try:
                self.wave.set_chords(self.last_chords or [])
                self.statusBar().showMessage(f"Chords (cached): {len(self.last_chords)} segments")
                QtCore.QTimer.singleShot(1200, lambda: self.statusBar().clearMessage())
            except Exception:
                pass
            return

        return self.start_chord_analysis(path, force=force)

    def start_chord_analysis(self, path: str, force: bool = False):
        # Skip if a worker is already running
        if hasattr(self, "chord_worker") and self.chord_worker and self.chord_worker.isRunning():
            return

        if getattr(self, "chords_locked", False) and not force:
            try:
                self.statusBar().showMessage("Chords locked — skipping auto analysis", 1500)
            except Exception:
                pass
            return

        # If we already have chords and not forcing, do not recompute
        try:
            have_cached = bool(self.last_chords) and len(self.last_chords) > 0
        except Exception:
            have_cached = False

        if have_cached and not force:
            try:
                self.wave.set_chords(self.last_chords or [])
                self.statusBar().showMessage(f"Chords (cached): {len(self.last_chords)} segments")
                QtCore.QTimer.singleShot(1200, lambda: self.statusBar().clearMessage())
            except Exception:
                pass
            return

        if not getattr(self, "last_key", None):
            self._pending_chords_waiting_for_key = True
            self.populate_key_async(path)
            self.statusBar().showMessage("Estimating key first…")
            return

        cw = ChordWorker(path, style=getattr(self, "analysis_style", "rock_pop"))
        try:
            cw.backend = getattr(self, 'backend', 'internal')
        except Exception:
            cw.backend = 'internal'
        cw.key_hint = dict(getattr(self, "last_key", {}) or {})
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
        """Normalize chord segments, then persist + paint."""
        try:
            segments = self._normalize_chords_for_view(segments)
        except Exception as ex:
            logging.error(f"_chords_ready normalize failed: {ex}")

        self.last_chords = list(segments)
        if hasattr(self, "wave") and self.wave is not None:
            self.wave.set_chords(self.last_chords)

        # best-effort session save & status
        try:
            self.save_session()
        except Exception:
            pass
        try:
            self.statusBar().showMessage(f"Chords: {len(self.last_chords)} segments")
            QtCore.QTimer.singleShot(1500, lambda: self.statusBar().clearMessage())
        except Exception:
            pass

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
        try:
            self._refresh_transport_icon()
        except Exception:
            pass
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
                self._clear_cached_analysis(keep_stems=False)
        except Exception:
            pass
        if not self.last_beats:
            self.populate_beats_async(self.current_path)
        if not self.last_chords:
            self.populate_chords_async(self.current_path, force=False)
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
