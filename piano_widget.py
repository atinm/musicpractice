#
# MIDI mapping
GLOBAL_MIDI_MIN = 21   # A0
GLOBAL_MIDI_MAX = 108  # C8
DEFAULT_VISIBLE_MIN = 24  # C1
VISIBLE_SPAN = 72         # C1..B6 (6 octaves)

# Debug: print keyboard rects being drawn (throttled)
DEBUG_PIANO_RECTS = True
"""
Piano roll widget for displaying note confidence with 88-key piano visualization.
"""

import numpy as np
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush
from PySide6.QtWidgets import QWidget

# Piano roll constants
WHITE_KEYS = [0,2,4,5,7,9,11]  # within octave
BLACK_KEYS = [1,3,6,8,10]

KEY_ORDER = [0,1,2,3,4,5,6,7,8,9,10,11]
# Visual proportions
BLACK_KEY_HEIGHT_RATIO = 0.6  # black keys are ~60% the height of white keys
# Visual proportions
BLACK_KEY_WIDTH_RATIO = 0.6   # black keys are ~60% the width of a white key
WHITE_DIVIDER_COLOR = QColor(190, 190, 190)
TOP_BORDER_COLOR = QColor(150, 150, 150)

def is_white(midi_pc):  # pitch class 0..11
    return midi_pc in WHITE_KEYS

class PianoRollWidget(QWidget):
    """
    Displays a horizontal 88-key piano with confidence bars rising from keys.
    Call set_data(note_conf, hop_length, sr) and update_playhead(sample_pos).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.note_conf = None  # (frames, 88)
        self.sr = 44100
        self.hop_length = 512
        self.sample_pos = 0
        self.setMinimumHeight(70)
        self.setAutoFillBackground(False)  # Disable auto-fill to use custom painting
        self.setAttribute(Qt.WA_OpaquePaintEvent)

        # Start hidden; parent view controls show/hide
        self.setVisible(False)
        self.update()

        # Debug throttle timestamp (epoch seconds)
        self._debug_last_print = 0.0

        # Rendering style: 'line' or 'bars'
        self.render_style = 'line'

        # Prefer flats (Db, Eb, ...) vs sharps (C#, D#, ...)
        self.prefer_flats = False

        # Key/scale info for biasing adjacent-note decisions
        self.key_name = None
        self.tonic_pc = None  # 0..11
        self.is_minor = False

        # Peak label threshold control
        self.auto_peak_threshold = True   # if True, compute from distribution each frame
        self.peak_label_threshold = 0.12  # fallback/manual threshold on normalized [0,1]
    def set_peak_threshold(self, value: float | None, auto: bool | None = None):
        """Set manual threshold for showing peak labels (normalized 0..1 after per-frame normalization).
        If auto is True, use automatic percentile-based threshold; if False, force manual value.
        If auto is None, only update the value.
        """
        if value is not None:
            try:
                self.peak_label_threshold = max(0.0, min(1.0, float(value)))
            except Exception:
                pass
        if auto is not None:
            self.auto_peak_threshold = bool(auto)
        self.update()

    def set_auto_peak_threshold(self, enabled: bool):
        self.auto_peak_threshold = bool(enabled)
        self.update()
    def set_notation_preference(self, prefer_flats: bool):
        """Force accidental style. True = flats, False = sharps."""
        self.prefer_flats = bool(prefer_flats)
        self.update()

    def set_key_signature(self, key_name: str | None):
        """Set accidental preference from a key signature name, e.g. 'Eb major', 'F minor'.
        Flats: F, Bb, Eb, Ab, Db, Gb, Cb (and relative minors Dm, Gm, Cm, Fm, Bbm, Ebm, Abm)
        Sharps: G, D, A, E, B, F#, C# (and relative minors Em, Bm, F#m, C#m, G#m, D#m, A#m)
        If key is ambiguous or None, leaves current preference unchanged.
        """
        if not key_name:
            return
        k = str(key_name).strip().lower()
        flat_keys = {"f", "bb", "eb", "ab", "db", "gb", "cb",
                     "d minor", "dm", "g minor", "gm", "c minor", "cm", "f minor", "fm",
                     "bb minor", "bbm", "eb minor", "ebm", "ab minor", "abm"}
        sharp_keys = {"g", "d", "a", "e", "b", "f#", "c#",
                      "e minor", "em", "b minor", "bm", "f# minor", "f#m", "c# minor", "c#m",
                      "g# minor", "g#m", "d# minor", "d#m", "a# minor", "a#m"}
        # quick heuristics for explicit accidentals
        if ("flat" in k) or ("b" in k and not "#" in k and any(root in k for root in ["f","bb","eb","ab","db","gb","cb"])):
            self.prefer_flats = True
        elif ("#" in k) or ("sharp" in k) or any(root in k for root in sharp_keys):
            self.prefer_flats = False
        elif any(root == k or k.startswith(root + " ") for root in flat_keys):
            self.prefer_flats = True
        elif any(root == k or k.startswith(root + " ") for root in sharp_keys):
            self.prefer_flats = False
        # else: leave unchanged
        self.update()

        # Parse tonic and mode for biasing
        try:
            # crude parse: split first token like 'Eb'/'F#'/'C', and detect 'minor' keyword
            parts = k.replace("major", "").strip().split()
            root = parts[0] if parts else k
            self.is_minor = ("minor" in k) or k.endswith("m")
            names = {"c":0,"c#":1,"db":1,"d":2,"d#":3,"eb":3,"e":4,"f":5,"f#":6,"gb":6,
                     "g":7,"g#":8,"ab":8,"a":9,"a#":10,"bb":10,"b":11}
            self.tonic_pc = names.get(root, None)
            self.key_name = key_name
        except Exception:
            self.tonic_pc = None
            self.key_name = key_name


    def _pcs_for_scale(self, tonic_pc: int, is_minor: bool):
        """Return a set of pitch classes (0..11) for the key's diatonic scale.
        Major: 0,2,4,5,7,9,11; Natural minor: 0,2,3,5,7,8,10 (relative to tonic).
        """
        major = [0, 2, 4, 5, 7, 9, 11]
        minor = [0, 2, 3, 5, 7, 8, 10]
        base = minor if is_minor else major
        return { (tonic_pc + x) % 12 for x in base }

    def set_data(self, note_conf, hop_length, sr):
        self.note_conf = note_conf
        self.hop_length = hop_length
        self.sr = sr
        self.update()

    def clear_data(self):
        """Clear the note confidence data and show basic piano."""
        self.note_conf = None
        self.update()

    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self.update()

    def update_playhead(self, sample_pos: int):
        self.sample_pos = sample_pos
        self.update()

    def _frame_idx(self):
        return int(self.sample_pos / self.hop_length) if self.hop_length else 0

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()

        # Always draw a solid background first
        painter.fillRect(0, 0, w, h, QColor(20, 20, 23))

        # Base/smaller fonts
        base_font = painter.font()
        try:
            base_pt = float(base_font.pointSizeF()) if base_font.pointSizeF() > 0 else float(base_font.pointSize())
        except Exception:
            base_pt = 10.0
        small_pt = max(7.0, base_pt - 2.0)
        tiny_pt  = max(6.0, small_pt - 1.0)

        # (Keyboard is now drawn after data window alignment, or via _draw_basic_piano for no-data case.)

        # If no data, draw a basic 6-octave keyboard and return
        if self.note_conf is None or self.note_conf.size == 0:
            self._draw_basic_piano(painter, w, h)
            painter.end()
            return

        kb_h = int(h * 0.25)         # keys take ~25% of widget height; graph gets ~75%
        # Reserve a few pixels at the top so the line stroke doesn't clip
        stroke_px = 2  # matches the pen width below
        top_pad = max(5, stroke_px + 3)   # extra headroom
        bar_h = max(1, h - kb_h - top_pad)  # bar region above keys
        base_y = h - kb_h                    # baseline at top of keyboard

        # --- Debug capture containers ---
        dbg_white_rects = []
        dbg_black_rects = []
        dbg_info = {"kb_h": kb_h, "base_y": base_y}

        # layout: 72 keys across width (6 octaves)
        key_w = w / float(VISIBLE_SPAN)

        # Determine visible MIDI range (default C1..B6, may be shifted up/down and centered)
        vis_min = DEFAULT_VISIBLE_MIN
        vis_max = vis_min + VISIBLE_SPAN - 1

        # draw confidence as a line graph (with light area fill) aligned to piano keys
        try:
            def _interp_vector(note_conf, hop_length, sample_pos):
                if note_conf is None or getattr(note_conf, 'size', 0) == 0:
                    return None
                f = sample_pos / float(max(1, hop_length))
                i0 = int(f)
                i0 = max(0, min(i0, note_conf.shape[0]-1))
                i1 = min(i0 + 1, note_conf.shape[0]-1)
                frac = float(f - i0)
                return (1.0 - frac) * note_conf[i0] + frac * note_conf[i1]

            conf = _interp_vector(self.note_conf, self.hop_length, self.sample_pos)
            if conf is not None:
                import numpy as _np
                # Ensure length-88 vector (A0..C8)
                if getattr(conf, 'shape', None) is not None and conf.shape[-1] != 88:
                    src_idx = _np.linspace(0.0, 1.0, num=conf.shape[-1], endpoint=True)
                    dst_idx = _np.linspace(0.0, 1.0, num=88, endpoint=True)
                    conf = _np.interp(dst_idx, src_idx, conf.astype(float))
                conf = _np.asarray(conf, dtype=float)
                # Remove a small floor and normalize to 99th percentile for proportional heights
                floor = float(_np.percentile(conf, 5)) if conf.size else 0.0
                conf = _np.clip(conf - floor, 0.0, None)
                peak = float(_np.percentile(conf, 99)) if conf.size else 1.0
                if not _np.isfinite(peak) or peak <= 1e-8:
                    peak = 1.0
                conf = conf / peak

                # Determine the visible 6-octave window (default C1..B6) and shift up if peak is above B6
                # conf is 88-long for A0..C8; map index to MIDI with GLOBAL_MIDI_MIN
                top_idx_full = int(_np.argmax(conf))
                top_midi = GLOBAL_MIDI_MIN + top_idx_full
                vis_min = DEFAULT_VISIBLE_MIN
                vis_max = vis_min + VISIBLE_SPAN - 1
                max_start = GLOBAL_MIDI_MAX - VISIBLE_SPAN + 1
                # Center the window around the strongest note where possible
                desired_start = int(top_midi) - (VISIBLE_SPAN // 2)
                vis_min = max(GLOBAL_MIDI_MIN, min(desired_start, max_start))
                vis_max = vis_min + VISIBLE_SPAN - 1
                # Ensure inclusion if clamped pushed the note outside
                if top_midi < vis_min:
                    vis_min = max(GLOBAL_MIDI_MIN, min(int(top_midi), max_start))
                    vis_max = vis_min + VISIBLE_SPAN - 1
                elif top_midi > vis_max:
                    vis_min = max(GLOBAL_MIDI_MIN, min(int(top_midi) - VISIBLE_SPAN + 1, max_start))
                    vis_max = vis_min + VISIBLE_SPAN - 1

                # Align the start of the window to a C so labels fall on white keys
                max_start = GLOBAL_MIDI_MAX - VISIBLE_SPAN + 1
                off = vis_min % 12
                vis_min = max(GLOBAL_MIDI_MIN, min(vis_min - off, max_start))
                vis_max = vis_min + VISIBLE_SPAN - 1

                # --- draw keyboard with continuous white base, then black keys, then white-only dividers ---
                # 1) Continuous white base across full keyboard area
                painter.fillRect(QtCore.QRectF(0, base_y, w, kb_h), QColor(240, 240, 240))
                dbg_white_rects.append((-1, 0.0, float(base_y), float(w), float(kb_h)))

                black_h = max(1.0, kb_h * BLACK_KEY_HEIGHT_RATIO)

                # Build equal-width WHITE layout using a running cursor anchored at x=0
                whites_idx = [i for i in range(VISIBLE_SPAN) if is_white((vis_min + i) % 12)]
                n_whites = len(whites_idx)
                ww = w / float(max(1, n_whites))
                # left edges of whites in *order* (length n_whites)
                white_edges = []
                x_cursor = 0.0
                for _ in range(n_whites):
                    white_edges.append(x_cursor)
                    x_cursor += ww
                # maps between semitone index ↔ white-order
                i_to_order = {i: k for k, i in enumerate(whites_idx)}
                order_to_i = {k: i for k, i in enumerate(whites_idx)}

                # 2) Draw black keys (shorter + narrower), centered between adjacent equal-width whites
                for i in range(VISIBLE_SPAN):
                    midi = vis_min + i
                    pc = midi % 12
                    if is_white(pc):
                        continue
                    # find preceding white in semitone space and its order position
                    prev_i = i - 1
                    while prev_i >= 0 and not is_white((vis_min + prev_i) % 12):
                        prev_i -= 1
                    # if we don't find both neighbors, skip
                    if prev_i < 0 or (prev_i not in i_to_order) or (i_to_order[prev_i] + 1 >= n_whites):
                        continue
                    left_ord = i_to_order[prev_i]
                    right_ord = left_ord + 1
                    # centers between these two white keys
                    x_left_white  = white_edges[left_ord]
                    x_right_white = white_edges[right_ord]
                    # center between the two *full-width* whites
                    cx = 0.5 * ((x_left_white + ww) + x_right_white)
                    bw = ww * BLACK_KEY_WIDTH_RATIO
                    xk = cx - (bw * 0.5)
                    rect = QtCore.QRectF(xk, float(base_y), bw, float(black_h))
                    painter.fillRect(rect, QColor(30, 30, 30))
                    painter.setPen(QPen(QColor(10, 10, 10)))
                    painter.drawRect(rect)
                    dbg_black_rects.append((i, float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height())))
                    dbg_info["black_h"] = float(black_h)

                # 3) Subtle top border separating graph from keys
                painter.setPen(QPen(TOP_BORDER_COLOR))
                painter.drawLine(0, int(base_y), int(w), int(base_y))

                # 4) Tail region: ensure the area under the black keys is white, then draw **white-only** dividers
                tail_y = base_y + black_h
                tail_h = max(0.0, kb_h - black_h)
                if tail_h > 0.0:
                    painter.fillRect(QtCore.QRectF(0, tail_y, w, tail_h), QColor(240, 240, 240))

                    painter.setPen(QPen(WHITE_DIVIDER_COLOR))
                    bottom_y = int(base_y + kb_h)
                    # full-height dividers at B–C and E–F; short dividers elsewhere
                    for ord_idx in range(1, n_whites):
                        # semitone index of the *right* white in this boundary
                        i_right = whites_idx[ord_idx]
                        pc_right = (vis_min + i_right) % 12
                        x_div = white_edges[ord_idx]
                        if pc_right in (0, 5):  # C or F (boundary after B or E)
                            painter.drawLine(x_div, int(base_y), x_div, bottom_y)
                        else:
                            painter.drawLine(x_div, int(tail_y), x_div, bottom_y)
                    # Bottom edge
                    painter.drawLine(0, bottom_y, int(w), bottom_y)

                # --- Debug print throttled ---
                import time as _time
                if DEBUG_PIANO_RECTS:
                    now = _time.time()
                    # Print at most ~4 times per second
                    if (now - getattr(self, "_debug_last_print", 0.0)) > 0.25:
                        self._debug_last_print = now
                        # Limit how many rects we print to keep output readable
                        WMAX = 24
                        BMAX = 24
                        def _fmt(items, maxn):
                            s = []
                            for (idx, x, y, w_, h_) in items[:maxn]:
                                s.append(f"[i={idx:02d} x={x:.1f} y={y:.1f} w={w_:.1f} h={h_:.1f}]")
                            more = len(items) - maxn
                            if more > 0:
                                s.append(f"... (+{more} more)")
                            return " ".join(s)
                        print(
                            "PIANO-RECTS",
                            f"kb_h={dbg_info.get('kb_h')} base_y={dbg_info.get('base_y')} black_h={dbg_info.get('black_h', 'n/a')}",
                            "\n  whites:", _fmt(dbg_white_rects, WMAX),
                            "\n  blacks:", _fmt(dbg_black_rects, BMAX),
                            f"\n  vis_min={vis_min} vis_max={vis_max} key_w={key_w:.2f}"
                        )

                # Slice the confidence vector EXACTLY to the visible MIDI window [vis_min..vis_max]
                start_idx = int(max(0, vis_min - GLOBAL_MIDI_MIN))
                end_idx   = int(max(0, vis_max  - GLOBAL_MIDI_MIN + 1))  # inclusive vis_max → +1 for slice end
                conf_vis = conf[start_idx:end_idx]
                # Ensure length equals the key span (VISIBLE_SPAN); pad if necessary at boundaries
                if conf_vis.shape[0] != VISIBLE_SPAN:
                    need = VISIBLE_SPAN - conf_vis.shape[0]
                    if need > 0:
                        conf_vis = _np.pad(conf_vis, (0, need), mode='constant')
                    else:
                        conf_vis = conf_vis[:VISIBLE_SPAN]

                # Recompute normalization over the visible window for proportional heights
                floor_v = float(_np.percentile(conf_vis, 5)) if conf_vis.size else 0.0
                conf_vis = _np.clip(conf_vis - floor_v, 0.0, None)
                peak_v = float(_np.percentile(conf_vis, 99)) if conf_vis.size else 1.0
                if not _np.isfinite(peak_v) or peak_v <= 1e-8:
                    peak_v = 1.0
                conf_vis = conf_vis / peak_v

                # Compute drawing/label threshold (auto percentile or manual) on visible window
                if self.auto_peak_threshold:
                    thr_label = max(self.peak_label_threshold, float(_np.percentile(conf_vis, 85)))
                else:
                    thr_label = float(self.peak_label_threshold)

                # Mask out low-confidence values so the fill hugs the baseline
                conf_draw = _np.where(conf_vis >= thr_label, conf_vis, 0.0)

                # If there is effectively no signal, skip drawing the line/fill
                if not _np.isfinite(conf_draw.max()) or float(conf_draw.max()) <= 1e-4:
                    pass  # leave baseline only
                else:
                    # X centers per visible *semitone* using equal-width white-key layout
                    xs_list = []
                    for i in range(VISIBLE_SPAN):
                        if i in i_to_order:
                            # white key center
                            xs_list.append((i_to_order[i] + 0.5) * ww)
                        else:
                            # black key: center between adjacent whites
                            prev_i = i - 1
                            while prev_i >= 0 and not is_white((vis_min + prev_i) % 12):
                                prev_i -= 1
                            if prev_i < 0 or (prev_i not in i_to_order) or (i_to_order[prev_i] + 1 >= n_whites):
                                # fallback: keep previous spacing if edge-case
                                xs_list.append((i * key_w) + (key_w * 0.5))
                            else:
                                left_ord = i_to_order[prev_i]
                                xs_list.append((left_ord + 1.0) * ww)  # midpoint of the two neighboring whites
                    xs = _np.asarray(xs_list, dtype=float)
                    ys = base_y - (bar_h * (conf_draw ** 0.9))
                    # Clamp to avoid clipping at the very top/bottom of the graph region
                    ys = _np.clip(ys, base_y - bar_h + stroke_px + 1, base_y - 1)

                    if self.render_style == 'line':
                        # Build a filled polygon to bottom of bar region for area under curve
                        pts = [QtCore.QPointF(xs[0], base_y)]
                        for x, y in zip(xs, ys):
                            pts.append(QtCore.QPointF(float(x), float(y)))
                        pts.append(QtCore.QPointF(xs[-1], base_y))
                        poly = QtGui.QPolygonF(pts)

                        # Fill (soft)
                        painter.setBrush(QBrush(QColor(120, 180, 255, 60)))
                        painter.setPen(Qt.NoPen)
                        painter.drawPolygon(poly)

                        # Stroke the confidence line
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor(120, 180, 255, 220), 2))
                        # Simple polyline across key centers
                        for i in range(1, VISIBLE_SPAN):
                            painter.drawLine(int(xs[i-1]), int(ys[i-1]), int(xs[i]), int(ys[i]))

                # Key-aware bias for labeling: prefer notes in the current key/scale
                weights12 = _np.ones(12, dtype=float)
                if self.tonic_pc is not None:
                    pcs = self._pcs_for_scale(int(self.tonic_pc), bool(self.is_minor))
                    for pc in pcs:
                        weights12[pc] = 1.15  # small boost for diatonic notes
                # Build biased confidence for peak picking
                pcs_vis = _np.array([(vis_min + i) % 12 for i in range(VISIBLE_SPAN)], dtype=int)
                conf_bias = conf_vis * weights12[pcs_vis]

                # Candidate indices above threshold
                cand = _np.where(conf_vis >= thr_label)[0]
                if cand.size > 0:
                    # Sort by biased confidence
                    cand = cand[_np.argsort(conf_bias[cand])[::-1]]
                    # Simple non-maximum suppression: keep peaks not adjacent to a kept one
                    keep = []
                    taken = _np.zeros(VISIBLE_SPAN, dtype=bool)
                    for vi in cand:
                        if taken[max(0, vi-1):min(VISIBLE_SPAN, vi+2)].any():
                            continue
                        keep.append(int(vi))
                        taken[max(0, vi-1):min(VISIBLE_SPAN, vi+2)] = True
                        if len(keep) >= 4:
                            break
                    if keep:
                        label_font = QtGui.QFont(base_font)
                        label_font.setPointSizeF(tiny_pt)
                        painter.setFont(label_font)
                        fm2 = painter.fontMetrics()
                        y_label = base_y - 4  # just above the keys
                        NAMES_SHARP = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                        NAMES_FLAT  = ["C","Db","D","Eb","E","F","Gb","G","Ab","A","Bb","B"]
                        for vi in keep:
                            midi = vis_min + int(vi)
                            pc = midi % 12
                            label = (NAMES_FLAT if self.prefer_flats else NAMES_SHARP)[pc]
                            x = float(xs[vi])
                            tw = fm2.horizontalAdvance(label)
                            painter.setPen(Qt.NoPen)
                            painter.setBrush(QBrush(QColor(10, 10, 10, 160)))
                            pad_x = 3; pad_y = 1
                            rect = QtCore.QRectF(x - (tw/2 + pad_x), y_label - fm2.ascent() - pad_y,
                                                 tw + 2*pad_x, fm2.height() + 2*pad_y)
                            painter.drawRoundedRect(rect, 3, 3)
                            painter.setBrush(Qt.NoBrush)
                            painter.setPen(QPen(QColor(200, 220, 255, 230)))
                            painter.drawText(QtCore.QPointF(x - tw/2, y_label), label)
                    else:
                        # Fallback: draw bars (original style)
                        thr = 0.06
                        top_k = 3
                        order = _np.argsort(conf_vis)[::-1][:top_k]
                        for i in order:
                            c = float(conf_vis[i])
                            if c <= thr:
                                continue
                        height = bar_h * (c ** 0.85)
                        x_center = float(xs[i])
                        bw = ww * 0.6
                        rect = QRectF(x_center - bw/2.0, base_y - height, bw, height)
                        painter.fillRect(rect, QColor(100, 180, 255, 220))
        except Exception as e:
            print(f"Error drawing confidence line: {e}")

        # playhead line (center) — restrict to the graph area (above keys)
        graph_top = max(0, int(base_y - bar_h))
        graph_bottom = int(base_y)
        painter.setPen(QPen(QColor(255, 255, 255, 120), 1))
        painter.drawLine(int(w/2), graph_top, int(w/2), graph_bottom)

        # label C keys for orientation (smaller font), follow the visible window
        painter.setPen(QPen(QColor(60, 60, 60)))
        key_font = QtGui.QFont(base_font)
        key_font.setPointSizeF(small_pt)
        painter.setFont(key_font)
        fm = painter.fontMetrics()
        text_y = base_y + kb_h - max(2, fm.descent() + 2)
        for i in range(VISIBLE_SPAN):
            midi = vis_min + i
            if midi % 12 == 0:  # C
                octave = midi // 12 - 1
                label = f"C{octave}"
                x_center = i * key_w + key_w * 0.5
                text_w = fm.horizontalAdvance(label)
                painter.drawText(int(x_center - text_w/2), int(text_y), label)

        painter.end()

    def _draw_basic_piano(self, painter, w, h):
        """Draw a basic piano when no data is available."""
        kb_h = int(h * 0.25)
        key_w = w / float(VISIBLE_SPAN)
        base_y = h - kb_h
        vis_min = DEFAULT_VISIBLE_MIN

        black_h = max(1.0, kb_h * BLACK_KEY_HEIGHT_RATIO)

        dbg_white_rects = []
        dbg_black_rects = []
        dbg_info = {"kb_h": kb_h, "base_y": base_y}

        # --- draw keyboard with continuous white base, then black keys, then white-only dividers ---
        # 1) Continuous white base across full keyboard area
        painter.fillRect(QtCore.QRectF(0, base_y, w, kb_h), QColor(240, 240, 240))
        dbg_white_rects.append((-1, 0.0, float(base_y), float(w), float(kb_h)))

        # Build equal-width WHITE layout using a running cursor anchored at x=0
        whites_idx = [i for i in range(VISIBLE_SPAN) if is_white((vis_min + i) % 12)]
        n_whites = len(whites_idx)
        ww = w / float(max(1, n_whites))
        white_edges = []
        x_cursor = 0.0
        for _ in range(n_whites):
            white_edges.append(x_cursor)
            x_cursor += ww
        i_to_order = {i: k for k, i in enumerate(whites_idx)}
        order_to_i = {k: i for k, i in enumerate(whites_idx)}

        # 2) Draw black keys (shorter + narrower), centered between adjacent equal-width whites
        for i in range(VISIBLE_SPAN):
            midi = vis_min + i
            pc = midi % 12
            if is_white(pc):
                continue
            # find preceding white in semitone space and its order position
            prev_i = i - 1
            while prev_i >= 0 and not is_white((vis_min + prev_i) % 12):
                prev_i -= 1
            # if we don't find both neighbors, skip
            if prev_i < 0 or (prev_i not in i_to_order) or (i_to_order[prev_i] + 1 >= n_whites):
                continue
            left_ord = i_to_order[prev_i]
            right_ord = left_ord + 1
            # centers between these two white keys
            x_left_white  = white_edges[left_ord]
            x_right_white = white_edges[right_ord]
            cx = 0.5 * ((x_left_white + ww) + x_right_white)
            bw = ww * BLACK_KEY_WIDTH_RATIO
            xk = cx - (bw * 0.5)
            rect = QtCore.QRectF(xk, float(base_y), bw, float(black_h))
            painter.fillRect(rect, QColor(30, 30, 30))
            painter.setPen(QPen(QColor(10, 10, 10)))
            painter.drawRect(rect)
            dbg_black_rects.append((i, float(rect.x()), float(rect.y()), float(rect.width()), float(rect.height())))
            dbg_info["black_h"] = float(black_h)

        # 3) Subtle top border at the start of the keys
        painter.setPen(QPen(TOP_BORDER_COLOR))
        painter.drawLine(0, int(base_y), int(w), int(base_y))

        # 4) Tail region: ensure the area under the black keys is white, then draw **white-only** dividers
        tail_y = base_y + black_h
        tail_h = max(0.0, kb_h - black_h)
        if tail_h > 0.0:
            painter.fillRect(QtCore.QRectF(0, tail_y, w, tail_h), QColor(240, 240, 240))

            painter.setPen(QPen(WHITE_DIVIDER_COLOR))
            bottom_y = int(base_y + kb_h)
            # full-height dividers at B–C and E–F; short dividers elsewhere
            for ord_idx in range(1, n_whites):
                # semitone index of the *right* white in this boundary
                i_right = whites_idx[ord_idx]
                pc_right = (vis_min + i_right) % 12
                x_div = white_edges[ord_idx]
                if pc_right in (0, 5):  # C or F (boundary after B or E)
                    painter.drawLine(x_div, int(base_y), x_div, bottom_y)
                else:
                    painter.drawLine(x_div, int(tail_y), x_div, bottom_y)
            # Bottom edge
            painter.drawLine(0, bottom_y, int(w), bottom_y)

        # --- Debug print throttled ---
        import time as _time
        if DEBUG_PIANO_RECTS:
            now = _time.time()
            # Print at most ~4 times per second
            if (now - getattr(self, "_debug_last_print", 0.0)) > 0.25:
                self._debug_last_print = now
                WMAX = 24
                BMAX = 24
                def _fmt(items, maxn):
                    s = []
                    for (idx, x, y, w_, h_) in items[:maxn]:
                        s.append(f"[i={idx:02d} x={x:.1f} y={y:.1f} w={w_:.1f} h={h_:.1f}]")
                    more = len(items) - maxn
                    if more > 0:
                        s.append(f"... (+{more} more)")
                    return " ".join(s)
                print(
                    "PIANO-RECTS",
                    f"kb_h={dbg_info.get('kb_h')} base_y={dbg_info.get('base_y')} black_h={dbg_info.get('black_h', 'n/a')}",
                    "\n  whites:", _fmt(dbg_white_rects, WMAX),
                    "\n  blacks:", _fmt(dbg_black_rects, BMAX),
                    f"\n  vis_min={vis_min} vis_max={vis_min + VISIBLE_SPAN - 1} key_w={key_w:.2f}"
                )

        # Label C keys following the default window (C1..B6)
        painter.setPen(QPen(QColor(60, 60, 60)))
        fm = painter.fontMetrics()
        text_y = base_y + kb_h - max(2, fm.descent() + 2)
        for i in range(VISIBLE_SPAN):
            midi = vis_min + i
            if midi % 12 == 0:
                octave = midi // 12 - 1
                label = f"C{octave}"
                x_center = i * key_w + key_w * 0.5
                text_w = fm.horizontalAdvance(label)
                painter.drawText(int(x_center - text_w/2), int(text_y), label)