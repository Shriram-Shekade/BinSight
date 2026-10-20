"""Live webcam demo: tray ROI, auto-trigger, overlay and tray animation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

from .config import UNSURE, Config, TriggerConfig, hex_to_bgr
from .model import Classifier, Result

WINDOW = "BinSight"
FONT = cv2.FONT_HERSHEY_SIMPLEX
GRID = 96  # ROI is shrunk to GRID x GRID grey for change detection
ANIM_SECONDS = 0.7
FLASH_SECONDS = 0.6
DESIGN_WIDTH = 960  # layout is designed at this width and scaled by display_width

# Trigger states
CAPTURING, ARMED, SETTLING, HELD = "capturing", "armed", "settling", "held"
STATE_TEXT = {
    CAPTURING: "Grabbing background - keep tray empty",
    ARMED: "Armed - place an item on the tray",
    SETTLING: "Hold still...",
    HELD: "Decided - remove the item",
}


class Trigger:
    """Fires once when something placed in the ROI has stayed still, then re-arms on clear."""

    def __init__(self, cfg: TriggerConfig):
        self.cfg = cfg
        self.state = CAPTURING
        self.begin_background(cfg.warmup_frames)

    def begin_background(self, skip: int = 0) -> None:
        self.state = CAPTURING
        self._skip = skip
        self._acc: list[np.ndarray] = []
        self._bg: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._still = 0
        self._clear = 0

    def hold(self) -> None:
        """Mark the current item as decided (after a manual SPACE too)."""
        self.state = HELD
        self._clear = 0

    @staticmethod
    def _prep(roi: np.ndarray) -> np.ndarray:
        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        grey = cv2.GaussianBlur(
            cv2.resize(grey, (GRID, GRID), interpolation=cv2.INTER_AREA), (5, 5), 0
        )
        grey = grey.astype(np.float32)
        return grey - grey.mean()  # cancels global exposure drift

    def update(self, roi: np.ndarray) -> bool:
        """Feed one ROI frame; True when it is time to classify."""
        cfg, cur = self.cfg, self._prep(roi)

        if self.state == CAPTURING:
            if self._skip > 0:
                self._skip -= 1
            else:
                self._acc.append(cur)
                if len(self._acc) >= cfg.background_frames:
                    self._bg = np.mean(self._acc, axis=0)
                    self.state = ARMED
            return False

        assert self._bg is not None
        changed = float((np.abs(cur - self._bg) > cfg.diff_level).mean())
        fire = False

        if self.state == ARMED:
            if changed > cfg.occupied_fraction:
                self.state, self._still = SETTLING, 0
        elif self.state == SETTLING:
            if changed < cfg.clear_fraction:
                self.state = ARMED
            else:
                moved = float((np.abs(cur - self._prev) > cfg.motion_level).mean())
                self._still = self._still + 1 if moved < cfg.still_fraction else 0
                if self._still >= cfg.still_frames:
                    self.hold()
                    fire = True
        elif self.state == HELD:
            self._clear = self._clear + 1 if changed < cfg.clear_fraction else 0
            if self._clear >= cfg.clear_frames:
                self.state = ARMED

        self._prev = cur
        return fire


@dataclass
class Animation:
    started: float
    start: tuple[int, int]
    compartment: str


def put(img, text, org, scale=0.6, color=(255, 255, 255), thick=1) -> None:
    """Text with a dark outline so it reads over any background."""
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):  # same thickness keeps glyph spacing equal
        cv2.putText(img, text, (x + dx, y + dy), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def shade(img, p0, p1, color=(0, 0, 0), alpha=0.6) -> None:
    """Translucent filled rectangle."""
    (x0, y0), (x1, y1) = p0, p1
    sub = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = cv2.addWeighted(sub, 1 - alpha, np.full_like(sub, color), alpha, 0)


def crop_16_9(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w * 9 > h * 16:  # too wide
        nw = h * 16 // 9
        return frame[:, (w - nw) // 2 : (w - nw) // 2 + nw]
    nh = w * 9 // 16
    return frame[(h - nh) // 2 : (h - nh) // 2 + nh]


class LiveApp:
    def __init__(self, cfg: Config, classifier: Classifier, threshold: float):
        self.cfg, self.clf, self.threshold = cfg, classifier, threshold
        self.w = cfg.display_width
        self.h = self.w * 9 // 16
        self.s = self.w / DESIGN_WIDTH
        side = int(self.h * cfg.roi_fraction)
        self.roi = ((self.w - side) // 2, (self.h - side) // 2, (self.w + side) // 2,
                    (self.h + side) // 2)  # fmt: skip
        self.panel_w = int(300 * self.s)
        self.trigger = Trigger(cfg.trigger)
        self.counts = dict.fromkeys(cfg.compartments, 0)
        self.colours = {k: hex_to_bgr(v) for k, v in cfg.colours.items()}
        self.result: Result | None = None
        self.anim: Animation | None = None
        self.landed_at = 0.0

    # --- decisions ---------------------------------------------------------

    def decide(self, raw: np.ndarray) -> None:
        """Classify the unmirrored ROI crop at native camera resolution."""
        k = raw.shape[1] / self.w
        x0, y0, x1, y1 = (int(v * k) for v in self.roi)
        crop = cv2.cvtColor(raw[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
        res = self.clf.classify(Image.fromarray(crop), self.threshold)
        self.result = res
        self.counts[res.compartment] += 1
        rx0, ry0, rx1, ry1 = self.roi
        self.anim = Animation(
            time.monotonic(), ((rx0 + rx1) // 2, (ry0 + ry1) // 2), res.compartment
        )
        self.trigger.hold()

        top3 = "  ".join(f"{n} {p:.2f}" for n, p in res.ranked()[:3])
        note = f"  ({res.reason})" if res.reason else ""
        print(
            f"{datetime.now():%H:%M:%S}  {res.compartment:<8} {res.top} {res.confidence:.2f}  "
            f"{res.latency_ms:.0f} ms  | {top3}{note}",
            flush=True,
        )

    def set_threshold(self, value: float) -> None:
        self.threshold = round(min(max(value, 0.05), 0.95), 2)
        print(f"threshold = {self.threshold:.2f}", flush=True)

    # --- drawing -----------------------------------------------------------

    def tiles(self) -> dict[str, tuple[int, int, int, int]]:
        """Compartment strip along the bottom of the video: name -> (x0, y0, x1, y1)."""
        names, gap = self.cfg.compartments, int(10 * self.s)
        tile_h = int(46 * self.s)
        tile_w = (self.w - gap * (len(names) + 1)) // len(names)
        y0 = self.h - tile_h - gap
        return {
            n: (gap + i * (tile_w + gap), y0, gap + i * (tile_w + gap) + tile_w, y0 + tile_h)
            for i, n in enumerate(names)
        }

    def draw_roi(self, img) -> None:
        x0, y0, x1, y1 = self.roi
        color = (255, 255, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 1, cv2.LINE_AA)
        arm = int(24 * self.s)
        for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            cv2.line(img, (cx, cy), (cx + dx * arm, cy), color, 3, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy * arm), color, 3, cv2.LINE_AA)
        put(img, "TRAY", (x0 + 6, y0 - 8), 0.5 * self.s, color)
        state = self.trigger.state
        hint = STATE_TEXT[state]
        hint_color = (0, 200, 255) if state in (CAPTURING, SETTLING) else (255, 255, 255)
        size = cv2.getTextSize(hint, FONT, 0.6 * self.s, 1)[0]
        put(img, hint, ((x0 + x1 - size[0]) // 2, y1 + int(24 * self.s)), 0.6 * self.s, hint_color)

    def draw_tiles(self, img, now: float) -> None:
        active = self.result.compartment if self.result else None
        since = now - self.landed_at
        for name, (x0, y0, x1, y1) in self.tiles().items():
            colour = self.colours[name]
            flash = name == active and since < FLASH_SECONDS
            shade(img, (x0, y0), (x1, y1), colour if flash else (0, 0, 0), 0.85 if flash else 0.55)
            cv2.rectangle(img, (x0, y0), (x1, y1), colour, 3 if name == active else 1, cv2.LINE_AA)
            size = cv2.getTextSize(name, FONT, 0.6 * self.s, 2)[0]
            text_colour = (0, 0, 0) if flash else colour
            cv2.putText(img, name, ((x0 + x1 - size[0]) // 2, (y0 + y1 + size[1]) // 2), FONT,
                        0.6 * self.s, text_colour, 2, cv2.LINE_AA)  # fmt: skip

    def draw_animation(self, img, now: float) -> None:
        if self.anim is None:
            return
        p = (now - self.anim.started) / ANIM_SECONDS
        if p >= 1:
            if self.landed_at < self.anim.started:
                self.landed_at = now
            self.anim = None
            return
        eased = p * p * (3 - 2 * p)
        x0, y0, x1, y1 = self.tiles()[self.anim.compartment]
        tx, ty = (x0 + x1) // 2, (y0 + y1) // 2
        sx, sy = self.anim.start
        cx, cy = int(sx + (tx - sx) * eased), int(sy + (ty - sy) * eased)
        half = int((26 - 12 * eased) * self.s)  # block shrinks as it drops
        colour = self.colours[self.anim.compartment]
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), colour, -1)
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), (255, 255, 255), 2)

    def draw_card(self, img) -> None:
        res = self.result
        if res is None:
            return
        s, colour = self.s, self.colours[res.compartment]
        x0, y0, w = int(12 * s), int(12 * s), int(300 * s)
        shade(img, (x0, y0), (x0 + w, y0 + int(212 * s)), alpha=0.65)
        cv2.rectangle(img, (x0, y0), (x0 + int(6 * s), y0 + int(212 * s)), colour, -1)
        tx = x0 + int(18 * s)

        put(img, res.compartment, (tx, y0 + int(38 * s)), 1.1 * s, colour, 2)
        put(img, f"{res.top}  {res.confidence * 100:.0f}%", (tx, y0 + int(64 * s)), 0.65 * s)
        bar_x0, bar_x1 = tx, x0 + w - int(12 * s)
        bar_y, bar_h = y0 + int(74 * s), int(14 * s)
        cv2.rectangle(img, (bar_x0, bar_y), (bar_x1, bar_y + bar_h), (70, 70, 70), -1)
        fill = bar_x0 + int((bar_x1 - bar_x0) * res.confidence)
        cv2.rectangle(img, (bar_x0, bar_y), (fill, bar_y + bar_h), colour, -1)
        tick = bar_x0 + int((bar_x1 - bar_x0) * self.threshold)
        cv2.line(img, (tick, bar_y - 4), (tick, bar_y + bar_h + 4), (255, 255, 255), 2)
        put(img, f"{res.latency_ms:.0f} ms", (tx, y0 + int(112 * s)), 0.55 * s, (200, 200, 200))
        if res.reason:
            put(img, res.reason, (tx + int(70 * s), y0 + int(112 * s)), 0.5 * s, (0, 160, 255))
        for i, (name, p) in enumerate(res.ranked()[:3]):
            y = y0 + int((142 + i * 24) * s)
            put(img, f"{i + 1}. {name}", (tx, y), 0.55 * s)
            put(img, f"{p:.2f}", (tx + int(190 * s), y), 0.55 * s)

    def render_panel(self, height: int) -> np.ndarray:
        s = self.s
        panel = np.full((height, self.panel_w, 3), (32, 32, 32), np.uint8)
        x = int(16 * s)
        put(panel, "BinSight", (x, int(40 * s)), 0.95 * s, (255, 255, 255), 2)
        short = self.clf.model_id.split("/")[-1]
        put(panel, short, (x, int(64 * s)), 0.45 * s, (170, 170, 170))
        put(
            panel,
            f"threshold {self.threshold:.2f}  (+/-)",
            (x, int(94 * s)),
            0.55 * s,
            (255, 255, 255),
        )

        total = sum(self.counts.values())
        put(panel, "THIS SESSION", (x, int(138 * s)), 0.5 * s, (170, 170, 170))
        bar_w = self.panel_w - 2 * x
        for i, name in enumerate(self.cfg.compartments):
            y = int((172 + i * 52) * s)
            n, colour = self.counts[name], self.colours[name]
            put(panel, name, (x, y), 0.6 * s, colour, 1)
            put(panel, str(n), (self.panel_w - x - int(30 * s), y), 0.7 * s, (255, 255, 255), 2)
            top = y + int(10 * s)
            cv2.rectangle(panel, (x, top), (x + bar_w, top + int(8 * s)), (60, 60, 60), -1)
            if total:
                cv2.rectangle(panel, (x, top), (x + int(bar_w * n / total), top + int(8 * s)),
                              colour, -1)  # fmt: skip

        rate = self.counts[UNSURE] / total * 100 if total else 0.0
        put(panel, f"total {total}", (x, int(448 * s)), 0.6 * s)
        put(panel, f"UNSURE rate {rate:.0f}%", (x, int(474 * s)), 0.6 * s, self.colours[UNSURE])
        put(panel, "SPACE classify  B background", (x, int(508 * s)), 0.42 * s, (150, 150, 150))
        put(panel, "+/- threshold   Q quit", (x, int(526 * s)), 0.42 * s, (150, 150, 150))
        return panel

    def render(self, frame: np.ndarray, now: float) -> np.ndarray:
        view = cv2.flip(frame, 1)  # mirrored for display only
        self.draw_roi(view)
        self.draw_card(view)
        self.draw_tiles(view, now)
        self.draw_animation(view, now)
        return np.hstack([view, self.render_panel(view.shape[0])])

    # --- main loop ---------------------------------------------------------

    def run(self) -> None:
        cap = cv2.VideoCapture(self.cfg.camera)
        if not cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.cfg.camera}. Allow camera access for your terminal in "
                "System Settings > Privacy & Security > Camera, or try --camera 1."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        x0, y0, x1, y1 = self.roi
        step = self.cfg.threshold_step
        try:
            while True:
                ok, raw = cap.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames.")
                raw = crop_16_9(raw)
                frame = cv2.resize(raw, (self.w, self.h), interpolation=cv2.INTER_AREA)

                if self.trigger.update(frame[y0:y1, x0:x1]):
                    self.decide(raw)

                cv2.imshow(WINDOW, self.render(frame, time.monotonic()))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key == ord(" "):
                    self.decide(raw)
                elif key in (ord("b"), ord("B")):
                    self.trigger.begin_background(0)
                    print("re-capturing background", flush=True)
                elif key in (ord("+"), ord("=")):
                    self.set_threshold(self.threshold + step)
                elif key in (ord("-"), ord("_")):
                    self.set_threshold(self.threshold - step)
        finally:
            cap.release()
            cv2.destroyAllWindows()


def run_live(cfg: Config, classifier: Classifier, threshold: float | None = None) -> None:
    LiveApp(cfg, classifier, cfg.threshold if threshold is None else threshold).run()
