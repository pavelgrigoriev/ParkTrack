import math
from typing import Callable

import cv2
import numpy as np

from .colors import Colors
from .detector import Detection
from .geometry import poly_center


class DrawingEngine:

    def __init__(
        self,
        fill_alpha: float = 0.45,
        border_thick: int = 2,
        bev_fill_alpha: float = 0.50,
    ):
        self.fill_alpha = fill_alpha
        self.border_thick = border_thick
        self.bev_fill_alpha = bev_fill_alpha

    def draw_camera(
        self,
        frame: np.ndarray,
        polygons: list,
        dets: list[Detection],
        is_busy: Callable[[str], bool],
        iof_data: dict[str, float],
        zone_hull: np.ndarray | None,
        frame_idx: int,
    ) -> np.ndarray:

        vis = frame.copy()
        overlay = frame.copy()

        for p in polygons:
            pts = np.int32(p["points"])
            cv2.fillPoly(overlay, [pts], Colors.fill(is_busy(p["label"])))
        cv2.addWeighted(overlay, self.fill_alpha, vis,
                        1.0 - self.fill_alpha, 0, vis)

        pulse = 1.0 + 0.4 * math.sin(frame_idx * 0.15)
        for p in polygons:
            pts = np.int32(p["points"])
            busy = is_busy(p["label"])
            thick = self.border_thick
            if busy:
                thick = max(2, int(self.border_thick * pulse + 0.5))
            cv2.polylines(vis, [pts], True,
                          Colors.border(busy), thick, cv2.LINE_AA)

        for p in polygons:
            pts = np.int32(p["points"])
            busy = is_busy(p["label"])
            cx, cy = poly_center(pts)
            short = p["label"].split("_")[-1]
            iof_val = iof_data.get(p["label"], 0.0)

            line1 = f"#{short}"
            line2 = f"{'BUSY' if busy else 'FREE'} {iof_val:.0%}"
            (tw1, th1), _ = cv2.getTextSize(
                line1, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            (tw2, th2), _ = cv2.getTextSize(
                line2, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)
            tw = max(tw1, tw2)
            pad = 4
            rx1, ry1 = cx - tw // 2 - pad, cy - th1 - pad - 2
            rx2, ry2 = cx + tw // 2 + pad, cy + th2 + pad + 2

            roi = vis[max(0, ry1):max(0, ry2), max(0, rx1):max(0, rx2)]
            if roi.size > 0:
                bg = np.full_like(roi, Colors.fill(busy))
                cv2.addWeighted(bg, 0.6, roi, 0.4, 0, roi)

            cv2.putText(vis, line1, (cx - tw1 // 2, cy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        Colors.WHITE, 1, cv2.LINE_AA)
            cv2.putText(vis, line2, (cx - tw2 // 2, cy + th2 + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34,
                        Colors.WHITE, 1, cv2.LINE_AA)

        if zone_hull is not None:
            cv2.polylines(vis, [zone_hull.reshape(-1, 1, 2)],
                          True, Colors.CYAN, 1, cv2.LINE_AA)

        for d in dets:
            x1, y1, x2, y2 = (int(round(v)) for v in d.bbox_xyxy)
            cv2.rectangle(vis, (x1, y1), (x2, y2),
                          Colors.YELLOW, 2, cv2.LINE_AA)
            gx, gy = d.ground_point
            cv2.circle(vis, (int(gx), int(gy)), 4,
                       (0, 0, 255), -1, cv2.LINE_AA)
            txt = f"{d.cls_name} {d.conf:.0%}"
            if d.track_id >= 0:
                txt = f"[{d.track_id}] {txt}"
            (tw, th), _ = cv2.getTextSize(
                txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6),
                          (x1 + tw + 4, y1), Colors.YELLOW, -1)
            cv2.putText(vis, txt, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        Colors.BLACK, 1, cv2.LINE_AA)

        return vis

    def draw_bev(
        self,
        bev: np.ndarray,
        warped: list,
        is_busy: Callable[[str], bool],
        iof_data: dict[str, float],
    ) -> np.ndarray:

        overlay = bev.copy()
        for wp in warped:
            cv2.fillPoly(overlay, [wp["pts"]],
                         Colors.fill(is_busy(wp["label"])))
        cv2.addWeighted(overlay, self.bev_fill_alpha, bev,
                        1 - self.bev_fill_alpha, 0, bev)

        for wp in warped:
            busy = is_busy(wp["label"])
            cv2.polylines(bev, [wp["pts"]], True,
                          Colors.border(busy), 2, cv2.LINE_AA)
            cx, cy = poly_center(wp["pts"])
            short = wp["label"].split("_")[-1]
            iof_v = iof_data.get(wp["label"], 0.0)
            status = f"{'X' if busy else 'O'} {iof_v:.0%}"
            cv2.putText(bev, f"#{short}", (cx - 12, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        Colors.WHITE, 1, cv2.LINE_AA)
            cv2.putText(bev, status, (cx - 15, cy + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        Colors.WHITE, 1, cv2.LINE_AA)
        return bev

    @staticmethod
    def draw_stats_panel(
        canvas, x0, y0,
        total, free, busy,
        iof_thresh,
        panel_w=260,
        play_fps=0.0, det_fps=0.0, det_count=0,
    ):
        panel_h = 170
        cv2.rectangle(canvas, (x0, y0),
                      (x0 + panel_w, y0 + panel_h), Colors.PANEL_BG, -1)
        cv2.rectangle(canvas, (x0, y0),
                      (x0 + panel_w, y0 + panel_h), Colors.WHITE, 1)

        cv2.putText(canvas, "PARKING MONITOR",
                    (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    Colors.YELLOW, 2, cv2.LINE_AA)
        cv2.line(canvas, (x0 + 8, y0 + 30),
                 (x0 + panel_w - 8, y0 + 30), Colors.GRAY_TEXT, 1)

        cv2.putText(canvas, f"TOTAL:      {total}",
                    (x0 + 12, y0 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    Colors.WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0 + 12, y0 + 60),
                      (x0 + 26, y0 + 74), Colors.FILL_FREE, -1)
        cv2.putText(canvas, f"FREE:       {free}",
                    (x0 + 32, y0 + 73), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    Colors.GREEN_TEXT, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0 + 12, y0 + 80),
                      (x0 + 26, y0 + 94), Colors.FILL_BUSY, -1)
        cv2.putText(canvas, f"BUSY:       {busy}",
                    (x0 + 32, y0 + 93), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    Colors.RED_TEXT, 1, cv2.LINE_AA)

        bx, by, bw, bh = x0 + 12, y0 + 105, panel_w - 24, 18
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (80, 80, 80), -1)
        fw = int(bw * free / total) if total > 0 else bw
        if fw > 0:
            cv2.rectangle(canvas, (bx, by), (bx + fw, by + bh),
                          Colors.FILL_FREE, -1)
        if bw - fw > 0:
            cv2.rectangle(canvas, (bx + fw, by), (bx + bw, by + bh),
                          Colors.FILL_BUSY, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), Colors.WHITE, 1)
        cv2.putText(canvas, f"{free}/{total} free",
                    (bx + 6, by + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    Colors.WHITE, 1, cv2.LINE_AA)

        y_f = y0 + 145
        cv2.putText(canvas, f"Play:{play_fps:.1f}",
                    (x0 + 12, y_f), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    Colors.GREEN_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas,
                    f"NN:{det_fps:.1f} (#{det_count}) IoF>{iof_thresh:.0%}",
                    (x0 + 100, y_f), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    Colors.ORANGE_TEXT, 1, cv2.LINE_AA)

    def compose(
        self, cam, bev,
        play_fps, det_fps, det_count,
        idx, n_det,
        total, free, busy,
        iof_thresh,
    ):
        ch, cw = cam.shape[:2]
        bh, bw = bev.shape[:2]
        scale = ch / bh
        nbw = min(int(bw * scale), int(cw * 0.40))
        bev_r = cv2.resize(bev, (nbw, ch))

        out = np.zeros((ch, cw + nbw, 3), np.uint8)
        out[:, :cw] = cam
        out[:, cw:] = bev_r
        cv2.line(out, (cw, 0), (cw, ch), Colors.WHITE, 2)

        info = (f"Frame:{idx}  Det:{n_det}  "
                f"Play:{play_fps:.1f}  NN:{det_fps:.1f}")
        cv2.putText(out, info, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    Colors.YELLOW, 2, cv2.LINE_AA)

        self.draw_stats_panel(out, 10, 35, total, free, busy,
                              iof_thresh, 290,
                              play_fps, det_fps, det_count)

        cv2.putText(out, "BIRD'S-EYE VIEW", (cw + 8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    Colors.YELLOW, 2, cv2.LINE_AA)
        self.draw_stats_panel(out, cw + 8, 35, total, free, busy,
                              iof_thresh, min(nbw - 16, 290),
                              play_fps, det_fps, det_count)
        return out