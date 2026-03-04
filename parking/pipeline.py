import os
import time

import cv2
import numpy as np

from .detector import Detector
from .drawing import DrawingEngine
from .fps_counter import FpsCounter
from .homography import HomographyTransform
from .occupancy import OccupancyTracker
from .xml_parser import parse_cvat_xml


class ParkingPipeline:

    def __init__(
        self, *,
        xml_path: str,
        model_path: str,
        video_path: str,
        output_path: str,
        device: str = "cuda:0",
        conf: float = 0.3,
        imgsz: int = 1280,
        show: bool = True,
        occ_thresh: float = 0.35,
        ewma_alpha: float = 0.40,
        iof_thresh: float = 0.15,
        dst_size=None,
        pad_zone: int = 60,
        fill_alpha: float = 0.45,
        border_thick: int = 2,
        bev_fill_alpha: float = 0.50,
        process_every: int = 1,
    ):
        self.video_path = video_path
        self.output_path = output_path
        self.show = show
        self.occ_thresh = occ_thresh
        self.iof_thresh = iof_thresh
        self.pad_zone = pad_zone
        self.process_every = max(1, process_every)

        print("[1/4] Парсинг XML …")
        self.rep_points, self.polygons, self.img_meta = parse_cvat_xml(xml_path)
        print(f"      Парковочных мест: {len(self.polygons)}")
        print(f"      Реперных точек:   {len(self.rep_points)}")

        print("[2/4] Гомография …")
        self.homo = HomographyTransform(
            self.rep_points, self.polygons, dst_size
        )

        print("[3/4] Детектор …")
        self.detector = Detector(model_path, device, conf, imgsz=imgsz)

        self.occ_tracker = OccupancyTracker(
            self.polygons, ewma_alpha, occ_thresh
        )
        self.drawer = DrawingEngine(fill_alpha, border_thick, bev_fill_alpha)

        self._zone_mask = None
        self._zone_hull = None
        self._frame_idx = 0
        self._last_dets = []

        print(f"[4/4] Готов.")
        print(f"      imgsz:         {imgsz}")
        print(f"      process_every: {self.process_every}")
        print(f"      occ_thresh:    {self.occ_thresh}")
        print(f"      ewma_alpha:    {ewma_alpha}")
        print(f"      iof_thresh:    {self.iof_thresh}\n")

    def _build_zone_mask(self, shape):
        h, w = shape[:2]
        all_pts = []
        for p in self.polygons:
            all_pts.extend(p["points"])
        hull = cv2.convexHull(np.float32(all_pts).reshape(-1, 1, 2))
        self._zone_hull = hull.reshape(-1, 2).astype(np.int32)
        self._zone_mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(self._zone_mask, self._zone_hull, 255)
        if self.pad_zone > 0:
            kern = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.pad_zone * 2, self.pad_zone * 2),
            )
            self._zone_mask = cv2.dilate(self._zone_mask, kern)

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Не удалось открыть: {self.video_path}")

        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Видео: {fw}x{fh} @ {fps_v:.1f} fps, "
              f"~{total_frames} кадров\n")

        ok, first = cap.read()
        if not ok:
            raise RuntimeError("Не прочитать первый кадр")
        self._build_zone_mask(first.shape)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        os.makedirs(os.path.dirname(self.output_path) or ".",
                    exist_ok=True)

        writer = None
        play_fps_c = FpsCounter(window=60)
        det_fps_c = FpsCounter(window=30)
        t_wall = time.time()
        det_count = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                self._frame_idx += 1

                do_detect = (self._frame_idx % self.process_every == 0)

                if do_detect:
                    t0 = time.time()
                    dets = self.detector.track(frame, self._zone_mask)
                    self.occ_tracker.update(dets)
                    self._last_dets = dets
                    det_count += 1
                    det_fps_c.tick()
                else:
                    dets = self._last_dets

                occ = self.occ_tracker.occ_data
                iof = self.occ_tracker.iof_data
                total, free, busy = self.occ_tracker.count_stats()

                def is_busy(label):
                    return occ.get(label, 0.0) >= self.occ_thresh

                cam_vis = self.drawer.draw_camera(
                    frame, self.polygons, dets,
                    is_busy, iof,
                    self._zone_hull, self._frame_idx,
                )

                bev_frame = self.homo.warp_frame(frame)
                bev_vis = self.drawer.draw_bev(
                    bev_frame, self.homo.warped, is_busy, iof,
                )

                play_fps_c.tick()

                composed = self.drawer.compose(
                    cam_vis, bev_vis,
                    play_fps_c.fps, det_fps_c.fps, det_count,
                    self._frame_idx, len(dets),
                    total, free, busy,
                    self.iof_thresh,
                )

                if writer is None:
                    oh, ow = composed.shape[:2]
                    writer = cv2.VideoWriter(
                        self.output_path,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps_v, (ow, oh),
                    )
                    print(f"Выход: {ow}x{oh}\n")
                writer.write(composed)

                if self.show:
                    disp = composed
                    if disp.shape[1] > 1920:
                        s = 1920 / disp.shape[1]
                        disp = cv2.resize(disp, None, fx=s, fy=s)
                    cv2.imshow("Parking Monitor", disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if self._frame_idx % 100 == 0:
                    el = time.time() - t_wall
                    print(
                        f"  #{self._frame_idx}/{total_frames}  "
                        f"det={len(dets)}  "
                        f"T={total} F={free} B={busy}  "
                        f"play={play_fps_c.fps:.1f}  "
                        f"nn={det_fps_c.fps:.1f}(#{det_count})  "
                        f"wall={el:.0f}s"
                    )

        finally:
            cap.release()
            if writer:
                writer.release()
            if self.show:
                cv2.destroyAllWindows()

            total, free, busy = self.occ_tracker.count_stats()
            el = time.time() - t_wall
            print(f"\n{'=' * 50}")
            print(f"Итог:  TOTAL={total}  FREE={free}  BUSY={busy}")
            print(f"Кадров: {self._frame_idx}  Детекций: {det_count}")
            print(f"Время: {el:.1f}s  "
                  f"Avg: {self._frame_idx / max(el, 1e-6):.1f} fps")
            print(f"Готово → {self.output_path}")