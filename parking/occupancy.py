import cv2
import numpy as np

from .detector import Detection


class OccupancyTracker:

    def __init__(
        self,
        polygons: list,
        ewma_alpha: float = 0.40,
        occ_thresh: float = 0.35,
    ):
        self.ewma_alpha = ewma_alpha
        self.occ_thresh = occ_thresh

        self._contours: dict[str, np.ndarray] = {}
        self._areas: dict[str, float] = {}

        for p in polygons:
            cnt = np.array(p["points"], dtype=np.int32).reshape(-1, 1, 2)
            self._contours[p["label"]] = cnt
            self._areas[p["label"]] = max(
                1.0, cv2.contourArea(cnt.astype(np.float32))
            )

        self._occ: dict[str, float] = {p["label"]: 0.0 for p in polygons}
        self._iof: dict[str, float] = {p["label"]: 0.0 for p in polygons}

    @property
    def occ_data(self) -> dict[str, float]:
        return dict(self._occ)

    @property
    def iof_data(self) -> dict[str, float]:
        return dict(self._iof)

    def is_busy(self, label: str) -> bool:
        return self._occ.get(label, 0.0) >= self.occ_thresh

    def count_stats(self) -> tuple[int, int, int]:
        total = len(self._occ)
        busy = sum(1 for v in self._occ.values() if v >= self.occ_thresh)
        return total, total - busy, busy

    def update(
        self, dets: list[Detection]
    ) -> tuple[dict[str, float], dict[str, float]]:
        a = self.ewma_alpha

        ground_pts = [d.ground_point for d in dets]
        det_polys = [
            np.array([
                [d.x1, d.y1], [d.x2, d.y1],
                [d.x2, d.y2], [d.x1, d.y2],
            ], dtype=np.int32)
            for d in dets
        ]

        occupied: set = set()
        for gx, gy in ground_pts:
            for label, cnt in self._contours.items():
                if cv2.pointPolygonTest(cnt, (float(gx), float(gy)), False) >= 0:
                    occupied.add(label)
                    break


        iof_map: dict[str, float] = {}
        for label, cnt in self._contours.items():
            spot_pts = cnt.reshape(-1, 2)
            area = self._areas[label]
            best = 0.0
            for bp in det_polys:
                inter = self._intersect(spot_pts, bp)
                best = max(best, inter / area)
            iof_map[label] = best

        new_occ: dict[str, float] = {}
        for label in self._contours:
            raw = 1.0 if label in occupied else 0.0
            prev = self._occ.get(label, 0.0)
            new_occ[label] = a * raw + (1.0 - a) * prev

        self._occ = new_occ
        self._iof = iof_map
        return new_occ, iof_map

    @staticmethod
    def _intersect(poly1: np.ndarray, poly2: np.ndarray) -> float:
        p1 = cv2.convexHull(poly1.reshape(-1, 1, 2).astype(np.float32))
        p2 = cv2.convexHull(poly2.reshape(-1, 1, 2).astype(np.float32))
        ret, _ = cv2.intersectConvexConvex(p1, p2)
        return max(0.0, float(ret))