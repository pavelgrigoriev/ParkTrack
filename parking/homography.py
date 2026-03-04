import cv2
import numpy as np

from .geometry import auto_dst_rect, order_points_cw, warp_pts


class HomographyTransform:

    def __init__(self, rep_points: dict, polygons: list, dst_size=None):
        keys = [f"rep_point_{i}" for i in range(1, 5)]
        miss = [k for k in keys if k not in rep_points]
        if miss:
            raise RuntimeError(f"Не хватает реперных точек: {miss}")

        src4 = order_points_cw([rep_points[k] for k in keys])

        if dst_size:
            W, H = dst_size
            dst4 = np.float32(
                [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]
            )
        else:
            W, H, dst4 = auto_dst_rect(src4)

        self.H_mat = cv2.getPerspectiveTransform(src4, dst4)
        self.bev_w = int(W)
        self.bev_h = int(H)

        self.warped = []
        for p in polygons:
            wp = np.int32(warp_pts(p["points"], self.H_mat))
            self.warped.append({
                "label": p["label"],
                "pts": wp,
                "cnt": wp.reshape(-1, 1, 2),
            })

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(
            frame, self.H_mat, (self.bev_w, self.bev_h)
        )