import cv2
import numpy as np


def order_points_cw(pts):
    arr = np.array(pts, dtype=np.float32)
    c   = arr.mean(axis=0)
    ang = np.arctan2(arr[:, 1] - c[1], arr[:, 0] - c[0])
    arr = arr[np.argsort(ang)]
    s   = arr.sum(axis=1)
    i   = int(np.argmin(s))
    arr = np.roll(arr, -i, axis=0)
    v1, v2 = arr[1] - arr[0], arr[3] - arr[0]
    if v1[0] * v2[1] - v1[1] * v2[0] > 0:
        arr = arr[[0, 3, 2, 1]]
    return arr


def auto_dst_rect(src4):
    tl, tr, br, bl = src4
    w = max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))
    h = max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))
    W, H = max(1, int(w + .5)), max(1, int(h + .5))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    return W, H, dst


def warp_pts(pts, H):
    a = np.float32(pts).reshape(-1, 1, 2)
    w = cv2.perspectiveTransform(a, H).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in w]


def poly_center(pts_int):
    M = cv2.moments(pts_int.astype(float))
    if M["m00"] == 0:
        return int(pts_int[:, 0].mean()), int(pts_int[:, 1].mean())
    return int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
