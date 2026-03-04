from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO

VEHICLE_CLASSES = [2, 5, 7]


@dataclass
class Detection:
    # оригинальные координаты (для отрисовки)
    x1: float
    y1: float
    x2: float
    y2: float
    # сжатые координаты (для логики occupancy)
    sx1: float
    sy1: float
    sx2: float
    sy2: float
    conf: float
    cls_id: int
    cls_name: str
    track_id: int = -1

    @property
    def bbox_xyxy(self):
        """Оригинальный bbox — для отрисовки."""
        return self.x1, self.y1, self.x2, self.y2

    @property
    def shrunk_xyxy(self):
        """Сжатый bbox — для occupancy."""
        return self.sx1, self.sy1, self.sx2, self.sy2

    @property
    def ground_point(self):
        """Нижний центр СЖАТОГО bbox."""
        return (self.sx1 + self.sx2) / 2.0, float(self.sy2)

    @property
    def center(self):
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


def _shrink_box(x1, y1, x2, y2, ratio):
    if isinstance(ratio, (tuple, list)):
        rx, ry = ratio
    else:
        rx = ry = ratio

    w = x2 - x1
    h = y2 - y1
    dx = w * rx
    dy = h * ry

    return (
        x1 + dx,
        y1 + dy,
        x2 - dx,
        y2 - dy,
    )


class Detector:

    def __init__(
            self,
            model_path: str,
            device: str = "cuda:0",
            conf: float = 0.3,
            classes: list[int] | None = None,
            imgsz: int = 1280,
            bbox_shrink: float = 0.10,
    ):
        self.model = YOLO(model_path)
        self.model.to(device)
        self.conf = conf
        self.classes = classes or list(VEHICLE_CLASSES)
        self.imgsz = imgsz
        self.bbox_shrink = bbox_shrink

        print(f"      Детектор: {model_path}  device={device}")
        print(f"      Классы:   {self.classes}  conf≥{self.conf}")
        print(f"      imgsz:    {self.imgsz}")
        print(f"      shrink:   {self.bbox_shrink}")

    def track(
            self,
            frame: np.ndarray,
            zone_mask: np.ndarray | None = None,
    ) -> list[Detection]:

        results = self.model.track(
            source=frame,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes
        names = results[0].names
        h, w = frame.shape[:2]
        n = len(boxes)
        if n == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = (
            boxes.id.cpu().numpy().astype(int)
            if boxes.id is not None
            else np.full(n, -1, dtype=int)
        )

        dets: list[Detection] = []
        for i in range(n):
            x1, y1, x2, y2 = xyxy[i]

            sx1, sy1, sx2, sy2 = _shrink_box(
                x1, y1, x2, y2, self.bbox_shrink
            )

            if zone_mask is not None:
                cx = int((sx1 + sx2) / 2)
                cy = int((sy1 + sy2) / 2)
                if not (0 <= cx < w and 0 <= cy < h
                        and zone_mask[cy, cx]):
                    continue

            dets.append(Detection(
                x1=float(x1), y1=float(y1),
                x2=float(x2), y2=float(y2),
                sx1=float(sx1), sy1=float(sy1),
                sx2=float(sx2), sy2=float(sy2),
                conf=float(confs[i]),
                cls_id=int(clss[i]),
                cls_name=names.get(int(clss[i]), str(clss[i])),
                track_id=int(ids[i]),
            ))
        return dets