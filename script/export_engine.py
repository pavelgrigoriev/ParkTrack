from ultralytics import YOLO

model = YOLO("../yolo26s.pt")

model.export(format="engine", imgsz=1920, int8=True)