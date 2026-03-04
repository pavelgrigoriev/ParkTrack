#!/usr/bin/env python3
"""
main.py — CLI точка входа в Parking Pipeline (ultralytics track)
"""

import argparse

from parking import ParkingPipeline


def main():
    ap = argparse.ArgumentParser(
        description="Parking Pipeline — async ultralytics track"
    )
    ap.add_argument("--xml",          required=True,
                    help="Путь к CVAT XML-разметке")
    ap.add_argument("--video",        required=True,
                    help="Путь к входному видео")
    ap.add_argument("--model",        default="yolo11s.pt",
                    help="Путь к весам YOLO модели")
    ap.add_argument("--output",       default="./output_parking.mp4",
                    help="Путь к выходному видео")
    ap.add_argument("--device",       default="cuda:0",
                    help="Устройство (cuda:0 / cpu)")
    ap.add_argument("--conf",         type=float, default=0.3,
                    help="Порог уверенности детектора")
    ap.add_argument("--imgsz",        type=int,   default=640,
                    help="Размер входа модели (пикселей)")
    ap.add_argument("--no-show",      action="store_true",
                    help="Не показывать окно cv2.imshow")
    ap.add_argument("--occ-thresh",   type=float, default=0.35,
                    help="EWMA порог бинарного решения 'занято'")
    ap.add_argument("--ewma-alpha",   type=float, default=0.40,
                    help="Скорость EWMA (0.1=плавно, 0.9=резко)")
    ap.add_argument("--iof-thresh",   type=float, default=0.15,
                    help="Доля площади места закрытая bbox (визуал)")
    ap.add_argument("--dst-size",     type=int, nargs=2, default=None,
                    help="Ширина Высота BEV-проекции")
    ap.add_argument("--pad-zone",     type=int,   default=60,
                    help="Расширение маски зоны детекции (пикс)")
    ap.add_argument("--fill-alpha",   type=float, default=0.45,
                    help="Прозрачность заливки полигонов на камере")
    ap.add_argument("--border-thick", type=int,   default=2,
                    help="Толщина границы полигонов")
    a = ap.parse_args()

    ParkingPipeline(
        xml_path=a.xml,
        model_path=a.model,
        video_path=a.video,
        output_path=a.output,
        device=a.device,
        conf=a.conf,
        imgsz=a.imgsz,
        show=not a.no_show,
        occ_thresh=a.occ_thresh,
        ewma_alpha=a.ewma_alpha,
        iof_thresh=a.iof_thresh,
        dst_size=a.dst_size,
        pad_zone=a.pad_zone,
        fill_alpha=a.fill_alpha,
        border_thick=a.border_thick,
    ).run()


if __name__ == "__main__":
    main()