"""탐지 가중치를 ONNX로 보낸다. 분류 ViT는 학습 체크포인트 그대로 둔다."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "runs" / "det_v8s_wound" / "weights" / "best.pt")
    args = parser.parse_args()
    YOLO(str(args.weights)).export(format="onnx", imgsz=640)


if __name__ == "__main__":
    main()
