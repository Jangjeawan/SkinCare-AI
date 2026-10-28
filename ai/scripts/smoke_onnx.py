"""내보낸 탐지 ONNX가 한 장에서 박스를 내는지 확인한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    result = YOLO(str(args.model)).predict(source=str(args.image), verbose=False)[0]
    count = 0 if result.boxes is None else len(result.boxes)
    print({"boxes": count})


if __name__ == "__main__":
    main()
