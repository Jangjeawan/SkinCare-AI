"""분류 원본 ID가 train/valid/test에 한 번씩만 들어가게 다시 나눈다.

Roboflow 증강본은 같은 `.rf.` 앞 ID로 묶는다. 클래스가 갈리는 ID는 뺀다.
valid/test는 원본당 이미지 1장만 두고, train은 그 ID의 증강본을 모두 쓴다.
파일은 복사하지 않고 하드링크로 연결한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path


def origin_id(path: Path) -> str:
    stem = path.stem
    return stem.split(".rf.")[0] if ".rf." in stem else stem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--dst", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--valid", type=float, default=0.15)
    args = parser.parse_args()

    grouped: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for split in ("train", "valid", "test"):
        split_dir = args.src / split
        if not split_dir.is_dir():
            continue
        for folder in split_dir.iterdir():
            if not folder.is_dir() or folder.name.lower() == "unlabeled":
                continue
            for path in folder.glob("*.jpg"):
                grouped[origin_id(path)].append((folder.name, path))

    clean: dict[str, tuple[str, list[Path]]] = {}
    dropped = 0
    for origin, items in grouped.items():
        counts = Counter(folder for folder, _ in items)
        if len(counts) != 1:
            dropped += 1
            continue
        folder = next(iter(counts))
        paths = sorted((path for _, path in items), key=lambda path: path.name)
        clean[origin] = (folder, paths)

    by_class: dict[str, list[str]] = defaultdict(list)
    for origin, (folder, _) in clean.items():
        by_class[folder].append(origin)

    rng = random.Random(args.seed)
    assignment: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for folder, origins in sorted(by_class.items()):
        origins = sorted(origins)
        rng.shuffle(origins)
        n = len(origins)
        n_train = int(n * args.train)
        n_valid = int(n * args.valid)
        if n >= 3:
            n_train = min(n_train, n - 2)
            n_valid = max(1, min(n_valid, n - n_train - 1))
        assignment["train"].extend(origins[:n_train])
        assignment["valid"].extend(origins[n_train : n_train + n_valid])
        assignment["test"].extend(origins[n_train + n_valid :])

    if args.dst.exists():
        raise SystemExit(f"이미 있습니다: {args.dst}")

    counts: dict[str, dict[str, int]] = {}
    for split, origins in assignment.items():
        for origin in origins:
            folder, paths = clean[origin]
            chosen = paths if split == "train" else paths[:1]
            dest_dir = args.dst / split / folder
            dest_dir.mkdir(parents=True, exist_ok=True)
            for source in chosen:
                digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:16]
                dest = dest_dir / f"{digest}.jpg"
                os.link(source, dest)
            counts.setdefault(split, Counter())[folder] += len(chosen)

    manifest = {
        "seed": args.seed,
        "origins": len(clean),
        "dropped_conflict_origins": dropped,
        "images": {split: dict(counter) for split, counter in counts.items()},
        "origin_counts": {split: len(origins) for split, origins in assignment.items()},
    }
    (args.dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
