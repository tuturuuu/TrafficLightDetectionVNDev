#!/usr/bin/env python3
"""Make a self-contained, fixed-size copy of the SeaDroneSee YOLO splits.

Every image is stretched to WIDTH x HEIGHT, without padding or cropping. YOLO
coordinates are normalized to each full image, so their bytes stay unchanged.
The source trees, including broken symlinks, are never changed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import yaml
from PIL import Image


SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    split: str
    name: str
    image: Path
    label: Path
    width: int
    height: int
    label_bytes: bytes
    n_boxes: int
    image_fallback: bool
    label_fallback: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path("data/seadrone_sequence"))
    parser.add_argument("--output-root", type=Path, default=Path("data/seadronesees_rz1230x933"))
    parser.add_argument(
        "--fallback-root", type=Path, default=Path("data/SeaDroneSees"),
        help="Resolve missing source links by unique basename in images/labels/{train,val}.",
    )
    parser.add_argument("--width", type=int, default=1230)
    parser.add_argument("--height", type=int, default=933)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--dry-run", action="store_true", help="Validate all inputs without writing output.")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.workers < 1:
        parser.error("width, height and workers must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("jpeg-quality must be between 1 and 100")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_roots(source: Path, output: Path, fallback: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    # lexists also rejects a broken destination symlink.
    if os.path.lexists(output):
        raise FileExistsError(f"Destination already exists; choose a new output-root: {output}")
    for input_root in (source, fallback):
        if output == input_root or output in input_root.parents or input_root in output.parents:
            raise ValueError(f"Source and destination must not overlap or nest: {input_root}, {output}")


def load_names(source: Path) -> tuple[Any, int]:
    config = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Source data.yaml must be a mapping")
    names = config.get("names")
    if isinstance(names, list) and names and all(isinstance(n, str) and n for n in names):
        return names, len(names)
    if isinstance(names, dict) and names:
        if any(type(k) is not int for k in names) or sorted(names) != list(range(len(names))):
            raise ValueError("Class-name keys must be contiguous integers beginning with 0")
        if not all(isinstance(n, str) and n for n in names.values()):
            raise ValueError("Every class name must be a nonempty string")
        return names, len(names)
    raise ValueError("Source data.yaml must provide a nonempty names list or mapping")


def fallback_index(root: Path, kind: str) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    extensions = IMAGE_EXTENSIONS if kind == "images" else {".txt"}
    for split in ("train", "val"):
        folder = root / kind / split
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in extensions and path.is_file():
                index.setdefault(path.name, []).append(path.resolve(strict=True))
    return index


def resolve_input(path: Path, index: dict[str, list[Path]]) -> tuple[Path, bool]:
    if path.is_file():
        return path.resolve(strict=True), False
    matches = index.get(path.name, [])
    if not matches:
        raise FileNotFoundError(f"Missing source file and no fallback with matching basename: {path}")
    if len(matches) != 1:
        raise ValueError(f"Ambiguous fallback for {path}: {matches}")
    return matches[0], True


def validate_labels(path: Path, class_count: int) -> tuple[bytes, int]:
    data = path.read_bytes()
    count = 0
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        context = f"{path}:{line_number}"
        if len(fields) != 5:
            raise ValueError(f"Expected YOLO detection label 'class xc yc w h' at {context}")
        try:
            values = [float(field) for field in fields]
        except ValueError as exc:
            raise ValueError(f"Non-numeric YOLO label at {context}") from exc
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite YOLO label at {context}")
        class_id, xc, yc, width, height = values
        if not class_id.is_integer() or not 0 <= class_id < class_count:
            raise ValueError(f"Class outside data.yaml names at {context}: {class_id}")
        if not all(0 <= value <= 1 for value in values[1:]) or width <= 0 or height <= 0:
            raise ValueError(f"Invalid normalized box at {context}")
        # Allow only the rounding error introduced by normalized text precision.
        if any(center - length / 2 < -1e-5 or center + length / 2 > 1 + 1e-5
               for center, length in ((xc, width), (yc, height))):
            raise ValueError(f"Normalized box extends beyond its image at {context}")
        count += 1
    return data, count


def inspect_sample(item: tuple[str, Path, Path], image_index: dict, label_index: dict,
                   class_count: int) -> Sample:
    split, image_link, label_link = item
    image_path, image_fallback = resolve_input(image_link, image_index)
    label_path, label_fallback = resolve_input(label_link, label_index)
    with Image.open(image_path) as image:
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image dimensions: {image_path}")
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"EXIF-rotated source needs explicit annotation handling: {image_path}")
        image.verify()
    label_bytes, n_boxes = validate_labels(label_path, class_count)
    return Sample(split, image_link.name, image_path, label_path, width, height,
                  label_bytes, n_boxes, image_fallback, label_fallback)


def load_split_manifest(source: Path, samples: list[Sample], output: Path) -> dict | None:
    path = source / "split_manifest.json"
    if not path.is_file():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise ValueError(f"Unsupported split manifest structure: {path}")
    expected = {(sample.split, sample.name) for sample in samples}
    found: set[tuple[str, str]] = set()
    result = copy.deepcopy(manifest)
    for sample in result["samples"]:
        if not isinstance(sample, dict) or not isinstance(sample.get("image"), str):
            raise ValueError(f"Invalid sample in split manifest: {sample}")
        key = (sample.get("split"), Path(sample["image"]).name)
        if key not in expected or key in found:
            raise ValueError(f"Unexpected or repeated split/name in split manifest: {key}")
        found.add(key)
        sample.setdefault("source_image", sample["image"])
        sample["image"] = str(output / key[0] / "images" / key[1])
    if found != expected:
        raise ValueError(f"Split manifest differs from source images; missing {len(expected - found)} samples")
    return result


def preflight(source: Path, output: Path, fallback: Path, workers: int) -> tuple[list[Sample], Any, dict | None]:
    names, class_count = load_names(source)
    image_index = fallback_index(fallback, "images")
    label_index = fallback_index(fallback, "labels")
    items = []
    all_names: set[str] = set()
    for split in SPLITS:
        image_dir, label_dir = source / split / "images", source / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source image/label directories for split: {split}")
        # iterdir includes broken symlinks, preserving membership of every split.
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if not images:
            raise ValueError(f"No images in split: {split}")
        expected_labels = {f"{image.stem}.txt" for image in images}
        if len(expected_labels) != len(images):
            raise ValueError(f"Images share a label stem in split: {split}")
        actual_labels = {p.name for p in label_dir.iterdir() if p.suffix.lower() == ".txt"}
        if expected_labels != actual_labels:
            raise ValueError(f"Image/label membership mismatch in {split}: "
                             f"missing={sorted(expected_labels - actual_labels)[:5]}, "
                             f"extra={sorted(actual_labels - expected_labels)[:5]}")
        for image in images:
            if image.name in all_names:
                raise ValueError(f"Image basename repeated across splits: {image.name}")
            all_names.add(image.name)
            items.append((split, image, label_dir / f"{image.stem}.txt"))
    samples: list[Sample] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = executor.map(lambda item: inspect_sample(item, image_index, label_index, class_count), items)
        for sample in results:
            samples.append(sample)
            if len(samples) % 100 == 0 or len(samples) == len(items):
                print(f"Preflight {len(samples)}/{len(items)}", flush=True)
    manifest = load_split_manifest(source, samples, output)
    return samples, names, manifest


def atomic_write(path: Path, data: bytes, *, replace: bool = False) -> None:
    if not replace and os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite: {path}")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            if handle.write(data) != len(data):
                raise OSError(f"Short write: {temporary}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, data: dict, *, replace: bool = False) -> None:
    atomic_write(path, (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"), replace=replace)


def resize_sample(sample: Sample, output: Path, width: int, height: int, quality: int) -> str:
    # Ignore orientation so coordinates refer to the same raw pixel grid validated above.
    image = cv2.imread(str(sample.image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if image is None:
        raise ValueError(f"OpenCV could not decode: {sample.image}")
    if image.shape[:2] != (sample.height, sample.width):
        raise ValueError(f"Source dimensions changed after preflight: {sample.image}")
    interpolation = (cv2.INTER_AREA if width <= sample.width and height <= sample.height
                     else cv2.INTER_LINEAR)
    resized = cv2.resize(image, (width, height), interpolation=interpolation)
    extension = Path(sample.name).suffix.lower()
    options = [cv2.IMWRITE_JPEG_QUALITY, quality] if extension in {".jpg", ".jpeg"} else []
    success, encoded = cv2.imencode(extension, resized, options)
    if not success:
        raise OSError(f"OpenCV could not encode: {sample.name}")
    image_dest = output / sample.split / "images" / sample.name
    label_dest = output / sample.split / "labels" / f"{Path(sample.name).stem}.txt"
    atomic_write(image_dest, encoded.tobytes())
    atomic_write(label_dest, sample.label_bytes)
    with Image.open(image_dest) as check:
        if check.size != (width, height):
            raise ValueError(f"Written dimensions are incorrect: {image_dest}")
        check.verify()
    if label_dest.read_bytes() != sample.label_bytes:
        raise ValueError(f"Written label differs from original: {label_dest}")
    for path in (image_dest, label_dest):
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError(f"Output must be an independent regular file: {path}")
    return "area" if interpolation == cv2.INTER_AREA else "linear"


def build_manifest(args: argparse.Namespace, source: Path, output: Path, fallback: Path,
                   samples: list[Sample]) -> dict:
    split_info = {}
    for split in SPLITS:
        entries = [sample for sample in samples if sample.split == split]
        split_info[split] = {
            "images": len(entries), "labels": len(entries),
            "boxes": sum(sample.n_boxes for sample in entries),
            "original_dimensions": dict(sorted(Counter(f"{s.width}x{s.height}" for s in entries).items())),
            "fallback_images": sum(sample.image_fallback for sample in entries),
            "fallback_labels": sum(sample.label_fallback for sample in entries),
        }
    return {
        "status": "running", "started_at": utc_now(), "source_root": str(source),
        "fallback_root": str(fallback), "output_root": str(output),
        "target_width": args.width, "target_height": args.height,
        "method": "full-frame anisotropic resize; no padding or cropping",
        "interpolation": "INTER_AREA when both axes shrink or stay equal; INTER_LINEAR otherwise",
        "jpeg_quality": args.jpeg_quality, "workers": args.workers,
        "labels": "YOLO normalized boxes copied byte-for-byte; full-frame scaling preserves normalized coordinates",
        "source_modified": False, "completed_images": 0, "total_images": len(samples),
        "splits": split_info,
        "samples": [
            {"split": s.split, "name": s.name, "source_image": str(s.image),
             "source_label": str(s.label), "original_width": s.width, "original_height": s.height,
             "n_boxes": s.n_boxes, "image_fallback": s.image_fallback,
             "label_fallback": s.label_fallback,
             "label_sha256": hashlib.sha256(s.label_bytes).hexdigest()}
            for s in samples
        ],
    }


def main() -> int:
    args = parse_args()
    source = args.source_root.expanduser().resolve()
    output_arg = args.output_root.expanduser().absolute()
    if os.path.lexists(output_arg):
        raise FileExistsError(f"Destination already exists: {output_arg}")
    output = output_arg.resolve()
    fallback = args.fallback_root.expanduser().resolve()
    validate_roots(source, output, fallback)
    cv2.setNumThreads(1)
    start = time.monotonic()
    samples, names, split_manifest = preflight(source, output, fallback, args.workers)
    manifest = build_manifest(args, source, output, fallback, samples)
    print(json.dumps({"preflight": "passed", "output": str(output), "splits": manifest["splits"]},
                     indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; no output files were created.", flush=True)
        return 0
    # Output creation is deliberately delayed until every input passes preflight.
    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / "resize_manifest.json"
    write_json(manifest_path, manifest)
    try:
        for split in SPLITS:
            (output / split / "images").mkdir(parents=True)
            (output / split / "labels").mkdir(parents=True)
        config = {"path": str(output), **{split: f"{split}/images" for split in SPLITS}, "names": names}
        atomic_write(output / "data.yaml", yaml.safe_dump(config, sort_keys=False, allow_unicode=True).encode("utf-8"))
        if split_manifest is not None:
            write_json(output / "split_manifest.json", split_manifest)
        interpolation_counts: Counter[str] = Counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(
                lambda sample: resize_sample(sample, output, args.width, args.height, args.jpeg_quality), samples)
            for completed, interpolation in enumerate(results, start=1):
                manifest["completed_images"] = completed
                interpolation_counts[interpolation] += 1
                if completed % 100 == 0 or completed == len(samples):
                    print(f"Resize {completed}/{len(samples)} | elapsed {time.monotonic() - start:.1f}s", flush=True)
        manifest.update(status="complete", finished_at=utc_now(),
                        elapsed_seconds=round(time.monotonic() - start, 3),
                        interpolation_counts=dict(interpolation_counts))
        write_json(manifest_path, manifest, replace=True)
    except BaseException as exc:
        manifest.update(status="failed", finished_at=utc_now(), error=f"{type(exc).__name__}: {exc}")
        write_json(manifest_path, manifest, replace=True)
        raise
    print(f"Complete: {len(samples)} images resized to {args.width}x{args.height} in {output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
