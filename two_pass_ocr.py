#!/usr/bin/env python3
"""Run dots.ocr twice to capture picture text and merge into final schema.

This script executes a layout OCR pass to detect blocks including pictures.
For each picture block a second OCR pass is run restricted to the picture
bounding box to obtain internal text which is attached as `picture-children`.

Outputs:
- JSON file containing the merged blocks.
- The markdown and image artifacts generated from the first pass are copied
  to the final output directory unchanged.

The script assumes the command line program `dots.ocr` is available.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def run_dots_ocr(image: Path, output_dir: Path, extra_args: List[str]) -> Path:
    """Execute dots.ocr with ``extra_args`` and return path to result JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["dots.ocr", str(image), "--output", str(output_dir), *extra_args]
    subprocess.run(cmd, check=True)
    return output_dir / "result.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def attach_picture_children(image: Path, blocks: List[Dict[str, Any]]) -> None:
    """Run OCR on each picture block and attach results as picture-children."""
    for block in blocks:
        if block.get("category") != "Picture" or "bbox" not in block:
            continue
        bbox = block["bbox"]
        bbox_arg = ",".join(map(str, bbox))
        pic_dir = image.parent / "_picture_temp"
        result_path = run_dots_ocr(
            image, pic_dir, ["--mode", "prompt_grounding_ocr", "--bbox", bbox_arg]
        )
        children = load_json(result_path)
        if children:
            block["picture-children"] = [
                {
                    "bbox": c.get("bbox"),
                    "text": c.get("text"),
                    "conf": c.get("conf"),
                    "category": "PictureText",
                    "source": "picture-ocr",
                }
                for c in children
            ]
        shutil.rmtree(pic_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to the input image")
    parser.add_argument("output", type=Path, help="Directory for final output")
    parser.add_argument(
        "--first-pass-args",
        nargs=argparse.REMAINDER,
        default=["--mode", "layout_all"],
        help="Extra args for initial dots.ocr call",
    )
    args = parser.parse_args()

    first_pass_dir = args.output / "first_pass"
    result_path = run_dots_ocr(args.image, first_pass_dir, args.first_pass_args)
    blocks: List[Dict[str, Any]] = load_json(result_path)

    attach_picture_children(args.image, blocks)

    args.output.mkdir(parents=True, exist_ok=True)
    save_json(blocks, args.output / f"{args.image.stem}.json")

    # copy over markdown and jpg produced by first pass if they exist
    for ext in (".md", ".jpg"):
        src = first_pass_dir / f"{args.image.stem}{ext}"
        if src.exists():
            shutil.copy(src, args.output / src.name)


if __name__ == "__main__":
    main()
