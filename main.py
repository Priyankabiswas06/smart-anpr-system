#!/usr/bin/env python3
"""
ANPR System — CLI entry point

Usage examples
--------------
# Single image
python main.py --image input/car.jpg

# All images in a folder
python main.py --folder input/

# Show last 20 logged detections
python main.py --history 20

# Search plate in the database
python main.py --search MH12

# Export all DB records to a new CSV
python main.py --export-all data/full_export.csv
"""

import argparse
import json
from pathlib import Path

from src import ANPRDetector, fetch_all, search_plate
from src.exporter import export_all_to_csv

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_images(detector: ANPRDetector, paths: list[Path]):
    all_detections = []
    for p in paths:
        print(f"\n{'='*50}")
        print(f"Processing: {p.name}")
        print("=" * 50)
        result = detector.process_image(str(p))
        all_detections.extend(result["detections"])

    print(f"\n✓ Done. Processed {len(paths)} image(s), "
          f"found {len(all_detections)} plate(s).")
    print("  Annotated images  → output/")
    print("  Database          → data/anpr.db")
    print("  CSV log           → data/detections.csv")
    return all_detections


def main():
    parser = argparse.ArgumentParser(
        description="Smart ANPR System — YOLO + EasyOCR"
    )
    parser.add_argument("--image",      type=str, help="Path to a single image")
    parser.add_argument("--folder",     type=str, help="Path to folder of images")
    parser.add_argument("--model",      type=str, default="yolov8n.pt",
                        help="YOLO model weights (default: yolov8n.pt)")
    parser.add_argument("--conf",       type=float, default=0.4,
                        help="Detection confidence threshold (default: 0.4)")
    parser.add_argument("--history",    type=int, metavar="N",
                        help="Show last N logged detections from DB")
    parser.add_argument("--search",     type=str, metavar="TEXT",
                        help="Search plate text in the database")
    parser.add_argument("--export-all", type=str, metavar="PATH",
                        help="Export all DB records to a CSV file")
    args = parser.parse_args()

    # ── Database queries (no detector needed) ──────────────────────────
    if args.history:
        rows = fetch_all(args.history)
        print(f"\nLast {args.history} detections:\n")
        for r in rows:
            print(f"  [{r['timestamp']}]  {r['image_name']}  →  "
                  f"{r['plate_text'] or '(no text)'}  "
                  f"(det:{r['det_confidence']:.2f} ocr:{r['ocr_confidence']:.2f})")
        return

    if args.search:
        rows = search_plate(args.search)
        print(f"\nResults for '{args.search}' ({len(rows)} match(es)):\n")
        for r in rows:
            print(f"  [{r['timestamp']}]  {r['image_name']}  →  "
                  f"{r['plate_text']}  bbox={r['bbox']}")
        return

    if args.export_all:
        rows = fetch_all(limit=100_000)
        export_all_to_csv(rows, args.export_all)
        print(f"Exported {len(rows)} records → {args.export_all}")
        return

    # ── Image processing ───────────────────────────────────────────────
    if not args.image and not args.folder:
        parser.print_help()
        return

    detector = ANPRDetector(model_path=args.model, conf_threshold=args.conf)

    if args.image:
        process_images(detector, [Path(args.image)])

    elif args.folder:
        folder = Path(args.folder)
        images = sorted([p for p in folder.iterdir() if p.suffix.lower() in SUPPORTED])
        if not images:
            print(f"No supported images found in {folder}")
            return
        process_images(detector, images)


if __name__ == "__main__":
    main()
