"""
test_pipeline.py — unit tests for ANPR modules (no GPU / model download needed)
Run with: python -m pytest test_pipeline.py -v
"""

import json
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── Database tests ────────────────────────────────────────────────────

def test_db_log_and_fetch(tmp_path, monkeypatch):
    from src import database
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")

    detection = {
        "image_name": "test.jpg",
        "plate_text": "MH12AB1234",
        "det_confidence": 0.87,
        "ocr_confidence": 0.94,
        "bbox": [100, 200, 300, 240],
    }
    database.log_detection(detection)
    rows = database.fetch_all(limit=10)
    assert len(rows) == 1
    assert rows[0]["plate_text"] == "MH12AB1234"


def test_db_search(tmp_path, monkeypatch):
    from src import database
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")

    for plate in ["MH12AB1234", "KA05CD5678", "MH14XY9999"]:
        database.log_detection({
            "image_name": "x.jpg", "plate_text": plate,
            "det_confidence": 0.9, "ocr_confidence": 0.9, "bbox": [0, 0, 1, 1],
        })

    results = database.search_plate("MH")
    assert len(results) == 2


# ── CSV export tests ──────────────────────────────────────────────────

def test_csv_export(tmp_path, monkeypatch):
    from src import exporter
    monkeypatch.setattr(exporter, "CSV_PATH", tmp_path / "test.csv")

    detection = {
        "image_name": "car.jpg",
        "plate_text": "DL3CAB0001",
        "det_confidence": 0.8,
        "ocr_confidence": 0.9,
        "bbox": [10, 20, 100, 50],
    }
    exporter.export_to_csv(detection)
    exporter.export_to_csv(detection)   # second row — no duplicate header

    lines = (tmp_path / "test.csv").read_text().splitlines()
    assert lines[0].startswith("timestamp")   # header present
    assert len(lines) == 3                     # header + 2 data rows
    assert "DL3CAB0001" in lines[1]


# ── Detector helper tests (no YOLO / EasyOCR download) ───────────────

def test_clean_plate_text():
    from src.detector import ANPRDetector
    with patch("src.detector.YOLO"), patch("src.detector.easyocr.Reader"):
        det = ANPRDetector.__new__(ANPRDetector)
        assert det._clean_plate_text("mh 12 ab 1234!") == "MH12AB1234"
        assert det._clean_plate_text("  KA-05 CD 5678 ") == "KA05CD5678"


def test_nms_removes_duplicate():
    from src.detector import ANPRDetector
    with patch("src.detector.YOLO"), patch("src.detector.easyocr.Reader"):
        det = ANPRDetector.__new__(ANPRDetector)
        plates = [
            {"x1": 10, "y1": 10, "x2": 100, "y2": 40, "confidence": 0.9},
            {"x1": 12, "y1": 12, "x2": 102, "y2": 42, "confidence": 0.5},  # heavy overlap
            {"x1": 300, "y1": 10, "x2": 400, "y2": 40, "confidence": 0.8}, # separate
        ]
        kept = det._nms(plates)
        assert len(kept) == 2
        assert kept[0]["confidence"] == 0.9
