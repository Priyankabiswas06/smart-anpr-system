# 🚗 Smart ANPR System

> Automatic Number Plate Recognition using **YOLOv8** for plate detection and **EasyOCR** for text extraction.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Plate Detection** | YOLOv8 (ultralytics) + contour-based fallback |
| **Text Recognition** | EasyOCR with pre-processing (CLAHE + Otsu threshold) |
| **Annotated Output** | Bounding boxes with plate text + confidence scores |
| **Database Logging** | SQLite — every detection is persisted automatically |
| **CSV Export** | Append-mode CSV log; bulk export via CLI |
| **CLI Interface** | Single image, batch folder, history search |

---

## 📁 Project Structure

```
anpr/
├── main.py              ← CLI entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── detector.py      ← YOLO + EasyOCR pipeline
│   ├── database.py      ← SQLite logging
│   └── exporter.py      ← CSV export
├── input/               ← Drop your images here
├── output/              ← Annotated images saved here
└── data/
    ├── anpr.db          ← SQLite database (auto-created)
    └── detections.csv   ← CSV log (auto-created)
```

---

