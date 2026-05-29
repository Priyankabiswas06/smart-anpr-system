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

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/smart-anpr.git
cd smart-anpr
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA first — see [pytorch.org](https://pytorch.org/get-started/locally/).  
> **CPU-only** works fine — detection is slower but fully functional.

### 2. Run on a single image

```bash
python main.py --image input/car.jpg
```

### 3. Run on a folder of images

```bash
python main.py --folder input/
```

### 4. View detection history

```bash
python main.py --history 20
```

### 5. Search a plate in the database

```bash
python main.py --search MH12
```

### 6. Export all records to CSV

```bash
python main.py --export-all data/full_export.csv
```

---

## ⚙️ CLI Options

| Flag | Description | Default |
|---|---|---|
| `--image PATH` | Single image file | — |
| `--folder PATH` | Process all images in folder | — |
| `--model PATH` | YOLO weights file | `yolov8n.pt` |
| `--conf FLOAT` | Detection confidence threshold | `0.4` |
| `--history N` | Show last N detections from DB | — |
| `--search TEXT` | Search plate text in DB | — |
| `--export-all PATH` | Bulk export DB → CSV | — |

---

## 🧠 How It Works

```
Input Image
     │
     ▼
┌─────────────────────┐
│   YOLO Detection    │  → finds plate bounding boxes
│   (YOLOv8n)         │
└────────┬────────────┘
         │ ROI crops
         ▼
┌─────────────────────┐
│  Image Pre-process  │  → resize 3×, CLAHE, Otsu threshold
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     EasyOCR         │  → extracts text + confidence
└────────┬────────────┘
         │
    ┌────┴──────┐
    ▼           ▼
SQLite DB    CSV Log
    │
    ▼
Annotated image saved to output/
```

**Fallback:** If YOLO finds no plates, a contour-based detector (OpenCV) activates automatically — useful for high-contrast, clean images.

---

## 🔧 Using a Custom Plate Model

For better accuracy, train or download a dedicated plate detection model:

- **Roboflow ANPR datasets** — [roboflow.com/object-detection](https://roboflow.com/object-detection)
- **CCPD dataset** — Chinese plate benchmark, widely used
- **OpenALPR** — pre-trained weights available

Then pass your model with:

```bash
python main.py --image input/car.jpg --model models/plate_detector.pt
```

---

## 📊 Output Example

**Annotated image** (`output/annotated_car.jpg`):
- Green bounding box around the plate
- Banner showing: `MH12AB1234  det:0.87  ocr:0.94`

**CSV row** (`data/detections.csv`):
```
timestamp,image_name,plate_text,det_confidence,ocr_confidence,x1,y1,x2,y2
2024-06-01T14:32:11,car.jpg,MH12AB1234,0.87,0.94,142,310,398,360
```

---

## 🛠️ Programmatic Use

```python
from src import ANPRDetector

detector = ANPRDetector(model_path="yolov8n.pt", conf_threshold=0.4)
result = detector.process_image("input/car.jpg")

for det in result["detections"]:
    print(det["plate_text"], det["ocr_confidence"])
```

---

## 📋 Requirements

- Python 3.10+
- 4 GB RAM minimum (8 GB recommended)
- GPU optional (CUDA 11.8+ supported)

---

## 📄 License

MIT License — free to use, modify, and distribute.
