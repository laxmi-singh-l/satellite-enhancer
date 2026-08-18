# Satellite IR Enhancement & Analysis

A Python pipeline that enhances low-quality infrared satellite imagery and generates semantic scene analysis reports. It combines super-resolution, land-cover segmentation, object detection, IR-to-RGB colorization, and natural-language scene summarization in a single workflow.

## Features

- **Super-resolution** — EDSR neural network (4x upscale) with OpenCV fallback
- **Land-cover segmentation** — DeepLabV3-based classification into 7 classes (water, forest, agriculture, urban, barren, wetland, grassland)
- **Object detection** — YOLOv8 for scene objects with contour-based fallback
- **IR-to-RGB colorization** — pix2pix UNet GAN with colormap fallback and optional segmentation overlay
- **Scene reporting** — JSON report with confidence scores, land-cover statistics, and natural-language descriptions (optional BLIP captioning)
- **Two interfaces** — CLI for batch processing, Streamlit dashboard for interactive use

## Requirements

- Python 3.9+
- CUDA-capable GPU (optional, significantly faster inference)

## Setup

1. **Clone and create a virtual environment:**

```bash
git clone <repo-url>
cd satellite-enhancer

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Download pretrained model weights (optional):**

```bash
python drawing/download_models.py
```

This downloads the EDSR super-resolution weights (~50 MB). If you skip this step, the pipeline automatically falls back to OpenCV-based enhancement (cubic interpolation + detail enhancement). The pix2pix colorization model has no public weights available; the pipeline falls back to an OpenCV inferno colormap with optional segmentation overlay.

## How It Works

```
Input IR Image
     │
     ▼
┌─────────────────────────────┐
│  1. Enhance                 │  Denoise (median blur + CLAHE) → Super-resolution (EDSR 4x or cubic upscale)
│     IRSuperResolution       │
└─────────────┬───────────────┘
              │
     ▼────────┴────────┐
┌────────────┐  ┌──────┴────────────┐
│  2. Segment│  │  3. Detect Objects │
│  DeepLabV3 │  │  YOLOv8 / contour  │
└─────┬──────┘  └──────┬────────────┘
      │                │
      ▼                │
┌─────────────┐        │
│  4. Colorize│◄───────┘
│  pix2pix /  │
│  colormap   │
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  5. Report  │  JSON + natural language description
│  SceneAnalyzer│
└─────────────┘
```

## Usage

### CLI (Command Line)

**Full pipeline** — enhance, segment, detect, colorize, and report:

```bash
python drawing/run.py --image path/to/ir_input.png --output results/
```

**Report only** — generate the JSON report without saving images:

```bash
python drawing/run.py --image path/to/ir_input.png --report-only
```

**Custom super-resolution scale:**

```bash
python drawing/run.py --image path/to/ir_input.png --output results/ --scale 4
```

**Example with the included demo image:**

```bash
python drawing/run.py --image ir_training_demo_dataset/input_lowres_ir.png --output results/
```

**CLI options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--image`, `-i` | (required) | Path to input IR image (PNG, JPG, TIFF) |
| `--output`, `-o` | `results` | Output directory for generated files |
| `--scale` | `4` | Super-resolution scale factor |
| `--report-only` | off | Skip image generation, only write `report.json` |

### Streamlit Dashboard

Start the interactive web app:

```bash
streamlit run dashboard/app.py
```

1. Open the URL shown in your terminal (typically `http://localhost:8501`)
2. Upload an IR satellite image via the sidebar
3. Click **Run Full Pipeline**
4. Explore results across 5 tabs: Overview, Enhancement, Colorization, Analysis, Report
5. Download the colorized RGB image or JSON report directly from the Report tab

### Programmatic Usage

```python
from models import IRSuperResolution, IR2RGB, LandCoverSegmenter
from analysis import SceneAnalyzer
from utils import load_image

# Load image
image = load_image("path/to/ir_image.png")

# Enhance (super-resolution)
enhancer = IRSuperResolution(scale=4)
enhanced = enhancer(image)

# Segment and detect
segmenter = LandCoverSegmenter()
analysis = segmenter.analyze(enhanced)
# analysis = {'segmentation_mask': ..., 'land_cover': {...}, 'objects': [...]}

# Colorize
colorizer = IR2RGB()
rgb = colorizer(enhanced, analysis['segmentation_mask'])

# Generate report
analyzer = SceneAnalyzer()
report = analyzer(
    land_cover=analysis['land_cover'],
    objects=analysis['objects'],
    rgb_image=rgb,
)
print(report.to_json())
```

## Output Files

The CLI saves these files in the output directory:

| File | Description |
|------|-------------|
| `01_enhanced_ir.png` | Super-resolved / enhanced IR image |
| `02_segmentation_map.png` | Color-coded land-cover segmentation |
| `03_segmentation_overlay.png` | Segmentation blended over the enhanced IR |
| `04_detections.png` | Detected objects with bounding boxes (if any) |
| `05_colorized_rgb.png` | IR-to-RGB colorized visualization |
| `report.json` | Full structured scene analysis report |

### Report JSON Structure

```json
{
  "scene_type": "urban",
  "land_cover": {
    "water": { "percentage": 0.07, "pixel_count": 46, "color": [60, 119, 181] },
    "urban": { "percentage": 99.68, "pixel_count": 65306, "color": [178, 34, 34] }
  },
  "objects_detected": [...],
  "object_counts": { "bright_anomaly": 39 },
  "description": "Urban area with 100% built-up coverage...",
  "confidence": 0.95,
  "time_analysis": "2026-08-19 21:42:48 UTC",
  "summary_stats": {
    "dominant_class": "urban",
    "dominant_percentage": 99.68,
    "natural_coverage": 0.32,
    "anthropogenic_coverage": 99.68,
    "class_diversity": 1
  }
}
```

## Fallback Behavior

When pretrained model weights are not available, each component gracefully degrades:

| Component | With Weights | Fallback |
|-----------|-------------|----------|
| Enhancement | EDSR neural network (4x) | OpenCV cubic interpolation + detail enhancement |
| Segmentation | DeepLabV3 (21-class COCO → 7 land-cover) | Threshold-based heuristic segmentation |
| Detection | YOLOv8 object detection | Bright-region contour detection |
| Colorization | pix2pix UNet GAN | OpenCV inferno colormap + optional seg overlay |

## Project Structure

```
satellite-enhancer/
├── analysis/
│   └── reporter.py          # SceneAnalyzer, SceneReport dataclass
├── dashboard/
│   ├── app.py                # Streamlit web app (full ML pipeline)
│   └── front_look.py         # UI mockup / design prototype
├── drawing/
│   ├── run.py                # CLI entry point
│   └── download_models.py    # Model weight downloader
├── models/
│   ├── enhance.py            # EDSR super-resolution network + IRSuperResolution
│   ├── detector.py           # DeepLabV3 segmenter + YOLO detector
│   ├── colorize.py           # pix2pix UNet GAN + PatchGAN discriminator
│   ├── super_resolution.py   # Re-export shim → enhance.py
│   └── segmenter.py          # Re-export shim → detector.py
├── utils/
│   ├── data_utils.py         # load_image, save_image, ensure_dir
│   └── visualization.py      # Overlay, comparison, detection drawing
├── ir_training_demo_dataset/ # Demo images for testing
├── requirements.txt
└── README.md
```

## Notes

- Without a GPU, inference is slower but fully functional on CPU.
- The EDSR model expects grayscale input; the pipeline handles conversion automatically.
- YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first use by ultralytics.
- The BLIP captioning model (optional) is loaded from HuggingFace transformers on first use.
