# Satellite IR Enhancement & Analysis

This project enhances low-quality infrared satellite imagery and generates a semantic scene analysis report. It combines super-resolution, land-cover segmentation, object detection, colorization, and natural-language scene summarization in a single workflow.

## Features

- Super-resolution enhancement for infrared satellite imagery
- Land-cover segmentation and percentage-based scene analysis
- Object detection for common scene elements
- IR-to-RGB colorization for visualization
- Structured scene report with confidence and descriptive summary
- Streamlit dashboard for interactive processing

## Project Structure

- `analysis/` — scene reporting and analysis logic
- `dashboard/` — Streamlit web app
- `drawing/` — CLI entry points and model download utilities
- `models/` — enhancement, segmentation, colorization, and detection models
- `utils/` — image loading, saving, and visualization helpers

## Requirements

- Python 3.9+
- pip
- CUDA-capable GPU is optional, but recommended for faster inference

Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Download pretrained model weights (recommended):

```bash
python drawing/download_models.py
```

If model files are missing, the pipeline will fall back to lightweight built-in methods.

## Workflow

The full processing pipeline works as follows:

1. Load an infrared satellite image
2. Enhance the image using denoising and super-resolution
3. Analyze land cover and detect objects
4. Colorize the enhanced image into an RGB-like visualization
5. Generate a scene report with land-cover statistics and description

## Running the CLI

Run the command-line pipeline with an input image:

```bash
python drawing/run.py --image path/to/ir_input.png --output results
```

Generate only the JSON report without saving images:

```bash
python drawing/run.py --image path/to/ir_input.png --report-only
```

## Running the Dashboard

Start the Streamlit app:

```bash
streamlit run dashboard/app.py
```

Then upload an infrared image through the web interface and run the full pipeline.

## Output Files

The CLI saves the following outputs in the chosen output directory:

- `01_enhanced_ir.png`
- `02_segmentation_map.png`
- `03_segmentation_overlay.png`
- `04_detections.png`
- `05_colorized_rgb.png`
- `report.json`

## Notes

- The project is designed for experimentation and demonstration, so model availability and quality can vary depending on your environment and downloaded weights.
- For better performance, ensure that your environment has the required PyTorch and OpenCV dependencies installed correctly.
