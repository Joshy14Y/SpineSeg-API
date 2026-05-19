# SpineSeg Inference API

Automated spine segmentation and Cobb angle estimation from X-ray images, built with FastAPI and a custom UNet architecture.

## What it does

Takes a grayscale X-ray image and returns:

- Annotated image with segmentation overlay, vertebra centers, spline, and Cobb angle lines
- Segmentation mask where pixel values correspond to vertebra IDs
- Cobb angle in degrees
- Per-vertebra segmentation confidence scores

## Requirements

- Python 3.10+
- CPU inference under 1s/image
- Model weights file (`.onnx`)

## Installation

```bash
git clone https://github.com/your-username/scoliosis-project.git
cd scoliosis-project

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place your model weights at `weights/u_net.onnx`.

## Configuration

Set the following environment variables or define them in a `.env` file:

| Variable       | Default                | Description              |
|----------------|------------------------|--------------------------|
| `PORT`         | `8080`                 | Server port              |
| `WEIGHTS_PATH` | —                      | Path to model weights    |
| `FRONTEND_URL` | —                      | Allowed CORS origin      |

## Running

```bash
python main.py
```

API docs available at `http://localhost:{PORT}/docs`.

## API

### `POST /segment`

Upload a spine X-ray and run the full inference pipeline.

```bash
curl -X POST http://localhost:{PORT}/segment \
  -F "image=@xray.jpg"
```

**Response**

```json
{
  "annotated_img": "<base64 PNG>",
  "mask_img": "<base64 PNG>",
  "cobb_angle": 24.7,
  "vertebrae": [
    { "uid": "a3f1", "class_id": 1, "confidence": 0.9821 }
  ]
}
```

**Decoding images**

```python
import base64
from PIL import Image
from io import BytesIO

image = Image.open(BytesIO(base64.b64decode(result["annotated_img"])))
```

*SpineSeg — Graduation Project, Computer Vision & Deep Learning*