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
- CUDA-capable GPU recommended (CPU inference ~60s/image)
- Model weights file (`.pth`)

## Installation

```bash
git clone https://github.com/your-username/scoliosis-project.git
cd scoliosis-project

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

Place your model weights at `weights/u_net.pth`.

## Configuration

Create a `.env` file in the project root:

```env
DEVICE=cuda
PORT=8080
```

| Variable       | Default               | Description                      |
|----------------|-----------------------|----------------------------------|
| `DEVICE`       | `cuda`                | PyTorch device (`cuda` or `cpu`) |
| `PORT`         | `8080`                | Server port                      |
| `WEIGHTS_PATH` | `weights/u_net.pth`   | Path to model weights            |

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
    { "id": 1, "confidence": 0.9821 }
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
