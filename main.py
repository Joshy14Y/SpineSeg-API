from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, UploadFile
from fastapi.responses import JSONResponse
import asyncio
from config.settings import settings
from inference.inference_pipeline import InferencePipeline
from utils.load_model import load_model
from utils.to_base64 import to_base64


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the model and initialize the inference pipeline at server startup."""
    if not settings.weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {settings.weights_path}")
    model = load_model(settings.weights_path, settings.device)
    application.state.pipeline = InferencePipeline(model)
    yield


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/")
def read_root():
    return f"Welcome to {settings.app_name}"


@app.post("/segment")
async def segmentation(request: Request, image: UploadFile):
    """Run the spine segmentation pipeline on an uploaded image."""
    pipeline = request.app.state.pipeline
    image_bytes = await image.read()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, pipeline, image_bytes)
    return JSONResponse(
        {
            "annotated_img": to_base64(result.annotated_img),
            "mask_img": to_base64(result.mask_img),
            "cobb_angle": result.cobb_angle,
            "vertebra": [v.to_dict() for v in result.vertebrae],
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
