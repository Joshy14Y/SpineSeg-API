from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from controllers import inference
from inference.inference_pipeline import InferencePipeline
from utils.load_model import load_model


@asynccontextmanager
async def lifespan(application: FastAPI):
    model = load_model(settings.weights_path)
    application.state.pipeline = InferencePipeline(model)
    yield


app = FastAPI(title="SpineSeg Inference API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url],
    allow_methods=["GET", "POST"],
)

app.include_router(inference.router)


@app.get("/")
async def root():
    """Health check and welcome message."""
    return {"message": "Welcome to SpineSeg API"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
