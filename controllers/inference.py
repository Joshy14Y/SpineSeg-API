import asyncio

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from utils.to_base64 import to_base64

router = APIRouter()


@router.post("/segment")
async def segmentation(request: Request, image: UploadFile):
    pipeline = request.app.state.pipeline
    image_bytes = await image.read()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, pipeline, image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return JSONResponse(
        {
            "annotated_img": to_base64(result.annotated_img),
            "mask_img": to_base64(result.mask_img),
            "cobb_angle": result.cobb_angle,
            "vertebrae": [v.to_dict() for v in result.vertebrae],
        }
    )
