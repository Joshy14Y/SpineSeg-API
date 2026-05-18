from io import BytesIO
import base64


def to_base64(image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
