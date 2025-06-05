from fastapi import APIRouter
from pydantic import BaseModel
from PIL import Image
import io, base64
from my_models.yolov5_model import detect_fire_v5
from my_models.yolov8_model import detect_objects_v8, find_closest_object

router = APIRouter()

class ImageRequest(BaseModel):
    image: str

@router.post("")
async def predict(request: ImageRequest):
    header, base64_data = request.image.split(",", 1) if "," in request.image else ("", request.image)
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    fire_center, fire_preds, fire_label = detect_fire_v5(image)
    if not fire_center:
        return {"message": "Fire or smoke detection failed"}

    object_preds = detect_objects_v8(image)
    cause_label, dist = find_closest_object(fire_center, object_preds)

    if cause_label:
        return {"message": f"Detected fire cause: {cause_label}, distance: {dist:.2f}"}
    else:
        return {"message": "Object detected, but cause not identified"}