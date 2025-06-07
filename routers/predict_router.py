import aiohttp
from PIL import Image
import io, base64
from my_models.yolov8_model import detect_objects_v8, find_closest_object
import traceback    

BACKEND_API_URL = "http://localhost:8080/alert/fire-cause/sms" 


async def post_fire_cause(fire_center, img_base64: str,  user_id: int):
    try:
        header, base64_data = img_base64.split(",", 1) if "," in img_base64 else ("", img_base64)
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 객체 감지
        object_preds = detect_objects_v8(image)
        cause_label, dist = find_closest_object(fire_center, object_preds)

        if cause_label is None:
            print("❗ 원인 감지 실패 또는 객체 없음")
            cause_label = "원인미상"
            dist = 0

        payload = {
            "userId" : user_id,
            "cause": cause_label,
            "distance": dist,
        }

        async with aiohttp.ClientSession() as session:
            res = await session.post(
                BACKEND_API_URL,
                json=payload,
                timeout=3
            )
            result = await res.text()
            print("✅ 원인 분석 전송 완료:", result)

    except Exception as e:
        print("❌ 원인 분석 중 오류 발생:", e)
        traceback.print_exc()  # 전체 스택 트레이스 출력