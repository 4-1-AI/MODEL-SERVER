from ultralytics import YOLO
import os

model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "find_cause.pt"))
yolov8_model = YOLO(model_path)

def detect_objects_v8(image):
    results = yolov8_model(image)[0]
    obj_preds = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        obj_preds.append((x1, y1, x2, y2, conf, cls_id))
    return obj_preds

def find_closest_object(fire_center, object_preds):
    fx, fy = fire_center
    min_dist = float('inf')
    cause_label = None
    names = yolov8_model.names
    for x1, y1, x2, y2, conf, cls_id in object_preds:
        label = names.get(cls_id, None)
        if label is None:
            continue
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            cause_label = label
    return cause_label, min_dist

def yolo_infer_and_draw(image):
    results = yolov8_model(image)
    return results[0].plot()