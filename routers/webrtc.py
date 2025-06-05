from fastapi import WebSocket, APIRouter
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from starlette.websockets import WebSocketDisconnect
from av import VideoFrame
from my_models.yolov8_model import yolo_model, yolo_infer_and_draw
import json

router = APIRouter()

class YOLOTrack(VideoStreamTrack):
    def __init__(self, track):
        super().__init__()
        self.track = track

    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        result_img = yolo_infer_and_draw(img)

        new_frame = VideoFrame.from_ndarray(result_img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

@router.websocket("")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    offer = await websocket.receive_text()
    offer = json.loads(offer)

    pc = RTCPeerConnection()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(YOLOTrack(track))

    await pc.setRemoteDescription(RTCSessionDescription(**offer))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    await websocket.send_text(json.dumps({"type": "answer", "sdp": pc.localDescription.sdp}))

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await pc.close()
