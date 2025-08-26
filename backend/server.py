from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from yolo_detector import YoloDetector
from tracker import Tracker
import json
import base64
import numpy as np
import cv2
import time

MODEL_PATH = "models/yolo11n.pt"

app=FastAPI()

detector = YoloDetector(MODEL_PATH, confidence=0.2)
tracker = Tracker()

@app.websocket("/ws")
async def websocket_endpoint ( websocket: WebSocket ):
    await websocket.accept()
    print("WebSocket Connection Accepted")
    try:
        while True:
            #1. Receive Data
            data_stream = await websocket.receive_txt()

            #2. Parse the whole JSON and Extract the "IMAGE"
            data = json.loads(data_stream)
            base64_image = data.get('image')
            if not base64_image:
                continue
            
            #3 Parse the IMAGE and transform to NumPy array
            image_bytes = base64.b64decode(base64_image)
            numpyarray = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(numpyarray, cv2.IMREAD_COLOR)

            if frame is None:
                print("Faile to Decode Image")
                continue
            #Log start end time for FPS
            start_time = time.perf_counter()
            #4 Now that you have a frame, start detections
            detections = detector.detect(frame)
            #5 Now that you have detections, run tracking
            tracked_objects = tracker.track(detections)

            end_time = time.perf_counter()
            fps = 1/ (end_time - start_time)

            tracking_results_to_return = []

            for tracking_id, bounding_box, label in tracked_objects:
                    #Convert NumPY Bounding Box to standard Python List:
                bbox_list = bounding_box.tolist() if isinstance(bounding_box, np.ndarray) else bounding_box
                tracking_results_to_return.append( [
                    tracking_id,
                    bbox_list,
                    label
                ])

            #Now, send the results as JSON
            await websocket.send_text(json.dumps(tracking_results_to_return))
        

    except WebSocketDisconnect:
        print("WebSocket(Client) Disconnected")
    except Exception as e:
        print("An Unknown Error Occured: {e}")

#Not essential, but a simple HTML page to show backend is up
@app.get("/")
async def get():
    return HTMLResponse("<h1> Chirag's Real-Time Object Tracking Server </h1> <p> Connect to the /ws endpoint for WS Communication. </p>")