import cv2
import time
from yolo_detector import YoloDetector
from tracker import Tracker

#First, Define the path to the model and video, for it to be accessible gloablly
MODEL_PATH = 'models/yolo11n.pt'
VIDEO_PATH = 'assets/football.mp4'

def main():
    #2. Start Capturing. Capture is a whole video. Below, we will do a capture.read() which will return EACH FRAME.
    capture = cv2.VideoCapture(0) #Give a static video path, or use webcam

    #3 If capture couldn't be opened, exit
    if not capture.isOpened():
        print (f'\n--------------------Video not opened at ${VIDEO_PATH}--------------------\n')
        exit()
    
    detector = YoloDetector(model_path=MODEL_PATH, confidence=0.2)
    tracker = Tracker()

    #4. Start a never ending While Loop
    while True:
        ret, frame = capture.read() #Grab TWO things, ret and frame, from capture.read(). Return is a boolean value.
        print (f"\n--------------------RET ${ret}, Frame ${frame}--------------------\n")
        #5. If ret is FALSE, meaning video ended, so end LOOP
        if not ret:
            break

        #Log the start time
        strart_time = time.perf_counter()

        #6. Now that we have ONE FRAME, pass this ONE image (or frame) to the detector. Jump to yolo_detector
        detections = detector.detect(frame)
        #Expect something like this back: [{[x1, y1, w, h], 1, 0.92}, {}, {}, {}, ..etc]
        #18. Now, send this detections object to the tracker's track function. Jump to Tracker.py
        tracked_objects = tracker.track(detections, frame)
        # print(f"\n--------------------Tracking IDs array: ${tracked_objects[0].track_id}--------------------\n")
        # print(f"\n--------------------Boxes: {tracked_objects[0].ltrb}--------------------\n")
        # print(f"\n--------------------Boxes: {tracked_objects[0].label}--------------------\n")
        #19 Get Tracking ID, and corresponding Box Coordinates.
        #Likely something like this: TrackingIDs = [1,2,3,8,6,1 ...etc] Boxes: [{30,20,4,5}, {20, 30, 4,5}, {}, {}, ...etc]

        #Done! Now, just drawing the boxes according to those coordinates left.
        end_time = time.perf_counter()
        fps = 1/(end_time - strart_time)
        print(f"\n--------------------FPS: {fps}\--------------------n")

        #Drawing Function
        for tracking_id, bounding_box, label in tracked_objects:
            x1, y1, x2, y2 = map(int, bounding_box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"ID: {tracking_id} | {label}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        #Show in a window
        cv2.imshow("Frame:", frame)

        #Exit Key
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key==27:
            break

    capture.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()