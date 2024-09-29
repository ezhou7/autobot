import cv2
import numpy as np
from collections import defaultdict

from ultralytics.trackers import BOTSORT 
from autobot.device import AutoBotDevice
from autobot.model import YoloModel
from autobot.utils.postprocess import post_process_torch


STANDARD_CAPTURE = 11


def video_capture():
    # Cam properties
    fps = 30.
    frame_width = 640
    frame_height = 640
    # Create capture
    capture = cv2.VideoCapture(STANDARD_CAPTURE)
    # Set camera properties
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    return capture


if __name__ == "__main__":
    device = AutoBotDevice()
    yolo = YoloModel(device)
    yolo.load("/home/orangepi/Documents/dev/models/yolov5s_relu.rknn")
    tracker = BOTSORT(args=None, frame_rate=30)

    track_history = defaultdict(lambda: [])

    cap = video_capture()
    while True:
        success, frame = cap.read()

        if not success:
            break

        output = yolo.infer([np.expand_dims(frame, 0)])
        print(output)
        # post_process_torch(output, frame)

        # output = yolo.track(frame)
        # boxes = output[0].boxes.xywh.cpu()
        # track_ids = output[0].boxes.id.int().cpu().tolist()
        # annotated_frame = output[0].plot()

        # for box, track_id in zip(boxes, track_ids):
        #     x, y, w, h = box
        #     track = track_history[track_id]
        #     track.append((float(x), float(y)))  # x, y center point
        #     if len(track) > 30:  # retain 90 tracks for 90 frames
        #         track.pop(0)

        #     # Draw the tracking lines
        #     points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
        #     cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
