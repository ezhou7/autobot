import cv2
import yaml
import numpy as np

from ultralytics.trackers import BOTSORT
from ultralytics.engine.results import Results
from autobot import get_resource
from autobot.common import VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT, FPS, STANDARD_CAPTURE
from autobot.device import AutoBotDevice
from autobot.model import YoloModel
from autobot.utils import Properties
from autobot.utils.postprocess import post_process_rknn, post_process_rknn_tracking, post_process_rknn_selecting
from autobot.utils.yolov5 import CLASSES


def video_capture():
    # Create capture
    capture = cv2.VideoCapture(STANDARD_CAPTURE)
    # Set camera properties
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, FPS)

    return capture


def load_tracker_model():
    botsort_args = yaml.safe_load(get_resource("botsort.yaml"))
    props = Properties(botsort_args)
    return BOTSORT(args=props, frame_rate=FPS)


class AutoBotTracker:
    def __init__(self, yolo_model_path: str):
        self.cap = video_capture()
        self.device = AutoBotDevice()
        self.yolo = self.__load_yolo_model(yolo_model_path)
        self.tracker = load_tracker_model()
    
    def __load_yolo_model(self, path: str):
        yolo = YoloModel(self.device)
        yolo.load(path)

        return yolo
    
    def track(self, stop_flag, msg_broker):
        currently_selected_id = -1
        while not stop_flag.is_set():
            success, frame = self.cap.read()
            frame = cv2.resize(frame, (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT))

            if not success:
                break

            output = self.yolo.infer([np.expand_dims(frame, 0)])
            boxes, classes, scores = post_process_rknn(output, frame)
            if len(boxes):
                new_boxes = np.hstack((
                    boxes, 
                    np.arange(0, boxes.shape[0]).reshape(boxes.shape[0], 1), 
                    scores.reshape(boxes.shape[0], 1),
                    classes.reshape(boxes.shape[0], 1)
                ))
                results = Results(
                    frame, "",
                    {i: cls for i, cls in enumerate(CLASSES)},
                    new_boxes,
                    scores
                )
                tracked_boxes = self.tracker.update(results.boxes, img=frame)
                fxc, fyc = post_process_rknn_tracking(tracked_boxes, frame)
                xc, yc = post_process_rknn_selecting(tracked_boxes, frame, currently_selected_id)

                if xc != -1 and yc != -1:
                    msg_broker.put("tracker_to_follower", (fxc, fyc, xc, yc))
                    msg_broker.put("tracker_to_lidar", (fxc, fyc, xc, yc))
            
            if not msg_broker.empty("user_input"):
                inp = msg_broker.get("user_input")
                if inp.isnumeric():
                    currently_selected_id = int(inp)

            cv2.imshow("YOLOv5 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()
