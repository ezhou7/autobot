import os
import cv2
import yaml
import numpy as np

from collections import defaultdict
from ultralytics.engine.results import Results, Boxes
from yolov5 import post_process, load_anchors, CLASSES

from ultralytics.trackers import BOTSORT
from ultralytics.utils.ops import xyxy2xywh
from autobot.device import AutoBotDevice
from autobot.model import YoloModel


def post_process_torch(results: list[Results], frame: np.ndarray):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates in xyxy format
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index
            class_name = result.names[cls]  # Get class name

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)  # draw label


def post_process_rknn(results: list, frame: np.ndarray):
    anchors = load_anchors()
    boxes, classes, scores = post_process(results, anchors)

    for box, class_index, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box  # Bounding box coordinates in xyxy format
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = CLASSES[class_index]  # Get class name

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_name} {score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)  # draw label
    
    return boxes, classes, scores


def post_process_rknn_tracking(boxes: np.ndarray, frame: np.ndarray):
    print(frame)
    for box in boxes:
        print(box)
        x1, y1, x2, y2, object_id, conf, class_id, track_id = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        print(x1, y1, x2, y2)
        cls = int(class_id)  # Class index

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{track_id}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)  # draw label


class BotSortArgs:
    def __init__(self, args_dict: dict):
        for k, v in args_dict.items():
            setattr(self, k, v)


if __name__ == "__main__":
    device = AutoBotDevice()
    yolo = YoloModel(device)
    yolo.load("/home/orangepi/Documents/dev/models/yolov5s_relu.rknn")
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "botsort.yaml"), 'r') as botfile:
        yaml_data = yaml.safe_load(botfile)

    args = BotSortArgs(yaml_data)
    tracker = BOTSORT(args=args, frame_rate=30)

    track_history = defaultdict(list)

    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bus.jpg")
    img = cv2.imread(img_path)

    output = yolo.infer([np.expand_dims(img, 0)])
    boxes, classes, scores = post_process_rknn(output, img)
    new_boxes = np.hstack((
        boxes, 
        np.arange(0, boxes.shape[0]).reshape(boxes.shape[0], 1), 
        scores.reshape(boxes.shape[0], 1),
        classes.reshape(boxes.shape[0], 1)
        ))
    # print(boxes)
    # print(new_boxes)
    # tracker.init_track(new_boxes, scores, classes, img=img)
    results = Results(
        img, img_path,
        {i: cls for i, cls in enumerate(CLASSES)},
        new_boxes,
        scores
    )
    tracked_boxes = tracker.update(results.boxes, img=img)
    print(tracked_boxes)
    post_process_rknn_tracking(tracked_boxes, img)

    cv2.imshow('YOLOv5 Detection', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
