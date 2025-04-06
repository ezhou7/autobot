import cv2
import yaml
import numpy as np
from collections import defaultdict
from queue import Queue
from threading import Event, Thread

from ultralytics.trackers import BOTSORT
from ultralytics.engine.results import Results
from autobot import get_resource
from autobot.device import AutoBotDevice
from autobot.model import YoloModel
from autobot.utils import Properties
from autobot.utils.postprocess import post_process_rknn, post_process_rknn_tracking, post_process_rknn_selecting
from autobot.utils.yolov5 import CLASSES


STANDARD_CAPTURE = 20


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


def yolo_thread(stop_flag: Event, input_queue: Queue, output_queue: Queue, cap: cv2.VideoCapture):
    device = AutoBotDevice()
    yolo = YoloModel(device)
    yolo.load("/home/orangepi/Documents/dev/models/yolov5s_relu.rknn")

    botsort_args = yaml.safe_load(get_resource("botsort.yaml"))
    props = Properties(botsort_args)
    tracker = BOTSORT(args=props, frame_rate=30)

    currently_selected_id = -1
    while not stop_flag.is_set():
        success, frame = cap.read()
        # frame = cv2.flip(frame, 0)
        frame = cv2.resize(frame, (640, 640))
        # print(frame.shape)

        if not success:
            break

        output = yolo.infer([np.expand_dims(frame, 0)])
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
            tracked_boxes = tracker.update(results.boxes, img=frame)
            post_process_rknn_tracking(tracked_boxes, frame)
            post_process_rknn_selecting(tracked_boxes, frame, currently_selected_id)
        
        if not input_queue.empty():
            inp = input_queue.get()
            if inp == 'q':
                break
            else:
                currently_selected_id = int(inp)

        cv2.imshow("YOLOv5 Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def input_thread(stop_flag: Event, input_queue: Queue):
    while not stop_flag.is_set():
        user_input = input("Enter 'q' to quit: ")
        input_queue.put(user_input)
        if user_input == 'q':
            stop_flag.set()
            break


if __name__ == "__main__":
    input_queue = Queue()
    stop_flag = Event()
    output_queue = Queue()

    cap = video_capture()
    input_thread_obj = Thread(target=input_thread, args=(stop_flag, input_queue))
    input_thread_obj.start()

    yolo_thread(stop_flag, input_queue, output_queue, cap)
    
    input_queue.put('q')
    input_thread_obj.join()

    cap.release()
    cv2.destroyAllWindows()
