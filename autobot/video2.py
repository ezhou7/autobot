import asyncio
import cv2
import yaml
import numpy as np
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


class AutoBotTracker:
    def __init__(self, yolo_model_path: str):
        self.cap = video_capture()
        self.device = AutoBotDevice()
        self.yolo = self.__load_yolo_model(yolo_model_path)
        self.tracker = self.__load_tracker_model()
    
    def __load_yolo_model(self, path: str):
        yolo = YoloModel(self.device)
        yolo.load(path)

        return yolo

    def __load_tracker_model(self):
        botsort_args = yaml.safe_load(get_resource("botsort.yaml"))
        props = Properties(botsort_args)
        return BOTSORT(args=props, frame_rate=30)
    
    def track(self, stop_flag, input_queue):
        currently_selected_id = -1
        while not stop_flag.is_set():
            success, frame = self.cap.read()
            frame = cv2.resize(frame, (640, 640))

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
    
    def destroy(self):
        self.cap.release()
        cv2.destroyAllWindows()


def execute_input_thread(stop_flag: Event, input_queue: Queue):
    while not stop_flag.is_set():
        user_input = input("Enter 'q' to quit: ")
        input_queue.put(user_input)
        if user_input == 'q':
            stop_flag.set()
            break


def async_thread(stop_flag: Event, input_queue: Queue):
    async def async_function():
        await asyncio.sleep(1)
        print("hello")
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(async_function())
    loop.close()


class Orchestrator:
    def __init__(self, thread_functions):
        self.input_queue = Queue()
        self.stop_flag = Event()
        self.thread_functions = thread_functions
        self.threads = self.__load_threads()
    
    def __load_threads(self):
        return [
            Thread(target=f, args=(self.stop_flag, self.input_queue))
            for f in self.thread_functions
        ]
    
    def execute(self):
        for thread in self.threads:
            thread.start()

        yolo_model_path = "/home/orangepi/Documents/dev/models/yolov5s_relu.rknn"
        autobot_tracker = AutoBotTracker(yolo_model_path=yolo_model_path)
        autobot_tracker.track(self.stop_flag, self.input_queue)

        self.input_queue.put('q')
        for thread in self.threads:
            thread.join()
        
        autobot_tracker.destroy()


if __name__ == "__main__":
    orchestrator = Orchestrator([execute_input_thread, async_thread])
    orchestrator.execute()
