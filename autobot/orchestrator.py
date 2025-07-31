from queue import Queue
from threading import Event, Thread

from autobot.tracker import AutoBotTracker
from autobot.common.queue import MessageBroker


class Orchestrator:
    def __init__(self, thread_functions):
        self.msg_broker = MessageBroker(topics=[
            "user_input",
            "tracker_to_follower",
            "tracker_to_lidar"
        ])
        self.stop_flag = Event()
        self.thread_functions = thread_functions
        self.threads = self.__load_threads()
    
    def __load_threads(self):
        return [
            Thread(target=f, args=(self.stop_flag, self.msg_broker))
            for f in self.thread_functions
        ]
    
    def execute(self):
        for thread in self.threads:
            thread.start()

        yolo_model_path = "/home/orangepi/Documents/dev/models/yolov5s_relu.rknn"
        autobot_tracker = AutoBotTracker(yolo_model_path=yolo_model_path)
        autobot_tracker.track(self.stop_flag, self.msg_broker)

        for thread in self.threads:
            thread.join()
        
        autobot_tracker.destroy()
