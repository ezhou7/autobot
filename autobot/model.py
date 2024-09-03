import numpy as np
from ultralytics import YOLO
from typing import Optional, Union

from autobot.device import AutoBotDevice


RKNN_MODEL_TYPE = "rknn"
TORCH_MODEL_TYPE = "torch"


class YoloModel:
    def __init__(self, device: AutoBotDevice):
        self.model: Optional[Union[YOLO, RKNNLite]] = None
        self.model_type = ""
        self.tracker = None
        self.device = device

    def load(self, model_path: str):
        hostname = self.device.get_host()

        if hostname in ["RK3576", "RK3588"]:
            from rknnlite.api import RKNNLite
            self.model = RKNNLite()
            self.model_type = RKNN_MODEL_TYPE

            self.model.load_rknn(model_path)
            status = self.model.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

            if status != 0:
                raise Exception("RKNN runtime initialization failed")
        else:
            self.model = YOLO(model_path)
            self.model_type = TORCH_MODEL_TYPE

    def infer(self, input_data: list[np.ndarray]):
        if self.model_type == RKNN_MODEL_TYPE:
            output = self.model.inference(inputs=input_data)
        elif self.model_type == TORCH_MODEL_TYPE:
            output = self.model.predict(input_data)
        else:
            raise Exception(f"Invalid model type={self.model_type} loaded")

        return output

    def track(self, input_data: np.ndarray):
        if self.model_type == TORCH_MODEL_TYPE:
            # if not self.tracker:
            #     self.tracker = self.model.track(input_data)
            output = self.model.track(input_data, persist=True, tracker="botsort.yaml")
        else:
            raise Exception(f"Tracker unavailable for model type={self.model_type}")

        return output
