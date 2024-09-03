import cv2
import numpy as np
from ultralytics.engine.results import Results


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
