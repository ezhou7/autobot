import cv2
import numpy as np
from ultralytics import YOLO

# --- Constants ---
VIDEO_CAPTURE_INDEX = 0  # 0 for webcam, or path to a video file
FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def tracker_process(stop_flag, target_queue, yolo_model_path='yolov8n.pt'):
    """
    This process runs the YOLO object tracker.
    It detects and tracks objects in a video stream and puts the coordinates
    of the primary target into a queue for the flight controller.
    """
    # Load the YOLO model
    model = YOLO(yolo_model_path)

    # Setup video capture
    cap = cv2.VideoCapture(VIDEO_CAPTURE_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)

    print("[Tracker] Tracker process started.")
    
    # Variable to store the ID of the object we are tracking
    target_id = None

    while not stop_flag.is_set():
        success, frame = cap.read()
        if not success:
            print("[Tracker] Failed to grab frame.")
            break

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        # and filtering for the 'person' class (class ID 0)
        results = model.track(frame, persist=True, verbose=False, classes=0)

        # Get the bounding boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id

        # --- Visualization and Tracking Logic ---
        annotated_frame = results[0].plot()

        if track_ids is not None:
            # If we don't have a target, pick the first one.
            if target_id is None and len(track_ids) > 0:
                target_id = track_ids[0].item()
                print(f"[Tracker] New target selected with ID: {target_id}")

            target_found_in_frame = False
            # Find the current position of our target.
            if target_id is not None:
                for i, current_id in enumerate(track_ids):
                    if current_id.item() == target_id:
                        x, y, w, h = boxes[i]
                        # Put coordinates in the queue for the drone
                        target_queue.put((x.item(), y.item()))
                        
                        # Draw the red circle for visualization
                        center_x, center_y = int(x), int(y)
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                        
                        target_found_in_frame = True
                        break

            # If our chosen target has disappeared from the frame.
            if not target_found_in_frame and target_id is not None:
                print(f"[Tracker] Lost target with ID: {target_id}. Searching for new target.")
                target_id = None
                target_queue.put((-1, -1))
                # # Clear the queue so the drone hovers
                # while not target_queue.empty():
                #     target_queue.get_nowait()

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    stop_flag.set()
    cap.release()
    cv2.destroyAllWindows()
    print("[Tracker] Tracker process stopped.")
