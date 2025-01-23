import math
import cv2
import cvzone
import dlib
from ultralytics import YOLO

# Initialize video capture
cap = cv2.VideoCapture(r"C:\Users\Salman\Downloads\WhatsApp Video 2024-09-04 at 9.42.36 PM.mp4")
cap.set(3, 1280)  # Set frame width
cap.set(4, 720)   # Set frame height

# Set up video writer to save output video with a higher FPS
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Define codec
out = cv2.VideoWriter('output_video_fast.mp4', fourcc, 200.0, (1280, 720))  # Increase FPS to 60

# Load YOLO model
model = YOLO(r'C:\python2024\Object detection with KerasCV\trained_yolov8l_model.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Initialize a list to store dlib trackers and a dictionary for IDs
trackers = []
tracker_ids = {}
id_counter = 0

# Function to find the closest existing tracker for a new detection
def find_closest_tracker(x1, y1, x2, y2, threshold=50):
    min_distance = float('inf')
    closest_tracker = None

    for tracker in trackers:
        pos = tracker.get_position()
        tx1, ty1, tx2, ty2 = int(pos.left()), int(pos.top()), int(pos.right()), int(pos.bottom())
        center_new = ((x1 + x2) / 2, (y1 + y2) / 2)
        center_existing = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
        distance = math.sqrt((center_new[0] - center_existing[0]) ** 2 + (center_new[1] - center_existing[1]) ** 2)
        if distance < min_distance and distance < threshold:
            min_distance = distance
            closest_tracker = tracker

    return closest_tracker

frame_skip = 5  # Skip every 2nd frame to speed up the video

while True:
    for _ in range(frame_skip):
        success, img = cap.read()  # Read the frame
        
    if not success or img is None or img.size == 0:
        print("Failed to capture image or image is empty")
        break

    results = model(img, stream=True)
    
    new_trackers = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if (currentClass == 'car' or currentClass == 'truck' or currentClass == 'bus') and conf > 0.3:
                # Find the closest existing tracker
                closest_tracker = find_closest_tracker(x1, y1, x2, y2)

                if closest_tracker is None:
                    # Initialize a new tracker if no close existing tracker is found
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(x1, y1, x2, y2)
                    tracker.start_track(img, rect)
                    new_trackers.append(tracker)
                    tracker_ids[tracker] = id_counter
                    id_counter += 1
                else:
                    # Update the closest tracker with the new position
                    rect = dlib.rectangle(x1, y1, x2, y2)
                    closest_tracker.start_track(img, rect)
                    new_trackers.append(closest_tracker)

    # Update existing trackers
    for tracker in new_trackers:
        tracker.update(img)
        pos = tracker.get_position()
        x1 = int(pos.left())
        y1 = int(pos.top())
        x2 = int(pos.right())
        y2 = int(pos.bottom())
        cvzone.cornerRect(img, (x1, y1, x2-x1, y2-y1), l=9)
        
        # Draw the ID associated with this tracker
        cvzone.putTextRect(img, f'ID:{tracker_ids[tracker]}', 
                           (max(0, x1), max(35, y1)), scale=1, offset=3, thickness=2)
    
    trackers = new_trackers

    # Write the processed frame to the video file
    out.write(img)

    # Display images
    if img is not None and img.size > 0:
        cv2.imshow("Image", img)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and writer
cap.release()
out.release()
cv2.destroyAllWindows()
