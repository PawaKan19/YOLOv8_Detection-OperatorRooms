import cv2
import time
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\runs\detect\train-15\weights\last.pt")

# Video source - เปลี่ยน path ไฟล์วิดีโอที่ต้องการตรวจจับ
video_path = r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\20260514_20260514145436_20260514150144_151033.mov"
cap = cv2.VideoCapture(video_path)

# ตรวจสอบว่าเปิดวิดีโอได้หรือไม่
if not cap.isOpened():
    print(f"Error: Cannot open video file: {video_path}")
    print("Please check if the video file exists at the specified path.")
    exit()

# Stability parameters
STABILITY_FRAMES = 10  # Frames needed for stable detection
PHONE_CLOSE_MARGIN = 12  # Pixel margin to treat boxes as touching
PHONE_STABILITY_FRAMES = 5
PHONE_STATUS_HOLD_TIME = 5  # Seconds to keep phone call status after call ends
SMOOTHING_ALPHA = 0.6  # Exponential moving average smoothing factor (0-1) - higher for more stability
HEAD_ABSENCE_TIMEOUT = 600  # 10 minutes before showing Nobody In Room
PHONE_MAX_RATIO = 0.15  # phone box must be < 15% of frame width/height
PHONE_MIN_SIZE = 10   # minimum pixel size for a phone box

# Stability counters
current_status = "Standby Loading"
last_person_seen = time.time()


# Phone call tracking
phone_call_start_time = None
phone_call_end_time = None
phone_call_duration = 0  # Final duration in seconds
phone_detect_counter = 0
stable_on_phone = False
show_phone_status = False

# Smoothed People boxes & counters
smoothed_people_boxes = []
people_counter = 0
last_stable_people_count = 0

# Smoothed bounding boxes for display
smoothed_head_boxes = []
smoothed_phone_boxes = []

def is_overlap_or_close(bbox_a, bbox_b, margin=PHONE_CLOSE_MARGIN):
    """Return True when two boxes overlap or are very close."""
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b

    expanded_b = (bx1 - margin, by1 - margin, bx2 + margin, by2 + margin)
    ex1, ey1, ex2, ey2 = expanded_b

    if ax2 < ex1 or ax1 > ex2 or ay2 < ey1 or ay1 > ey2:
        return False
    return True

def smooth_boxes(current_boxes, smoothed_boxes, alpha=SMOOTHING_ALPHA):
    """Apply exponential moving average to smooth bounding boxes."""
    if len(current_boxes) == 0:
        return []
    
    if len(smoothed_boxes) == 0:
        return current_boxes.copy()
    
    # Simple matching based on IoU or proximity
    result = []
    used_indices = set()
    
    for curr_box in current_boxes:
        best_match_idx = None
        best_iou = 0
        
        for i, smooth_box in enumerate(smoothed_boxes):
            if i in used_indices:
                continue
            
            # Calculate IoU
            iou = calculate_iou(curr_box, smooth_box)
            if iou > best_iou and iou > 0.3:
                best_iou = iou
                best_match_idx = i
        
        if best_match_idx is not None:
            # Smooth with matched box
            smoothed = alpha * curr_box + (1 - alpha) * smoothed_boxes[best_match_idx]
            result.append(smoothed)
            used_indices.add(best_match_idx)
        else:
            # New detection, add as-is
            result.append(curr_box)
    
    return result

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two boxes."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    # Calculate intersection
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Calculate union
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection
    results = model(frame, conf=0.5)
    annotated_frame = frame.copy()
    frame_h, frame_w = frame.shape[:2]

    # Extract bounding boxes by class
    people_boxes = []
    head_boxes = []
    phone_boxes = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = str(model.names[class_id])
            bbox = box.xyxy[0].cpu().numpy()
            
            if class_name == "People":
                people_boxes.append(bbox)
            elif class_name == "Head":
                head_boxes.append(bbox)
            elif class_name == "Phone":
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                if bw < PHONE_MIN_SIZE or bh < PHONE_MIN_SIZE:
                    continue
                if bw > frame_w * PHONE_MAX_RATIO or bh > frame_h * PHONE_MAX_RATIO:
                    continue
                phone_boxes.append(bbox)
    
    # Apply temporal smoothing to bounding boxes
    smoothed_head_boxes = smooth_boxes(head_boxes, smoothed_head_boxes)
    smoothed_phone_boxes = smooth_boxes(phone_boxes, smoothed_phone_boxes)
    
    # Phone call detection: Phone overlap/touch with Head (use smoothed boxes)
    detected_phone_head = False
    for head_box in smoothed_head_boxes:
        for phone_box in smoothed_phone_boxes:
            if is_overlap_or_close(phone_box, head_box):
                detected_phone_head = True
                break
        if detected_phone_head:
            break

    if detected_phone_head:
        phone_detect_counter += 1
    else:
        phone_detect_counter = max(0, phone_detect_counter - 1)

    stable_on_phone = phone_detect_counter >= PHONE_STABILITY_FRAMES
    if stable_on_phone:
        if phone_call_start_time is None:
            phone_call_start_time = time.time()
        show_phone_status = True
        phone_call_end_time = None
        phone_call_duration = 0
    else:
        if phone_call_start_time is not None and phone_call_end_time is None:
            phone_call_end_time = time.time()
            # Save final duration when call ends
            phone_call_duration = int(phone_call_end_time - phone_call_start_time)
        
        # Keep showing status for 5 seconds after call ends
        if phone_call_end_time is not None:
            if time.time() - phone_call_end_time < PHONE_STATUS_HOLD_TIME:
                show_phone_status = True
            else:
                show_phone_status = False
                phone_call_start_time = None
                phone_call_end_time = None
                phone_call_duration = 0
        else:
            show_phone_status = False
    
    # Draw bounding boxes (use smoothed boxes for stability)
    for box in smoothed_phone_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(annotated_frame, "Phone", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    for box in smoothed_head_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(annotated_frame, "Head", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    
    # Display status
    frame_height, frame_width = frame.shape[:2]
    
    # Status text
    # Update people/head stability counter
    visible_person_boxes = len(smoothed_people_boxes) + len(smoothed_head_boxes)
    if visible_person_boxes > 0:
        people_counter = min(people_counter + 1, STABILITY_FRAMES)
        last_person_seen = time.time()
    else:
        people_counter = max(people_counter - 1, 0)

    if people_counter >= STABILITY_FRAMES:
        last_stable_people_count = max(len(smoothed_people_boxes), len(smoothed_head_boxes))

    stable_people = people_counter >= STABILITY_FRAMES
    time_since_person = time.time() - last_person_seen
    person_absent = time_since_person >= HEAD_ABSENCE_TIMEOUT

    if person_absent:
        current_status = "Nobody In Room"
        last_stable_people_count = 0
    else:
        current_status = "Standby Loading"

    status_color = (0, 0, 255) if person_absent else (0, 255, 0)
    text_size = cv2.getTextSize(current_status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(annotated_frame, current_status,
                (frame_width - text_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    # Employee count: hold last stable count until absence timeout
    display_people_count = last_stable_people_count
    employee_count = f"Employee: {display_people_count} people"
    text_size = cv2.getTextSize(employee_count, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(annotated_frame, employee_count, 
                (frame_width - text_size[0] - 10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Phone call status (show for 5 seconds after call ends)
    if show_phone_status:
        if phone_call_end_time is None:
            # Call is still active - count elapsed time
            phone_call_seconds = int(time.time() - phone_call_start_time)
        else:
            # Call ended - show final duration (stop counting)
            phone_call_seconds = phone_call_duration
        phone_status = f"Phone Call: {phone_call_seconds} seconds"
        text_size = cv2.getTextSize(phone_status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.putText(annotated_frame, phone_status, 
                (frame_width - text_size[0] - 10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Display frame
    cv2.imshow("YOLOv8 Detection", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()