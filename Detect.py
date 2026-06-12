import cv2
import time
import numpy as np
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO(r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\runs\detect\train-4\weights\last.pt")

# Video source
video_path = r"D:\Intern Project\YOLOv8_Detection-OperatorRooms\_โรงงานบางโคล่_20260513135249_55169326.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video file: {video_path}")
    exit()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
SMOOTHING_ALPHA        = 0.2
IOU_MATCH_THRESHOLD    = 0.4
CENTROID_FALLBACK      = 80

HEAD_MAX_RATIO         = 0.50
HEAD_MIN_SIZE          = 20
HEAD_ASPECT_MIN        = 0.25
HEAD_ASPECT_MAX        = 3.0

PEOPLE_MAX_RATIO       = 0.48
PEOPLE_MIN_SIZE        = 30
PEOPLE_ASPECT_MIN      = 0.25
PEOPLE_ASPECT_MAX      = 3.0
PEOPLE_NMS_IOU         = 0.20
PEOPLE_CENTROID_MAX    = 80

PHONE_MAX_RATIO        = 0.15
PHONE_MIN_SIZE         = 10
PHONE_CLOSE_MARGIN     = 40
PHONE_STABILITY_FRAMES = 6
PHONE_STATUS_HOLD_TIME = 8
PHONE_HOLD_FRAMES      = 3

STATUS_DEBOUNCE_FRAMES = 10
NOBODY_TIMEOUT_SEC   = 600

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────
smoothed_head_boxes    = []
smoothed_phone_boxes   = []

last_person_seen       = time.time()
current_status         = "Standby Loading"
pending_status         = "Standby Loading"
status_debounce        = 0

phone_detect_counter   = 0
phone_hold_counter     = 0
show_phone_status      = False
phone_timers           = {}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a1 = max(0, x2 - x1) * max(0, y2 - y1)
    a2 = max(0, x4 - x3) * max(0, y4 - y3)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def calculate_iomin(box1, box2):
    """Intersection over min(area) — catches partial containment."""
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    ix1, iy1 = max(x1, x3), max(y1, y3)
    ix2, iy2 = min(x2, x4), min(y2, y4)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter    = (ix2 - ix1) * (iy2 - iy1)
    a1       = max(0, x2 - x1) * max(0, y2 - y1)
    a2       = max(0, x4 - x3) * max(0, y4 - y3)
    min_area = min(a1, a2)
    return inter / min_area if min_area > 0 else 0.0


def smooth_boxes(current_boxes, smoothed_boxes, alpha=SMOOTHING_ALPHA):
    """EMA-smooth bounding boxes with IoU + centroid-distance matching."""
    if not current_boxes:
        return []
    processed = [np.array(b, dtype=float) for b in current_boxes]
    if not smoothed_boxes:
        return [b.copy() for b in processed]

    result, used = [], set()
    for curr in processed:
        best_idx, best_iou = None, 0
        for idx, smooth in enumerate(smoothed_boxes):
            if idx in used:
                continue
            iou = calculate_iou(curr, smooth)
            if iou > best_iou and iou > IOU_MATCH_THRESHOLD:
                best_iou, best_idx = iou, idx
        if best_idx is None:
            cx, cy = (curr[0]+curr[2])/2, (curr[1]+curr[3])/2
            best_dist = float("inf")
            for idx, smooth in enumerate(smoothed_boxes):
                if idx in used:
                    continue
                scx = (smooth[0]+smooth[2])/2
                scy = (smooth[1]+smooth[3])/2
                dist = ((cx-scx)**2 + (cy-scy)**2)**0.5
                if dist < CENTROID_FALLBACK and dist < best_dist:
                    best_dist, best_idx = dist, idx
        if best_idx is not None:
            blended = alpha * curr + (1 - alpha) * smoothed_boxes[best_idx]
            result.append(blended.copy())
            used.add(best_idx)
        else:
            result.append(curr.copy())
    return result


def is_overlap_or_close(box_a, box_b, margin=PHONE_CLOSE_MARGIN):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    return not (ax2 < bx1-margin or ax1 > bx2+margin or
                ay2 < by1-margin or ay1 > by2+margin)


def people_nms(raw):
    """IoMin-based NMS + centroid dedup for People detections."""
    raw.sort(key=lambda x: x[2], reverse=True)
    keep, suppressed = [], set()
    for i, (tid_i, bbox_i, conf_i) in enumerate(raw):
        if i in suppressed:
            continue
        keep.append((tid_i, bbox_i, conf_i))
        for j in range(i+1, len(raw)):
            if j not in suppressed:
                b2 = raw[j][1]
                if max(calculate_iou(bbox_i, b2), calculate_iomin(bbox_i, b2)) > PEOPLE_NMS_IOU:
                    suppressed.add(j)
    result = []
    for tid_i, bbox_i, conf_i in keep:
        cx_i = (bbox_i[0]+bbox_i[2])/2
        cy_i = (bbox_i[1]+bbox_i[3])/2
        dup = any(
            ((cx_i-(b[1][0]+b[1][2])/2)**2 + (cy_i-(b[1][1]+b[1][3])/2)**2)
            < PEOPLE_CENTROID_MAX**2
            for b in result)
        if not dup:
            result.append((tid_i, bbox_i, conf_i))
    return result

def center_inside(inner, outer):
  cx=(inner[0]+inner[2])/2
  cy=(inner[1]+inner[3])/2
  return outer[0]<=cx<=outer[2] and outer[1]<=cy<=outer[3]

# ─────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    results = model.track(frame, conf=0.15, persist=True, tracker="bytetrack.yaml")

    head_raw, phone_raw, people_raw = [], [], []

    for result in results:
        for box in result.boxes:
            cls  = int(box.cls[0])
            name = model.names[cls]
            bbox = box.xyxy[0].cpu().numpy().copy()
            conf = float(box.conf[0])
            bw   = bbox[2] - bbox[0]
            bh   = bbox[3] - bbox[1]

            if name == "Head":
                if bw < HEAD_MIN_SIZE or bh < HEAD_MIN_SIZE:
                    continue
                if bw > frame_w * HEAD_MAX_RATIO or bh > frame_h * HEAD_MAX_RATIO:
                    continue
                aspect = bw / max(bh, 1)
                if not (HEAD_ASPECT_MIN <= aspect <= HEAD_ASPECT_MAX):
                    continue
                head_raw.append(bbox)

            elif name == "Phone":
                if bw < PHONE_MIN_SIZE or bh < PHONE_MIN_SIZE:
                    continue
                if bw > frame_w * PHONE_MAX_RATIO or bh > frame_h * PHONE_MAX_RATIO:
                    continue
                phone_raw.append(bbox)

            elif name == "People":
                if bw < PEOPLE_MIN_SIZE or bh < PEOPLE_MIN_SIZE:
                    continue
                if bw > frame_w * PEOPLE_MAX_RATIO or bh > frame_h * PEOPLE_MAX_RATIO:
                    continue
                aspect = bw / max(bh, 1)
                if not (PEOPLE_ASPECT_MIN <= aspect <= PEOPLE_ASPECT_MAX):
                    continue
                tid = int(box.id[0]) if box.id is not None else -1
                people_raw.append((tid, bbox, conf))

    smoothed_head_boxes  = smooth_boxes(head_raw,  smoothed_head_boxes)
    smoothed_phone_boxes = smooth_boxes(phone_raw, smoothed_phone_boxes)
    people_display       = people_nms(people_raw)
    employee_count       = len(people_display)

    if employee_count > 0 or smoothed_head_boxes:
        last_person_seen = time.time()

    time_since_person = time.time() - last_person_seen
    person_absent     = time_since_person >= NOBODY_TIMEOUT_SEC

    if employee_count > 0 :
        candidate_status = "Standby Loading"
    elif person_absent:
        candidate_status = "Nobody Here"
    else:
        candidate_status = "Standby Loading"

    if candidate_status == pending_status:
        status_debounce = min(status_debounce + 1, STATUS_DEBOUNCE_FRAMES)
    else:
        pending_status  = candidate_status
        status_debounce = 1
    if status_debounce >= STATUS_DEBOUNCE_FRAMES:
        current_status = pending_status

    # Phone detection: trigger when ALL 3 classes overlap, OR Head+Phone overlap
    detected_phone_head = False

    # Condition 1: People + Head (inside People) + Phone (near Head)
    if people_display and not detected_phone_head:
        for _ptid, p_box, _pc in people_display:
            for hb in smoothed_head_boxes:
                if calculate_iou(hb, p_box) < 0.05:
                    continue
                for pb in smoothed_phone_boxes:
                    if is_overlap_or_close(pb, hb):
                        detected_phone_head = True
                        break
                if detected_phone_head:
                    break
            if detected_phone_head:
                break

    # Condition 2: Head + Phone overlap (no People box required)
    if not detected_phone_head:
        for hb in smoothed_head_boxes:
            for pb in smoothed_phone_boxes:
                if is_overlap_or_close(pb, hb):
                    detected_phone_head = True
                    break
            if detected_phone_head:
                break

    if detected_phone_head:
        phone_detect_counter = min(phone_detect_counter + 1, PHONE_STABILITY_FRAMES)
        phone_hold_counter   = PHONE_HOLD_FRAMES
    else:
        if phone_hold_counter > 0:
            phone_hold_counter -= 1
        else:
            if (phone_detect_counter >= PHONE_STABILITY_FRAMES
                    and phone_call_start_time is not None
                    and phone_call_end_time is None):
                phone_call_end_time = time.time()
                phone_call_duration = int(phone_call_end_time - phone_call_start_time)
            phone_detect_counter = max(0, phone_detect_counter - 1)

    stable_on_phone = phone_detect_counter >= PHONE_STABILITY_FRAMES
    if stable_on_phone:
        if phone_call_start_time is None:
            phone_call_start_time = time.time()
            phone_call_end_time   = None
            phone_call_duration   = 0
        show_phone_status = True
    else:
        if phone_call_end_time is not None:
            if time.time() - phone_call_end_time < PHONE_STATUS_HOLD_TIME:
                show_phone_status = True
            else:
                show_phone_status     = False
                phone_call_start_time = None
                phone_call_end_time   = None
                phone_call_duration   = 0
        else:
            show_phone_status = False

    # Drawing
    annotated = frame.copy()

    for box in smoothed_phone_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 0), 2)

    for box in smoothed_head_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)

    for tid, box, conf in people_display:
        x1, y1, x2, y2 = map(int, box)
        has_head  = any(center_inside(hb, box) for hb in smoothed_head_boxes)
        has_phone = any(center_inside(pb, box) for pb in smoothed_phone_boxes)
        tags  = ("Head " if has_head else "") + ("Phone" if has_phone else "")
        label = "P#" + str(tid) + " (" + f"{conf:.2f}" + ")" + (" [" + tags.strip() + "]" if tags.strip() else "")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(y1 - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    text_color = (0, 255, 0) if current_status != "Nobody Here" else (0, 0, 255)
    tw = cv2.getTextSize(current_status, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0][0]
    cv2.putText(annotated, current_status,
                (frame_w - tw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)

    emp_text = "Employee: " + str(employee_count)
    tw = cv2.getTextSize(emp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
    cv2.putText(annotated, emp_text,
                (frame_w - tw - 10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if show_phone_status:
        seconds = (int(time.time() - phone_call_start_time)
                   if phone_call_end_time is None else phone_call_duration)
        ph_text = "Phone Call: " + str(seconds) + " seconds"
        tw = cv2.getTextSize(ph_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]
        cv2.putText(annotated, ph_text,
                    (frame_w - tw - 10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("YOLOv8 Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()