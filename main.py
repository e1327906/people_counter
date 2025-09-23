import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import numpy as np
import time

# =============================================================================
# CONFIGURATION SECTION - Easily modify detection areas and line properties
# =============================================================================

# Detection area configuration
AREA_CONFIG = {
    'type': 'horizontal',  # Options: 'vertical', 'horizontal', 'diagonal', 'custom'
    'extension': 500,    # How much to extend the areas (pixels)
    'width': 30,         # Width of detection zones
    'position': {        # Base position for areas
        'x_center': 500,
        'y_center': 250,
        'spacing': 80    # Distance between area1 and area2
    }
}

# Polyline visual configuration
POLYLINE_CONFIG = {
    'thickness': 2,      # Line thickness
    'color_area1': (0, 255, 0),    # Green for area1
    'color_area2': (255, 0, 0),    # Blue for area2
    'style': 'solid'     # Options: 'solid', 'dashed' (future enhancement)
}

def generate_detection_areas(config):
    """Generate detection areas based on configuration"""
    area_type = config['type']
    extension = config['extension']
    width = config['width']
    pos = config['position']
    
    if area_type == 'vertical':
        # Vertical detection lines (original style but configurable)
        y_top = pos['y_center'] - extension
        y_bottom = pos['y_center'] + extension
        
        area1 = [
            (pos['x_center'] - width//2, y_top),
            (pos['x_center'] - width//2, y_bottom),
            (pos['x_center'] + width//2, y_bottom),
            (pos['x_center'] + width//2, y_top)
        ]
        
        area2 = [
            (pos['x_center'] - width//2 + pos['spacing'], y_top),
            (pos['x_center'] - width//2 + pos['spacing'], y_bottom),
            (pos['x_center'] + width//2 + pos['spacing'], y_bottom),
            (pos['x_center'] + width//2 + pos['spacing'], y_top)
        ]
        
    elif area_type == 'horizontal':
        # Horizontal detection lines
        x_left = pos['x_center'] - extension
        x_right = pos['x_center'] + extension
        
        # Area1 is the upper horizontal line (for people entering from top)
        area1 = [
            (x_left, pos['y_center'] - width//2),
            (x_left, pos['y_center'] + width//2),
            (x_right, pos['y_center'] + width//2),
            (x_right, pos['y_center'] - width//2)
        ]
        
        # Area2 is the lower horizontal line (for people exiting to bottom)
        area2 = [
            (x_left, pos['y_center'] + pos['spacing'] - width//2),
            (x_left, pos['y_center'] + pos['spacing'] + width//2),
            (x_right, pos['y_center'] + pos['spacing'] + width//2),
            (x_right, pos['y_center'] + pos['spacing'] - width//2)
        ]
        
    elif area_type == 'diagonal':
        # Diagonal detection lines
        area1 = [
            (pos['x_center'] - extension, pos['y_center'] - extension),
            (pos['x_center'] - extension + width, pos['y_center'] - extension + width),
            (pos['x_center'] + extension, pos['y_center'] + extension),
            (pos['x_center'] + extension - width, pos['y_center'] + extension - width)
        ]
        
        area2 = [
            (pos['x_center'] - extension + pos['spacing'], pos['y_center'] - extension + pos['spacing']),
            (pos['x_center'] - extension + width + pos['spacing'], pos['y_center'] - extension + width + pos['spacing']),
            (pos['x_center'] + extension + pos['spacing'], pos['y_center'] + extension + pos['spacing']),
            (pos['x_center'] + extension - width + pos['spacing'], pos['y_center'] + extension - width + pos['spacing'])
        ]
        
    else:  # custom - use original coordinates as fallback
        area1 = [(494, 200), (505, 499), (578, 496), (530, 200)]
        area2 = [(548, 200), (600, 496), (637, 493), (574, 200)]
    
    return area1, area2

# Initialize YOLO model
model = YOLO('yolov8s.pt')

def people_counter(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

def load_class_list(file_path):
    with open(file_path, "r") as file:
        return file.read().split("\n")

def process_frame(frame, model, class_list, tracker, area1, area2, going_out, going_in, counter1, counter2, detection_type='vertical'):
    frame = cv2.resize(frame, (1020, 500), interpolation=cv2.INTER_AREA)
    results = model.predict(frame)
    boxes_data = results[0].boxes.data
    px = pd.DataFrame(boxes_data).astype("float")

    detected_objects = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        if 'person' in class_list[d]:
            detected_objects.append([x1, y1, x2, y2])

    objects_bbs_ids = tracker.update(detected_objects)
    
    # Draw all detected people first (for debugging)
    for bbox in objects_bbs_ids:
        x3, y3, x4, y4, obj_id = bbox
        # Draw basic detection rectangle for all people
        cv2.rectangle(frame, (x3, y3), (x4, y4), (128, 128, 128), 1)
        cv2.putText(frame, f'ID:{obj_id}', (x3, y3-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Use multiple detection points for better accuracy
        center_x = (x3 + x4) // 2
        bottom_y = y4
        center_y = (y3 + y4) // 2
        
        detection_points = [
            (center_x, bottom_y),      # Bottom center (feet)
            (center_x, center_y),      # Body center
            (center_x, y3 + 20)        # Upper body
        ]
        
        # Check if any detection point is in either area
        in_area1 = any(cv2.pointPolygonTest(np.array(area1, np.int32), point, False) >= 0 for point in detection_points)
        in_area2 = any(cv2.pointPolygonTest(np.array(area2, np.int32), point, False) >= 0 for point in detection_points)
        
        # Adjust detection logic based on area type
        if detection_type == 'horizontal':
            # For horizontal lines: detect top-to-bottom movement
            # Area1 is upper line, Area2 is lower line
            
            # Track when person enters area1 (upper line)
            if in_area1:
                going_in[obj_id] = (center_x, bottom_y)
                cv2.circle(frame, (center_x, bottom_y), 6, (0, 255, 255), -1)  # Yellow dot
                
            # Track when person enters area2 (lower line)  
            if in_area2:
                going_out[obj_id] = (center_x, bottom_y)
                cv2.circle(frame, (center_x, bottom_y), 6, (255, 0, 255), -1)  # Magenta dot
            
            # Count when person goes from area1 to area2 (downward = IN)
            if obj_id in going_in and in_area2:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)  # Green rectangle
                cvzone.putTextRect(frame, f'IN-{obj_id}', (x3, y3-20), 1, 1)
                if obj_id not in counter1:
                    counter1.append(obj_id)
                    print(f"Person {obj_id} counted as IN (downward)")
                    
            # Count when person goes from area2 to area1 (upward = OUT)
            if obj_id in going_out and in_area1:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)  # Red rectangle
                cvzone.putTextRect(frame, f'OUT-{obj_id}', (x3, y3-20), 1, 1)
                if obj_id not in counter2:
                    counter2.append(obj_id)
                    print(f"Person {obj_id} counted as OUT (upward)")
                    
        else:
            # Original vertical detection logic (improved)
            if in_area2:
                going_out[obj_id] = (center_x, bottom_y)
                cv2.circle(frame, (center_x, bottom_y), 6, (255, 0, 255), -1)
                
            if in_area1:
                going_in[obj_id] = (center_x, bottom_y)
                cv2.circle(frame, (center_x, bottom_y), 6, (0, 255, 255), -1)
            
            # Count crossings
            if obj_id in going_out and in_area1:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)
                cvzone.putTextRect(frame, f'IN-{obj_id}', (x3, y3-20), 1, 1)
                if obj_id not in counter1:
                    counter1.append(obj_id)
                    print(f"Person {obj_id} counted as IN")

            if obj_id in going_in and in_area2:
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                cvzone.putTextRect(frame, f'OUT-{obj_id}', (x3, y3-20), 1, 1)
                if obj_id not in counter2:
                    counter2.append(obj_id)
                    print(f"Person {obj_id} counted as OUT")

    return frame, len(counter1), len(counter2)

def main():
    cv2.namedWindow('people_counter')
    cv2.setMouseCallback('people_counter', people_counter)
    cap = cv2.VideoCapture('video/test_1.mp4')

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    class_list = load_class_list("coco.txt")
    tracker = Tracker()

    # Generate detection areas based on configuration
    area1, area2 = generate_detection_areas(AREA_CONFIG)
    print(f"Detection type: {AREA_CONFIG['type']}")
    print(f"Area1: {area1}")
    print(f"Area2: {area2}")
    
    going_out, going_in = {}, {}
    counter1, counter2 = [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    #delay = int(1000 / fps)  # Target delay in milliseconds
    delay = int(0 / fps)  # Target delay in milliseconds
    print(f"Video FPS: {fps}, Target delay: {delay}ms")

    while True:
        # Start timing the frame processing
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video or encountered an error.")
            break

        frame, out_count, in_count = process_frame(frame, model, class_list, tracker, area1, area2, going_out, going_in, counter1, counter2, AREA_CONFIG['type'])

        cv2.putText(frame, f'In: {in_count}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Out: {out_count}', (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f'Detected: {len(tracker.center_points)}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        print(f'In: {in_count}, Out: {out_count}, Active IDs: {list(tracker.center_points.keys())}')
        
        # Draw configurable polylines
        cv2.polylines(frame, [np.array(area1, np.int32)], True, POLYLINE_CONFIG['color_area1'], POLYLINE_CONFIG['thickness'])
        cv2.polylines(frame, [np.array(area2, np.int32)], True, POLYLINE_CONFIG['color_area2'], POLYLINE_CONFIG['thickness'])
        
        # Add labels for areas
        cv2.putText(frame, 'AREA1', (area1[0][0], area1[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, POLYLINE_CONFIG['color_area1'], 1)
        cv2.putText(frame, 'AREA2', (area2[0][0], area2[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, POLYLINE_CONFIG['color_area2'], 1)

        cv2.imshow("people_counter", frame)
        
        # Calculate processing time and adjust delay
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        adjusted_delay = max(1, delay - int(processing_time))  # Ensure minimum 1ms delay
        
        # Optional: Display processing stats (remove this line if not needed)
        if processing_time > delay:
            print(f"Processing time ({processing_time:.1f}ms) exceeds target delay ({delay}ms)")
        
        if cv2.waitKey(adjusted_delay) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()