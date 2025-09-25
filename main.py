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
    'width': 150,        # VERY LARGE width to cover more vertical space - increased from 100 to 150
    'position': {        # Base position for areas
        'x_center': 500,
        'y_center': 250,  # Center of frame - people from top/bottom should cross here
        'spacing': 30    # Reduced spacing further from 40 to 30 for better overlap
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
        # Horizontal detection lines - positioned for bottom area where people actually walk
        x_left = 0  # Full width of frame
        x_right = 1000
        
        # Area1 is the first crossing line (people coming from outside frame)
        area1 = [
            (x_left, 400),   # Y=400 (upper line for bottom area)
            (x_left, 430),   # Y=430 
            (x_right, 430),
            (x_right, 400)
        ]
        
        # Area2 is the second crossing line 
        area2 = [
            (x_left, 320),   # Y=320 (lower line - closer to detection area)
            (x_left, 350),   # Y=350
            (x_right, 350),
            (x_right, 320)
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

# Global tracking for person states
person_states = {}

class PersonState:
    def __init__(self):
        self.last_area = None  # 1 for area1, 2 for area2, None for neither
        self.counted = False   # Prevent double counting
        self.frames_since_count = 0

def process_frame(frame, model, class_list, tracker, area1, area2, going_out, going_in, counter1, counter2, detection_type='vertical'):
    global person_states
    
    frame = cv2.resize(frame, (1020, 500), interpolation=cv2.INTER_AREA)
    results = model.predict(frame, classes=[0], verbose=False)  # Only detect persons (class 0), reduce verbose output
    boxes_data = results[0].boxes.data
    px = pd.DataFrame(boxes_data).astype("float")

    detected_objects = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        if 'person' in class_list[d]:
            detected_objects.append([x1, y1, x2, y2])

    objects_bbs_ids = tracker.update(detected_objects)
    
    # Clean up states for people no longer detected
    current_ids = [bbox[4] for bbox in objects_bbs_ids]
    person_states = {k: v for k, v in person_states.items() if k in current_ids}
    
    # Process each detected person
    for bbox in objects_bbs_ids:
        x3, y3, x4, y4, obj_id = bbox
        
        # Initialize person state if new
        if obj_id not in person_states:
            person_states[obj_id] = PersonState()
        
        state = person_states[obj_id]
        state.frames_since_count += 1
        
        # Draw basic detection rectangle
        cv2.rectangle(frame, (x3, y3), (x4, y4), (128, 128, 128), 1)
        cv2.putText(frame, f'ID:{obj_id}', (x3, y3-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Use center point for detection
        center_x = (x3 + x4) // 2
        center_y = (y3 + y4) // 2
        detection_point = (center_x, center_y)
        
        # Check which area the person is in
        in_area1 = cv2.pointPolygonTest(np.array(area1, np.int32), detection_point, False) >= 0
        in_area2 = cv2.pointPolygonTest(np.array(area2, np.int32), detection_point, False) >= 0
        
        # Determine current area - prioritize area1 if in both
        current_area = None
        if in_area1 and in_area2:
            # If in both areas, use the one closest to center
            dist1 = cv2.pointPolygonTest(np.array(area1, np.int32), detection_point, True)
            dist2 = cv2.pointPolygonTest(np.array(area2, np.int32), detection_point, True)
            current_area = 1 if abs(dist1) < abs(dist2) else 2
            cv2.circle(frame, detection_point, 8, (255, 255, 0), -1)  # Yellow for overlapping
        elif in_area1:
            current_area = 1
            cv2.circle(frame, detection_point, 6, (0, 255, 255), -1)  # Yellow dot
        elif in_area2:
            current_area = 2
            cv2.circle(frame, detection_point, 6, (255, 0, 255), -1)  # Magenta dot
        
        # Detection logic based on area transitions
        if detection_type == 'horizontal':
            # Area1 is upper line, Area2 is lower line
            
            if state.last_area is not None and current_area is not None and state.last_area != current_area:
                # Only count if enough frames have passed since last count
                if state.frames_since_count > 5:
                    
                    if state.last_area == 1 and current_area == 2:  # area1 → area2 = IN
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)  # Green rectangle
                        cvzone.putTextRect(frame, f'IN-{obj_id}', (x3, y3-20), 1, 1)
                        if obj_id not in counter1:
                            counter1.append(obj_id)
                            state.counted = True
                            state.frames_since_count = 0
                            print(f"✅ Person {obj_id} counted as IN (area1→area2)")
                            
                    elif state.last_area == 2 and current_area == 1:  # area2 → area1 = OUT
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)  # Red rectangle
                        cvzone.putTextRect(frame, f'OUT-{obj_id}', (x3, y3-20), 1, 1)
                        if obj_id not in counter2:
                            counter2.append(obj_id)
                            state.counted = True
                            state.frames_since_count = 0
                            print(f"✅ Person {obj_id} counted as OUT (area2→area1)")
                            
        else:
            # Vertical detection logic
            if state.last_area is not None and current_area is not None and state.last_area != current_area:
                if state.frames_since_count > 5:
                    
                    if state.last_area == 2 and current_area == 1:  # area2 → area1 = IN
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 3)
                        cvzone.putTextRect(frame, f'IN-{obj_id}', (x3, y3-20), 1, 1)
                        if obj_id not in counter1:
                            counter1.append(obj_id)
                            state.counted = True
                            state.frames_since_count = 0
                            print(f"✅ Person {obj_id} counted as IN")

                    elif state.last_area == 1 and current_area == 2:  # area1 → area2 = OUT
                        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)
                        cvzone.putTextRect(frame, f'OUT-{obj_id}', (x3, y3-20), 1, 1)
                        if obj_id not in counter2:
                            counter2.append(obj_id)
                            state.counted = True
                            state.frames_since_count = 0
                            print(f"✅ Person {obj_id} counted as OUT")
        
        # Update last area only when clearly in an area
        if current_area is not None:
            state.last_area = current_area
            
        # Show comprehensive debug info
        debug_text = f'A:{current_area} L:{state.last_area} F:{state.frames_since_count}'
        cv2.putText(frame, debug_text, (x3, y4+10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        # Show position coordinates
        pos_text = f'({center_x},{center_y})'
        cv2.putText(frame, pos_text, (x3, y4+25), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        # Print detailed info for people not in areas (to help reposition areas)
        if current_area is None:
            print(f"🔍 ID:{obj_id} at ({center_x},{center_y}) - NOT in any area (Area1: Y=235-265, Area2: Y=315-345)")
        else:
            print(f"✅ ID:{obj_id} at ({center_x},{center_y}) - IN Area{current_area}")

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