import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import cvzone
import numpy as np

# Initialize YOLO model
model = YOLO('yolov8s.pt')

def people_counter(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

def load_class_list(file_path):
    with open(file_path, "r") as file:
        return file.read().split("\n")

def process_frame(frame, model, class_list, tracker, area1, area2, going_out, going_in, counter1, counter2):
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    boxes_data = results[0].boxes.data
    px = pd.DataFrame(boxes_data).astype("float")

    detected_objects = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        if 'person' in class_list[d]:
            detected_objects.append([x1, y1, x2, y2])

    objects_bbs_ids = tracker.update(detected_objects)
    for bbox in objects_bbs_ids:
        x3, y3, x4, y4, obj_id = bbox
        if cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
            going_out[obj_id] = (x4, y4)
        if obj_id in going_out and cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
            cv2.circle(frame, (x4, y4), 4, (0, 255, 0), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
            cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 1)
            if obj_id not in counter1:
                counter1.append(obj_id)

        if cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False) >= 0:
            going_in[obj_id] = (x4, y4)
        if obj_id in going_in and cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False) >= 0:
            cv2.circle(frame, (x4, y4), 4, (0, 255, 0), -1)
            cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f'{obj_id}', (x3, y3), 1, 1)
            if obj_id not in counter2:
                counter2.append(obj_id)

    return frame, len(counter1), len(counter2)

def main():
    cv2.namedWindow('people_counter')
    cv2.setMouseCallback('people_counter', people_counter)
    cap = cv2.VideoCapture('E:/yolov8-students-counting-lobby-main/video/p.mp4')

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    class_list = load_class_list("coco.txt")
    tracker = Tracker()

    area1 = [(494, 289), (505, 499), (578, 496), (530, 292)]
    area2 = [(548, 290), (600, 496), (637, 493), (574, 288)]
    going_out, going_in = {}, {}
    counter1, counter2 = [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached the end of the video or encountered an error.")
            break

        frame, out_count, in_count = process_frame(frame, model, class_list, tracker, area1, area2, going_out, going_in, counter1, counter2)

        cv2.putText(frame, f'In: {in_count}', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Out: {out_count}', (20, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

        cv2.imshow("people_counter", frame)
        if cv2.waitKey(delay) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()