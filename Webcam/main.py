import cvzone
from ultralytics import YOLO
import cv2
import math

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

while True:
    success, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:

        # Get the Bounding Box Information
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # box information in xyxy format
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(x1, y1, x2, y2)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            x_center, y_center = int((x1 + x2) / 2), int((y1 + y2) / 2)

            w, h = x2 - x1, y2 - y1  # calculate width and height of Bounding Box

            # Rectangle Plot using cvzone
            bbox = int(x1), int(y1), int(w), int(h)
            cvzone.cornerRect(frame, bbox=bbox, l=15)

            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence score
            print(conf)
            cvzone.putTextRect(frame, "center", (max(0, x_center+15), max(35, y_center+15 )), scale=1, thickness=1)
            #Draw circle at the center of the bounding box
            cv2.circle(frame, (x_center, y_center), 5, (0, 255, 0), 2)
            #Get the Class Information
            cls = int(box.cls[0])  # Class Label
            cvzone.putTextRect(frame, f"{cls}{conf}", (max(0, x1), max(35, y1)), scale=1, thickness=1)
    cv2.imshow("Image", frame)
    cv2.waitKey(1)
