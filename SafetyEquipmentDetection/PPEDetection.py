import math

import cvzone
import cv2
from ultralytics import YOLO

model = YOLO("ppe.pt")
cap = cv2.VideoCapture(0)
classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']
while True:
    success, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width, height = x2 - x1, y2 - y1
            cls = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100  # Confidence score
            if classNames[cls] == 'Mask':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                x_center, y_center = int((x2 + x1) / 2), int((y2 + y1) / 2)
                cv2.circle(frame, (x_center, y_center), 3, (255, 0, 255), 3)
                cvzone.putTextRect(frame, f"{classNames[cls]}{conf}", (max(0, x1), max(35, y1)), scale=0.8, thickness=2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                x_center, y_center = int((x2 + x1) / 2), int((y2 + y1) / 2)
                cv2.circle(frame, (x_center, y_center), 3, (255, 0, 0), 3)
                cvzone.putTextRect(frame, f"{classNames[cls]}{conf}", (max(0, x1), max(35, y1)), scale=0.8, thickness=2)

    cv2.imshow("Image", frame)
    cv2.waitKey(1)
