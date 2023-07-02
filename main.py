from ultralytics import YOLO
import cv2
import cvzone
from sort import Sort
import numpy as np

model = YOLO('E:\Object Detection\Chapter 5 - Running Yolo\yolov8n.pt')

cap = cv2.VideoCapture("../Videos/people.mp4")

mask = cv2.imread('E:\Object Detection\python\mask.png')

tracker = Sort(22)

lineUp = (105, 161), (296, 161)
lineDown = (527, 489), (735, 489)

up = []
down = []

while 2023:
    done, frm = cap.read()
    # frm = cv2.flip(frm, 1)
    roi = cv2.bitwise_and(cv2.resize(mask, (frm.shape[1], frm.shape[0])), frm)
    res = model(roi, stream=True)
    res = next(iter(res))
    names = res[0].names
    detects = np.empty((0, 5))
    cv2.line(frm, lineUp[0], lineUp[1], [0, 0, 255], 4)
    cv2.line(frm, lineDown[0], lineDown[1], [0, 0, 255], 4)

    for box in res.boxes:
        conf, name = round(float(box.conf[0]) * 100) / 100, res[0].names[int(box.cls[0])]
        if name == 'person' and conf>0.42:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # cv2.rectangle(frm, (x1, y1), (x2, y2), [0, 255, 0], 2)
            # cvzone.putTextRect(frm, f"{name} {conf}", (max(0, x1), max(35, y1-15)), 1, 2)

            arr = np.array([x1, y1, x2, y2, conf])
            detects = np.vstack([detects, arr])
    trackResults = tracker.update(detects)
    for track_res in trackResults:
        x1, y1, x2, y2, Id = track_res
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2-x1, y2-y1
        cv2.rectangle(frm, (x1, y1), (x2, y2), [0, 0, 255], 2)
        cvzone.putTextRect(frm, f"ID {int(Id)}", (max(0, x1), max(35, y1 - 15)), 1, 2)

        cp = x1+w//2, y1+h//2
        cv2.circle(frm, cp, 6, [255, 0, 0], -1)

        if cp[0]>lineUp[0][0] and cp[0]<lineUp[1][0] and cp[1] and\
        cp[1]>lineUp[0][1]-25 and cp[1]<lineUp[0][1]+25:
            if Id not in up:
                up.append(Id)
                cv2.line(frm, lineUp[0], lineUp[1], [0, 255, 0], 4)

        if cp[0]>lineDown[0][0] and cp[0]<lineDown[1][0] and cp[1] and\
        cp[1]>lineDown[0][1]-25 and cp[1]<lineDown[0][1]+25:
            if Id not in down:
                down.append(Id)
                cv2.line(frm, lineDown[0], lineDown[1], [0, 255, 0], 4)

    cvzone.putTextRect(frm, f"{len(up)} persons up", (frm.shape[1]-420, 100), 3, 3)
    cvzone.putTextRect(frm, f"{len(down)} persons Down", (frm.shape[1]-420, 150), 3, 3)

    cv2.imshow('people', frm)
    # cv2.imshow('roi', roi)
    if cv2.waitKey(22) == ord('q'): break