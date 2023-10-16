import cv2
import numpy as np
import os
import argparse
import imutils
import time
import pandas as pd

data = {'Time_Step': [], 'Class': []}
df = pd.DataFrame(data)

net = cv2.dnn.readNet("/model/yolov3.weights", "/model/yolov3.cfg")
classes = []
with open("/model/coco.names", "r") as f:
    classes = f.read().splitlines()

cap = cv2.VideoCapture('/src/test.mp4')

frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
# fps = int(data.get(cv2.CAP_PROP_FPS))
# seconds = int(frames / fps)

for flag in range(60):
    _, img = cap.read()
    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416,416), (0,0,0), swapRB = True, crop = False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size = (len(boxes), 3))

    label_name = []
    
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, 10)
        cv2.putText(img, label + " " + confidence, (x,y+20), font, 5, (255,255,255), 5)
        # print(label)
        # print(cap.get(cv2.CAP_PROP_POS_MSEC))
        label_name.append(label)

    df = df.append({'Time_Step': cap.get(cv2.CAP_PROP_POS_MSEC), 'Class': label_name}, ignore_index=True)


    img = cv2.resize(img, (800,800))
    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()

df.to_csv('output.csv', index=False)

