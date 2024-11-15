import cv2
import numpy as np
import zipfile
import os

weights_path = r""
cfg_path = r""
image_path = r""
coco_names_path = r""

if weights_path.endswith('.zip') and not os.path.exists(weights_path.replace('.zip', '')):
    with zipfile.ZipFile(weights_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(weights_path))

net = cv2.dnn.readNet(r"", cfg_path)

layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]

img = cv2.imread(image_path)
height, width, channels = img.shape

blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

with open(coco_names_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

for i in indices.flatten():
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]
    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = (0, 255, 0)

    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

screen_width = 1920
screen_height = 1080
x_offset = (screen_width - width) // 2
y_offset = (screen_height - height) // 2

cv2.imshow('Image', img)
cv2.moveWindow('Image', x_offset, y_offset)
cv2.waitKey(0)
cv2.destroyAllWindows()
