# YOLO Object Detection with OpenCV

This project demonstrates how to use YOLO (You Only Look Once) for object detection on images using OpenCV. The script processes an input image and identifies objects by bounding boxes and labels.

---

## Features
- Automatically extracts weights from a ZIP file if necessary.
- Uses the YOLO algorithm for object detection.
- Draws bounding boxes around detected objects and labels them with class names and confidence scores.
- Supports non-maximum suppression (NMS) to eliminate overlapping boxes.

---

## Prerequisites

- Python 3.x
- OpenCV
- Numpy

### Files Required:
1. **YOLO Weights File (`.weights`)** - Pre-trained weights for YOLO.
2. **YOLO Configuration File (`.cfg`)** - YOLO model configuration.
3. **COCO Class Names File (`.names`)** - File containing class names.
4. **Input Image (`.jpg` or `.png`)** - Image on which object detection will be performed.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python numpy
   ```

3. **Prepare your files:**
   - Place the weights, config, and names files in the project folder.
   - Ensure your input image is available at the specified path.

---

## Usage

1. **Update the paths:**
   Open the script and update the paths for the following variables:
   ```python
   weights_path = r"<path-to-yolo-weights-file>"
   cfg_path = r"<path-to-yolo-config-file>"
   image_path = r"<path-to-image-file>"
   coco_names_path = r"<path-to-coco-names-file>"
   ```

2. **Run the script:**
   ```bash
   python yolo_object_detection.py
   ```

3. **View the results:**
   The script will display the image with bounding boxes and labels.

---

## Code Walkthrough

### Weight Extraction
Automatically extracts weights if provided in a ZIP file:
```python
if weights_path.endswith('.zip') and not os.path.exists(weights_path.replace('.zip', '')):
    with zipfile.ZipFile(weights_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(weights_path))
```

### YOLO Model Setup
Load the YOLO model and prepare layers for output:
```python
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
unconnected_out_layers = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in unconnected_out_layers.flatten()]
```

### Object Detection
Process the image and perform object detection:
```python
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
```

### Post-Processing
Use non-maximum suppression (NMS) to refine detections:
```python
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```

### Visualization
Display bounding boxes and labels on the image:
```python
cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
```

---

## Troubleshooting

1. **Invalid file paths:**
   - Ensure all paths (weights, config, names, and image) are correct and accessible.

2. **Missing dependencies:**
   - Install missing packages using `pip install <package-name>`.

3. **Unrecognized objects:**
   - Verify that your COCO class names file matches the model.

---

## Contributing

Feel free to submit issues or pull requests to enhance this project. For significant changes, please open an issue first to discuss your ideas.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments
- [YOLO](https://pjreddie.com/darknet/yolo/)
- [OpenCV Documentation](https://docs.opencv.org/)
- COCO Dataset

---

Thank you for using this project! If you encounter any issues, feel free to open an issue on GitHub.

