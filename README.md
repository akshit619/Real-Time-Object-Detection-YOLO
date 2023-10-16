# Real-Time-Object-Detection-and-Classification
Object detection using YOLOv3 object detector

## YOLO for Object Detection
Object detection is a computer vision task that involves both localizing one or more objects within an image and classifying each object in the image.

The approach involves a single deep convolutional neural network (originally a version of GoogLeNet, later updated and called DarkNet based on VGG) that splits the input into a grid of cells and each cell directly predicts a bounding box and object classification. The result is a large number of candidate bounding boxes that are consolidated into a final prediction by a post-processing step.


We’ll be using YOLOv3 in this project, in particular, YOLO trained on the COCO dataset.

The COCO dataset consists of 80 labels, including, but not limited to:

- People
- Bicycles
- Cars and trucks
- Airplanes
- Stop signs and fire hydrants
- Animals, including cats, dogs, birds, horses, cows, and sheep, to name a few
- Kitchen and dining objects, such as wine glasses, cups, forks, knives, spoons, etc.
…and much more!

You can find a full list of what YOLO trained on the COCO dataset can detect <a href="https://github.com/pjreddie/darknet/blob/master/data/coco.names" target="_blank"><b>using this link.</b></a>

- yolo-coco : The YOLOv3 object detector pre-trained (on the COCO dataset) model files. These were trained by the <a href="https://pjreddie.com/darknet/yolo/" target="_blank"> <b>Darknet team.</b> </a>

## Installation

- `pip install numpy`
- `pip install opencv-python`

##  YOLO object detection in video streams
Here we make frame by frame prediction on <a href="https://github.com/vivekagarwal2349/Real-Time-Object-Detection-and-Classification/blob/main/test.mp4" target="_blank"><b>test video</b></a>

<img src="https://github.com/vivekagarwal2349/Real-Time-Object-Detection-and-Classification/blob/main/test_result.gif">

Labels predicted on each frame get updated on <a href="https://github.com/vivekagarwal2349/Real-Time-Object-Detection-and-Classification/blob/main/output.csv" target="_blank"><b>CSV file</b></a> with timestep.

| Time_Step          | Class                                                       |
|--------------------|-------------------------------------------------------------|
| 0.0                | "['car', 'person', 'person', 'person', 'person', 'person']" |
| 40.0               | "['car', 'person', 'person', 'person', 'person', 'person']" |
| 80.0               | "['car', 'person', 'person', 'person', 'person', 'person']" |
| 120.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 160.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 200.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 240.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 280.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 320.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 360.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 400.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |
| 440.0              | "['person', 'car', 'person', 'person', 'person', 'person']" |

## Limitation:
### Arguably the largest limitation and drawback of the YOLO object detector is that:

- It does not always handle small objects well
- It especially does not handle objects grouped close together
- The reason for this limitation is due to the YOLO algorithm itself:

The YOLO object detector divides an input image into an SxS grid where each cell in the grid predicts only a single object.
If there exist multiple, small objects in a single cell then YOLO will be unable to detect them, ultimately leading to missed object detections.
Therefore, if you know your dataset consists of many small objects grouped close together then you should not use the YOLO object detector.

In terms of small objects, Faster R-CNN tends to work the best; however, it’s also the slowest.



