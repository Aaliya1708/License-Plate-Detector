# License-Plate-Detector
Automated Number Plate Detection is a very important problem in our day to day world, from tolls to security cameras these have enumerous applications.
We present you the solution of Mosaic'21 Problem Statement 2, which asks us to make a algorithm which detections the Vehicle Number from the image of its number plate.
like so.
<p align="center">
<img src="sample_plates/plate_1.jpg?raw=true" align="center"/>

<h6 align="center">image</h6>
<h2 align="center">HR26DK6475</h2>

<h2 align="center">prediction</h6>
</p>

Salient Features
- Asfd
- ASdf
- asfd

## Our Approach

We have used Yolo v5 to get bounding boxes for characters. We trained it on a custom dataset of 1000 license plates (augmented to 5000 images).
We have also used another Yolo v5 to get bounding boxes of license plates from full images as explained in the salient features. Here we used a dataset of 300 images only.
Finally for character recognition we had trained a simple CNN model with 36 output layers, trained on [CHAR-47 dataset](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/) with a bit of data augmentation.

Our architectture is a robust and the most efficient one compared to Unets or Faster RCNN because we have inculcated state-of-the-art Yolov5 models.
Below is a flow chart of our approach.

<p align="center">
<img src="curves/approach.png?raw=true" align="center"/>
</p>

## Results

## Training Metrics
|Model Name| F1 Score| Giou loss | Validation |
|--|--|--|--|
|License Plate Detector|![F1_score](curves/license_train_f1.png?raw=true)|![Giou_loss](curves/license_train_loss.png?raw=true)|![Giou_val_loss](curves/license_val_loss.png?raw=true)|
|Character Segmentation|![F1_score](curves/character_train_f1.png?raw=true)|![Giou_loss](curves/character_train_loss.png?raw=true)|![Giou_val_loss](curves/character_val_loss.png?raw=true)|

