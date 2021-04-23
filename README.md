# License-Plate-Detector
Automated Number Plate Detection is a very important problem in our day to day world, from tolls to security cameras these have enumerous applications.
We present you the solution of Mosaic'21 Problem Statement 2, which asks us to make a algorithm which detections the Vehicle Number from the image of its number plate.
like so.

<img src="sample_plates/plate_1.jpg?raw=true" align="center"/>
<p align="center">
image
<h2 align="center">HR26DK6475</h2>
prediction
</p>
## Our Approach


## Training Metrics
|Model Name| F1 Score| Giou loss | Validation |
|--|--|--|--|
|License Plate Detector|![F1_score](curves/license_train_f1.png?raw=true)|![Giou_loss](curves/license_train_loss.png?raw=true)|![Giou_val_loss](curves/license_val_loss.png?raw=true)|
|Character Segmentation|![F1_score](curves/character_train_f1.png?raw=true)|![Giou_loss](curves/character_train_loss.png?raw=true)|![Giou_val_loss](curves/character_val_loss.png?raw=true)|

