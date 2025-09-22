# meow 😺

- M : **M**ulti-Staged 
- E : **E**pidermis 
- O : **O**bservational 
- W : **W**indowed interface 

![icon](icon/logo.png) 

### Description 

Acne Detection and Severity Model using Region of Interests (ROIs) with 
Ultralytic's SAHI (Sliced Aided Hyper Inference) for Small Object Detection and tkinter GUI written in Python.

#### Two-staged Acne Detection with ROIs

* Train acne severity dataset on R-CNN classifier (Resnet-18)
* Use YOLO detector as ROIs first stage detection
* Second stage R-CNN to detect and classify acne with SAHI's small 
object detection

Todo:
- [x] Train 300 epoch and adjustments
- [x] SAHI
- [x] Hyperparameter tuning using Ultralytics `Tuner`
- [x] Data augmentation for better blurring and generalization
- [x] Metrics report 
- [x] Train Efficientnet-B0 with Adam Optimizer 
- [x] Crop ROIs and inference the classification (one stage detection, second stage classifcation)

> Resnet-18 is faster and efficient as classifier compared to YOLO

> It's not the most efficient way unless you use YOLO in classification mode
