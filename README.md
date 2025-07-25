# meow

(Multi-Staged Epidermis Observational Windowed Interface)
R-CNN Inspired Ance Detection and Severity Model using PyTorch and OpenCV.

Ideas :

#### Two-staged Acne Detectionwith ROIs

* Train acne severity dataset on R-CNN classifier (Resnet-18)
* Use the yolo detector as ROIs first stage 
* Second stage R-CNN to detect the acne

>> Becomes an R-CNN acne severity with SAHI inferential detection for screenshot
(To make it faster requires more adjustments so focus on this first)

> Resnet-18 is faster and efficient as classifier compared to Yolo
> It's not the most efficient way unless you use YOLO in classification mode

Todo:
- [x] Train 300 epoch 
- [x] SAHI
- [x] Hyperparameter tuning using Ultralytics `Tuner`
- [x] Data augmentation for better blurring and generalization
- [x] Make metrics report 
    - [x] Understand the metrics
- [x] Train Efficientnet-B0 with Adam Optimizer 
- [x] Crop ROIs and inference the classification (one stage detection, second stage classifcation)


