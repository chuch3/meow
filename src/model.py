import typing

from tqdm import tqdm

import pandas as pd
import cv2 as cv
from deepface import DeepFace
from sahi.predict import (AutoDetectionModel, get_prediction,
                          get_sliced_prediction)
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import utils.constant as const

from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import pil_to_tensor
from torch.utils.data import DataLoader


def train_rcnn():
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Following the pretrained resnet size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=const.NORM_IMAGENET_MEAN,
            std=const.NORM_IMAGENET_STD
        )
    ])

    train_data = datasets.ImageFolder(
        root=f"{const.DATA_CLASSIFY_PATH}/train",
        transform=transform,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=const.BATCH_SIZE,
        shuffle=True,
    )

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights
    ).to(device)

    # Overwriting previous final layer's input to our desired IGA class
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features, len(train_data.classes)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    epochs = 300

    # Training loop
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        current_loss = 0

        for i, data in enumerate(train_loader):
            images, labels = data

            # Reset the gradient parameters
            optimizer.zero_grad()

            # Foward + Backward + Optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()

            running_loss += loss.item()

            if i % const.BATCH_SIZE == const.BATCH_SIZE - 1:
                current_loss = loss.item()
                print(
                    f"epoch : {epoch+1} / {epochs} "
                    "---- "
                    f"batch : {i+1} "
                    "---- "
                    f"current loss : {current_loss:.4f}"
                    "---- "
                    f"running loss : {running_loss:.4f}"
                )
                current_loss = 0

        torch.save(model.state_dict(), const.SAVE_RESNET_NAME)


def inference_rcnn(file_path: str):
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=const.BEST_MODEL_PATH,
        confidence_threshold=0.2,
        device=const.DEVICE
    )

    classification = models.efficientnet_b0().to(device)
    classification.classifier[-1] = nn.Linear(
        classification.classifier[-1].in_features, 4
    )

    saved_dict = torch.load(const.SAVE_RESNET_NAME, weights_only=True)

    classification.load_state_dict(saved_dict)
    classification.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Following the pretrained resnet size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=const.NORM_IMAGENET_MEAN,
            std=const.NORM_IMAGENET_STD
        )
    ])

    severity_map = ("light", "mild", "moderate", "severe")

    slice_height = 250
    slice_width = 250
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2

    image = Image.open(file_path)
    annotator = Annotator(image)

    results = get_sliced_prediction(
        file_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio
    )

    pred_list = results.object_prediction_list

    for idx, _ in enumerate(results.object_prediction_list):
        confidence = pred_list[idx].score.value

        box = pred_list[idx].bbox.to_xyxy()

        cropped_image = image.crop(box)
        input = transform(cropped_image).unsqueeze(0)

        # Context manager that disables gradient calculation during inference
        with torch.no_grad():
            output = classification(input)
            predicted_class = torch.argmax(output, dim=1).item()

        annotator.box_label(
            box=box,
            label=f'{severity_map[predicted_class]} : {confidence:.2f}',
            color=const.BOX_COLOR
        )

    return annotator.result()

# ------------------ Training & Evaluation ------------------ #


def train_tuned():
    model = YOLO(const.MODEL_PATH)

    # Hyperparameter tuning with subset for faster results
    tuned_model = model.tune(
        data=const.DATA_CONFIG_PATH,
        epochs=10,
        iterations=10,
    )

    """
    # Adjusted Data Augmentation
    final_model = tuned_model.train(
        data=const.DATA_CONFIG_PATH,
        device=const.DEVICE,
        rect=True,
        imgsz=640,  # Increasing input for better smaller object detection
        epochs=300,
        batch=-1,  # Dynamic batch size
        amp=True,  # Mixed precision to speed up and memory during training
        auto_augment=None,
    )

    final_model.save(const.SAVE_MODEL_PATH)
    """


def train_base_epoch():
    model = YOLO(const.RESUME_MODEL_PATH)
    resume_model = model.train(
        data=const.DATA_CONFIG_PATH,
        device=const.DEVICE,
        rect=True,
        imgsz=640,  # Increasing input for better smaller object detection
        epochs=2e0,
        batch=-1,  # Dynamic batch size
        amp=True,  # Mixed precision to speed up and memory during training
        resume=True,
        warmup_epochs=0,
    )
    resume_model.save(const.SAVE_MODEL_PATH)


def model_eval():
    model = YOLO(const.RESUME_MODEL_PATH)
    results = model.val(data=const.DATA_CONFIG_PATH, plots=True, save_txt=True)

    confidences = []

    print(f"MAP at IoU=0.50: {results.box.map50:.4f}")
    print(f"Mean AP at IoU=0.75: {results.box.map75:.4f}")
    print(f"Mean precision: {results.box.mp:.4f}")
    print("-------------------------------------------")
    print(f"Mean recall: {results.box.mr:.4f}")


# ------------------ Models  ------------------ #


def realtime_acne_yolov11():
    model = YOLO(const.BEST_MODEL_PATH)
    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, const.FRAME_RES[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, const.FRAME_RES[1])

    while True:
        ret, frame = cap.read()

        results = model.predict(frame, conf=0.3)

        if not ret:
            print("failed to grab frame")
            break

        for r in results:
            annotator = Annotator(frame)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                annotator.box_label(
                    b,
                    f'{model.names[0]} {box.conf.item():.2f}',
                    color=const.BOX_COLOR
                )

        img = annotator.result()
        cv.imshow('YOLOv11 Acne Detection', img)

        if cv.waitKey(1) & const.BIT_MASK == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def realtime_severity():

    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
    else:
        device = "cpu"

    detection = YOLO(const.BEST_MODEL_PATH)

    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, const.FRAME_RES[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, const.FRAME_RES[1])

    classification = models.efficientnet_b0().to(device)
    classification.classifier[-1] = nn.Linear(
        classification.classifier[-1].in_features, 4
    )

    saved_dict = torch.load(const.SAVE_RESNET_NAME, weights_only=True)

    classification.load_state_dict(saved_dict)
    classification.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Following the pretrained resnet size
        transforms.ToTensor(),
        transforms.Normalize(
            mean=const.NORM_IMAGENET_MEAN,
            std=const.NORM_IMAGENET_STD
        )
    ])

    severity_map = ("light", "mild", "moderate", "severe")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("failed to grab frame")
            break

        results = detection.predict(frame, conf=0.3)

        for r in results:
            annotator = Annotator(frame)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]

                image = Image.fromarray(frame)
                cropped_image = image.crop(tuple(map(int, b)))
                input = transform(cropped_image).unsqueeze(0)

                # Context manager that disables gradient calculation inference
                with torch.no_grad():
                    output = classification(input)
                    predicted_class = torch.argmax(output, dim=1).item()

                annotator.box_label(
                    box=b,
                    label=f'{severity_map[predicted_class]} '
                          f': {box.conf.item():.2f} ',
                    color=const.BOX_COLOR
                )

        img = annotator.result()

        cv.imshow('Cascade CNNs Acne Severity', img)

        if cv.waitKey(1) & const.BIT_MASK == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def realtime_sahi_acne_yolov11():
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=const.BEST_MODEL_PATH,
        confidence_threshold=0.3,
        device=const.DEVICE
    )

    cap = cv.VideoCapture(0)

    cap.set(cv.CAP_PROP_FRAME_WIDTH, const.FRAME_RES[0])
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, const.FRAME_RES[1])

    slice_height = 600
    slice_width = 600
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2

    while True:
        ret, frame = cap.read()

        if not ret:
            print("failed to grab frame")
            break

        annotator = Annotator(frame)

        results = get_sliced_prediction(
            frame,
            detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio
        )
        pred_list = results.object_prediction_list

        for idx, _ in enumerate(results.object_prediction_list):
            confidence = pred_list[idx].score.value
            name = pred_list[idx].category.name

            bbox = pred_list[idx].bbox
            b = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
            annotator.box_label(
                box=b,
                label=f'{name} : {confidence:.2f}',
                color=const.BOX_COLOR
            )

        img = annotator.result()

        cv.imshow('Yolov11+SAHI Acne Detection', img)

        if cv.waitKey(1) & const.BIT_MASK == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def sahi_yolov11(file_path: str):
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=const.BEST_MODEL_PATH,
        confidence_threshold=0.3,
        device=const.DEVICE
    )

    slice_height = 200
    slice_width = 200
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2

    image = Image.open(file_path)
    annotator = Annotator(image)

    results = get_sliced_prediction(
        file_path,
        detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio
    )

    pred_list = results.object_prediction_list

    for idx, _ in enumerate(results.object_prediction_list):
        confidence = pred_list[idx].score.value
        name = pred_list[idx].category.name

        bbox = pred_list[idx].bbox
        b = int(bbox.minx), int(bbox.miny), int(bbox.maxx), int(bbox.maxy)
        annotator.box_label(
            box=b,
            label=f'{name} : {confidence:.2f}',
            color=const.BOX_COLOR
        )

    info = (
        slice_height, slice_width, overlap_height_ratio, overlap_width_ratio
    )

    return annotator.result(), info


#------------------------------------------------------------------------#


def deep_face_video():
    DeepFace.stream(db_path=const.STREAM_PATH)


def deep_face_analyze(file_path: str) -> list[dict[str, typing.Any]]:
    return DeepFace.analyze(
        file_path,
        actions=['age', 'gender', 'race', 'emotion'],
        enforce_detection=False
    )

#------------------------------------------------------------------------#


