import os

BIT_MASK = 0xFF
BATCH_SIZE = 64
NORM_IMAGENET_MEAN = [0.485, 0.456, 0.406]
NORM_IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE: str = "cpu"

BG_COLOR: str = "#1a1919"
FG_COLOR: str = "#FFFFFF"

BOX_COLOR:  tuple = (255, 102, 102)
ROOT_RES:   tuple = (1900, 600)
FRAME_RES:  tuple = (1000, 800)
OUTPUT_RES: tuple = (500, 400)
ICON_RES:   tuple = (206, 206)

DATA_DIR:   tuple = ("dataset", "acne04yolov11")
RESUME_MODEL_DIR: tuple = ("runs", "detect", "retrain", "weights")
MODEL_DIR:  str = "models"
IMG_DIR:    str = "faces"
ICON_DIR:   str = "icon"
STREAM_DIR: str = "video"


DATA_CONFIG_NAME:   str = "data.yaml"
DATA_CLASSIFY_NAME: str = "acne_severity"
SAVE_NAME:          str = "save"
SAVE_RESNET_NAME:   str = "severity_resnet18.pth"
MODEL_NAME:         str = "yolo11n.pt"
RESUME_MODEL_NAME:  str = "last.pt"
BEST_MODEL_NAME:    str = "best.pt"
ICON_NAME:          str = "logo.png"
IMG_NAME:           str = "saved.jpg"


SAVE_MODEL_PATH: str = os.path.realpath(
    os.path.join("..", *MODEL_DIR, SAVE_NAME)
)
DATA_CLASSIFY_PATH: str = os.path.realpath(
    os.path.join("..", DATA_DIR[0], DATA_CLASSIFY_NAME)
)
DATA_CONFIG_PATH: str = os.path.realpath(
    os.path.join("..", *DATA_DIR, DATA_CONFIG_NAME)
)
RESUME_MODEL_PATH: str = os.path.realpath(
    os.path.join(*RESUME_MODEL_DIR, RESUME_MODEL_NAME)
)
BEST_MODEL_PATH: str = os.path.realpath(
    os.path.join(*RESUME_MODEL_DIR, BEST_MODEL_NAME)
)
MODEL_PATH:       str = os.path.realpath(
    os.path.join("..", MODEL_DIR, MODEL_NAME)
)
ICON_PATH:        str = os.path.realpath(
    os.path.join("..", ICON_DIR, ICON_NAME)
)
IMG_PATH:         str = os.path.realpath(
    os.path.join("..", IMG_DIR, IMG_NAME)
)
STREAM_PATH:      str = os.path.realpath(
    os.path.join("..", STREAM_DIR)
)
