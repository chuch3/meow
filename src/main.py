import os
import tkinter as tk
import typing
from tkinter import filedialog

import cv2 as cv
from deepface import DeepFace
from PIL import Image, ImageTk
from sahi.predict import (AutoDetectionModel, get_prediction,
                          get_sliced_prediction)
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

import model
import interface
import utils.constant as const


# ------------------ Models ------------------ #


def image_analyzer() -> None:
    try:
        interface.display_window()
    except KeyboardInterrupt as e:
        print(f"error! Video capture has exited abruptly! {e}")


def main() -> None:
    image_analyzer()


if __name__ == "__main__":
    main()
