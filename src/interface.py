import os
import tkinter as tk
import typing
import cv2 as cv
import numpy as np

import utils.constant as const
import model

from tkinter import filedialog
from PIL import Image, ImageTk


def display_result(
        file_path: str,
        image_label: tk.Label,
        result_label: tk.Label,
        type_model: str,
):
    if type_model == "deepface":
        panel = Image.open(file_path)
        image = panel.resize(const.OUTPUT_RES)
        photo = ImageTk.PhotoImage(image)

        image_label.config(image=photo)
        image_label.photo = photo

        result = model.deep_face_analyze(file_path)[0]
        table = {k: float(v) for k, v in result['gender'].items()}
        result_label.config(
            text=f"Age: {result['age']}\n"
                 f"Gender: {table}\n"
                 f"Race: {result['dominant_race']}\n"
                 f"Emotion: {result['dominant_emotion']}"
        )
    elif type_model == "yolo":
        result, info = model.sahi_yolov11(file_path)

        image = Image.fromarray(np.uint8(result)).convert('RGB')
        image = image.resize(const.OUTPUT_RES)

        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.photo = photo

        result_label.config(
            text=f"Slice Height : {info[0]}\n"
                 f"Slice Width : {info[1]}\n"
                 f"Overlap Height : {info[2]}\n"
                 f"Overlap Width : {info[3]}\n"
        )
    elif type_model == "roi":
        result = model.inference_rcnn(file_path)

        image = Image.fromarray(np.uint8(result)).convert('RGB')
        image = image.resize(const.OUTPUT_RES)

        photo = ImageTk.PhotoImage(image)
        image_label.config(image=photo)
        image_label.photo = photo



def screenshot_video(
        image_label: tk.Label,
        result_label: tk.Label
) -> None:
    try:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Video capture is not initialized!")

        cap.set(cv.CAP_PROP_FRAME_WIDTH, const.FRAME_RES[0])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, const.FRAME_RES[1])

        while True:
            ret, frame = cap.read()

            if not ret:
                raise ValueError("Unable to capture frame!")
                break

            cv.imshow("Recording", frame)

            key_input = cv.waitKey(1) & const.BIT_MASK

            if key_input == ord('s'):
                cv.imwrite(filename=const.IMG_PATH, img=frame)
                print(
                    f"\n\n -------- "
                    f"\x1b[1mImage saved at {const.IMG_PATH}.\x1b[0m"
                    f"--------\n"
                )
                cv.destroyAllWindows()
                display_result(
                    const.IMG_PATH, image_label, result_label, "yolo"
                )
                break
            elif key_input == ord('q'):
                break

        cv.destroyAllWindows()
        cap.release()

    except ValueError as e:
        print(f"error! {e}!")


def choose_file(image_label: tk.Label, result_label: tk.Label, model: str):
    chosen_file_path = filedialog.askopenfilename()
    if chosen_file_path:
        display_result(chosen_file_path, image_label, result_label, model)


def display_window() -> None:
    """ ------------------ Initializing window ------------------ """

    root = tk.Tk()
    root.title("Acne Detection with YOLOv11 and DeepFace")
    root.geometry("x".join(str(val) for val in const.ROOT_RES))
    root.configure(bg=const.BG_COLOR)

    canvas = tk.Canvas(root, bg=const.BG_COLOR, highlightthickness=0)
    canvas.pack(side="left", fill="both", expand=True)

    scroll_bar = tk.Scrollbar(root, command=canvas.yview)
    scroll_bar.pack(side="left", fill='y')
    canvas.configure(yscrollcommand=scroll_bar.set)

    canvas.bind(
        '<Configure>',
        lambda x: canvas.configure(scrollregion=canvas.bbox('all'))
    )

    x_center = const.ROOT_RES[0] / 2
    y_center = const.ROOT_RES[1] / 2

    frame = tk.Frame(canvas, bg=const.BG_COLOR)

    canvas.create_window((x_center, y_center), window=frame, anchor="center")

    """ ------------------ Initializing Labels ------------------ """

    title_label = tk.Label(
        frame,
        text="MEOW \n"
             "(Multi-Staged Epidermis Observational Windowed Interface) \n",
        fg=const.FG_COLOR, bg=const.BG_COLOR,
        font=("Arial", 20)
    )
    title_label.pack(pady=10)

    icon = Image.open(const.ICON_PATH)
    icon = icon.resize(const.ICON_RES)
    load_icon = ImageTk.PhotoImage(icon)
    icon_label = tk.Label(frame, image=load_icon)
    icon_label.pack(pady=25, side="top")

    root.iconphoto(False, load_icon)

    image_label = tk.Label(frame, bg=const.BG_COLOR)
    image_label.pack()

    result_label = tk.Label(
        frame,
        fg=const.FG_COLOR,
        bg=const.BG_COLOR,
        font=("Arial", 14)
    )
    result_label.pack(pady=25)

    """ ------------------ Initializing Buttons ------------------ """

    button = tk.Button(
        frame, text="Deepface Inference",
        command=lambda: choose_file(image_label, result_label, "deepface"),
        bg="#4B4E5C", fg=const.FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="Yolov11+SAHI Inference",
        command=lambda: choose_file(image_label, result_label, "yolo"),
        bg="#4B4E5C", fg=const.FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="R-CNN-like Acne Severity Inference",
        command=lambda: choose_file(image_label, result_label, "roi"),
        bg="#4B4E5C", fg=const.FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="Acne Detection Real-Time Screenshot",
        command=lambda: screenshot_video(image_label, result_label),

        bg="#4B4E5C", fg=const.FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="Real-Time DeepFace Facial Detection",
        command=lambda: model.deep_face_video(),
        bg="#4B4E5C", fg=const.FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="Real-Time Yolov11 Acne Detection",
        command=lambda: model.realtime_acne_yolov11(),
        bg="#4B4E5C", fg=const.FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    button = tk.Button(
        frame, text="Real-Time R-CNN-Like Acne Severity",
        command=lambda: model.realtime_severity(),
        bg="#4B4E5C", fg=const.FG_COLOR, font=("Arial", 14)
    )
    button.pack(pady=20)

    """ ------------------ Run window ------------------ """

    root.mainloop()
