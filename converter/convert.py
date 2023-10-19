import os
from math import sqrt
import imghdr

import numpy as np
import cv2

FROM_DIR_NAME = "from"
TO_DIR_NAME = "to"

IMG_MAX_SIZE = 8e4


def adjust_size(img: cv2.Mat) -> cv2.Mat:
    width = img.shape[1]
    height = img.shape[0]
    size = width * height

    if size < IMG_MAX_SIZE:
        return img

    scale = sqrt(IMG_MAX_SIZE / size)
    width = int(width * scale)
    height = int(height * scale)

    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def convert(img: cv2.Mat) -> cv2.Mat:
    img = adjust_size(img)
    (_, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if np.sum(img == 0) < np.sum(img == 255):
        img = ~img

    return img


def convert_by_path(in_img_path: str, out_img_path: str):
    img = cv2.imread(in_img_path, cv2.IMREAD_GRAYSCALE)
    img = convert(img)
    cv2.imwrite(out_img_path, img)


if __name__ == "__main__":
    curr_dir_path = os.path.dirname(os.path.realpath(__file__))
    from_dir_path = f"{curr_dir_path}/{FROM_DIR_NAME}"
    to_dir_path = f"{curr_dir_path}/{TO_DIR_NAME}"

    for from_img_name in os.listdir(from_dir_path):
        from_img_path = f"{from_dir_path}/{from_img_name}"

        if imghdr.what(from_img_path) == None:
            continue

        to_img_name = from_img_name.replace(" ", "")
        to_img_name = os.path.splitext(to_img_name)[0] + '.png'

        to_img_path = f"{to_dir_path}/{to_img_name}"

        img = cv2.imread(from_img_path, cv2.IMREAD_GRAYSCALE)
        img = convert(img)
        cv2.imwrite(to_img_path, img)
