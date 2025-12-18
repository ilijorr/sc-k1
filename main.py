# SV 73/2022
# Ilija Jordanovski
from typing import Dict, Tuple
import os
import cv2
import numpy as np


def calculate_mae(
        predictions: Dict[str, int],
        csv_path:    str
        ) -> float:
    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    errors_abs = []
    for _, row in df.iterrows():
        image_name, true_value = row['image_name'], row['coins_value']

        if image_name in predictions:
            predicted_value = predictions[image_name]
            errors_abs.append(abs(true_value - predicted_value))

    return sum(errors_abs) / len(errors_abs) if errors_abs else 0


def parse_args() -> (str, bool):
    import sys

    if len(sys.argv) < 2:
        print("Data directory path not provided")
        sys.exit(1)

    data_dir = sys.argv[1]

    if not os.path.isdir(data_dir):
        print("Provided path is not a directory")
        sys.exit(1)

    dev = False
    if len(sys.argv) > 2:
        if sys.argv[2] == '-d':
            dev = True

    return data_dir, dev


def heuristic(
        yellow_coins: int,
        red_coins:    int,
        star_coins:   int
        ) -> int:
    YELLOW_COIN_VALUE = 1
    RED_COIN_VALUE = 2
    STAR_COIN_VALUE = 5
    return yellow_coins * YELLOW_COIN_VALUE \
        + red_coins * RED_COIN_VALUE \
        + star_coins * STAR_COIN_VALUE


def create_mask(
        hsv_image:   np.ndarray,
        min:         np.ndarray,
        max:         np.ndarray,
        kernel_size: Tuple[int, int]
        ) -> np.ndarray:
    mask = cv2.inRange(hsv_image, min, max)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def is_coin(
        contour:          np.ndarray,
        size_min:         int,
        size_max:         int,
        min_aspect_ratio: float,
        smoothness:       float
        ) -> bool:
    area = cv2.contourArea(contour)
    if area < size_min or area > size_max:
        return False

    _, _, width, height = cv2.boundingRect(contour)

    contour_aspect_ratio = max(width, height) / min(width, height)
    if contour_aspect_ratio > min_aspect_ratio:
        return False

    # ------ ai-assisted code ------
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter == 0:
        return False

    if (perimeter / np.sqrt(area)) > smoothness:
        return False  # too many rough edges, probably a brick
    # ------------------------------

    return True


def is_yellow_coin(contour: np.ndarray) -> bool:
    return is_coin(
            contour, 1000, 2900, 2.2, 4.0
            )


def is_red_coin(contour: np.ndarray) -> bool:
    return is_coin(
            contour, 800, 1500, 2.2, 4.0
            )


def is_star_coin(contour: np.ndarray) -> bool:
    return is_coin(
            contour, 8000, 15000, 2.2, 4.0
            )


def detect_coins(
        image: np.ndarray,
        dev:   bool = False
        ) -> Tuple[int, int, int]:
    image = cv2.GaussianBlur(image, (3, 3), 0)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow_min = np.array([15, 100, 100])
    yellow_max = np.array([35, 255, 255])
    yellow_kernel_size = (6, 6)
    yellow_mask = create_mask(
            image, yellow_min, yellow_max, yellow_kernel_size)

    red_min = np.array([0, 120, 70])
    red_max = np.array([10, 255, 255])
    red_kernel_size = (7, 7)
    red_mask = create_mask(
            image, red_min, red_max, red_kernel_size)

    yellow_coins = 0
    red_coins = 0
    star_coins = 0

    yellow_contours, _ = cv2.findContours(yellow_mask,
                                          cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    red_contours, _ = cv2.findContours(red_mask,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

    for yellow_contour in yellow_contours:
        if is_yellow_coin(yellow_contour):
            yellow_coins += 1
        elif is_star_coin(yellow_contour):
            star_coins += 1

    for red_contour in red_contours:
        if is_red_coin(red_contour):
            red_coins += 1

    return yellow_coins, red_coins, star_coins


if __name__ == "__main__":
    data_dir, dev = parse_args()

    predictions = {}
    image_files = [f for f in os.listdir(data_dir) if f.endswith('jpg')]
    for image in image_files:
        image_data = cv2.imread(os.path.join(data_dir, image))
        yellow, red, star = detect_coins(image_data)

        predictions[image] = heuristic(yellow, red, star)

    csv_path = os.path.join(data_dir, "coin_value_count.csv")
print(calculate_mae(predictions, csv_path))
