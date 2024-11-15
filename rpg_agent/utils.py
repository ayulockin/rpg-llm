import base64
import io
import os
import json
import numpy as np

import mss
from PIL import (
    Image, ImageDraw, ImageGrab
)
import cv2

import weave


def encode_image(image: Image.Image) -> str:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = base64.b64encode(byte_arr.getvalue()).decode("utf-8")
    encoded_string = f"data:image/png;base64,{encoded_string}"
    return str(encoded_string)


def draw_bbox(image, bbox_data, image_size=(1920, 1080)):
    # Convert the image from PIL to OpenCV format
    open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Draw bounding box
    element = bbox_data["element"]
    bbox = bbox_data["bbox"]
    confidence = bbox_data["confidence"]

    x_min, y_min = int(bbox[0]), int(bbox[1])
    x_max, y_max = int(bbox[2]), int(bbox[3])

    # Draw rectangle and label
    color = (0, 255, 0)  # Green color for bounding box
    cv2.rectangle(open_cv_image, (x_min, y_min), (x_max, y_max), color, 2)
    label = f"{element} ({confidence:.2f})"
    cv2.putText(open_cv_image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Convert the image back from OpenCV format (BGR) to PIL format (RGB)
    result_image = Image.fromarray(cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB))
    return result_image


def get_bbox_center(bbox):
    """
    Calculate the center of a bounding box.

    Args:
        bbox (list or tuple): A list or tuple in the format [x_min, y_min, x_max, y_max].

    Returns:
        tuple: The (x_center, y_center) coordinates of the bounding box center.
    """
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return int(x_center), int(y_center)


@weave.op()
def parse_json(self, text: str) -> dict:
    """
    Parse JSON from text that starts with ```json and ends with ```
    """
    import re
    
    # Use regex to extract JSON content between ```json and ``` markers
    pattern = r"```json(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    
    if not match:
        return json.loads(text)

    # Extract and parse the JSON content
    json_str = match.group(1).strip()
    return json.loads(json_str)


@weave.op()
def save_combined_image(before_img, after_img, x, y, count, save_dir):
    """Save concatenated before and after images with coordinates annotation."""
    # Convert OpenCV images to PIL format for combining
    before_pil = Image.fromarray(cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB))
    after_pil = Image.fromarray(cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB))

    # Concatenate images side by side
    combined_image = Image.new('RGB', (before_pil.width + after_pil.width, before_pil.height))
    combined_image.paste(before_pil, (0, 0))
    combined_image.paste(after_pil, (before_pil.width, 0))

    # Annotate the image with coordinates
    draw = ImageDraw.Draw(combined_image)
    annotation_text = f"Change detected at ({x}, {y})"
    draw.text((10, 10), annotation_text, fill="red")

    # Save the combined image
    filename = f"{save_dir}/detection_{count}_{x}_{y}.png"
    os.makedirs(save_dir, exist_ok=True)
    combined_image.save(filename)
    print(f"Saved detection image: {filename}")

    return combined_image


@weave.op()
def get_game_window(use_image_grab: bool = True, monitor_index: int = 2):
    """Get the current game window.
    """
    if use_image_grab:
        return ImageGrab.grab() # defaults to whole window capture
    with mss.mss() as sct:
        monitors = sct.monitors
        extended_display = monitors[monitor_index]
        screenshot = sct.grab(extended_display)
        return Image.frombytes(
            "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
        )
