import base64
import io
import os

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


def get_game_window():
    """Get the current game window.
    """
    return ImageGrab.grab() # defaults to whole window capture
