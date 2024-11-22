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
    (x_min, y_min), (x_max, y_max) = bbox[0], bbox[1]
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
        image = ImageGrab.grab() # defaults to whole window capture
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        return image

    with mss.mss() as sct:
        monitors = sct.monitors
        extended_display = monitors[monitor_index]
        screenshot = sct.grab(extended_display)
        return Image.frombytes(
            "RGB", screenshot.size, screenshot.bgra, "raw", "BGRX"
        )
 

@weave.op()
def get_template_match(frame: np.ndarray, template_img_path: str, method='TM_SQDIFF_NORMED'):
    """
    Perform template matching on an input frame using the provided template image.
    
    Args:
        frame: Input PIL Image or numpy array to search in (grayscale)
        template_img_path: Path to template image file to search for
        method: Template matching method to use (default: TM_SQDIFF_NORMED)
        
    Returns:
        dict containing:
            bbox: Tuple of (top_left, bottom_right) coordinates
            match_result: The full matching result matrix
    """
    # Convert PIL Image to numpy array if needed
    if hasattr(frame, 'convert'):
        frame = np.array(frame.convert('RGB'))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if isinstance(frame, np.ndarray):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Load and convert template image
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    assert template_img is not None, "Template image could not be read"
    
    # Get template dimensions
    w, h = template_img.shape[::-1]

    # Get the matching method
    match_method = getattr(cv2, method)
    
    # Apply template matching
    res = cv2.matchTemplate(frame, template_img, match_method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # For TM_SQDIFF/TM_SQDIFF_NORMED, minimum value is best match
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    
    return {
        "bbox": (top_left, bottom_right),
        "match_result": Image.fromarray((res * 255).astype(np.uint8)),
    }
