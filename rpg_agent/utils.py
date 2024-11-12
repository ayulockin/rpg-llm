import base64
import io

from PIL import Image


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
