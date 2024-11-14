import base64
import io

from PIL import Image


def encode_image(
    image: Image.Image,
    use_standard_encoding: bool = False,
    include_media_type: bool = True,
) -> str:
    byte_arr = io.BytesIO()
    image.save(byte_arr, format="PNG")
    encoded_string = (
        base64.b64encode(byte_arr.getvalue())
        if not use_standard_encoding
        else base64.standard_b64encode(byte_arr.getvalue())
    )
    encoded_string = encoded_string.decode("utf-8")
    if include_media_type:
        encoded_string = f"data:image/png;base64,{encoded_string}"
    return str(encoded_string)
