# onnx_preprocessor.py

import numpy as np
from PIL import Image
from typing import List, Union


# -------------------------------------------------
# CLIP (MobileVLM) constants
# -------------------------------------------------
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

IMAGE_SIZE = 336


# -------------------------------------------------
# expand to square (padding)
# -------------------------------------------------
def expand2square_np(img: np.ndarray, bg_color: np.ndarray) -> np.ndarray:
    """
    img: (H, W, C), float32 or uint8
    bg_color: (3,) RGB  
    """
    h, w, c = img.shape

    if h == w:
        return img

    size = max(h, w)

    result = np.ones((size, size, c), dtype=img.dtype) * bg_color
    if w > h:
        y_offset = (w - h) // 2
        result[y_offset:y_offset + h, :, :] = img
    else:
        x_offset = (h - w) // 2
        result[:, x_offset:x_offset + w, :] = img

    return result


def resize_bilinear_np_336(img: np.ndarray) -> np.ndarray:
    """
    img: (H, W, C), uint8 or float32
    return: (336, 336, C), same dtype as input
    """

    out_h, out_w = 336, 336
    in_h, in_w, channels = img.shape

    # dtype 보존
    input_dtype = img.dtype

    # 계산은 float32로
    img = img.astype(np.float32)

    output = np.zeros((out_h, out_w, channels), dtype=np.float32)

    scale_x = in_w / out_w
    scale_y = in_h / out_h

    for y in range(out_h):  # library 사용하기
        for x in range(out_w):

            # source 좌표 (pixel center align)
            src_x = (x + 0.5) * scale_x - 0.5
            src_y = (y + 0.5) * scale_y - 0.5

            x0 = int(np.floor(src_x))
            y0 = int(np.floor(src_y))
            x1 = min(x0 + 1, in_w - 1)
            y1 = min(y0 + 1, in_h - 1)

            dx = src_x - x0
            dy = src_y - y0

            # boundary clamp (순서 유지)
            x0 = max(x0, 0)
            y0 = max(y0, 0)

            # 4 neighbor pixels
            top_left     = img[y0, x0]
            top_right    = img[y0, x1]
            bottom_left  = img[y1, x0]
            bottom_right = img[y1, x1]

            # bilinear interpolation
            top    = top_left * (1.0 - dx) + top_right * dx
            bottom = bottom_left * (1.0 - dx) + bottom_right * dx
            value  = top * (1.0 - dy) + bottom * dy

            output[y, x] = value

    # dtype 복원
    if input_dtype == np.uint8:
        output = np.clip(output, 0, 255).astype(np.uint8)

    return output


# -------------------------------------------------
# single image preprocessing
# -------------------------------------------------
def preprocess_image(
    image: Union[str, Image.Image]
) -> np.ndarray:

    # 1. load image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("Input must be PIL.Image or image path")

    img = np.array(image)
    # img.tofile("python_png.bin")
    # image.save("image.png")

    # 3. padding color (uint8)
    bg_color = (CLIP_MEAN * 255.0).astype(np.uint8)

    img = expand2square_np(img, bg_color)

    # 5. resize
    # pil_img = Image.fromarray(img) 
    # pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR) # (336, 336, 3) 
    # img = np.array(pil_img) # uint8
    img = resize_bilinear_np_336(img)

    # 6. rescale
    img = img.astype(np.float32) / 255.0

    # 7. normalize
    img = (img - CLIP_MEAN) / CLIP_STD

    # 8. HWC -> CHW
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)

    return img


# -------------------------------------------------
# batch processing
# -------------------------------------------------
def preprocess_batch(
    images: List[Union[str, Image.Image]]
) -> np.ndarray:
    """
    Input:
        list of images
    Output:
        np.ndarray (B, 3, 336, 336)
    """
    processed = [preprocess_image(img) for img in images]
    return np.stack(processed, axis=0)
