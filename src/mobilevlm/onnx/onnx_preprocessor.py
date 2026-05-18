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


# -------------------------------------------------
# single image preprocessing
# -------------------------------------------------
def preprocess_image(
    image: Union[str, Image.Image]
) -> np.ndarray:
    """
    Input:
        image: PIL Image or image path
    Output:
        np.ndarray (3, 336, 336) float32
    """

    # 1. load image
    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise TypeError("Input must be PIL.Image or image path")

    # 2. PIL -> numpy (HWC)
    img = np.array(image).astype(np.float32)

    # 3. padding color (mean * 255): (122, 116, 104)
    bg_color = CLIP_MEAN * 255.0

    # 4. expand to square
    img = expand2square_np(img, bg_color)

    # 5. resize (CLIP uses bicubic)
    pil_img = Image.fromarray(img.astype(np.uint8))
    pil_img = pil_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
    img = np.array(pil_img).astype(np.float32)

    # 6. rescale
    img = img / 255.0

    # 7. normalize
    img = (img - CLIP_MEAN) / CLIP_STD

    # 8. HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    return img.astype(np.float32)


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
