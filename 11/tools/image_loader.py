import base64
import cv2
import numpy as np
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def to_base64(img: np.ndarray, fmt: str = ".jpg") -> str:
    """Encode BGR numpy array to base64 string for Claude vision."""
    success, buf = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("Failed to encode image")
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8")


def image_info(img: np.ndarray) -> dict:
    """Return basic image metadata."""
    h, w = img.shape[:2]
    channels = img.shape[2] if img.ndim == 3 else 1
    return {"width": w, "height": h, "channels": channels}


def preprocess(img: np.ndarray, resize_max: int = 1024) -> np.ndarray:
    """Optionally downscale very large images to speed up matching."""
    h, w = img.shape[:2]
    if max(h, w) > resize_max:
        scale = resize_max / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img
