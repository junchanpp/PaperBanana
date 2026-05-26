# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Image utility functions for processing and converting images
"""

import base64
import io
import numpy as np
from PIL import Image


def convert_png_b64_to_jpg_b64(png_b64_str: str) -> str:
    """
    Convert a PNG base64 string to a JPG base64 string.
    
    Args:
        png_b64_str: Base64 encoded PNG image string
        
    Returns:
        Base64 encoded JPG image string, or None if conversion fails
    """
    try:
        if not png_b64_str or len(png_b64_str) < 10:
            print(f"⚠️  Invalid base64 string (too short): {png_b64_str[:50] if png_b64_str else 'None'}")
            return None
            
        img = Image.open(io.BytesIO(base64.b64decode(png_b64_str))).convert("RGB")
        out_io = io.BytesIO()
        img.save(out_io, format="JPEG", quality=95)
        return base64.b64encode(out_io.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"❌ Error converting image: {e}")
        print(f"   Input preview: {png_b64_str[:100] if png_b64_str else 'None'}")
        return None


def compute_preservation_diff(original, upscaled, threshold: int = 20) -> dict:
    """Compare an upscaled image against its original to visualize what changed.

    The upscaled image is downscaled back to the original size, then compared
    pixel-by-pixel. Pixels whose max per-channel difference exceeds `threshold`
    (0-255) are considered "changed" and tinted red in the overlay.

    Args:
        original: PIL.Image of the source.
        upscaled: PIL.Image of the model output (any size).
        threshold: per-pixel max-channel diff above which a pixel counts as changed.

    Returns:
        dict with:
          mad: float, mean absolute pixel difference (0-255).
          changed_ratio: float, fraction of pixels exceeding threshold (0-1).
          overlay_png_bytes: bytes, original-size PNG with changed pixels tinted red.
    """
    orig = original.convert("RGB")
    up = upscaled.convert("RGB")
    if up.size != orig.size:
        up = up.resize(orig.size, Image.LANCZOS)

    a = np.asarray(orig, dtype=np.int16)
    b = np.asarray(up, dtype=np.int16)
    abs_diff = np.abs(a - b)
    mad = float(abs_diff.mean())
    per_pixel_max = abs_diff.max(axis=2)
    mask = per_pixel_max > threshold
    changed_ratio = float(mask.mean())

    overlay = np.asarray(orig, dtype=np.uint8).copy()
    red = np.array([255, 0, 0], dtype=np.float32)
    if mask.any():
        blended = 0.5 * overlay[mask].astype(np.float32) + 0.5 * red
        overlay[mask] = blended.astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return {
        "mad": mad,
        "changed_ratio": changed_ratio,
        "overlay_png_bytes": buf.getvalue(),
    }
