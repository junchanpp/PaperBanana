"""Plain-assert tests for compute_preservation_diff (run with venv python; no pytest needed)."""
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from PIL import Image
from utils.image_utils import compute_preservation_diff


def _solid(color, size=(64, 48)):
    return Image.new("RGB", size, color)


def test_identical_has_no_changes():
    img = _solid((200, 180, 160))
    out = compute_preservation_diff(img, img.copy())
    assert out["mad"] == 0.0, f"expected mad 0, got {out['mad']}"
    assert out["changed_ratio"] == 0.0, f"expected 0 changed, got {out['changed_ratio']}"
    # overlay must decode as a same-size PNG
    ov = Image.open(BytesIO(out["overlay_png_bytes"]))
    assert ov.size == img.size


def test_modified_region_is_flagged():
    base = _solid((255, 255, 255))
    mod = base.copy()
    # paint a 20x20 black square -> clearly changed pixels
    for x in range(20):
        for y in range(20):
            mod.putpixel((x, y), (0, 0, 0))
    out = compute_preservation_diff(base, mod)
    assert out["changed_ratio"] > 0.0, "modified image must report changed pixels"
    # 20x20 changed out of 64x48 = 400/3072 ~= 0.13
    assert 0.10 < out["changed_ratio"] < 0.16, f"unexpected ratio {out['changed_ratio']}"
    assert out["mad"] > 0.0


def test_upscaled_is_downscaled_to_match():
    base = _solid((120, 120, 120), size=(50, 40))
    big = base.resize((200, 160), Image.LANCZOS)  # 4x, same content
    out = compute_preservation_diff(base, big)
    assert out["changed_ratio"] == 0.0, "pure resize of a solid color must show no change"


if __name__ == "__main__":
    test_identical_has_no_changes()
    test_modified_region_is_flagged()
    test_upscaled_is_downscaled_to_match()
    print("ALL PASS")
