"""Text on OpenCV images, including Japanese.

cv2.putText only has ASCII Hershey fonts. For the exhibition screen
("係員を呼んでください") a CJK TrueType font is drawn with PIL when one is
installed; otherwise the English fallback text is drawn with cv2.
"""

import os

import cv2
import numpy as np

# Searched in order. Windows ships Meiryo / Yu Gothic / MS Gothic; Linux
# distributions ship Noto CJK (fonts-noto-cjk) or IPA fonts.
CJK_FONT_PATHS = [
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
    "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
]

_font_cache = {}


def find_cjk_font_path():
    for font_path in CJK_FONT_PATHS:
        if os.path.exists(font_path):
            return font_path
    return None


def _get_font(size):
    if size in _font_cache:
        return _font_cache[size]
    font = None
    font_path = find_cjk_font_path()
    if font_path is not None:
        try:
            from PIL import ImageFont

            font = ImageFont.truetype(font_path, size)
        except Exception:  # noqa: BLE001 -- fall back to cv2 text
            font = None
    _font_cache[size] = font
    return font


def draw_text(image, text, org, size=32, color=(0, 0, 0), fallback_text=None):
    """Draw `text` with its top-left corner at org on an RGB uint8 image (in
    place, also returned). size is the font height in pixels. Non-ASCII text
    needs a CJK font; without one fallback_text (ASCII) is drawn instead."""
    font = _get_font(size) if not text.isascii() else None
    if font is not None:
        from PIL import Image, ImageDraw

        pil_image = Image.fromarray(image)
        ImageDraw.Draw(pil_image).text(
            org, text, font=font, fill=tuple(int(c) for c in color)
        )
        image[...] = np.asarray(pil_image)
        return image
    if not text.isascii():
        text = (
            fallback_text
            if fallback_text is not None
            else text.encode("ascii", "replace").decode()
        )
    scale = size / 30.0
    thickness = max(1, int(round(size / 16)))
    cv2.putText(
        image,
        text,
        (int(org[0]), int(org[1] + size * 0.8)),
        cv2.FONT_HERSHEY_DUPLEX,
        scale,
        tuple(int(c) for c in color),
        thickness,
    )
    return image
