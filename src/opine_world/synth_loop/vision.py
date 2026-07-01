"""Frame rendering and ASCII representation for the exploration LLM.

Converts ArcEngineEnv.get_frame() color-index arrays to base64 PNG and compact ASCII grids.
"""
from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLOR_PALETTE = {
    0:  (0xFF, 0xFF, 0xFF),
    1:  (0xCC, 0xCC, 0xCC),
    2:  (0x99, 0x99, 0x99),
    3:  (0x66, 0x66, 0x66),
    4:  (0x33, 0x33, 0x33),
    5:  (0x00, 0x00, 0x00),
    6:  (0xE5, 0x3A, 0xA3),
    7:  (0xFF, 0x7B, 0xCC),
    8:  (0xF9, 0x3C, 0x31),
    9:  (0x1E, 0x93, 0xFF),
    10: (0x88, 0xD8, 0xF1),
    11: (0xFF, 0xDC, 0x00),
    12: (0xFF, 0x85, 0x1B),
    13: (0x92, 0x12, 0x31),
    14: (0x4F, 0xCC, 0x30),
    15: (0xA3, 0x56, 0xD6),
}
DEFAULT_COLOR = (0x88, 0x88, 0x88)

SCALE_FACTOR = 15


def _val_to_char(v: int) -> str:
    if v < 10:
        return str(v)
    if v < 16:
        return chr(ord("A") + v - 10)
    return "?"


def frame_to_ascii(frame: np.ndarray, separator: str = "") -> str:
    """One character per cell, rows separated by newlines."""
    arr = np.asarray(frame)
    lines = []
    for row in arr:
        cells = [_val_to_char(int(v)) for v in row]
        lines.append(separator.join(cells))
    return "\n".join(lines)


def diff_to_ascii(
    before: np.ndarray, after: np.ndarray, separator: str = "",
) -> str:
    """ASCII diff: '.' for unchanged cells, new char for changed cells. Shapes clipped to intersection."""
    b = np.asarray(before)
    a = np.asarray(after)
    min_rows = min(b.shape[0], a.shape[0])
    min_cols = min(b.shape[1], a.shape[1])

    lines = []
    for r in range(min_rows):
        cells = []
        for c in range(min_cols):
            bv = int(b[r, c])
            av = int(a[r, c])
            cells.append("." if bv == av else _val_to_char(av))
        lines.append(separator.join(cells))
    return "\n".join(lines)


def render_frame_png_b64(
    frame: np.ndarray,
    description: str = "",
    with_grid: bool = True,
    with_coords: bool = True,
) -> str:
    """Render a 2D color-index frame to a base64-encoded PNG with optional gridlines and coordinate labels."""
    arr = np.asarray(frame, dtype=np.uint8)
    rows, cols = arr.shape

    border = SCALE_FACTOR if with_coords else 0
    scaled_width = cols * SCALE_FACTOR + border
    scaled_height = rows * SCALE_FACTOR + border

    description_height = 40 if description else 0
    img = Image.new(
        "RGB",
        (scaled_width, scaled_height + description_height),
        color=(0, 0, 0),
    )
    pixels = img.load()

    for y in range(rows):
        for x in range(cols):
            color = COLOR_PALETTE.get(int(arr[y, x]), DEFAULT_COLOR)
            for i in range(SCALE_FACTOR):
                for j in range(SCALE_FACTOR):
                    pixels[x * SCALE_FACTOR + j + border,
                           y * SCALE_FACTOR + i + border] = color

    draw = ImageDraw.Draw(img)

    if with_grid:
        line_color = (96, 96, 96)
        for k_h in range(rows + 1):
            y = k_h * SCALE_FACTOR + border
            draw.line([(border, y), (scaled_width - 1, y)],
                      fill=line_color, width=1)
        for k_v in range(cols + 1):
            x = k_v * SCALE_FACTOR + border
            draw.line([(x, border), (x, scaled_height - 1)],
                      fill=line_color, width=1)

    if with_coords:
        font = ImageFont.load_default()
        text_color = (255, 255, 255)
        y_col = SCALE_FACTOR / 2.0
        for col_idx in range(cols):
            if col_idx % 5 != 0 and col_idx != cols - 1:
                continue
            x_col = col_idx * SCALE_FACTOR + SCALE_FACTOR / 2.0 + border
            draw.text((x_col, y_col), str(col_idx), font=font,
                      fill=text_color, anchor="mm")
        x_row = SCALE_FACTOR / 2.0
        for row_idx in range(rows):
            if row_idx % 5 != 0 and row_idx != rows - 1:
                continue
            y_row = row_idx * SCALE_FACTOR + SCALE_FACTOR / 2.0 + border
            draw.text((x_row, y_row), str(row_idx), font=font,
                      fill=text_color, anchor="mm")

    if description:
        font = ImageFont.load_default()
        draw.text(
            (scaled_width / 2, scaled_height + description_height / 2),
            description, font=font, fill=(255, 255, 255), anchor="mm",
        )

    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def render_frame_to_file(
    frame: np.ndarray, path: str, description: str = "",
    with_grid: bool = True, with_coords: bool = True,
) -> str:
    b64 = render_frame_png_b64(frame, description, with_grid, with_coords)
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return b64
