# -*- coding: utf-8 -*-
"""
Utility functions for MangaTranslate: Unicode image IO, geometry, and reading order.
"""
from pathlib import Path
from typing import List, Tuple, Dict, Any
import copy
import cv2
import numpy as np


def imread_unicode(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from disk supporting Windows Unicode and multi-byte paths."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Empty image file: {p}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"Failed to decode image: {p}")
    return img


def imwrite_unicode(path: Path | str, img: np.ndarray, params: list | None = None) -> None:
    """Write an image to disk supporting Windows Unicode paths."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower() or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise OSError(f"Failed to encode image to {ext}: {p}")
    buf.tofile(str(p))


def sort_reading_order(
    boxes: List[Tuple[int, int, int, int]], 
    row_height: int = 300
) -> List[int]:
    """
    Sort bounding boxes in standard Japanese manga reading order:
    Top panels to bottom panels, and right to left within each panel/row band.
    
    Returns:
        List of original indices in sorted reading order.
    """
    if not boxes:
        return []
    
    indexed_boxes = list(enumerate(boxes))
    # Heuristic: group into vertical bands, then sort primarily right-to-left (-center_x)
    def sort_key(item):
        idx, (x1, y1, x2, y2) = item
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        band = int(center_y // row_height)
        return (band, -center_x, center_y)

    indexed_boxes.sort(key=sort_key)
    return [idx for idx, box in indexed_boxes]


def pad_box(
    box: Tuple[int, int, int, int], 
    pad: int, 
    max_w: int, 
    max_h: int
) -> Tuple[int, int, int, int]:
    """Expand bounding box by pad pixels, clamped to image dimensions."""
    x1, y1, x2, y2 = box
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(max_w, x2 + pad),
        min(max_h, y2 + pad)
    )


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
