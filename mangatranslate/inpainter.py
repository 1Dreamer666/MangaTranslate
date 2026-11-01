# -*- coding: utf-8 -*-
"""
Inpainting module for erasing original Japanese text while strictly preserving speech bubble borders.
"""
from typing import List, Tuple
import cv2
import numpy as np


class MangaInpainter:
    """Removes text inside detected speech bubble regions preserving bubble contours."""

    def __init__(
        self,
        method: str = "telea",
        inpaint_radius: int = 5,
        dilate_kernel: int = 5,
        threshold: int = 210,
    ):
        self.method = method.lower()
        self.inpaint_radius = inpaint_radius
        self.dilate_kernel = dilate_kernel
        self.threshold = threshold

    def inpaint_bubbles(
        self,
        img: np.ndarray,
        boxes: List[Tuple[int, int, int, int]],
    ) -> np.ndarray:
        """
        Inpaint text inside the specified bounding box regions.
        Uses connected component filtering so outer bubble borders are NEVER erased.

        Args:
            img: BGR input image (H, W, 3).
            boxes: List of (x1, y1, x2, y2) bounding boxes.

        Returns:
            Inpainted BGR image with text removed and bubble borders preserved.
        """
        output = img.copy()
        h, w = output.shape[:2]

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.dilate_kernel, self.dilate_kernel)
        )

        for x1, y1, x2, y2 in boxes:
            x1_c, y1_c = max(0, x1), max(0, y1)
            x2_c, y2_c = min(w, x2), min(h, y2)
            if x2_c <= x1_c or y2_c <= y1_c:
                continue

            crop = output[y1_c:y2_c, x1_c:x2_c]
            ch, cw = crop.shape[:2]
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            mean_val = np.mean(gray)
            is_light_bg = mean_val >= 128

            if is_light_bg:
                # Normal bubble: dark text on light/white background
                binary = (gray < self.threshold).astype(np.uint8)
            else:
                # Inverted bubble: light text on dark background
                binary = (gray > (255 - self.threshold)).astype(np.uint8)

            if np.count_nonzero(binary) == 0:
                continue

            # Connected component analysis to separate isolated text glyphs from outer bubble outline
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            text_mask = np.zeros((ch, cw), dtype=np.uint8)

            for i in range(1, num_labels):
                sx, sy, sw, sh, sarea = stats[i]
                
                # Check if this component touches the crop boundary (it's the bubble outline or external drawing!)
                touches_border = (sx <= 2 or sy <= 2 or (sx + sw) >= cw - 2 or (sy + sh) >= ch - 2)
                # Check if component is gigantic relative to crop
                is_giant = (sw > cw * 0.85 and sh > ch * 0.85)

                if touches_border or is_giant:
                    # Ignore bubble borders and frames!
                    continue

                text_mask[labels == i] = 255

            # If connected component filtering eliminated everything (e.g. text touches crop edge),
            # fall back to eroding boundary of binary mask
            if np.count_nonzero(text_mask) == 0:
                border_mask = np.zeros((ch, cw), dtype=np.uint8)
                border_mask[4:-4, 4:-4] = 255
                text_mask = cv2.bitwise_and(binary, border_mask)

            if np.count_nonzero(text_mask) == 0:
                continue

            # Dilate text mask slightly to absorb glyph anti-aliasing edges
            dilated_mask = cv2.dilate(text_mask, kernel, iterations=1)

            # Inpaint using Fast Marching (Telea)
            flags = cv2.INPAINT_TELEA if self.method == "telea" else cv2.INPAINT_NS
            inpainted_crop = cv2.inpaint(
                crop, dilated_mask, inpaintRadius=self.inpaint_radius, flags=flags
            )

            # If background of non-text area is predominantly uniform white/light, fill cleanly
            bg_pixels = gray[dilated_mask == 0]
            if bg_pixels.size > 0 and np.std(bg_pixels) < 14.0:
                median_bg = int(np.median(bg_pixels))
                inpainted_crop[dilated_mask > 0] = [median_bg, median_bg, median_bg]

            output[y1_c:y2_c, x1_c:x2_c] = inpainted_crop

        return output
