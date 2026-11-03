# -*- coding: utf-8 -*-
"""
CJK Typesetting engine for manga speech bubbles: auto vertical/horizontal layout, font size fitting, and stroke.
"""
from pathlib import Path
from typing import List, Tuple, Union, Optional, Any
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class MangaTypesetter:
    """Typesets translated Chinese text into manga speech bubbles."""

    # Punctuation marks that should not start a line
    NO_START_PUNCT = set("，。！？、；：）》」』”’…—～~!?")

    # Symbols commonly used in manga dialogue that Microsoft YaHei / SimHei
    # do not cover. They are drawn with a symbol font instead of becoming
    # tofu boxes (口) in the middle of a bubble.
    SYMBOL_FALLBACK_CHARS = set("♡♥♦♣♠♪♬✕❤")

    def __init__(
        self,
        font_path: Optional[Union[str, Path]] = None,
        fallback_font: Optional[Union[str, Path]] = None,
        default_color: Tuple[int, int, int] = (0, 0, 0),
        stroke_color: Tuple[int, int, int] = (255, 255, 255),
        stroke_width: int = 2,
    ):
        self.font_path = self._resolve_font(font_path, fallback_font)
        self.symbol_font_path = self._resolve_symbol_font()
        self._symbol_fonts: dict = {}
        self.default_color = default_color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width

    @staticmethod
    def _resolve_symbol_font() -> Optional[str]:
        """Locate a font that covers common manga symbols (♡ ♥ ♦ ♪ ...)."""
        candidates = [
            Path(r"C:\Windows\Fonts\seguisym.ttf"),
            Path(r"C:\Windows\Fonts\seguiemj.ttf"),
            Path(r"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path(r"/usr/share/fonts/opentype/noto/NotoSansSymbols2-Regular.ttf"),
        ]
        for c in candidates:
            if c.is_file():
                return str(c)
        return None

    def _is_symbol_char(self, ch: str) -> bool:
        """Whether a character must be rendered with the symbol fallback font."""
        return bool(self.symbol_font_path) and ch in self.SYMBOL_FALLBACK_CHARS

    def _symbol_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Load (and cache) the symbol font at a given pixel size."""
        if size not in self._symbol_fonts:
            self._symbol_fonts[size] = ImageFont.truetype(self.symbol_font_path, size)
        return self._symbol_fonts[size]

    def _font_runs(self, text: str) -> List[Tuple[str, bool]]:
        """Split text into runs of (characters, uses_symbol_font)."""
        runs: List[Tuple[str, bool]] = []
        buf: List[str] = []
        current_is_symbol: Optional[bool] = None
        for ch in text:
            is_symbol = self._is_symbol_char(ch)
            if buf and is_symbol != current_is_symbol:
                runs.append(("".join(buf), bool(current_is_symbol)))
                buf = []
            buf.append(ch)
            current_is_symbol = is_symbol
        if buf:
            runs.append(("".join(buf), bool(current_is_symbol)))
        return runs

    def _measure_text(self, text: str, font: ImageFont.FreeTypeFont, size: int) -> int:
        """Measure text width across mixed CJK and symbol-font runs."""
        total = 0
        for run, is_symbol in self._font_runs(text):
            run_font = self._symbol_font(size) if is_symbol else font
            bbox = run_font.getbbox(run)
            total += bbox[2] - bbox[0]
        return total

    @staticmethod
    def _resolve_font(primary: Optional[Union[str, Path]], fallback: Optional[Union[str, Path]]) -> str:
        candidates = []
        if primary:
            candidates.append(Path(primary))
        if fallback:
            candidates.append(Path(fallback))
        # Common system CJK fonts on Windows
        candidates.extend([
            Path(r"C:\Windows\Fonts\msyh.ttc"),      # Microsoft YaHei
            Path(r"C:\Windows\Fonts\msyhbd.ttc"),    # YaHei Bold
            Path(r"C:\Windows\Fonts\simhei.ttf"),    # SimHei
            Path(r"C:\Windows\Fonts\simsun.ttc"),    # SimSun
            Path(r"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path(r"/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        ])
        for c in candidates:
            if c and c.is_file():
                return str(c)
        raise FileNotFoundError("Could not find a valid CJK font on the system.")

    def typeset_bubble(
        self,
        img: np.ndarray,
        box: Tuple[int, int, int, int],
        text: str,
        orientation: str = "auto",
        padding_ratio: float = 0.15,
    ) -> np.ndarray:
        """
        Render translated text inside a bubble bounding box onto the image.

        Args:
            img: BGR numpy image.
            box: (x1, y1, x2, y2) bounding box.
            text: Translated Chinese text.
            orientation: 'auto', 'vertical', or 'horizontal'.
            padding_ratio: Safe margin around the box border.

        Returns:
            BGR numpy image with typeset text.
        """
        text = text.strip()
        if not text:
            return img

        x1, y1, x2, y2 = box
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 10 or bh <= 10:
            return img

        # Determine orientation
        aspect_ratio = bw / float(bh)
        if orientation == "auto":
            is_vertical = aspect_ratio < 0.95
        else:
            is_vertical = (orientation == "vertical")

        # Determine text and stroke color based on local background
        crop_gray = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        mean_bg = np.mean(crop_gray)
        if mean_bg >= 128:
            fill_color = self.default_color          # Black text
            stroke_col = self.stroke_color           # White stroke
        else:
            fill_color = (255, 255, 255)            # White text
            stroke_col = (0, 0, 0)                  # Black stroke

        # Usable bubble width and height
        pad_x = int(bw * padding_ratio)
        pad_y = int(bh * padding_ratio)
        target_w = max(10, bw - 2 * pad_x)
        target_h = max(10, bh - 2 * pad_y)

        # Find optimal font size and layout
        best_size, layout_blocks = self._fit_text(
            text=text,
            is_vertical=is_vertical,
            target_w=target_w,
            target_h=target_h,
            min_size=14,
            max_size=64,
        )

        font = ImageFont.truetype(self.font_path, best_size)

        # Convert OpenCV BGR to PIL Image for crisp Unicode text rendering
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Calculate bounding dimensions of the formatted layout
        if is_vertical:
            # Columns list: e.g. [['让', '你'], ['久', '等'], ['啦', '！']]
            # In vertical, columns flow right to left!
            cols = layout_blocks
            col_w = int(best_size * 1.25)
            line_h = int(best_size * 1.1)
            total_block_w = len(cols) * col_w
            max_col_len = max(len(c) for c in cols)
            total_block_h = max_col_len * line_h

            start_x = x1 + pad_x + (target_w - total_block_w) // 2
            start_y = y1 + pad_y + (target_h - total_block_h) // 2

            # Right-to-left column layout
            for col_idx, col_chars in enumerate(cols):
                # Rightmost column first:
                cx = start_x + (len(cols) - 1 - col_idx) * col_w
                # Vertically center this individual column if shorter than max
                col_offset_y = (total_block_h - len(col_chars) * line_h) // 2
                for row_idx, char in enumerate(col_chars):
                    cy = start_y + col_offset_y + row_idx * line_h
                    char_font = self._symbol_font(best_size) if self._is_symbol_char(char) else font
                    draw.text(
                        (cx, cy),
                        char,
                        font=char_font,
                        fill=fill_color,
                        stroke_width=self.stroke_width,
                        stroke_fill=stroke_col,
                    )
        else:
            # Horizontal lines
            lines = layout_blocks
            line_h = int(best_size * 1.2)
            total_block_h = len(lines) * line_h
            start_y = y1 + pad_y + (target_h - total_block_h) // 2

            for row_idx, line_text in enumerate(lines):
                # Measure line width (mixed CJK/symbol) for centering
                lw = self._measure_text(line_text, font, best_size)
                lx = x1 + pad_x + (target_w - lw) // 2
                ly = start_y + row_idx * line_h
                cx = lx
                # Draw contiguous runs separately so each uses the right font,
                # while keeping stroke merging intact within a run.
                for run, is_symbol in self._font_runs(line_text):
                    run_font = self._symbol_font(best_size) if is_symbol else font
                    draw.text(
                        (cx, ly),
                        run,
                        font=run_font,
                        fill=fill_color,
                        stroke_width=self.stroke_width,
                        stroke_fill=stroke_col,
                    )
                    bbox = run_font.getbbox(run)
                    cx += bbox[2] - bbox[0]

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _fit_text(
        self,
        text: str,
        is_vertical: bool,
        target_w: int,
        target_h: int,
        min_size: int = 14,
        max_size: int = 64,
    ) -> Tuple[int, List[Any]]:
        """Binary search / evaluate to find the largest font size that fits into the box."""
        best_size = min_size
        best_layout = [list(text)] if is_vertical else [text]

        for size in range(max_size, min_size - 1, -2):
            if is_vertical:
                layout = self._wrap_vertical(text, size, target_h)
                col_w = int(size * 1.25)
                line_h = int(size * 1.1)
                total_w = len(layout) * col_w
                max_h = max(len(c) for c in layout) * line_h
                if total_w <= target_w and max_h <= target_h:
                    return size, layout
            else:
                font = ImageFont.truetype(self.font_path, size)
                layout = self._wrap_horizontal(text, font, target_w)
                line_h = int(size * 1.2)
                total_h = len(layout) * line_h
                max_w = max(self._measure_text(l, font, size) for l in layout)
                if total_h <= target_h and max_w <= target_w:
                    return size, layout

        return best_size, best_layout

    def _wrap_vertical(self, text: str, font_size: int, max_h: int) -> List[List[str]]:
        """Wrap text into vertical columns."""
        line_h = int(font_size * 1.1)
        max_chars_per_col = max(1, max_h // line_h)
        
        cols: List[List[str]] = []
        current_col: List[str] = []
        
        for char in text:
            if len(current_col) >= max_chars_per_col:
                # Avoid orphan punctuation at start of new column if possible
                if char in self.NO_START_PUNCT and len(current_col) > 1:
                    current_col.append(char)
                    cols.append(current_col)
                    current_col = []
                    continue
                cols.append(current_col)
                current_col = []
            current_col.append(char)

        if current_col:
            cols.append(current_col)
        return cols if cols else [[text]]

    def _wrap_horizontal(self, text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
        """Wrap text into horizontal lines."""
        lines: List[str] = []
        current_line = ""

        for char in text:
            test_line = current_line + char
            line_w = self._measure_text(test_line, font, font.size)
            if line_w > max_w and current_line:
                lines.append(current_line)
                current_line = char
            else:
                current_line = test_line

        if current_line:
            lines.append(current_line)
        return lines if lines else [text]
