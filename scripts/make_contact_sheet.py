#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a labeled contact sheet from translated pages or QA comparison images.
"""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


def create_contact_sheet(
    input_dir: Path | str,
    output_path: Path | str,
    columns: int = 5,
    thumb_width: int = 360,
    label_height: int = 28,
):
    in_dir = Path(input_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in in_dir.glob("*.*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"} and not p.name.startswith(".")])
    if not files:
        print(f"No image files found in {in_dir}")
        return

    thumbs = []
    for p in files:
        im = Image.open(p).convert("RGB")
        w, h = im.size
        th = int(h * thumb_width / w)
        thumbs.append((p.stem, im.resize((thumb_width, th), Image.LANCZOS)))

    cell_w = thumb_width
    cell_h = max(im.size[1] for _, im in thumbs) + label_height
    cols = columns
    rows = (len(thumbs) + cols - 1) // cols
    margin = 16

    sheet_w = cols * cell_w + (cols + 1) * margin
    sheet_h = rows * cell_h + (rows + 1) * margin
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for idx, (stem, thumb) in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)
        sheet.paste(thumb, (x, y))
        draw.text((x + 4, y + thumb.size[1] + 4), stem[:30], fill="black", font=font)

    sheet.save(out_file, quality=90)
    print(f"Contact sheet ({len(thumbs)} pages) saved -> {out_file}")


def main():
    ap = argparse.ArgumentParser(description="Create a grid contact sheet of translated manga pages.")
    ap.add_argument("--input-dir", "-i", required=True, help="Directory containing images")
    ap.add_argument("--output", "-o", required=True, help="Output contact sheet JPEG path")
    ap.add_argument("--columns", "-c", type=int, default=5, help="Number of columns in the grid")
    ap.add_argument("--thumb-width", "-w", type=int, default=360, help="Thumbnail width in pixels")
    args = ap.parse_args()
    create_contact_sheet(args.input_dir, args.output, args.columns, args.thumb_width)


if __name__ == "__main__":
    main()
