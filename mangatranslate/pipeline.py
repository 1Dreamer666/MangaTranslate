# -*- coding: utf-8 -*-
"""
Main pipeline coordinating detection, OCR, translation, inpainting, and typesetting.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import time
import cv2
import numpy as np

from .utils import imread_unicode, imwrite_unicode, sort_reading_order, pad_box, deep_merge
from .detector import BubbleDetector
from .ocr import MangaOcrWrapper
from .translator import GeminiTranslator
from .inpainter import MangaInpainter
from .typesetter import MangaTypesetter

DEFAULT_CONFIG: Dict[str, Any] = {
    "api": {
        "base_url": "http://127.0.0.1:8045/v1",
        "api_key": "sk-42c3360b3d73446ab1c87b7ce35f9b99",
        "model": "gemini-3.8-flash-tiered",
        "temperature": 0.2,
        "timeout": 60,
        "max_retries": 3,
    },
    "detector": {
        "model_path": "models/detector/comic-speech-bubble-detector.pt",
        "conf": 0.25,
        "device": "cuda",
    },
    "ocr": {
        "model_path": "models/ocr/manga-ocr-base",
        "device": "cuda",
    },
    "inpainter": {
        "method": "telea",
        "inpaint_radius": 5,
        "dilate_kernel": 5,
        "threshold": 210,
    },
    "typesetter": {
        "font_path": "C:/Windows/Fonts/msyh.ttc",
        "fallback_font": "C:/Windows/Fonts/simhei.ttf",
        "vertical": "auto",
        "text_color": [0, 0, 0],
        "stroke_color": [255, 255, 255],
        "stroke_width": 2,
    },
    "pipeline": {
        "output_dir": "outputs",
        "save_json": True,
        "save_qa": True,
    },
}


class MangaTranslatePipeline:
    """End-to-end pipeline for translating manga pages."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, config_path: Optional[Union[str, Path]] = None):
        cfg = copy_cfg = dict(DEFAULT_CONFIG)
        if config_path and Path(config_path).is_file():
            loaded = json.loads(Path(config_path).read_text(encoding="utf-8"))
            cfg = deep_merge(cfg, loaded)
        if config:
            cfg = deep_merge(cfg, config)
        self.cfg = cfg

        print("Initializing MangaTranslatePipeline components...")
        self.detector = BubbleDetector(
            model_path=self.cfg["detector"]["model_path"],
            conf=self.cfg["detector"]["conf"],
            device=self.cfg["detector"]["device"],
        )
        self.ocr = MangaOcrWrapper(
            model_path=self.cfg["ocr"]["model_path"],
            device=self.cfg["ocr"]["device"],
        )
        self.translator = GeminiTranslator(
            base_url=self.cfg["api"]["base_url"],
            api_key=self.cfg["api"]["api_key"],
            model=self.cfg["api"]["model"],
            temperature=self.cfg["api"]["temperature"],
            timeout=self.cfg["api"]["timeout"],
            max_retries=self.cfg["api"]["max_retries"],
        )
        self.inpainter = MangaInpainter(
            method=self.cfg["inpainter"]["method"],
            inpaint_radius=self.cfg["inpainter"]["inpaint_radius"],
            dilate_kernel=self.cfg["inpainter"]["dilate_kernel"],
            threshold=self.cfg["inpainter"]["threshold"],
        )
        self.typesetter = MangaTypesetter(
            font_path=self.cfg["typesetter"].get("font_path"),
            fallback_font=self.cfg["typesetter"].get("fallback_font"),
            default_color=tuple(self.cfg["typesetter"].get("text_color", [0, 0, 0])),
            stroke_color=tuple(self.cfg["typesetter"].get("stroke_color", [255, 255, 255])),
            stroke_width=self.cfg["typesetter"].get("stroke_width", 2),
        )
        print("All pipeline components loaded successfully!")

    def process_page(
        self,
        img_path: Union[str, Path],
        output_path: Union[str, Path],
        qa_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Process a single manga page end-to-end."""
        img_path = Path(img_path)
        output_path = Path(output_path)
        t_start = time.time()

        # 1. Load image
        img = imread_unicode(img_path)
        h, w = img.shape[:2]

        # 2. Detect text regions
        detections = self.detector.detect(img)
        if not detections:
            print(f"[{img_path.name}] No text bubbles detected, copying original.")
            imwrite_unicode(output_path, img)
            return {"page": img_path.name, "boxes": 0, "seconds": round(time.time() - t_start, 2)}

        # 3. Sort boxes in manga reading order (right-to-left, top-to-bottom)
        raw_boxes = [d.box for d in detections]
        order = sort_reading_order(raw_boxes)
        sorted_detections = [detections[i] for i in order]

        # 4. OCR
        crops = []
        valid_items = []
        for det in sorted_detections:
            x1, y1, x2, y2 = pad_box(det.box, pad=4, max_w=w, max_h=h)
            crop = img[y1:y2, x1:x2]
            text = self.ocr.recognize(crop)
            # Filter out empty or single punctuation noise
            if len(text.strip()) > 0:
                valid_items.append({
                    "box": det.box,
                    "padded_box": (x1, y1, x2, y2),
                    "cls": det.cls_name,
                    "conf": round(det.conf, 2),
                    "original": text,
                })

        if not valid_items:
            print(f"[{img_path.name}] Detected boxes had no readable text.")
            imwrite_unicode(output_path, img)
            return {"page": img_path.name, "boxes": 0, "seconds": round(time.time() - t_start, 2)}

        # 5. Translate in batch
        orig_lines = [item["original"] for item in valid_items]
        translations = self.translator.translate_lines(orig_lines, context_hint=f"Manga page: {img_path.name}")
        for item, trans in zip(valid_items, translations):
            item["translation"] = trans

        # 6. Inpaint bubbles
        inpaint_boxes = [item["padded_box"] for item in valid_items]
        inpainted_img = self.inpainter.inpaint_bubbles(img, inpaint_boxes)

        # 7. Typeset translations
        translated_img = inpainted_img.copy()
        for item in valid_items:
            translated_img = self.typesetter.typeset_bubble(
                translated_img,
                box=item["box"],
                text=item["translation"],
                orientation=self.cfg["typesetter"].get("vertical", "auto"),
            )

        # 8. Save output image
        imwrite_unicode(output_path, translated_img)

        # 9. Save JSON if requested
        if json_path:
            json_p = Path(json_path)
            json_p.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "page": img_path.name,
                "width": w,
                "height": h,
                "items": valid_items,
                "seconds": round(time.time() - t_start, 2),
            }
            json_p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        # 10. Save QA comparison if requested
        if qa_path:
            qa_p = Path(qa_path)
            qa_p.parent.mkdir(parents=True, exist_ok=True)
            # 2-panel comparison: [Original, Translated]
            sep = np.full((h, 10, 3), 255, dtype=np.uint8)
            compare_img = np.hstack([img, sep, translated_img])
            # Scale down if very wide to prevent huge QA files
            if compare_img.shape[1] > 2400:
                scale = 2400 / compare_img.shape[1]
                compare_img = cv2.resize(
                    compare_img, 
                    (2400, int(compare_img.shape[0] * scale)), 
                    interpolation=cv2.INTER_AREA
                )
            imwrite_unicode(qa_p, compare_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        elapsed = time.time() - t_start
        print(f"[{img_path.name}] Done: {len(valid_items)} bubbles translated in {elapsed:.1f}s -> {output_path.name}")
        return {
            "page": img_path.name,
            "bubbles": len(valid_items),
            "seconds": round(elapsed, 2),
            "output": str(output_path),
        }

    def process_batch(
        self,
        input_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        glob_pattern: str = "*.jpg",
        force: bool = False,
    ) -> List[Dict[str, Any]]:
        """Process a directory of manga pages with resume capability."""
        in_p = Path(input_dir)
        out_p = Path(output_dir or self.cfg["pipeline"]["output_dir"])
        out_p.mkdir(parents=True, exist_ok=True)
        qa_dir = out_p / "qa" if self.cfg["pipeline"].get("save_qa", True) else None
        json_dir = out_p / "json" if self.cfg["pipeline"].get("save_json", True) else None

        pages = sorted([p for p in in_p.glob(glob_pattern) if p.is_file()])
        if not pages:
            # Try recursive or other extensions
            pages = sorted([p for p in in_p.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        
        print(f"Processing batch of {len(pages)} pages from {in_p} -> {out_p}")
        summary = []
        t_batch_start = time.time()

        for idx, page in enumerate(pages, 1):
            out_file = out_p / f"{page.stem}.png"
            qa_file = qa_dir / f"{page.stem}_compare.jpg" if qa_dir else None
            json_file = json_dir / f"{page.stem}.json" if json_dir else None

            if not force and out_file.is_file():
                print(f"[{idx}/{len(pages)}] Skipping already translated: {page.name}")
                summary.append({"page": page.name, "status": "skipped"})
                continue

            print(f"[{idx}/{len(pages)}] Translating: {page.name} ...")
            try:
                res = self.process_page(
                    img_path=page,
                    output_path=out_file,
                    qa_path=qa_file,
                    json_path=json_file,
                )
                res["status"] = "success"
                summary.append(res)
            except Exception as e:
                print(f"[{idx}/{len(pages)}] Error on {page.name}: {e}")
                summary.append({"page": page.name, "status": f"error: {e}"})

        # Write summary
        total_time = time.time() - t_batch_start
        (out_p / "summary.json").write_text(
            json.dumps({"total_pages": len(pages), "elapsed_seconds": round(total_time, 1), "pages": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"Batch completed in {total_time / 60:.1f} minutes! Summary saved to {out_p / 'summary.json'}")
        return summary
