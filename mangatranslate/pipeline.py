# -*- coding: utf-8 -*-
"""
Main pipeline coordinating detection, OCR, positional JSON translation, inpainting, and typesetting.
Supports whole-chapter narrative continuity and strict bounding box verification.
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
        "chapter_context": True,
    },
}


class MangaTranslatePipeline:
    """End-to-end pipeline for translating manga pages with positional JSON integrity."""

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

    def extract_page_items(self, img_path: Union[str, Path]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Load image, detect speech bubbles, sort reading order, and run OCR."""
        img_path = Path(img_path)
        img = imread_unicode(img_path)
        h, w = img.shape[:2]

        detections = self.detector.detect(img)
        if not detections:
            return img, []

        raw_boxes = [d.box for d in detections]
        order = sort_reading_order(raw_boxes)
        sorted_detections = [detections[i] for i in order]

        items = []
        for det_idx, det in enumerate(sorted_detections, 1):
            x1, y1, x2, y2 = pad_box(det.box, pad=4, max_w=w, max_h=h)
            crop = img[y1:y2, x1:x2]
            text = self.ocr.recognize(crop)
            if len(text.strip()) > 0:
                items.append({
                    "id": det_idx,
                    "page": img_path.name,
                    "order": det_idx,
                    "box": [int(v) for v in det.box],
                    "padded_box": [int(v) for v in (x1, y1, x2, y2)],
                    "cls": det.cls_name,
                    "conf": round(det.conf, 2),
                    "original": text,
                    "text": text,
                })
        return img, items

    def process_page(
        self,
        img_path: Union[str, Path],
        output_path: Union[str, Path],
        qa_path: Optional[Union[str, Path]] = None,
        json_path: Optional[Union[str, Path]] = None,
        pre_translated_items: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Process a single manga page end-to-end with positional JSON verification."""
        img_path = Path(img_path)
        output_path = Path(output_path)
        t_start = time.time()

        img, valid_items = self.extract_page_items(img_path)
        h, w = img.shape[:2]

        if not valid_items:
            print(f"[{img_path.name}] No readable text bubbles, copying original.")
            imwrite_unicode(output_path, img)
            return {"page": img_path.name, "bubbles": 0, "seconds": round(time.time() - t_start, 2)}

        # Translate if not already supplied by whole-chapter coordinator
        if pre_translated_items is not None:
            # Map pre-translated items by box or id
            verified_items = pre_translated_items
        else:
            context = f"单页漫画翻译：{img_path.name}。请根据画面阅读顺序进行自然贴切的汉化。"
            verified_items = self.translator.translate_items(valid_items, context_hint=context)

        # Inpaint bubbles using verified positions
        inpaint_boxes = [it["padded_box"] for it in verified_items]
        inpainted_img = self.inpainter.inpaint_bubbles(img, inpaint_boxes)

        # Typeset translations onto the clean bubbles
        translated_img = inpainted_img.copy()
        for item in verified_items:
            translated_img = self.typesetter.typeset_bubble(
                translated_img,
                box=item["box"],
                text=item.get("translation", item.get("original", "")),
                orientation=self.cfg["typesetter"].get("vertical", "auto"),
            )

        # Save output image
        imwrite_unicode(output_path, translated_img)

        # Save verified JSON metadata
        if json_path:
            json_p = Path(json_path)
            json_p.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "page": img_path.name,
                "width": w,
                "height": h,
                "items": verified_items,
                "seconds": round(time.time() - t_start, 2),
            }
            json_p.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

        # Save QA comparison
        if qa_path:
            qa_p = Path(qa_path)
            qa_p.parent.mkdir(parents=True, exist_ok=True)
            sep = np.full((h, 10, 3), 255, dtype=np.uint8)
            compare_img = np.hstack([img, sep, translated_img])
            if compare_img.shape[1] > 2400:
                scale = 2400 / compare_img.shape[1]
                compare_img = cv2.resize(
                    compare_img, 
                    (2400, int(compare_img.shape[0] * scale)), 
                    interpolation=cv2.INTER_AREA
                )
            imwrite_unicode(qa_p, compare_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

        elapsed = time.time() - t_start
        print(f"[{img_path.name}] Done: {len(verified_items)} bubbles translated with verified positions in {elapsed:.1f}s -> {output_path.name}")
        return {
            "page": img_path.name,
            "bubbles": len(verified_items),
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
        """
        Whole-Chapter / Batch Coordinator:
        1. Pre-scans all pages in reading order to extract all dialogue boxes.
        2. Submits the entire chapter's dialogue stream with global narrative context to the LLM.
        3. Verifies and reconciles 100% of positional coordinates and IDs.
        4. Distributes verified dialogue back to each page for inpainting and typesetting.
        """
        in_p = Path(input_dir)
        out_p = Path(output_dir or self.cfg["pipeline"]["output_dir"])
        out_p.mkdir(parents=True, exist_ok=True)
        qa_dir = out_p / "qa" if self.cfg["pipeline"].get("save_qa", True) else None
        json_dir = out_p / "json" if self.cfg["pipeline"].get("save_json", True) else None

        pages = sorted([p for p in in_p.glob(glob_pattern) if p.is_file()])
        if not pages:
            pages = sorted([p for p in in_p.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])

        print(f"=== Whole-Chapter Manga Translation Pipeline ===")
        print(f"Directory: {in_p} ({len(pages)} pages) -> Output: {out_p}")
        t_batch_start = time.time()

        # Step 1: Pre-scan and extract dialogue from all pages
        print(f"\n[Phase 1/3] Extracting dialogue and coordinates across all {len(pages)} pages...")
        all_chapter_items: List[Dict[str, Any]] = []
        page_images: Dict[str, np.ndarray] = {}
        pages_to_process = []
        global_id_counter = 1

        for idx, page in enumerate(pages, 1):
            out_file = out_p / f"{page.stem}.png"
            if not force and out_file.is_file():
                print(f"  [{idx}/{len(pages)}] Skipping already processed: {page.name}")
                continue

            pages_to_process.append(page)
            img, items = self.extract_page_items(page)
            page_images[page.name] = img
            for it in items:
                it["global_id"] = global_id_counter
                it["id"] = global_id_counter
                global_id_counter += 1
                all_chapter_items.append(it)
            print(f"  [{idx}/{len(pages)}] {page.name}: {len(items)} dialogue bubbles extracted.")

        if not pages_to_process:
            print("All pages are already processed. Exiting.")
            return []

        # Step 2: Whole-chapter narrative translation with position validation
        print(f"\n[Phase 2/3] Translating {len(all_chapter_items)} items with global narrative context...")
        chapter_context = (
            f"日漫章节全篇连贯翻译：共 {len(pages_to_process)} 页，包含 {len(all_chapter_items)} 处台词与拟声词。\n"
            "要求：统筹理解全话剧情起伏、对话来回交锋与角色情感关系，确保全书人称、敬语、口癖与专有名词完全统一连贯。"
        )

        translated_items = self.translator.translate_items(
            all_chapter_items,
            context_hint=chapter_context,
            max_chunk_size=35,
        )

        # Group translated items by page name
        page_item_map: Dict[str, List[Dict[str, Any]]] = {}
        for it in translated_items:
            p_name = it.get("page", "")
            page_item_map.setdefault(p_name, []).append(it)

        # Step 3: Inpaint and typeset each page
        print(f"\n[Phase 3/3] Inpainting bubbles and typesetting verified translations...")
        summary = []
        for idx, page in enumerate(pages_to_process, 1):
            out_file = out_p / f"{page.stem}.png"
            qa_file = qa_dir / f"{page.stem}_compare.jpg" if qa_dir else None
            json_file = json_dir / f"{page.stem}.json" if json_dir else None

            items_for_page = page_item_map.get(page.name, [])
            img = page_images.get(page.name)
            if img is None:
                img = imread_unicode(page)

            h, w = img.shape[:2]
            t_page = time.time()

            if items_for_page:
                inpaint_boxes = [it["padded_box"] for it in items_for_page]
                inpainted = self.inpainter.inpaint_bubbles(img, inpaint_boxes)
                translated_img = inpainted.copy()
                for item in items_for_page:
                    translated_img = self.typesetter.typeset_bubble(
                        translated_img,
                        box=item["box"],
                        text=item.get("translation", item.get("original", "")),
                        orientation=self.cfg["typesetter"].get("vertical", "auto"),
                    )
            else:
                translated_img = img

            imwrite_unicode(out_file, translated_img)

            if json_file:
                json_p = Path(json_file)
                json_p.parent.mkdir(parents=True, exist_ok=True)
                json_p.write_text(json.dumps({
                    "page": page.name,
                    "width": w,
                    "height": h,
                    "items": items_for_page,
                    "seconds": round(time.time() - t_page, 2),
                }, ensure_ascii=False, indent=2), encoding="utf-8")

            if qa_file:
                qa_p = Path(qa_file)
                qa_p.parent.mkdir(parents=True, exist_ok=True)
                sep = np.full((h, 10, 3), 255, dtype=np.uint8)
                compare_img = np.hstack([img, sep, translated_img])
                if compare_img.shape[1] > 2400:
                    sc = 2400 / compare_img.shape[1]
                    compare_img = cv2.resize(compare_img, (2400, int(compare_img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
                imwrite_unicode(qa_p, compare_img, [cv2.IMWRITE_JPEG_QUALITY, 90])

            el = time.time() - t_page
            print(f"  [{idx}/{len(pages_to_process)}] {page.name}: {len(items_for_page)} bubbles typeset in {el:.1f}s -> {out_file.name}")
            summary.append({"page": page.name, "bubbles": len(items_for_page), "seconds": round(el, 2), "status": "success"})

        total_time = time.time() - t_batch_start
        (out_p / "summary.json").write_text(
            json.dumps({"total_pages": len(pages_to_process), "elapsed_seconds": round(total_time, 1), "pages": summary}, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"\nWhole-chapter translation finished in {total_time / 60:.1f} minutes! Summary saved to {out_p / 'summary.json'}")
        return summary
