#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main CLI entrance for MangaTranslate.
Translates a single manga page or an entire directory of manga pages.
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path so mangatranslate is importable
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from mangatranslate.pipeline import MangaTranslatePipeline, DEFAULT_CONFIG
from mangatranslate.utils import deep_merge


def main():
    parser = argparse.ArgumentParser(
        description="MangaTranslate: Automatic Japanese Manga Translation & Typesetting Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to an input image file or a directory containing manga pages."
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory for translated pages (default: outputs/)"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.example.json",
        help="Path to JSON configuration file (default: config.example.json)"
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="Override OpenAI-compatible translation API base URL"
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override translation API key"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override translation model (e.g. gemini-3.8-flash-tiered)"
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=None,
        help="Compute device for detector and OCR (default: cuda if available)"
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Path to custom TTF/TTC Chinese font"
    )
    parser.add_argument(
        "--glob",
        default="*.jpg",
        help="File pattern when input is a directory (default: *.jpg)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-translation of pages that already exist in output"
    )
    parser.add_argument(
        "--no-qa",
        action="store_true",
        help="Disable generating side-by-side comparison QA images"
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="Disable exporting intermediate OCR and translation JSON records"
    )
    parser.add_argument(
        "--contact-sheet",
        action="store_true",
        help="Generate a combined contact sheet after batch processing"
    )

    args = parser.parse_args()

    # Load configuration
    cfg_override = {}
    if args.api_base:
        cfg_override.setdefault("api", {})["base_url"] = args.api_base
    if args.api_key:
        cfg_override.setdefault("api", {})["api_key"] = args.api_key
    if args.model:
        cfg_override.setdefault("api", {})["model"] = args.model
    if args.device:
        cfg_override.setdefault("detector", {})["device"] = args.device
        cfg_override.setdefault("ocr", {})["device"] = args.device
    if args.font:
        cfg_override.setdefault("typesetter", {})["font_path"] = args.font
    if args.output:
        cfg_override.setdefault("pipeline", {})["output_dir"] = args.output
    if args.no_qa:
        cfg_override.setdefault("pipeline", {})["save_qa"] = False
    if args.no_json:
        cfg_override.setdefault("pipeline", {})["save_json"] = False

    pipeline = MangaTranslatePipeline(
        config=cfg_override,
        config_path=args.config if Path(args.config).is_file() else None
    )

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input path '{in_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if in_path.is_file():
        # Single page mode
        out_dir = Path(args.output or pipeline.cfg["pipeline"]["output_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{in_path.stem}.png"
        qa_file = (out_dir / "qa" / f"{in_path.stem}_compare.jpg") if not args.no_qa else None
        json_file = (out_dir / "json" / f"{in_path.stem}.json") if not args.no_json else None
        
        print(f"Translating single page: {in_path} -> {out_file}")
        pipeline.process_page(
            img_path=in_path,
            output_path=out_file,
            qa_path=qa_file,
            json_path=json_file,
        )
    else:
        # Batch directory mode
        out_dir = Path(args.output or pipeline.cfg["pipeline"]["output_dir"])
        pipeline.process_batch(
            input_dir=in_path,
            output_dir=out_dir,
            glob_pattern=args.glob,
            force=args.force,
        )

        if args.contact_sheet:
            from scripts.make_contact_sheet import create_contact_sheet
            sheet_path = out_dir / "contact_sheet.jpg"
            create_contact_sheet(out_dir, sheet_path)


if __name__ == "__main__":
    main()
