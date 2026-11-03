---
name: manga-translate
description: "Use when the user requests 日漫汉化, 漫画翻译, 嵌字, 修图, manga translation, inpainting, CJK typesetting, or batch chapter processing with this repository's MangaTranslate pipeline; load this skill before running any translation, typesetting, or font-fallback task."
---

# MangaTranslate Skill (AI Agent Guide)

MangaTranslate is an end-to-end Japanese manga translation pipeline: bubble/text detection, manga-ocr, context-aware LLM translation, contour-preserving inpainting, and vertical/horizontal CJK typesetting.

## 1. Prerequisites & Verification Gates

Always run commands from the repository root. Do not start a translation run until the diagnostic check passes:

1. **Check environment, models, and API**:
   ```powershell
   python scripts/check_setup.py --config config.example.json
   ```
   - Verifies Python packages and CUDA/CPU inference.
   - Verifies the bubble detector (`models/detector/comic-speech-bubble-detector.pt`).
   - Verifies Manga-OCR (`models/ocr/manga-ocr-base/model.safetensors`).
   - Verifies the OpenAI-compatible translation gateway (`http://127.0.0.1:8045/v1`).
2. If models are missing on a fresh machine:
   ```powershell
   .\scripts\download_models.ps1
   ```
3. Do not run OCR, translation, or model inference for a typesetting-only change. Reuse the existing `outputs/json/<stem>.json` records and inpainted/background images when available.

---

## 2. CLI Command Guide

### 2.1 Single Page
```powershell
python scripts/translate.py `
  --input "example/002.jpg" `
  --output "outputs/single_test"
```

### 2.2 Batch Chapter
```powershell
python scripts/translate.py `
  --input "example/(C91) 泣かないで! 桃ちゃん!! 1 (別スキャン)" `
  --output "outputs/chapter_01" `
  --glob "*.jpg" `
  --contact-sheet
```

- Batch mode submits the whole chapter as one coherent dialogue stream and preserves positional JSON (`id`, `page`, `box`) for layout safety.
- Already processed pages are skipped unless `--force` is passed.
- Final pages, JSON metadata, QA comparisons, and the contact sheet are written under the chosen `--output` directory.

---

## 3. Typesetting & Font Fallback

- `typesetter.font_path` selects the main CJK font (Windows defaults to Microsoft YaHei; use `--font` to override).
- `typesetter.fallback_font` is the secondary CJK font used when the primary path is unavailable.
- Symbol fallback is automatic and requires no configuration. When a character is in `SYMBOL_FALLBACK_CHARS` (`♡ ♥ ♦ ♣ ♠ ♪ ♬ ✕ ❤`), the typesetter draws that character with a dedicated symbol font:
  - Windows: `C:\Windows\Fonts\seguisym.ttf`, then `seguiemj.ttf`.
  - Linux: DejaVu Sans, then Noto Sans Symbols 2.
- This prevents hearts, diamonds, and music notes from becoming tofu boxes (`口`) with fonts such as Microsoft YaHei or SimHei.
- Horizontal text is split into contiguous font runs and measured per run, so stroke merging inside a run is preserved and no seams appear between CJK and symbol glyphs.
- If a new symbol still renders as `口`, add it to `SYMBOL_FALLBACK_CHARS` in `mangatranslate/typesetter.py` and make sure a symbol font exists on the system.

---

## 4. Hard Constraints

- Never commit `outputs/`, `models/`, `example/`, or `.cache/`; they are runtime, weight, or local-test assets and are gitignored.
- Do not commit logs, temporary test scripts, or generated images.
- Do not claim success if `scripts/check_setup.py` fails or the expected output/QA files are missing; report the blocker instead.
- Keep changes scoped: a typesetting fix should touch `mangatranslate/typesetter.py` and documentation, not OCR/translation logic.
- Never force-push. Preserve the existing CLI API and normal Chinese text behavior when changing the typesetter.

---

## 5. 中文速查摘要

- **自检**：`python scripts/check_setup.py --config config.example.json`
- **单页翻译**：`python scripts/translate.py -i <图片> -o outputs/single_test`
- **整话批量**：`python scripts/translate.py -i <文件夹> -o outputs/chapter_01 --glob "*.jpg" --contact-sheet`
- **字体**：`typesetter.font_path` 指定中文字体，`typesetter.fallback_font` 为备用字体；`♡ ♥ ♦ ♣ ♠ ♪ ♬ ✕ ❤` 会自动切换符号字体，不再出现 `口`。
- **约束**：不提交 `outputs/`、`models/`、`example/`、`.cache/`；仅修嵌字时复用已有 JSON 与底图，不重跑 OCR/翻译。
