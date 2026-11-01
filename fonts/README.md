# Font Guidance for MangaTranslate

MangaTranslate uses CJK TrueType/OpenType (`.ttf` / `.ttc` / `.otf`) fonts to render translated Chinese text into speech bubbles.

## Windows Default Fonts
On Windows 10 / 11, the pipeline automatically detects and uses standard system CJK fonts:
- `C:\Windows\Fonts\msyh.ttc` (Microsoft YaHei / 微软雅黑)
- `C:\Windows\Fonts\msyhbd.ttc` (Microsoft YaHei Bold)
- `C:\Windows\Fonts\simhei.ttf` (SimHei / 黑体)

## Recommended Free & Open Source Fonts
If running on Linux, macOS, or if you prefer a comic-style typeface:

1. **Source Han Sans / Noto Sans CJK (思源黑体)**:
   - Official release: [Google Noto CJK](https://github.com/notofonts/noto-cjk) / [Adobe Source Han Sans](https://github.com/adobe-fonts/source-han-sans)
   - License: SIL Open Font License 1.1

2. **ZCOOL KuHei / 酷黑 (Comic Style)**:
   - Free commercial comic/title font.
   - [ZCOOL Fonts](https://github.com/googlefonts/zcool-kuaile)

3. **Custom Font Configuration**:
   Pass your desired font via CLI or `config.json`:
   ```powershell
   python scripts/translate.py --input page.jpg --font "fonts/your_font.ttf"
   ```
