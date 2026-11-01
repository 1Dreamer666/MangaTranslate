# -*- coding: utf-8 -*-
"""
MangaTranslate: AI-powered Japanese Manga Translation Pipeline.
"""
__version__ = "0.1.0"

from .detector import BubbleDetector, Detection
from .ocr import MangaOcrWrapper
from .translator import GeminiTranslator
from .inpainter import MangaInpainter
from .typesetter import MangaTypesetter
from .pipeline import MangaTranslatePipeline

__all__ = [
    "BubbleDetector",
    "Detection",
    "MangaOcrWrapper",
    "GeminiTranslator",
    "MangaInpainter",
    "MangaTypesetter",
    "MangaTranslatePipeline",
]
