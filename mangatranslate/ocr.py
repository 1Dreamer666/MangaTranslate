# -*- coding: utf-8 -*-
"""
Japanese manga OCR module wrapping manga-ocr.
"""
from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image
import torch
import manga_ocr


class MangaOcrWrapper:
    """Wrapper for MangaOcr with support for local safetensors model directories."""

    def __init__(
        self,
        model_path: Union[str, Path] = "models/ocr/manga-ocr-base",
        device: str = "cuda",
        force_cpu: bool = False,
    ):
        self.model_path = Path(model_path)
        
        # Verify local model directory if a path is passed
        if self.model_path.exists() and self.model_path.is_dir():
            safe_weights = self.model_path / "model.safetensors"
            bin_weights = self.model_path / "pytorch_model.bin"
            
            # If safetensors is missing but bin exists, auto-convert to prevent torch 2.5 CVE error
            if not safe_weights.is_file() and bin_weights.is_file():
                self._convert_bin_to_safetensors(bin_weights, safe_weights)
            
            init_target = str(self.model_path)
        else:
            # Fall back to HuggingFace hub id
            init_target = str(model_path)

        use_cpu = force_cpu or (device == "cpu") or (not torch.cuda.is_available())
        self.ocr = manga_ocr.MangaOcr(init_target, force_cpu=use_cpu)

    @staticmethod
    def _convert_bin_to_safetensors(bin_path: Path, safe_path: Path):
        """Converts PyTorch bin weights to safetensors to bypass torch.load restriction in transformers."""
        from safetensors.torch import save_file
        state_dict = torch.load(bin_path, map_location="cpu", weights_only=True)
        # Clone tied weights to satisfy safetensors duplicate memory check
        for key in ["decoder.cls.predictions.decoder.weight", "decoder.cls.predictions.decoder.bias"]:
            if key in state_dict:
                state_dict[key] = state_dict[key].clone()
        save_file(state_dict, safe_path)

    def recognize(self, crop: Union[Image.Image, np.ndarray]) -> str:
        """Recognize Japanese text from an image crop."""
        if isinstance(crop, np.ndarray):
            # Convert BGR/Gray numpy to PIL RGB
            if len(crop.shape) == 2:
                img = Image.fromarray(crop).convert("RGB")
            else:
                img = Image.fromarray(crop[:, :, ::-1]).convert("RGB")
        elif isinstance(crop, Image.Image):
            img = crop.convert("RGB")
        else:
            raise TypeError(f"Expected PIL Image or numpy array, got {type(crop)}")

        text = self.ocr(img)
        # Normalize text: strip whitespace, remove stray spaces between kanji/kana
        text = text.strip().replace(" ", "")
        return text

    def batch_recognize(self, crops: List[Union[Image.Image, np.ndarray]]) -> List[str]:
        """Recognize text from a list of crops."""
        return [self.recognize(c) for c in crops]
