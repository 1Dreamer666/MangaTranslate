# -*- coding: utf-8 -*-
"""
Bubble and text detection module using Ultralytics YOLO comic-speech-bubble-detector.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union
import numpy as np
import torch
from ultralytics import YOLO


@dataclass
class Detection:
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2
    conf: float
    cls_id: int
    cls_name: str


class BubbleDetector:
    """Detects speech bubbles and standalone text blocks in manga pages."""

    CLASS_NAMES = {0: "text_bubble", 1: "text_free"}

    def __init__(
        self,
        model_path: Union[str, Path] = "models/detector/comic-speech-bubble-detector.pt",
        conf: float = 0.25,
        device: str = "cuda",
    ):
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Bubble detector model not found at '{self.model_path}'. "
                "Please run scripts/download_models.ps1 or specify a valid path in config."
            )
        
        target_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        self.device = target_device
        self.conf = conf
        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

    def detect(
        self, 
        image: Union[np.ndarray, str, Path],
        conf: float | None = None
    ) -> List[Detection]:
        """
        Run bubble and text detection on an image.

        Args:
            image: BGR numpy image or file path.
            conf: Optional confidence threshold override.

        Returns:
            List of Detection objects sorted roughly in reading order.
        """
        threshold = conf if conf is not None else self.conf
        results = self.model(image, conf=threshold, device=self.device, verbose=False)
        
        detections: List[Detection] = []
        if not results:
            return detections
        
        res = results[0]
        if res.boxes is None:
            return detections
        
        for box in res.boxes:
            xyxy = [int(v) for v in box.xyxy[0].tolist()]
            score = float(box.conf[0])
            cid = int(box.cls[0])
            cname = self.CLASS_NAMES.get(cid, f"class_{cid}")
            
            # Filter zero/degenerate area boxes
            if xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1]:
                detections.append(Detection(
                    box=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    conf=score,
                    cls_id=cid,
                    cls_name=cname
                ))
        
        return detections
