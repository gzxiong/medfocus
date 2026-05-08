"""MedSAM (HuggingFace `flaviagiammarino/medsam-vit-base`) wrapper.

The wrapper does two things:
  - hold a single (model, processor) pair so we don't reload weights per call;
  - turn an input box prompt into a tight refined bbox via MedSAM mask prediction.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor


def _mask_to_box(mask: np.ndarray) -> Optional[tuple[int, int, int, int]]:
    if mask.dtype != np.bool_:
        mask = mask > 0.0
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _clip_box(box: tuple[int, int, int, int], w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, 0, w - 1))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


class MedSAMClient:
    """Cached MedSAM wrapper.

    Methods
    -------
    refine_box(img, raw_box) : run MedSAM on a single box prompt and return the
        tightest bbox enclosing the predicted mask. Falls back to `raw_box` if
        MedSAM yields an empty mask.
    refine_boxes(img, raw_boxes) : batched form.
    """

    def __init__(
        self,
        model_id: str = "flaviagiammarino/medsam-vit-base",
        device: Optional[str | torch.device] = None,
        multimask_output: bool = True,
        mask_threshold: float = 0.5,
    ):
        self.model_id = model_id
        self.device = torch.device(device) if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(self.device).eval()
        self.multimask_output = multimask_output
        self.mask_threshold = float(mask_threshold)

    @torch.inference_mode()
    def refine_boxes(
        self,
        img: Image.Image,
        raw_boxes: Iterable[tuple[int, int, int, int]],
    ) -> list[tuple[int, int, int, int]]:
        if img.mode != "RGB":
            img = img.convert("RGB")
        W, H = img.size
        raw_boxes = [_clip_box(tuple(map(int, b)), W, H) for b in raw_boxes]
        if not raw_boxes:
            return []

        boxes_f = [[float(x) for x in b] for b in raw_boxes]
        inputs = self.processor(images=img, input_boxes=[boxes_f], return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, multimask_output=self.multimask_output)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.sigmoid().detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
            binarize=False,
        )[0]
        iou = outputs.iou_scores[0].detach().cpu()

        refined: list[tuple[int, int, int, int]] = []
        for bi in range(masks.shape[0]):
            best_m = int(torch.argmax(iou[bi]).item()) if iou.numel() else 0
            mask_np = masks[bi, best_m].numpy() > self.mask_threshold
            tight = _mask_to_box(mask_np) or raw_boxes[bi]
            refined.append(_clip_box(tight, W, H))
        return refined

    def refine_box(self, img: Image.Image, raw_box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return self.refine_boxes(img, [raw_box])[0]
