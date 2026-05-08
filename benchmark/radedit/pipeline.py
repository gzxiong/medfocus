"""RadEdit (Perez-Garcia et al., 2024) wrapper.

The RadEdit pipeline is a custom Diffusers pipeline that takes:
  - an image
  - a binary `edit_mask` over the region to modify
  - a binary `keep_mask` over the region to preserve
  - a text prompt describing the desired post-edit content

For MedGround-Bench we need both directions:
  - foreground edit: `edit_mask = annotated_box`, prompt = "No {attribute}"
  - background edit: `edit_mask = inverse(annotated_box)`, same or "no abnormality" prompt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch
from PIL import Image, ImageDraw, ImageOps

from medfocus.data.io import resize_pad


Mode = Literal["fg", "bg"]


@dataclass
class RadEditConfig:
    radedit_id: str = "microsoft/radedit"
    vae_id: str = "stabilityai/sdxl-vae"
    text_encoder_id: str = "microsoft/BiomedVLP-BioViL-T"
    image_size: int = 224
    num_inference_steps: int = 200
    guidance_scale: float = 7.5
    skip_ratio: float = 0.3
    seed: int = 0


class RadEditPipeline:
    """Lazy-loaded RadEdit inpainting wrapper."""

    def __init__(self, cfg: RadEditConfig | None = None, device: Optional[str | torch.device] = None):
        self.cfg = cfg or RadEditConfig()
        self.device = torch.device(device) if device else (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._pipe = None  # built on first call

    def _build(self):
        from diffusers import (
            AutoencoderKL,
            DDIMScheduler,
            DiffusionPipeline,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from transformers import AutoModel, AutoTokenizer

        unet = UNet2DConditionModel.from_pretrained(self.cfg.radedit_id, subfolder="unet")
        vae = AutoencoderKL.from_pretrained(self.cfg.vae_id)
        text_encoder = AutoModel.from_pretrained(self.cfg.text_encoder_id, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.text_encoder_id, model_max_length=128, trust_remote_code=True
        )
        scheduler = DDIMScheduler(
            beta_schedule="linear",
            clip_sample=False,
            prediction_type="epsilon",
            timestep_spacing="trailing",
            steps_offset=1,
        )
        gen = StableDiffusionPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet, scheduler=scheduler,
            safety_checker=None, requires_safety_checker=False, feature_extractor=None,
        ).to(self.device)
        self._pipe = DiffusionPipeline.from_pipe(gen, custom_pipeline=self.cfg.radedit_id)

    def _ensure_built(self):
        if self._pipe is None:
            self._build()

    @torch.inference_mode()
    def inpaint(
        self,
        img: Image.Image,
        boxes: Iterable[tuple[int, int, int, int]],
        prompt: str,
        *,
        mode: Mode,
        seed: Optional[int] = None,
    ) -> Image.Image:
        """Run one inpainting forward.

        `mode="fg"` edits the union of `boxes`; `mode="bg"` edits everything outside.
        Boxes are in the image's native pixel coordinates.
        """
        self._ensure_built()
        cfg = self.cfg

        edit_mask = Image.new("L", img.size, 0)
        draw = ImageDraw.Draw(edit_mask)
        for b in boxes:
            draw.rectangle(b, fill=255)
        if mode == "bg":
            edit_mask = ImageOps.invert(edit_mask)
        keep_mask = ImageOps.invert(edit_mask)

        input_image = resize_pad(img.convert("RGB"), cfg.image_size)
        edit_mask = resize_pad(edit_mask, cfg.image_size)
        keep_mask = resize_pad(keep_mask, cfg.image_size, fill=255)

        torch.manual_seed(cfg.seed if seed is None else seed)
        edited = self._pipe(
            prompt,
            weights=[cfg.guidance_scale],
            image=input_image,
            edit_mask=edit_mask,
            keep_mask=keep_mask,
            num_inference_steps=cfg.num_inference_steps,
            invert_prompt="",
            skip_ratio=cfg.skip_ratio,
        )
        return edited.images[0] if hasattr(edited, "images") else edited[0]
