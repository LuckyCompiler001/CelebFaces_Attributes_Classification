from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from utils.constant import ATTRIBUTES
from wrapper_plot import create_bar_plot, track_scalar

LOGGER = logging.getLogger("probe")

TARGET_ATTRS = ["Male", "Young", "Wearing_Necktie"]
EXPECTATION = "Logits should change <0.08 when only background pixels are edited."


@dataclass
class ProbeSettings:
    threshold: float = 0.08
    max_batches: int = 50
    max_samples: int = 0
    blur_kernel_size: int = 31
    background_mix: float = 0.75  # fraction of blurred pixels kept (rest replaced by mean)
    face_radius_x: float = 0.42
    face_radius_y: float = 0.55
    face_y_offset: float = 0.1
    log_scalar_name: str = "probe/background_sensitivity"


def _select_dataloader(params: Parameters) -> torch.utils.data.DataLoader:
    datamodule = CelebADataModule(params.data_param)
    datamodule.setup("fit")
    return datamodule.val_dataloader()


def _load_model(params: Parameters, device: torch.device) -> Tuple[Classification, torch.device]:
    ckpt_path = Path(params.inference_param.ckpt_path)
    if ckpt_path.is_file():
        model = Classification.load_from_checkpoint(
            ckpt_path,
            config=params.train_param,
            attr_dict=ATTRIBUTES,
        )
        LOGGER.info("Loaded checkpoint from %s", ckpt_path)
    else:
        LOGGER.warning(
            "Checkpoint %s not found; running probe with randomly initialised weights.", ckpt_path
        )
        model = Classification(params.train_param, ATTRIBUTES)

    try:
        model.to(device)
    except RuntimeError as exc:
        LOGGER.warning("CUDA init failed (%s); falling back to CPU.", exc)
        device = torch.device("cpu")
        model.to(device)

    model.eval()
    return model, device


def _estimate_face_mask(
    height: int,
    width: int,
    batch_size: int,
    device: torch.device,
    rx: float,
    ry: float,
    y_offset: float,
) -> torch.Tensor:
    ys = torch.linspace(-1.0, 1.0, steps=height, device=device)
    xs = torch.linspace(-1.0, 1.0, steps=width, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    shifted_y = yy - y_offset
    ellipse = (xx / rx) ** 2 + (shifted_y / ry) ** 2 <= 1.0
    mask = ellipse.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


def _box_blur(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return images
    pad = kernel_size // 2
    kernel = torch.ones((images.size(1), 1, kernel_size, kernel_size), device=images.device, dtype=images.dtype)
    kernel /= kernel_size * kernel_size
    return F.conv2d(images, kernel, padding=pad, groups=images.size(1))


def _apply_background_edit(
    images: torch.Tensor,
    face_mask: torch.Tensor,
    kernel_size: int,
    mix: float,
) -> torch.Tensor:
    face_mask = face_mask.unsqueeze(1).to(images.dtype)
    background_mask = 1.0 - face_mask
    blurred = _box_blur(images, kernel_size)
    mean_color = images.mean(dim=(2, 3), keepdim=True)
    replaced_bg = mix * blurred + (1.0 - mix) * mean_color
    edited = face_mask * images + background_mask * replaced_bg
    return edited


@torch.no_grad()
def run_probe(settings: ProbeSettings) -> dict:
    params = Parameters()
    loader = _select_dataloader(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = _load_model(params, device)

    attr_to_idx = {name: idx for idx, name in ATTRIBUTES.items()}
    attr_indices = [attr_to_idx[name] for name in TARGET_ATTRS]

    per_attr_sum = torch.zeros(len(TARGET_ATTRS), dtype=torch.float64)
    sample_count = 0

    for batch_idx, batch in enumerate(loader):
        if settings.max_batches and batch_idx >= settings.max_batches:
            break

        images, _targets = batch
        if settings.max_samples and sample_count >= settings.max_samples:
            break

        if settings.max_samples:
            take = min(images.size(0), settings.max_samples - sample_count)
            images = images[:take]

        images = images.to(device)
        b, _, h, w = images.shape

        face_mask = _estimate_face_mask(
            h,
            w,
            b,
            images.device,
            settings.face_radius_x,
            settings.face_radius_y,
            settings.face_y_offset,
        )

        edited_images = _apply_background_edit(
            images,
            face_mask,
            settings.blur_kernel_size,
            settings.background_mix,
        )

        clean_logits = model(images)
        bg_logits = model(edited_images)

        clean_sel = clean_logits[:, attr_indices]
        bg_sel = bg_logits[:, attr_indices]

        delta = (clean_sel - bg_sel).abs()
        per_attr_sum += delta.detach().double().sum(dim=0).cpu()
        sample_count += images.size(0)

    if sample_count == 0:
        raise RuntimeError("Probe did not process any samples; ensure the dataset is available.")

    per_attr_mean = per_attr_sum / sample_count
    aggregate = float(per_attr_mean.mean().item())

    fig, _ = create_bar_plot(
        categories=TARGET_ATTRS,
        values=[float(x) for x in per_attr_mean],
        title="Background Sensitivity per attribute",
        x_label="Attribute",
        y_label="|Δ logit|",
    )
    fig.add_hline(y=settings.threshold, line_dash="dash", line_color="red")
    fig.add_annotation(
        x=0.5,
        y=settings.threshold,
        text=f"threshold={settings.threshold:.2f}",
        showarrow=False,
        xref="paper",
        yshift=10,
    )

    track_scalar(name=settings.log_scalar_name, value=aggregate)
    return {
        "metric_data1": round(aggregate, 4),
        "threshold": settings.threshold,
        "expectation": EXPECTATION,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = run_probe(ProbeSettings())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
