from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import torch

from datamodules.celebadatamodule import CelebADataModule
from hparams import Parameters
from lightningmodules.classification import Classification
from utils.constant import ATTRIBUTES
from wrapper_plot import create_bar_plot, track_scalar

LOGGER = logging.getLogger("probe")

TARGET_ATTRS = ["Wearing_Lipstick", "Smiling", "Mouth_Slightly_Open"]
EXPECTATION = (
    "Mouth occlusion should reduce logits at least 0.15 more than masking an equally "
    "sized context patch."
)
IOU_TOLERANCE = 0.05


@dataclass
class ProbeSettings:
    """Configuration with sensible defaults so the probe runs without CLI args."""

    threshold: float = 0.15
    max_batches: int = 50
    max_samples: int = 0  # 0 → use all samples made available by dataloader
    mask_fill: float = 0.0
    context_attempts: int = 32
    jitter_frac: float = 0.03  # jitter applied to mouth box anchors
    log_scalar_name: str = "probe/delta_mouth_context"


def _select_dataloader(params: Parameters) -> torch.utils.data.DataLoader:
    """Use the existing CelebA datamodule without wrapping the dataset."""

    datamodule = CelebADataModule(params.data_param)
    datamodule.setup("fit")
    return datamodule.val_dataloader()


def _load_model(params: Parameters, device: torch.device) -> Tuple[Classification, torch.device]:
    """Load the Lightning module using the same configuration as training."""

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


def _sample_mouth_box(
    height: int,
    width: int,
    rng: np.random.Generator,
    jitter_frac: float,
) -> Tuple[float, float, float, float]:
    """CelebA faces are aligned, so a deterministic region with slight jitter suffices."""

    base_x0, base_x1 = 0.32 * width, 0.68 * width
    base_y0, base_y1 = 0.55 * height, 0.90 * height

    jitter_x = rng.uniform(-jitter_frac, jitter_frac) * width
    jitter_y = rng.uniform(-jitter_frac, jitter_frac) * height

    x0 = np.clip(base_x0 + jitter_x, 0, width - 2)
    x1 = np.clip(base_x1 + jitter_x, x0 + 1, width)
    y0 = np.clip(base_y0 + jitter_y, 0, height - 2)
    y1 = np.clip(base_y1 + jitter_y, y0 + 1, height)
    return float(x0), float(y0), float(x1), float(y1)


def _sample_context_box(
    mouth_box: Tuple[float, float, float, float],
    height: int,
    width: int,
    rng: np.random.Generator,
    attempts: int,
) -> Tuple[float, float, float, float]:
    """Sample a context patch of identical size that minimally overlaps the mouth."""

    mx0, my0, mx1, my1 = mouth_box
    box_w = max(2.0, mx1 - mx0)
    box_h = max(2.0, my1 - my0)

    for _ in range(max(1, attempts)):
        x0 = float(rng.uniform(0, max(1.0, width - box_w)))
        y0 = float(rng.uniform(0, max(1.0, height - box_h)))
        candidate = (x0, y0, min(width, x0 + box_w), min(height, y0 + box_h))
        if _iou(candidate, mouth_box) <= IOU_TOLERANCE:
            return candidate

    # Fallback: shift upwards if random sampling failed
    y0 = max(0.0, my0 - box_h - 1.0)
    return (mx0, y0, min(width, mx0 + box_w), min(height, y0 + box_h))


def _iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
        return 0.0
    inter_area = (inter_x1 - inter_x0) * (inter_y1 - inter_y0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter_area / max(1e-6, area_a + area_b - inter_area)


def _mask(images: torch.Tensor, boxes: Sequence[Tuple[float, float, float, float]], fill: float):
    masked = images.clone()
    c = masked.shape[1]
    fill_tensor = torch.full((c,), fill, dtype=masked.dtype, device=masked.device)
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        x0_i, y0_i = int(round(x0)), int(round(y0))
        x1_i, y1_i = int(round(x1)), int(round(y1))
        if x1_i <= x0_i or y1_i <= y0_i:
            continue
        masked[idx, :, y0_i:y1_i, x0_i:x1_i] = fill_tensor.view(-1, 1, 1)
    return masked


@torch.no_grad()
def run_probe(settings: ProbeSettings) -> dict:
    params = Parameters()
    loader = _select_dataloader(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, device = _load_model(params, device)

    attr_to_idx = {name: idx for idx, name in ATTRIBUTES.items()}
    attr_indices = [attr_to_idx[name] for name in TARGET_ATTRS]

    per_attr_sums = torch.zeros(len(TARGET_ATTRS), dtype=torch.float64)
    sample_count = 0
    rng = np.random.default_rng(7)

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
        mouth_boxes = [
            _sample_mouth_box(h, w, rng, settings.jitter_frac) for _ in range(images.size(0))
        ]
        context_boxes = [
            _sample_context_box(box, h, w, rng, settings.context_attempts)
            for box in mouth_boxes
        ]

        clean_logits = model(images)
        mouth_logits = model(_mask(images, mouth_boxes, settings.mask_fill))
        context_logits = model(_mask(images, context_boxes, settings.mask_fill))

        clean_sel = clean_logits[:, attr_indices]
        mouth_sel = mouth_logits[:, attr_indices]
        context_sel = context_logits[:, attr_indices]

        delta = (clean_sel - mouth_sel) - (clean_sel - context_sel)
        per_attr_sums += delta.detach().double().sum(dim=0).cpu()
        sample_count += images.size(0)

    if sample_count == 0:
        raise RuntimeError("Probe did not process any samples; ensure the dataset is available.")

    per_attr_mean = per_attr_sums / sample_count
    aggregate_metric = float(per_attr_mean.mean().item())

    fig, _ = create_bar_plot(
        categories=TARGET_ATTRS,
        values=[float(x) for x in per_attr_mean],
        title="Δmouth−Δcontext per attribute",
        x_label="Attribute",
        y_label="Δmouth−Δcontext",
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

    track_scalar(name=settings.log_scalar_name, value=aggregate_metric)
    return {
        "metric_data1": round(aggregate_metric, 4),
        "threshold": settings.threshold,
        "expectation": EXPECTATION,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = run_probe(ProbeSettings())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
