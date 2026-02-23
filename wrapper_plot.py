
"""
wrapper_plot.py

Plotting + Tracking Wrapper (Plotly + Aim)

Design goals (as requested)
- Aim tracking is ALWAYS ON by default.
- A single Aim Run is always initialized (unless Aim is not installed).
- Every figure created by this module is ALWAYS tracked to Aim.
- Callers do NOT need to pass log_name; a default name is auto-generated.
- You can manually turn tracking OFF by calling `set_tracking(False)` or setting
  environment variable `AIM_PLOTS_ENABLED=0`.

How to use
- Import any create_* function; it will track to Aim automatically.
- Optionally call `set_tracking(False)` to disable tracking globally.

Notes
- This module does NOT require callers to know about Aim.
- If Aim is not installed, plots are still created (tracked=False).
- To ensure Aim logs always go to the same place, this wrapper pins the Aim repo
  to the project root (by default, the parent directory of this file).

Environment variables
- AIM_PLOTS_ENABLED: "1" (default) enables tracking; "0" disables.
- AIM_REPO: optional explicit path for Aim repo (overrides auto-detected root).
- AIM_EXPERIMENT: experiment name (default: "plots")

Helpers return:
    (fig: plotly.graph_objects.Figure, tracked: bool)
"""

from __future__ import annotations

import atexit
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Mapping, Sequence, Tuple

import numpy as np
import plotly.graph_objects as go

LOGGER = logging.getLogger(__name__)

# -----------------------------
# Optional Aim dependency
# -----------------------------
try:
    from aim import Figure, Run
except Exception:  # Aim is optional
    Figure = None  # type: ignore
    Run = None  # type: ignore

# -----------------------------
# Global state
# -----------------------------
_AIM_RUN: Run | None = None
_TRACKING_ENABLED: bool = True  # default ON


def set_tracking(enabled: bool) -> None:
    """
    Manually enable/disable Aim tracking at runtime.

    Example:
        from wrapper_plot import set_tracking
        set_tracking(False)  # disable Aim tracking
    """
    global _TRACKING_ENABLED
    _TRACKING_ENABLED = bool(enabled)


def _env_flag(name: str, default: str = "1") -> bool:
    v = os.getenv(name, default).strip().lower()
    return v not in {"0", "false", "no", "off", ""}


def _get_repo_path() -> str:
    """
    Always store Aim repo in the same directory as this wrapper_plot.py,
    unless AIM_REPO is explicitly set.
    """
    repo = os.getenv("AIM_REPO", "").strip()
    if repo:
        return str(Path(repo).expanduser().resolve())
    return str(Path(__file__).resolve().parent)


def _get_experiment_name() -> str:
    return os.getenv("AIM_EXPERIMENT", "plots").strip() or "plots"


def _get_aim_run() -> Run | None:
    """
    Initialize Aim Run (singleton). Always attempts to start Aim when enabled.
    """
    global _AIM_RUN

    # Global enable switch: env var overrides runtime default
    if not _env_flag("AIM_PLOTS_ENABLED", "1"):
        return None
    if not _TRACKING_ENABLED:
        return None

    if Run is None:
        # Aim not installed; still allow plotting.
        return None

    if _AIM_RUN is not None:
        return _AIM_RUN

    try:
        repo_path = _get_repo_path()
        experiment = _get_experiment_name()
        _AIM_RUN = Run(repo=repo_path, experiment=experiment)
        return _AIM_RUN
    except Exception:
        LOGGER.exception("Failed to initialize Aim run; plots will not be tracked.")
        _AIM_RUN = None
        return None


def _close_aim_run() -> None:
    global _AIM_RUN
    if _AIM_RUN is not None:
        try:
            _AIM_RUN.close()
        except Exception:
            LOGGER.exception("Failed to close Aim run.")
        finally:
            _AIM_RUN = None


if Run is not None:
    atexit.register(_close_aim_run)


def _auto_name(prefix: str, title: str | None = None) -> str:
    """
    Generate a stable-ish name if caller doesn't supply one.
    Aim groups by name, so we keep it readable and consistent.
    """
    if title:
        # sanitize
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-", "/"} else "_" for ch in title.strip())
        safe = "_".join([p for p in safe.split("_") if p])
        safe = safe[:80] if safe else "untitled"
        return f"{prefix}/{safe}"
    # fallback unique
    return f"{prefix}/{uuid.uuid4().hex[:10]}"


def _track_figure(
    *,
    figure: go.Figure,
    name: str | None,
    step: int,
    context: Mapping[str, object] | None,
    title: str | None,
    prefix: str,
) -> bool:
    """
    Always track if Aim is available and tracking is enabled.
    If `name` is None/empty, auto-generate one.
    """
    if Figure is None:
        return False
    run = _get_aim_run()
    if run is None:
        return False

    final_name = name.strip() if isinstance(name, str) and name.strip() else _auto_name(prefix, title)

    try:
        run.track(Figure(figure), name=final_name, step=step, context=context or {})
        return True
    except Exception:
        LOGGER.exception("Failed to track plot '%s'", final_name)
        return False


def track_scalar(
    *,
    name: str,
    value: float,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> bool:
    """
    Track a scalar value to Aim (always-on if enabled).
    Useful for logging probe conclusions like ΔP_good mean.
    """
    run = _get_aim_run()
    if run is None:
        return False
    if not name or not isinstance(value, (int, float)) or np.isnan(value):
        return False
    try:
        run.track(float(value), name=name, step=step, context=context or {})
        return True
    except Exception:
        LOGGER.exception("Failed to track scalar '%s'", name)
        return False


# -----------------------------
# Plot creators
# -----------------------------
def create_bar_plot(
    *,
    categories: Sequence[str],
    values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    log_name: str | None = None,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> Tuple[go.Figure, bool]:
    """Create a categorical bar plot and ALWAYS attempt to track it to Aim."""

    try:
        fig = go.Figure(data=[go.Bar(x=list(categories), y=list(values))])
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    except Exception:
        LOGGER.exception("Failed to render bar plot '%s'", title)
        return go.Figure(), False

    tracked = _track_figure(
        figure=fig,
        name=log_name,
        step=step,
        context=context,
        title=title,
        prefix="bar",
    )
    return fig, tracked


def create_line_plot(
    *,
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    log_name: str | None = None,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> Tuple[go.Figure, bool]:
    """Create a line chart and ALWAYS attempt to track it to Aim."""

    try:
        fig = go.Figure(data=[go.Scatter(x=list(x), y=list(y), mode="lines+markers")])
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    except Exception:
        LOGGER.exception("Failed to render line plot '%s'", title)
        return go.Figure(), False

    tracked = _track_figure(
        figure=fig,
        name=log_name,
        step=step,
        context=context,
        title=title,
        prefix="line",
    )
    return fig, tracked


def create_scatter_plot(
    *,
    x: Sequence[float],
    y: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    color: Sequence[float] | None = None,
    log_name: str | None = None,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> Tuple[go.Figure, bool]:
    """Create a scatter plot and ALWAYS attempt to track it to Aim."""

    try:
        marker_args: dict = {"size": 8}
        if color is not None:
            marker_args["color"] = list(color)
            marker_args["colorscale"] = "Viridis"
            marker_args["showscale"] = True
        fig = go.Figure(
            data=[go.Scatter(x=list(x), y=list(y), mode="markers", marker=marker_args)]
        )
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    except Exception:
        LOGGER.exception("Failed to render scatter plot '%s'", title)
        return go.Figure(), False

    tracked = _track_figure(
        figure=fig,
        name=log_name,
        step=step,
        context=context,
        title=title,
        prefix="scatter",
    )
    return fig, tracked


def create_histogram(
    *,
    values: Sequence[float],
    title: str,
    x_label: str,
    y_label: str,
    nbins: int | None = None,
    log_name: str | None = None,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> Tuple[go.Figure, bool]:
    """Create a histogram and ALWAYS attempt to track it to Aim."""

    try:
        fig = go.Figure(data=[go.Histogram(x=list(values), nbinsx=nbins)])
        fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label)
    except Exception:
        LOGGER.exception("Failed to render histogram '%s'", title)
        return go.Figure(), False

    tracked = _track_figure(
        figure=fig,
        name=log_name,
        step=step,
        context=context,
        title=title,
        prefix="hist",
    )
    return fig, tracked


def create_box_plot(
    *,
    categories: Sequence[str],
    series: Mapping[str, Sequence[float]],
    title: str,
    y_label: str,
    log_name: str | None = None,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> Tuple[go.Figure, bool]:
    """Create box plots and ALWAYS attempt to track them to Aim."""

    try:
        traces = []
        for name in categories:
            traces.append(go.Box(y=list(series.get(name, [])), name=name, boxmean=True))
        fig = go.Figure(data=traces)
        fig.update_layout(title=title, yaxis_title=y_label)
    except Exception:
        LOGGER.exception("Failed to render box plot '%s'", title)
        return go.Figure(), False

    tracked = _track_figure(
        figure=fig,
        name=log_name,
        step=step,
        context=context,
        title=title,
        prefix="box",
    )
    return fig, tracked


def create_heatmap(
    *,
    z: Sequence[Sequence[float]],
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    log_name: str | None = None,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> Tuple[go.Figure, bool]:
    """Create a heatmap and ALWAYS attempt to track it to Aim."""

    try:
        z_array = np.array(z, dtype=float)
        fig = go.Figure(data=go.Heatmap(z=z_array, x=list(x_labels), y=list(y_labels)))
        fig.update_layout(title=title)
    except Exception:
        LOGGER.exception("Failed to render heatmap '%s'", title)
        return go.Figure(), False

    tracked = _track_figure(
        figure=fig,
        name=log_name,
        step=step,
        context=context,
        title=title,
        prefix="heatmap",
    )
    return fig, tracked


def create_pie_chart(
    *,
    labels: Sequence[str],
    values: Sequence[float],
    title: str,
    log_name: str | None = None,
    step: int = 0,
    context: Mapping[str, object] | None = None,
) -> Tuple[go.Figure, bool]:
    """Create a pie chart and ALWAYS attempt to track it to Aim."""

    try:
        fig = go.Figure(data=[go.Pie(labels=list(labels), values=list(values))])
        fig.update_layout(title=title)
    except Exception:
        LOGGER.exception("Failed to render pie chart '%s'", title)
        return go.Figure(), False

    tracked = _track_figure(
        figure=fig,
        name=log_name,
        step=step,
        context=context,
        title=title,
        prefix="pie",
    )
    return fig, tracked