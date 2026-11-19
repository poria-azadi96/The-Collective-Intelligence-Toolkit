"""Shared publication-grade Matplotlib styling and animated GIF export helpers."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

PALETTE: dict[str, str] = {
    "cobalt": "#1D63B8",
    "teal": "#0D7377",
    "amber": "#C87D1E",
    "crimson": "#C62828",
    "slate": "#475569",
    "muted": "#94A3B8",
    "ink": "#0F172A",
    "grid": "#E2E8F0",
    "bg": "#FFFFFF",
    "card": "#F8FAFC",
    "sky": "#0284C7",
    "violet": "#6D28D9",
}


def apply_publication_style(dpi: int = 130) -> None:
    """Configure consistent publication-grade Matplotlib defaults."""
    matplotlib.rcParams.update(
        {
            "figure.dpi": dpi,
            "figure.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["bg"],
            "axes.edgecolor": PALETTE["ink"],
            "axes.linewidth": 1.1,
            "axes.titlesize": 11.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 10.5,
            "axes.labelweight": "bold",
            "font.sans-serif": ["DejaVu Sans", "Arial", "sans-serif"],
            "xtick.color": PALETTE["ink"],
            "ytick.color": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "legend.frameon": True,
            "legend.facecolor": PALETTE["bg"],
            "legend.edgecolor": PALETTE["grid"],
            "savefig.facecolor": PALETTE["bg"],
        }
    )


def style_axes(
    ax: plt.Axes,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    grid_axis: str | None = "both",
) -> plt.Axes:
    """Apply clean publication styling to a single Matplotlib Axes."""
    ax.set_facecolor(PALETTE["bg"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["ink"])
    ax.spines["bottom"].set_color(PALETTE["ink"])
    if grid_axis in ("both", "x", "y"):
        ax.grid(True, axis=grid_axis, linestyle="--", alpha=0.55, color=PALETTE["grid"])
    if title is not None:
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=9, color=PALETTE["ink"])
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", color=PALETTE["ink"])
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold", color=PALETTE["ink"])
    return ax


def fig_to_pil_image(fig: plt.Figure, dpi: int = 115) -> Image.Image:
    """Render a Matplotlib Figure into an RGB PIL Image without leaving disk artifacts."""
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=dpi,
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    buf.seek(0)
    with Image.open(buf) as im:
        rgb = im.convert("RGB")
    return rgb


def save_gif_from_frames(
    frames: Iterable[Image.Image],
    out_path: str | Path,
    duration_ms: int = 110,
    loop: int = 0,
    hold_last_frames: int = 6,
) -> Path:
    """Save an iterable of PIL Images as a palette-optimized looping animated GIF."""
    frame_list = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=256) for f in frames]
    if not frame_list:
        raise ValueError("Cannot save GIF from an empty frame sequence.")
    if hold_last_frames > 0:
        frame_list.extend([frame_list[-1]] * hold_last_frames)
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame_list[0].save(
        target,
        save_all=True,
        append_images=frame_list[1:],
        duration=duration_ms,
        loop=loop,
        optimize=True,
    )
    return target


def save_figure(fig: plt.Figure, out_path: str | Path, dpi: int = 240, close: bool = True) -> Path:
    """Save a Matplotlib figure as a high-resolution PNG and optionally close it."""
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        target,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
    )
    if close:
        plt.close(fig)
    return target
