"""
Usage:
    py -3.13 plot_training_loss.py  input.log  output.png
"""

import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Parse ──────────────────────────────────────────────────────────────────────
COMPLETED = re.compile(r"\[EPOCH\s+(\d+)/\d+\].*?Completed", re.I)
AVG_LOSS  = re.compile(r"Avg Total Loss\s*:\s*([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)")

def parse(path: str):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    epochs, losses = [], []
    for i, line in enumerate(lines):
        m = COMPLETED.search(line)
        if m:
            for j in range(i, min(i + 5, len(lines))):
                lm = AVG_LOSS.search(lines[j])
                if lm:
                    epochs.append(int(m.group(1)))
                    losses.append(float(lm.group(1)))
                    break
    if not epochs:
        sys.exit("No epoch entries found — check your log format.")
    return np.array(epochs), np.array(losses)

# ── Plot ───────────────────────────────────────────────────────────────────────
def plot(epochs, losses, out_path):
    plt.style.use("default")
    matplotlib.rcParams.update({
        "font.family":         "serif",
        "font.serif":          ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size":           10,
        "text.color":          "black",
        "axes.labelcolor":     "black",
        "axes.edgecolor":      "black",
        "axes.facecolor":      "white",
        "figure.facecolor":    "white",
        "xtick.color":         "black",
        "ytick.color":         "black",
        "axes.labelsize":      12,
        "axes.titlesize":      13,
        "axes.linewidth":      0.8,
        "xtick.labelsize":     10,
        "ytick.labelsize":     10,
        "xtick.direction":     "in",
        "ytick.direction":     "in",
        "xtick.major.size":    5,
        "ytick.major.size":    5,
        "xtick.minor.size":    2.5,
        "ytick.minor.size":    2.5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "legend.fontsize":     9.5,
        "legend.frameon":      True,
        "legend.framealpha":   0.93,
        "legend.edgecolor":    "0.65",
    })

    mask = epochs >= 10
    epochs, losses = epochs[mask], losses[mask]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    ax.fill_between(epochs, losses, losses.min(),
                    color="#2166ac", alpha=0.08, zorder=1)
    ax.plot(epochs, losses,
            color="#2166ac", linewidth=1.0,
            label="Total Loss",
            zorder=3)

    # Best-loss marker + annotation
    bi = int(np.argmin(losses))
    ax.scatter(epochs[bi], losses[bi],
               color="#d6604d", edgecolors="white", linewidths=0.8,
               s=70, zorder=5,
               label=f"Best: {losses[bi]:.4f}  (epoch {epochs[bi]})")
    ax.annotate(
        f"epoch {epochs[bi]},  loss = {losses[bi]:.4f}",
        xy=(epochs[bi], losses[bi]),
        xytext=(18, 12), textcoords="offset points",
        fontsize=8.5, color="#d6604d",
        arrowprops=dict(arrowstyle="-", color="#d6604d", lw=0.8),
    )

    # Axes
    ax.set_xlabel("Epoch", labelpad=7)
    ax.set_ylabel("Average Total Loss", labelpad=7)
    ax.set_title("Training Loss Curve", pad=10)

    ax.set_xlim(10, epochs[-1])
    ypad = (losses.max() - losses.min()) * 0.10
    ax.set_ylim(losses.min() - ypad, losses.max() + ypad)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4f"))

    ax.grid(True, which="major", linestyle="--", linewidth=0.45, color="0.78", zorder=0)
    ax.grid(True, which="minor", linestyle=":",  linewidth=0.28, color="0.88", zorder=0)

    ax.legend(loc="upper right", borderpad=0.8, handlelength=2.0)

    # Stats box (top-left)
    stats = (
        f"Epochs:  {epochs[0]}–{epochs[-1]}\n"
        f"Initial: {losses[0]:.4f}\n"
        f"Final:   {losses[-1]:.4f}\n"
        f"Best:    {losses[bi]:.4f}\n"
        f"Drop:    {losses[0] - losses.min():.4f}"
    )
    ax.text(0.017, 0.97, stats,
            transform=ax.transAxes,
            fontsize=8, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="0.70", linewidth=0.7, alpha=0.92))

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out_path}")
    plt.close(fig)

# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: py -3.13 plot_training_loss.py  input.log  output.png")
    epochs, losses = parse(sys.argv[1])
    print(f"Parsed {len(epochs)} epochs  |  loss {losses[0]:.4f} → {losses[-1]:.4f}")
    plot(epochs, losses, sys.argv[2])