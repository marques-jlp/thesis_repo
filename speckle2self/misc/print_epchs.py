"""
Usage:
    py -3.13 compare_epochs.py
"""

import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
RAW_DIR    = r"C:\P2P\Dissertation\cropped_pngs4\Almeida\cross-section"
EPOCHS_DIR = r"D:\speckle2self-results\git_val\260509-lrelu"
SUBTITLE = "lrelu"
# ══════════════════════════════════════════════════════════════════════════════

# ── One explicit sorted list per source ────────────────────────────────────────
def read_list(input_list, path_name):
    for file in Path(path_name).glob("*png"):
        input_list.append(file)
    return sorted(input_list)[::25]

raw_list    = []
raw_list = read_list(raw_list, RAW_DIR)



ep25_list = []
ep25_list = read_list(ep25_list, EPOCHS_DIR + r"\epoch25")

ep50_list = []
ep50_list = read_list(ep50_list, EPOCHS_DIR + r"\epoch50")

ep100_list = []
ep100_list = read_list(ep100_list, EPOCHS_DIR + r"\epoch100")

ep150_list = []
ep150_list = read_list(ep150_list, EPOCHS_DIR + r"\epoch150")

ep199_list = []
ep199_list = read_list(ep199_list, EPOCHS_DIR + r"\epoch199")


all_lists  = [raw_list, ep25_list, ep50_list, ep100_list, ep150_list, ep199_list]
all_titles = ["Raw", "Epoch 25", "Epoch 50", "Epoch 100", "Epoch 150", "Epoch 199"]

n_slices = len(raw_list)

# ── Read helper ────────────────────────────────────────────────────────────────
def read(path):
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 7))
fig.suptitle(f"{SUBTITLE}: Raw vs. Denoised — Epoch Comparison", fontsize=13)
plt.subplots_adjust(bottom=0.12, hspace=0.35, wspace=0.05)

handles = []
for ax, lst, title in zip(axes.flat, all_lists, all_titles):
    im = ax.imshow(read(lst[0]), cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    handles.append(im)

# ── Slider ─────────────────────────────────────────────────────────────────────
ax_slider = plt.axes([0.2, 0.03, 0.6, 0.025])
slider = Slider(ax_slider, "Slice", 0, n_slices - 1, valinit=0, valstep=1)

def update(val):
    i = int(slider.val)
    handles[0].set_data(read(raw_list[i]))
    handles[1].set_data(read(ep25_list[i]))
    handles[2].set_data(read(ep50_list[i]))
    handles[3].set_data(read(ep100_list[i]))
    handles[4].set_data(read(ep150_list[i]))
    handles[5].set_data(read(ep199_list[i]))
    fig.canvas.draw_idle()

slider.on_changed(update)

from matplotlib.animation import FuncAnimation


plt.show()