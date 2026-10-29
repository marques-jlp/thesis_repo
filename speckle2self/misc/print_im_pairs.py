import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import cv2

from matplotlib.widgets import Button, Slider


currdir = os.getcwd()
start_dir = os.path.dirname(os.path.dirname(os.path.dirname(currdir)))
start_dir = os.path.join(start_dir, r"experiments\speckle2self-results\s2f_val")



raw_dir = r"C:\P2P\Dissertation\Cropped_PNGs2\Almeida\cross-section"
epoch = "49"
dn_dir = r"C:\P2P\Dissertation\experiments\speckle2self-results\s2f_val\16-05-2026\epoch" + epoch


raw_list = []
dn_list = []

for file in Path(raw_dir).glob("*.png"):
    raw_list.append(file)



for file in Path(dn_dir).glob('*.png'):
    dn_list.append(file)

raw_list = sorted(raw_list)[::260] # Trial and error until the the number of elements was 10
dn_list = sorted(dn_list)[::260]

# Load the first pair to initialize
idx = 0 
raw_im = cv2.imread(str(raw_list[idx]), cv2.IMREAD_GRAYSCALE)
dn_im = cv2.imread(str(dn_list[idx]), cv2.IMREAD_GRAYSCALE)

fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(bottom=0.2) # Make room for the slider

im1 = ax1.imshow(raw_im, cmap='gray')
ax1.set_title('Raw')

im2 = ax2.imshow(dn_im, cmap='gray')
ax2.set_title('Denoised')

ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03]) # [left, bottom, width, height]
slider = Slider(ax_slider, 'Slice', 0, len(raw_list) -1, valinit=0, valstep=1)

def update(val):
    i = int(slider.val)

    raw_im = cv2.imread(str(raw_list[i]), cv2.IMREAD_GRAYSCALE)
    dn_im = cv2.imread(str(dn_list[i]), cv2.IMREAD_GRAYSCALE)

    im1.set_data(raw_im)
    im2.set_data(dn_im)

    fig.canvas.draw_idle() # Redraw only what changed

slider.on_changed(update)

plt.show()