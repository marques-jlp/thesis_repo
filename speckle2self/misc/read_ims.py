import cv2
from tkinter import filedialog
from tkinter import *
import os
import numpy as np

root = Tk()

currdir = os.getcwd()
start_dir = os.path.dirname(os.path.dirname(os.path.dirname(currdir)))
start_dir = os.path.join(start_dir, r"experiments\speckle2self-results\s2f_val")

root.filename = filedialog.askopenfilename(initialdir=start_dir,
                                             title = 'Select file',
                                             filetypes=(('PNG Files', '*.png'),
                                                        ('All Files', '*.*')   
                                             ))

im_dir = root.filename
im = cv2.imread(im_dir)
im_array = np.array(im)
print(f'The minimum value in the image: {np.min(im_array)}') #It print 0