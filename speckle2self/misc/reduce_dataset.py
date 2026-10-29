import os 
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

data_dir = r"D:\Cropped_PNGs2"
output_dir = os.path.dirname(data_dir)
output_dir = os.path.join(output_dir, "cropped_reduced")
os.makedirs(output_dir, exist_ok=True)


data = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith(".png"):
            data_path = os.path.join(root, file)
            data.append(data_path)

kept_data = sorted(data)[::10]

output_data = []

for idx in range(len(kept_data)):
    name_change = kept_data[idx].replace("Cropped_PNGs2", "cropped_reduced")
    dirname, filename = os.path.split(name_change)
    os.makedirs(dirname, exist_ok=True)

    im = cv2.imread(kept_data[idx])
    


    cv2.imwrite(name_change, im)
    print(f"Saved a PGN to the path: {name_change}")

