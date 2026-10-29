import os
import cv2
import torch
import numpy as np
import argparse
from networks.srn.net import SpeckleReductionNet
from utils.image_ops import linear_normalization


def load_model(model_path, device):
    model = SpeckleReductionNet().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model


def load_png_images(data_path):
    """
    Load images from either a single PNG file or a directory of PNGs.
    Recursively searches subdirectories matching the training dataset structure.
    Returns a list of (image_array, image_path) tuples.
    """
    image_entries = []

    if os.path.isfile(data_path):
        # Single PNG file
        if data_path.lower().endswith('.png'):
            img = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to load image: {data_path}")
            image_entries.append((img, data_path))
        else:
            raise ValueError(f"Expected a .png file, got: {data_path}")

    elif os.path.isdir(data_path):
        # Directory — recursively find all PNGs
        png_paths = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.lower().endswith('.png'):
                    png_paths.append(os.path.join(root, file))

        png_paths.sort()

        if len(png_paths) == 0:
            raise ValueError(f"No PNG files found in: {data_path}")

        print(f"[INFO] Found {len(png_paths)} PNG files in {data_path}")

        for path in png_paths:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARNING] Failed to load image, skipping: {path}")
                continue
            image_entries.append((img, path))

    else:
        raise FileNotFoundError(f"Data path not found: {data_path}")

    return image_entries


def visualize_result(norm_input, output):
    """
    Display input and gamma-corrected output in two separate windows.
    Gamma is adjustable from 1.0 to 3.0 (step 0.1) using a short slider.
    Gamma correction helps align denoised results with human perception,
    but ideal values may vary across ultrasound device types.
    """
    win_input = "Input"
    win_output = "Output (Gamma corrected)"
    slider_name = 'Gamma x10'

    cv2.namedWindow(win_input, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_output, cv2.WINDOW_NORMAL)

    cv2.moveWindow(win_input, 100, 100)
    cv2.moveWindow(win_output, 750, 100)

    def nothing(x): pass

    if cv2.getTrackbarPos(slider_name, win_output) == -1:
        cv2.createTrackbar(slider_name, win_output, 15, 30, nothing)

    input_uint8 = (np.clip(norm_input, 0, 1) * 255).astype(np.uint8)
    cv2.imshow(win_input, input_uint8)

    while True:
        slider_val = cv2.getTrackbarPos(slider_name, win_output)
        gamma = max(slider_val / 10.0, 1.0)

        output_clipped = np.clip(output, 0, 1)
        output_gamma = np.power(output_clipped, gamma)
        output_uint8 = (output_gamma * 255).astype(np.uint8)

        cv2.imshow(win_output, output_uint8)

        key = cv2.waitKey(1)
        if key != -1:
            break

    cv2.destroyWindow(win_input)
    cv2.destroyWindow(win_output)

    return output_gamma


def run_inference(model, image_entries, device, output_dir, visualize=False):
    """
    Run inference on a list of (image_array, image_path) tuples.
    Saves each output as a PNG mirroring the input directory structure.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_list = []

    for i, (img_input, img_path) in enumerate(image_entries):
        print(f"[{i+1}/{len(image_entries)}] Processing: {img_path}")

        norm_input = linear_normalization(img_input.astype(np.float32))
        tensor_input = torch.tensor(norm_input).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor, _, _ = model(tensor_input, tensor_input, tensor_input)
            output_tensor = torch.clamp(output_tensor, 0, 1)
            output = output_tensor.squeeze().cpu().numpy()

        if visualize:
            output = visualize_result(norm_input, output)

        # Save output PNG preserving the filename
        filename = os.path.splitext(os.path.basename(img_path))[0]
        save_path = os.path.join(output_dir, f"{filename}_denoised.png")
        output_uint8 = (np.clip(output, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(save_path, output_uint8)
        print(f"[✓] Saved: {save_path}")

        output_list.append(output)

    return np.stack(output_list, axis=0)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = load_model(args.model_path, device)
    print(f"[INFO] Model loaded from: {args.model_path}")

    image_entries = load_png_images(args.data_path)
    print(f"[INFO] Loaded {len(image_entries)} images")

    run_inference(
        model=model,
        image_entries=image_entries,
        device=device,
        output_dir=args.output_dir,
        visualize=args.visualize
    )

    print(f"[✓] All done. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SRN model inference on PNG images.")
    parser.add_argument('--data_path', type=str, required=True, help="Path to a PNG file or directory of PNG files")
    parser.add_argument('--model_path', type=str, required=True, help="Path to model .pth file")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save denoised PNG outputs")
    parser.add_argument('--visualize', action='store_true', help="Whether to visualize each result interactively")
    args = parser.parse_args()

    main(args)