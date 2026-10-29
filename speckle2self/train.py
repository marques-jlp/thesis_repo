import os
import yaml
import torch
from torch.utils.data import random_split
from networks.srn.net import SpeckleReductionNet
from utils.datasets import DenoisingDatasetCCA, DenoisingDatasetSimulator, DenoisingDatasetPNG
from utils.training_utils import init_random_seed, train_loop
import argparse


def main(cfg_path, data_dir=None, output_dir=None):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    init_random_seed(cfg.get('seed', 777))

    # Override paths if provided (useful for cluster)
    if data_dir is not None:
        cfg['data']['image_dir'] = data_dir
    if output_dir is not None:
        cfg['training']['output_dir'] = output_dir

    # Add timestamp to output_dir so every run gets its own folder
    from utils.training_utils import get_timestamp
    cfg['training']['output_dir'] = os.path.join(cfg['training']['output_dir'], get_timestamp())

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg['training']['output_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['training']['output_dir'], "save_model"), exist_ok=True)

    # Dataset
    dataset_type = cfg['data'].get('type', 'png').lower()
    if dataset_type == 'cca':
        dataset = DenoisingDatasetCCA(cfg['data']['image_dir'], cfg['data']['interp_method'])
    elif dataset_type == 'simulator':
        dataset = DenoisingDatasetSimulator(cfg['data']['image_dir'], cfg['data']['interp_method'])
    elif dataset_type == 'png':
        dataset = DenoisingDatasetPNG(cfg['data']['image_dir'], cfg['data']['interp_method'])
    else:
        raise ValueError("Unsupported dataset type.")

    # Split dataset 80/10/10
    total      = len(dataset)
    train_size = int(0.8 * total)
    val_size   = int(0.1 * total)
    test_size  = total - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(cfg.get('seed', 777))
    )

    print(f"[INFO] Dataset split — Train: {train_size} | Val: {val_size} | Test: {test_size}")

    model = SpeckleReductionNet().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg['training']['lr'],
        betas=(cfg['training']['b1'], cfg['training']['b2'])
    )

    train_loop(cfg, train_dataset, val_dataset, test_dataset, model, optimizer, device, cfg['training']['output_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/params_png_200epochs.yaml', help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='../../cropped_pngs4', help='Override data directory (for Cluster)')
    parser.add_argument('--output_dir', type=str, default='../../speckle2self-results', help='Override output directory (for Cluster)')

    args = parser.parse_args()
    main(args.config, args.data_dir, args.output_dir)