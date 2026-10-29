import os
import time
import random
import logging
from datetime import datetime

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def init_random_seed(seed=777):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def build_dataloader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )


def get_loss_fn(loss_type):
    if loss_type == 'L1':
        return torch.nn.L1Loss()
    elif loss_type == 'L2':
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")


def get_timestamp():
    """Get current timestamp in format YYMMDD-HHMMSS"""
    return datetime.now().strftime("%y%m%d-%H%M%S")


def get_session_id():
    """Alias for get_timestamp for semantic clarity when naming sessions"""
    return get_timestamp()


class FlushHandler(logging.FileHandler):
    """File handler that flushes after every log record, essential for SLURM environments."""
    def emit(self, record):
        super().emit(record)
        self.flush()


def setup_logger(logger_name, log_dir, phase, level=logging.INFO, screen=False, tofile=True):
    """Set up logger to write to file and/or console with immediate flushing.

    Args:
        logger_name: Name of the logger
        log_dir: Directory to save log files
        phase: Phase name (e.g., 'train', 'val')
        level: Logging level
        screen: Whether to log to console
        tofile: Whether to log to file

    Returns:
        logger: Configured logger instance
    """
    lg = logging.getLogger(logger_name)
    lg.setLevel(level)

    for handler in lg.handlers[:]:
        lg.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

    if tofile:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{phase}_{get_timestamp()}.log")
        fh = FlushHandler(log_file, mode="w")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        lg.addHandler(fh)

    if screen:
        sh = logging.StreamHandler()
        sh.setLevel(level)
        sh.setFormatter(formatter)
        lg.addHandler(sh)

    return lg


def save_comparison(raw_tensor, pred_tensor, save_path):
    raw  = raw_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.squeeze().cpu().numpy()

    raw  = np.clip(raw,  0, 1)
    pred = np.clip(pred, 0, 1)

    raw_uint8  = (raw  * 255).astype(np.uint8)
    pred_uint8 = (pred * 255).astype(np.uint8)

    combined = np.concatenate([raw_uint8, pred_uint8], axis=1)
    cv2.imwrite(save_path, combined)


def train_loop(cfg, train_dataset, val_dataset, test_dataset, model, optimizer, device, output_dir):
    train_loader = build_dataloader(train_dataset, cfg['training']['batch_size'])
    val_loader   = build_dataloader(val_dataset,   cfg['training']['batch_size'])
    test_loader  = build_dataloader(test_dataset,  cfg['training']['batch_size'])

    writer = SummaryWriter(log_dir=output_dir)

    logger = setup_logger(
        logger_name='training',
        log_dir=output_dir,
        phase='train',
        screen=True,
        tofile=True
    )

    logger.info(f"Training started")
    logger.info(f"Total epochs: {cfg['training']['epoch']}")
    logger.info(f"Batch size: {cfg['training']['batch_size']}")

    loss_rec     = get_loss_fn(cfg['training']['loss_rec']).to(device)
    loss_consist = get_loss_fn(cfg['training']['loss_consist']).to(device)

    session_id = get_session_id()
    session_save_dir = os.path.join(output_dir, "save_model", session_id)
    os.makedirs(session_save_dir, exist_ok=True)
    logger.info(f"Model checkpoints will be saved to: {session_save_dir}")

    # Create visual output directories
    val_vis_dir  = os.path.join(output_dir, 'val_visuals')
    test_vis_dir = os.path.join(output_dir, 'test_visuals')
    os.makedirs(val_vis_dir,  exist_ok=True)
    os.makedirs(test_vis_dir, exist_ok=True)

    batch_num    = 0
    total_epochs = cfg['training']['epoch']

    # Track losses across epochs for the final plot
    history = {
        'train_total':   [], 'train_rec':   [], 'train_consist':   [],
        'val_total':     [], 'val_rec':     [], 'val_consist':     [],
    }

    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"[EPOCH {epoch+1}/{total_epochs}] Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")

        # ------------------------------------------------------------------ #
        #  TRAINING                                                            #
        # ------------------------------------------------------------------ #
        model.train()
        epoch_total_loss   = 0.0
        epoch_rec_loss     = 0.0
        epoch_consist_loss = 0.0
        num_batches        = 0

        for i_batch, sample in enumerate(train_loader):
            img_low  = sample['image_low'].to(device)
            img_high = sample['image_high'].to(device)
            img_mid  = sample['image_mid'].to(device)
            label    = sample.get('image_clean', None)
            if label is not None:
                label = label.to(device)

            optimizer.zero_grad()
            pred_hr, pred_lr, pred_mid = model(img_high, img_low, img_mid)

            rec_loss = (
                cfg['training']['High_res_weight_rec'] * loss_rec(pred_hr, img_high) +
                cfg['training']['Low_res_weight_rec']  * loss_rec(pred_lr, img_low)  +
                cfg['training']['Mid_res_weight_rec']  * loss_rec(pred_mid, img_mid)
            )

            consist_loss = (
                cfg['training']['Hl_res_weight_consist'] * loss_consist(pred_hr, pred_lr)  +
                cfg['training']['HM_res_weight_consist'] * loss_consist(pred_hr, pred_mid) +
                cfg['training']['ML_res_weight_consist'] * loss_consist(pred_lr, pred_mid)
            )

            total_loss = rec_loss + consist_loss
            total_loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/train_total', total_loss.item(), batch_num)
            batch_num += 1

            epoch_total_loss   += total_loss.item()
            epoch_rec_loss     += rec_loss.item()
            epoch_consist_loss += consist_loss.item()
            num_batches        += 1

            if batch_num % cfg['training']['print_every'] == 0:
                logger.info(f"Batch {batch_num}, Loss: {total_loss.item():.4f}, Rec: {rec_loss.item():.4f}, Consist: {consist_loss.item():.4f}")

            if batch_num % cfg['training']['visualize_every'] == 0:
                writer.add_image('img_lowRes', img_low[0],  epoch)
                writer.add_image('img_highRes', img_high[0], epoch)
                writer.add_image('img_midRes', img_mid[0],  epoch)
                writer.add_image('pred_hr',    pred_hr[0],  epoch)
                writer.add_image('pred_lr',    pred_lr[0],  epoch)
                writer.add_image('pred_mid',   pred_mid[0], epoch)
                if label is not None:
                    writer.add_image('label', label[0], epoch)

            if batch_num % cfg['training']['checkpoint_every'] == 0 and epoch >= cfg['training']['save_after_epoch']:
                save_path = os.path.join(session_save_dir, f"model_{epoch}.pth")
                torch.save(model.state_dict(), save_path)
                logger.info(f"Model saved at: {save_path}")

        avg_train_total   = epoch_total_loss   / num_batches if num_batches > 0 else 0
        avg_train_rec     = epoch_rec_loss     / num_batches if num_batches > 0 else 0
        avg_train_consist = epoch_consist_loss / num_batches if num_batches > 0 else 0

        history['train_total'].append(avg_train_total)
        history['train_rec'].append(avg_train_rec)
        history['train_consist'].append(avg_train_consist)

        writer.add_scalar('Epoch/train_total',   avg_train_total,   epoch)
        writer.add_scalar('Epoch/train_rec',     avg_train_rec,     epoch)
        writer.add_scalar('Epoch/train_consist', avg_train_consist, epoch)

        # ------------------------------------------------------------------ #
        #  VALIDATION                                                          #
        # ------------------------------------------------------------------ #
        model.eval()
        val_total_loss   = 0.0
        val_rec_loss     = 0.0
        val_consist_loss = 0.0
        val_batches      = 0

        with torch.no_grad():
            for sample in val_loader:
                img_low  = sample['image_low'].to(device)
                img_high = sample['image_high'].to(device)
                img_mid  = sample['image_mid'].to(device)

                pred_hr, pred_lr, pred_mid = model(img_high, img_low, img_mid)

                rec_loss = (
                    cfg['training']['High_res_weight_rec'] * loss_rec(pred_hr, img_high) +
                    cfg['training']['Low_res_weight_rec']  * loss_rec(pred_lr, img_low)  +
                    cfg['training']['Mid_res_weight_rec']  * loss_rec(pred_mid, img_mid)
                )

                consist_loss = (
                    cfg['training']['Hl_res_weight_consist'] * loss_consist(pred_hr, pred_lr)  +
                    cfg['training']['HM_res_weight_consist'] * loss_consist(pred_hr, pred_mid) +
                    cfg['training']['ML_res_weight_consist'] * loss_consist(pred_lr, pred_mid)
                )

                total_loss = rec_loss + consist_loss

                val_total_loss   += total_loss.item()
                val_rec_loss     += rec_loss.item()
                val_consist_loss += consist_loss.item()
                val_batches      += 1

        avg_val_total   = val_total_loss   / val_batches if val_batches > 0 else 0
        avg_val_rec     = val_rec_loss     / val_batches if val_batches > 0 else 0
        avg_val_consist = val_consist_loss / val_batches if val_batches > 0 else 0

        history['val_total'].append(avg_val_total)
        history['val_rec'].append(avg_val_rec)
        history['val_consist'].append(avg_val_consist)

        writer.add_scalar('Epoch/val_total',   avg_val_total,   epoch)
        writer.add_scalar('Epoch/val_rec',     avg_val_rec,     epoch)
        writer.add_scalar('Epoch/val_consist', avg_val_consist, epoch)

        # ------------------------------------------------------------------ #
        #  PER EPOCH VISUAL — VALIDATION                                       #
        # ------------------------------------------------------------------ #
        val_sample  = next(iter(val_loader))
        rand_idx    = random.randint(0, val_sample['image_high'].shape[0] - 1)
        val_img     = val_sample['image_high'][rand_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            val_pred, _, _ = model(val_img, val_img, val_img)

        save_comparison(
            raw_tensor=val_sample['image_high'][rand_idx],
            pred_tensor=val_pred,
            save_path=os.path.join(val_vis_dir, f'epoch_{epoch+1:04d}.png'),
        )

        # ------------------------------------------------------------------ #
        #  PER EPOCH VISUAL — TEST                                             #
        # ------------------------------------------------------------------ #
        test_sample = next(iter(test_loader))
        rand_idx    = random.randint(0, test_sample['image_high'].shape[0] - 1)
        test_img    = test_sample['image_high'][rand_idx].unsqueeze(0).to(device)

        with torch.no_grad():
            test_pred, _, _ = model(test_img, test_img, test_img)

        save_comparison(
            raw_tensor=test_sample['image_high'][rand_idx],
            pred_tensor=test_pred,
            save_path=os.path.join(test_vis_dir, f'epoch_{epoch+1:04d}.png'),
        )

        # ------------------------------------------------------------------ #
        #  EPOCH SUMMARY                                                       #
        # ------------------------------------------------------------------ #
        epoch_elapsed = time.time() - epoch_start_time
        logger.info(f"[EPOCH {epoch+1}/{total_epochs}] Completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"  Time: {epoch_elapsed:.2f}s")
        logger.info(f"  Train — Total: {avg_train_total:.4f} | Rec: {avg_train_rec:.4f} | Consist: {avg_train_consist:.4f}")
        logger.info(f"  Val   — Total: {avg_val_total:.4f}   | Rec: {avg_val_rec:.4f}   | Consist: {avg_val_consist:.4f}")
        logger.info(f"{'='*80}\n")

    # ---------------------------------------------------------------------- #
    #  TEST EVALUATION                                                         #
    # ---------------------------------------------------------------------- #
    logger.info("Running final evaluation on test set...")
    model.eval()
    test_total_loss   = 0.0
    test_rec_loss     = 0.0
    test_consist_loss = 0.0
    test_batches      = 0

    with torch.no_grad():
        for sample in test_loader:
            img_low  = sample['image_low'].to(device)
            img_high = sample['image_high'].to(device)
            img_mid  = sample['image_mid'].to(device)

            pred_hr, pred_lr, pred_mid = model(img_high, img_low, img_mid)

            rec_loss = (
                cfg['training']['High_res_weight_rec'] * loss_rec(pred_hr, img_high) +
                cfg['training']['Low_res_weight_rec']  * loss_rec(pred_lr, img_low)  +
                cfg['training']['Mid_res_weight_rec']  * loss_rec(pred_mid, img_mid)
            )

            consist_loss = (
                cfg['training']['Hl_res_weight_consist'] * loss_consist(pred_hr, pred_lr)  +
                cfg['training']['HM_res_weight_consist'] * loss_consist(pred_hr, pred_mid) +
                cfg['training']['ML_res_weight_consist'] * loss_consist(pred_lr, pred_mid)
            )

            total_loss = rec_loss + consist_loss

            test_total_loss   += total_loss.item()
            test_rec_loss     += rec_loss.item()
            test_consist_loss += consist_loss.item()
            test_batches      += 1

    avg_test_total   = test_total_loss   / test_batches if test_batches > 0 else 0
    avg_test_rec     = test_rec_loss     / test_batches if test_batches > 0 else 0
    avg_test_consist = test_consist_loss / test_batches if test_batches > 0 else 0

    logger.info(f"Test — Total: {avg_test_total:.4f} | Rec: {avg_test_rec:.4f} | Consist: {avg_test_consist:.4f}")

    # ---------------------------------------------------------------------- #
    #  LOSS PLOT                                                               #
    # ---------------------------------------------------------------------- #
    epochs = range(1, total_epochs + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training vs Validation Loss', fontsize=14)

    axes[0].plot(epochs, history['train_total'], label='Train', color='steelblue')
    axes[0].plot(epochs, history['val_total'],   label='Val',   color='coral')
    axes[0].axhline(y=avg_test_total, color='green', linestyle='--', label=f'Test: {avg_test_total:.4f}')
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_rec'], label='Train', color='steelblue')
    axes[1].plot(epochs, history['val_rec'],   label='Val',   color='coral')
    axes[1].axhline(y=avg_test_rec, color='green', linestyle='--', label=f'Test: {avg_test_rec:.4f}')
    axes[1].set_title('Reconstruction Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs, history['train_consist'], label='Train', color='steelblue')
    axes[2].plot(epochs, history['val_consist'],   label='Val',   color='coral')
    axes[2].axhline(y=avg_test_consist, color='green', linestyle='--', label=f'Test: {avg_test_consist:.4f}')
    axes[2].set_title('Consistency Loss')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curves saved to: {plot_path}")

    writer.close()