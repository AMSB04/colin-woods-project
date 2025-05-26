import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from datetime import datetime
from tqdm import tqdm
from typing import Tuple, Optional, Dict
import argparse
from torch.cuda.amp import GradScaler, autocast

from dataset import MRISliceDataset
from networks.generator import SwinUNetGenerator
from networks.discriminator import Discriminator
from utils.model_util import save_checkpoint, prepare_images_for_logging, init_weights_gan, get_device
import logging
from utils.logger import Logger

def train_swin_unet_gan(
    root_dir: str,
    device: torch.device,
    epochs: int = 200,
    batch_size: int = 4,
    lr: float = 2e-4,
    beta1: float = 0.5,
    lambda_cycle: float = 10.0,
    lambda_identity: float = 0.5,
    lambda_paired: float = 1.0,
    save_interval: int = 10,
    checkpoint_dir: str = 'checkpoints',
    log_dir: str = 'logs',
    target_size: Tuple[int, int] = (256, 256),
    paired: bool = True,
    pretrained_gen: Optional[str] = None,
    pretrained_disc: Optional[str] = None
) -> None:
    """
    Train a SwinUNet-based CycleGAN for 3T to 7T MRI synthesis.

    Args:
        root_dir (str): Path to the dataset directory.
        device (torch.device): Device to run the training on.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for optimizers.
        beta1 (float): Beta1 for Adam optimizer.
        lambda_cycle (float): Weight for cycle consistency loss.
        lambda_identity (float): Weight for identity loss.
        lambda_paired (float): Weight for L1 paired loss (used when paired=True).
        save_interval (int): Interval for saving checkpoints.
        checkpoint_dir (str): Directory to save model checkpoints.
        log_dir (str): Directory for logging metrics and images.
        target_size (Tuple[int, int]): Target image resolution.
        paired (bool): Whether paired training with L1 loss is used.
        pretrained_gen (Optional[str]): Path to pretrained generator weights.
        pretrained_disc (Optional[str]): Path to pretrained discriminator weights.
    """
    
    # Setup directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize logger
    logger = Logger(
        log_dir=log_dir,
        experiment_name="swin_gan",
        file_log_level=logging.INFO,
        console_log_level=logging.INFO
    )

    # Initialize datasets
    train_dataset = MRISliceDataset(
        root_dir=root_dir,
        mode='train',
        paired=paired,
        target_size=target_size,
        augment=True
    )
    
    val_dataset = MRISliceDataset(
        root_dir=root_dir,
        mode='val',
        paired=paired,
        target_size=target_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=MRISliceDataset.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=MRISliceDataset.collate_fn
    )

    # Initialize models
    generator = SwinUNetGenerator(
        img_size=target_size[0],
        in_chans=1,
        out_chans=1,
        embed_dim=64,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8
    ).to(device)

    discriminator = Discriminator(in_channels=2).to(device)

    # Initialize weights
    init_weights_gan(generator)
    init_weights_gan(discriminator)

    # Load pretrained weights if provided
    if pretrained_gen:
        generator.load_state_dict(torch.load(pretrained_gen, map_location=device))
        logger.log_text("init", f"Loaded generator weights from {pretrained_gen}")
    if pretrained_disc:
        discriminator.load_state_dict(torch.load(pretrained_disc, map_location=device))
        logger.log_text("init", f"Loaded discriminator weights from {pretrained_disc}")

    # Loss functions
    criterion_gan = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()
    criterion_paired = nn.L1Loss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Learning rate schedulers
    def lr_lambda(epoch):
        decay_start = epochs // 2
        if epoch < decay_start:
            return 1.0
        return max(0.0, 1.0 - (epoch - decay_start) / (epochs - decay_start))

    lr_scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
    lr_scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lr_lambda)

    # Mixed precision training
    scaler = GradScaler()

    # Log hyperparameters
    hparams = {
        'lr': lr,
        'batch_size': batch_size,
        'lambda_cycle': lambda_cycle,
        'lambda_identity': lambda_identity,
        'lambda_paired': lambda_paired,
        'paired': paired
    }
    logger.log_hyperparams(hparams, {})

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        
        epoch_losses = {
            'G': 0.0,
            'D': 0.0,
            'cycle': 0.0,
            'identity': 0.0,
            'paired': 0.0
        }

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            real_A, real_B = batch['A'].to(device), batch['B'].to(device)
            batch_size = real_A.size(0)
            valid = torch.ones((batch_size, 1, 30, 30), device=device)
            fake = torch.zeros((batch_size, 1, 30, 30), device=device)

            # --- Generator Update ---
            optimizer_G.zero_grad()
            
            with autocast():
                fake_B = generator(real_A)
                loss_GAN = criterion_gan(discriminator(torch.cat((real_A, fake_B), 1)), valid)
                loss_paired = criterion_paired(fake_B, real_B) * lambda_paired if paired else 0.0
                loss_cycle = criterion_cycle(generator(fake_B), real_A) * lambda_cycle if lambda_cycle > 0 else 0
                loss_identity = criterion_identity(generator(real_B), real_B) * lambda_identity if lambda_identity > 0 else 0
                loss_G = loss_GAN + loss_cycle + loss_identity + loss_paired

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)

            # --- Discriminator Update ---
            optimizer_D.zero_grad()
            
            with autocast():
                loss_real = criterion_gan(discriminator(torch.cat((real_A, real_B), 1)), valid)
                loss_fake = criterion_gan(discriminator(torch.cat((real_A, fake_B.detach()), 1)), fake)
                loss_D = (loss_real + loss_fake) * 0.5

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            
            scaler.update()

            # Update metrics
            epoch_losses['G'] += loss_G.item()
            epoch_losses['D'] += loss_D.item()
            epoch_losses['cycle'] += loss_cycle if isinstance(loss_cycle, float) else loss_cycle.item()
            epoch_losses['identity'] += loss_identity if isinstance(loss_identity, float) else loss_identity.item()
            epoch_losses['paired'] += loss_paired if isinstance(loss_paired, float) else loss_paired.item()

            progress_bar.set_postfix({
                'G': loss_G.item(),
                'D': loss_D.item(),
                'Cyc': epoch_losses['cycle'] / (batch_idx + 1),
                'Id': epoch_losses['identity'] / (batch_idx + 1),
                'Pair': epoch_losses['paired'] / (batch_idx + 1)
            })

            # Log sample images
            if batch_idx == 0 and epoch % 5 == 0:
                image_grid = prepare_images_for_logging(real_A, fake_B, real_B)
                logger.log_image("samples", image_grid, epoch)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        # Log epoch metrics
        for k in epoch_losses:
            epoch_losses[k] /= len(train_loader)
            logger.log_scalar(f'Loss/{k}', epoch_losses[k], epoch)
        
        logger.log_scalar('LR/Generator', optimizer_G.param_groups[0]['lr'], epoch)
        logger.log_scalar('LR/Discriminator', optimizer_D.param_groups[0]['lr'], epoch)

        # Validation
        if (epoch + 1) % 5 == 0:
            val_metrics = validate(
                generator=generator,
                discriminator=discriminator,
                val_loader=val_loader,
                device=device,
                criterion_gan=criterion_gan,
                criterion_cycle=criterion_cycle,
                criterion_identity=criterion_identity,
                criterion_paired=criterion_paired,
                lambda_cycle=lambda_cycle,
                lambda_identity=lambda_identity,
                lambda_paired=lambda_paired,
                paired=paired
            )
            
            logger.log_metrics(val_metrics, epoch, prefix='Val')
            
            # Save best model
            val_loss = sum(val_metrics.values())
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.log_text("checkpoint", f"New best model at epoch {epoch} with val loss {val_loss:.4f}")
                save_checkpoint(
                    model=generator,
                    optimizer=optimizer_G,
                    epoch=epoch,
                    checkpoint_dir=checkpoint_dir,
                    model_name='generator',
                    is_best=True
                )
                save_checkpoint(
                    model=discriminator,
                    optimizer=optimizer_D,
                    epoch=epoch,
                    checkpoint_dir=checkpoint_dir,
                    model_name='discriminator',
                    is_best=True
                )
        
        # Save checkpoints
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            save_checkpoint(
                model=generator,
                optimizer=optimizer_G,
                epoch=epoch,
                checkpoint_dir=checkpoint_dir,
                model_name='generator'
            )
            save_checkpoint(
                model=discriminator,
                optimizer=optimizer_D,
                epoch=epoch,
                checkpoint_dir=checkpoint_dir,
                model_name='discriminator'
            )
    
    logger.close()

def validate(
    generator: nn.Module,
    discriminator: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    criterion_gan: nn.Module,
    criterion_cycle: nn.Module,
    criterion_identity: nn.Module,
    criterion_paired: nn.Module,
    lambda_cycle: float,
    lambda_identity: float,
    lambda_paired: float,
    paired: bool
) -> Dict[str, float]:
    """
    Evaluate the performance of the generator and discriminator on the validation set.

    Args:
        generator (nn.Module): Generator model (e.g., SwinUNet).
        discriminator (nn.Module): Discriminator model (e.g., PatchGAN).
        val_loader (DataLoader): Validation data loader yielding paired batches of domain A and B.
        device (torch.device): Computation device (CPU or CUDA).
        criterion_gan (nn.Module): Loss function for adversarial loss (e.g., BCEWithLogitsLoss).
        criterion_cycle (nn.Module): Loss function for cycle consistency (e.g., L1Loss).
        criterion_identity (nn.Module): Loss function for identity mapping (e.g., L1Loss).
        criterion_paired (nn.Module): Loss function for paired image translation (e.g., L1Loss).
        lambda_cycle (float): Weight for the cycle consistency loss.
        lambda_identity (float): Weight for the identity loss.
        lambda_paired (float): Weight for the paired L1 loss.
        paired (bool): Whether to include paired loss (True for supervised setting).

    Returns:
        Dict[str, float]: Averaged validation losses for each component ('gan', 'cycle', 'identity', 'paired').
    """
    generator.eval()
    discriminator.eval()
    metrics = {
        'gan': 0.0,
        'cycle': 0.0,
        'identity': 0.0,
        'paired': 0.0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            real_A, real_B = batch['A'].to(device), batch['B'].to(device)
            valid = torch.ones((real_A.size(0), 1, 30, 30), device=device)
            
            # Forward pass
            fake_B = generator(real_A)
            
            # GAN loss
            pred = discriminator(torch.cat((real_A, fake_B), 1))
            metrics['gan'] += criterion_gan(pred, valid).item()
            
            # Cycle loss
            if lambda_cycle > 0:
                recovered_A = generator(fake_B)
                metrics['cycle'] += criterion_cycle(recovered_A, real_A).item() * lambda_cycle
            
            # Identity loss
            if lambda_identity > 0:
                identity_B = generator(real_B)
                metrics['identity'] += criterion_identity(identity_B, real_B).item() * lambda_identity
            
            # Paired loss
            if paired and lambda_paired > 0:
                metrics['paired'] += criterion_paired(fake_B, real_B).item() * lambda_paired
    
    # Average losses over all batches
    return {k: v / len(val_loader) for k, v in metrics.items()}

def parse_args():
    """
    Parse command-line arguments for training the model.

    Returns:
        argparse.Namespace: Parsed arguments including dataset path, training hyperparameters,
                            loss weights, logging paths, and model checkpoint options.
    """
    parser = argparse.ArgumentParser(description='Train SwinUNet GAN for MRI translation')
    parser.add_argument('--root_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta1')
    parser.add_argument('--lambda_cycle', type=float, default=10.0, help='Cycle loss weight')
    parser.add_argument('--lambda_identity', type=float, default=0.5, help='Identity loss weight')
    parser.add_argument('--lambda_paired', type=float, default=1.0, help='Paired L1 loss weight')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint save interval')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='TensorBoard log directory')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256], help='Image size (H W)')
    parser.add_argument('--unpaired', action='store_true', help='Use unpaired training')
    parser.add_argument('--pretrained_gen', type=str, default=None, help='Pretrained generator path')
    parser.add_argument('--pretrained_disc', type=str, default=None, help='Pretrained discriminator path')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    
    train_swin_unet_gan(
        root_dir=args.root_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta1=args.beta1,
        lambda_cycle=args.lambda_cycle,
        lambda_identity=args.lambda_identity,
        lambda_paired=args.lambda_paired,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        target_size=tuple(args.target_size),
        paired=not args.unpaired,
        pretrained_gen=args.pretrained_gen,
        pretrained_disc=args.pretrained_disc
    )