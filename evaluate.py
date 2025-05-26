import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Tuple

from dataset import MRISliceDataset
from networks.generator import SwinUNetGenerator
from utils.model_util import denormalize, calculate_metrics, get_device, prepare_images_for_logging
from utils.logger import Logger
import logging

def evaluate_generator(
    root_dir: str,
    checkpoint_path: str,
    device: torch.device,
    output_dir: str = 'evaluation_results',
    batch_size: int = 4,
    target_size: Tuple[int, int] = (256, 256),
    paired: bool = True,
    save_images: bool = True,
    metrics: Tuple[str] = ('mse', 'psnr', 'ssim')
) -> Dict[str, float]:
    """Evaluate a trained generator model on test data.
    
    Args:
        root_dir: Path to dataset root directory
        checkpoint_path: Path to generator checkpoint
        device: Device to run evaluation on
        output_dir: Directory to save evaluation results
        batch_size: Evaluation batch size
        target_size: Target image size
        paired: Whether dataset is paired
        save_images: Whether to save output images
        metrics: Tuple of metrics to compute
        
    Returns:
        Dictionary of metric names and values
    """
    # Initialize logger
    logger = Logger(
        log_dir=output_dir,
        experiment_name="evaluation",
        file_log_level=logging.INFO,
        console_log_level=logging.INFO,
        tensorboard=True
    )
    
    # Setup directories
    os.makedirs(output_dir, exist_ok=True)
    if save_images:
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    
    # Initialize dataset and loader
    test_dataset = MRISliceDataset(
        root_dir=root_dir,
        mode='test',
        paired=paired,
        target_size=target_size,
        augment=False,
        return_metadata=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=MRISliceDataset.collate_fn
    )
    
    # Initialize model
    generator = SwinUNetGenerator(
        img_size=target_size[0],
        in_chans=1,
        out_chans=1,
        embed_dim=64,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8
    ).to(device)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state_dict)
    generator.eval()
    
    # Initialize metrics
    metric_results = {metric: 0.0 for metric in metrics}
    total_samples = 0
    
    # Evaluation loop
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc='Evaluating')):
            real_A = batch['A'].to(device)
            real_B = batch['B'].to(device)
            batch_size = real_A.size(0)
            total_samples += batch_size
            
            # Generate fake images
            fake_B = generator(real_A)
            
            # Calculate metrics using model_util function
            batch_metrics = calculate_metrics(real_B, fake_B, metrics)
            for metric in batch_metrics:
                metric_results[metric] += batch_metrics[metric] * batch_size
            
            # Save sample images and log to tensorboard
            if save_images and batch_idx == 0:
                # Prepare grid image for logging
                image_grid = prepare_images_for_logging(real_A, fake_B, real_B)
    
                # Ensure proper shape for logging (H,W,C)
                if len(image_grid.shape) == 3 and image_grid.shape[-1] == 1:
                     image_grid = image_grid.squeeze(-1)  # Remove single channel for grayscale
    
                logger.log_image("sample_comparison", image_grid, 0)
    
                # Save individual images
                save_comparison_images(
                    real_A=real_A,
                    fake_B=fake_B,
                    real_B=real_B,
                    output_dir=os.path.join(output_dir, 'images'),
                    metadata=batch['metadata'][0] if 'metadata' in batch else None
                )
    
    # Average metrics
    for metric in metric_results:
        metric_results[metric] /= total_samples
    
    # Log metrics to both file and tensorboard
    logger.log_metrics(metric_results, 0)
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        for metric, value in metric_results.items():
            f.write(f'{metric}: {value:.4f}\n')
            logger.log_text("metrics", f"{metric}: {value:.4f}")
    
    logger.close()
    return metric_results

def save_comparison_images(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    real_B: torch.Tensor,
    output_dir: str,
    metadata: Dict = None
) -> None:
    """
    Save side-by-side comparison images: input (3T), output (fake 7T), and ground truth (real 7T).

    Args:
        real_A (torch.Tensor): Input 3T MRI slices.
        fake_B (torch.Tensor): Synthesized 7T MRI slices.
        real_B (torch.Tensor): Ground truth 7T MRI slices.
        output_dir (str): Directory to save the comparison images.
        metadata (Dict, optional): Metadata dict to extract filenames (e.g., from 'filename_3T').
    """
    # Convert tensors to numpy and denormalize using model_util function
    real_A_np = (denormalize(real_A).cpu().numpy().squeeze() * 255).astype(np.uint8)
    fake_B_np = (denormalize(fake_B).cpu().numpy().squeeze() * 255).astype(np.uint8)
    real_B_np = (denormalize(real_B).cpu().numpy().squeeze() * 255).astype(np.uint8)
    
    # Save each image in batch
    for i in range(real_A_np.shape[0]):
        # Create filename from metadata if available
        if metadata and 'filename_3T' in metadata:
            base_name = os.path.splitext(metadata['filename_3T'])[0]
            filename = f"{base_name}_comparison.png"
        else:
            filename = f"sample_{i}_comparison.png"
        
        # Stack images horizontally
        comparison = np.hstack([
            real_A_np[i],
            fake_B_np[i],
            real_B_np[i]
        ])
        
        # Ensure proper shape for PIL (H,W) for grayscale
        if len(comparison.shape) == 2:
            img = Image.fromarray(comparison, mode='L')
        else:
            img = Image.fromarray(comparison)
        
        img.save(os.path.join(output_dir, filename))

def parse_args():
    """
    Parse command line arguments for evaluation script.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Evaluate SwinUNet Generator for MRI translation')
    parser.add_argument('--root_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Generator checkpoint path')
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--target_size', type=int, nargs=2, default=[256, 256], help='Image size (H W)')
    parser.add_argument('--unpaired', action='store_true', help='Use unpaired evaluation')
    parser.add_argument('--no_save_images', action='store_true', help='Disable saving sample images')
    parser.add_argument('--metrics', nargs='+', default=['mse', 'psnr', 'ssim'], 
                        choices=['mse', 'psnr', 'ssim'], help='Metrics to compute')
    
    return parser.parse_args()

if __name__ == "__main__":
    import logging
    args = parse_args()
    device = get_device()
    
    results = evaluate_generator(
        root_dir=args.root_dir,
        checkpoint_path=args.checkpoint,
        device=device,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        target_size=tuple(args.target_size),
        paired=not args.unpaired,
        save_images=not args.no_save_images,
        metrics=tuple(args.metrics)
    )
    
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric.upper()}: {value:.4f}")