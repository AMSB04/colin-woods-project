import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import os
from torchvision.utils import make_grid
import torchvision
from PIL import Image

def init_weights(net: nn.Module, init_type: str = 'normal', init_gain: float = 0.02) -> None:
    """
    Initialize weights of a network using the specified method.

    Args:
        net (nn.Module): The network to initialize.
        init_type (str): The type of initialization ('normal', 'xavier', 'kaiming', 'orthogonal').
        init_gain (float): Scaling factor for weight initialization.
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'Initialization method {init_type} is not implemented')
            
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    print(f'Initialize network with {init_type} initialization')
    net.apply(init_func)

def init_weights_gan(m: nn.Module) -> None:
    """
    Initialize weights specifically for GAN components.

    Args:
        m (nn.Module): Layer/module in the model.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_dir: str,
    model_name: str,
    is_best: bool = False,
    additional_info: Optional[Dict] = None
) -> None:
    """
    Save model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current training epoch.
        checkpoint_dir (str): Directory to save the checkpoint.
        model_name (str): Name prefix for the saved model.
        is_best (bool): If True, also save as best model.
        additional_info (Optional[Dict]): Any additional info to save.
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'additional_info': additional_info or {}
    }
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = os.path.join(checkpoint_dir, f'{model_name}_latest.pth')
    torch.save(state, filename)
    
    if is_best:
        best_filename = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
        torch.save(state, best_filename)

def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: torch.device
) -> Tuple[nn.Module, torch.optim.Optimizer, int, Dict]:
    """
    Load a model checkpoint from file.

    Args:
        model (nn.Module): The model to load weights into.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer to load state into (optional).
        checkpoint_path (str): Path to the checkpoint file.
        device (torch.device): Device to map the checkpoint.

    Returns:
        Tuple containing updated model, optimizer, starting epoch, and additional info.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    epoch = checkpoint.get('epoch', 0)
    additional_info = checkpoint.get('additional_info', {})
    
    return model, optimizer, epoch, additional_info

def denormalize(tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """
    Denormalize a tensor image with given mean and std.

    Args:
        tensor (torch.Tensor): Normalized image tensor.
        mean (float or list): Mean used during normalization.
        std (float or list): Std used during normalization.

    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    if isinstance(mean, (list, tuple)):
        mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    if isinstance(std, (list, tuple)):
        std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    
    return tensor * std + mean

def prepare_images_for_logging(
    real_A: torch.Tensor,
    fake_B: torch.Tensor,
    real_B: torch.Tensor,
    n_samples: int = 3
) -> np.ndarray:
    """
    Create a numpy grid of real and generated images for logging or visualization.

    Args:
        real_A (torch.Tensor): Input images.
        fake_B (torch.Tensor): Generated images.
        real_B (torch.Tensor): Ground truth images.
        n_samples (int): Number of samples to include.

    Returns:
        np.ndarray: A single image grid (H x W x C).
    """
    # Select and denormalize images
    real_A = denormalize(real_A[:n_samples])
    fake_B = denormalize(fake_B[:n_samples])
    real_B = denormalize(real_B[:n_samples])
    
    # Convert to numpy and scale to 0-255
    real_A_np = (real_A.cpu().numpy() * 255).astype(np.uint8)
    fake_B_np = (fake_B.cpu().numpy() * 255).astype(np.uint8)
    real_B_np = (real_B.cpu().numpy() * 255).astype(np.uint8)
    
    # Stack images horizontally for comparison
    comparisons = []
    for a, f, b in zip(real_A_np, fake_B_np, real_B_np):
        comparison = np.hstack([a.squeeze(), f.squeeze(), b.squeeze()])
        comparisons.append(comparison)
    
    # Stack all comparisons vertically
    grid = np.vstack(comparisons)
    
    # Add channel dimension if missing (for grayscale)
    if len(grid.shape) == 2:
        grid = np.expand_dims(grid, axis=-1)
    
    return grid

def calculate_metrics(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    metrics: Tuple[str] = ('mse', 'psnr', 'ssim')
) -> Dict[str, float]:
    """
    Compute quality metrics between real and generated images.

    Args:
        real_images (torch.Tensor): Ground truth images.
        fake_images (torch.Tensor): Generated images.
        metrics (Tuple[str]): Metrics to compute ('mse', 'psnr', 'ssim').

    Returns:
        Dict[str, float]: Dictionary of metric results.
    """
    results = {}
    real_images = denormalize(real_images)
    fake_images = denormalize(fake_images)
    
    if 'mse' in metrics:
        mse = nn.MSELoss()(fake_images, real_images).item()
        results['mse'] = mse
    
    if 'psnr' in metrics:
        max_pixel = 1.0  # Since we denormalized
        psnr = 10 * torch.log10(max_pixel**2 / nn.MSELoss()(fake_images, real_images)).item()
        results['psnr'] = psnr
    
    if 'ssim' in metrics:
        from skimage.metrics import structural_similarity as ssim
        real_np = real_images.cpu().numpy().squeeze()
        fake_np = fake_images.cpu().numpy().squeeze()
        
        ssim_values = []
        for r, f in zip(real_np, fake_np):
            ssim_values.append(ssim(r, f, data_range=1.0))
        
        results['ssim'] = np.mean(ssim_values)
    
    return results

def get_device() -> torch.device:
    """
    Return the best available torch device (CUDA > MPS > CPU).

    Returns:
        torch.device: Available device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility across torch, numpy, and environment.

    Args:
        seed (int): Random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count number of parameters in a model.

    Args:
        model (nn.Module): Model to inspect.
        trainable_only (bool): Count only trainable parameters if True.

    Returns:
        int: Total number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def create_model_summary(model: nn.Module, input_size: Tuple[int, ...]) -> str:
    """
    Generate a model summary using torchinfo.

    Args:
        model (nn.Module): Model to summarize.
        input_size (Tuple[int, ...]): Size of input tensor.

    Returns:
        str: Summary string.
    """
    from torchinfo import summary
    return str(summary(
        model,
        input_size=input_size,
        device='cpu',
        verbose=0,
        col_names=("input_size", "output_size", "num_params", "kernel_size"),
    ))