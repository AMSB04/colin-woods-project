import os
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class Logger:
    """
    A unified logger for training and evaluation that handles:
    - TensorBoard logging
    - File logging
    - Console logging
    - Image/plot saving

    Attributes:
        experiment_name (str): Name of the experiment and subdirectory for logs.
        log_dir (str): Root directory for saving logs.
        logger (logging.Logger): Internal logger for file and console output.
        tensorboard_writer (SummaryWriter): TensorBoard writer instance.
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: Optional[str] = None,
        file_log_level: int = logging.INFO,
        console_log_level: int = logging.INFO,
        tensorboard: bool = True
    ):
        """
        Initialize the Logger object, create log directories, and set up handlers.

        Args:
            log_dir (str): Base directory for saving logs.
            experiment_name (Optional[str]): Custom experiment name. If None, a timestamped name is generated.
            file_log_level (int): Logging level for the file handler.
            console_log_level (int): Logging level for the console handler.
            tensorboard (bool): Whether to enable TensorBoard logging.
        """
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name or f"exp_{timestamp}"
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, "images"), exist_ok=True)
        
        # Initialize file and console logging
        self.logger = logging.getLogger(self.experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(self.log_dir, "training.log"))
        file_handler.setLevel(file_log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # TensorBoard writer
        self.tensorboard_writer = None
        if tensorboard:
            tb_dir = os.path.join(self.log_dir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tb_dir)
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        Log a scalar value to TensorBoard and file.

        Args:
            tag (str): Name of the scalar metric (e.g., 'loss', 'accuracy').
            value (float): Scalar value to log.
            step (int): Training step or epoch number.
        """
        self.logger.info(f"{tag}: {value:.4f} (step {step})")
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, value, step)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "") -> None:
        """
        Log multiple scalar metrics at once.

        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values.
            step (int): Training step or epoch.
            prefix (str): Optional prefix to add before each metric name.
        """
        for name, value in metrics.items():
            full_tag = f"{prefix}/{name}" if prefix else name
            self.log_scalar(full_tag, value, step)
    
    def log_image(self, tag: str, image: np.ndarray, step: int) -> None:
        """
        Save and log an image to TensorBoard and disk.

        Args:
            tag (str): Label for the image.
            image (np.ndarray): Image array (H x W), (H x W x 1), or (H x W x 3).
            step (int): Step number for TensorBoard and filename.
        """
        # Save to disk
        img_path = os.path.join(self.log_dir, "images", f"{tag}_step{step}.png")
        
        # Handle grayscale images
        if len(image.shape) == 2:  # Grayscale (H,W)
            img = Image.fromarray(image.astype(np.uint8), mode='L')
        elif len(image.shape) == 3 and image.shape[-1] == 1:  # Grayscale (H,W,1)
            img = Image.fromarray(image.squeeze(-1).astype(np.uint8), mode='L')
        else:  # RGB or other
            img = Image.fromarray(image.astype(np.uint8))
        
        img.save(img_path)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            if len(image.shape) == 2:  # Grayscale (H,W)
                image = np.expand_dims(image, axis=0)  # Convert to (1,H,W)
            elif len(image.shape) == 3 and image.shape[-1] == 1:  # Grayscale (H,W,1)
                image = image.transpose(2, 0, 1)  # Convert to (1,H,W)
            self.tensorboard_writer.add_image(tag, image, step)

    
    def log_figure(self, tag: str, figure: plt.Figure, step: int) -> None:
        """
        Save and log a matplotlib figure.

        Args:
            tag (str): Label for the figure.
            figure (plt.Figure): Matplotlib figure object.
            step (int): Step number for TensorBoard and filename.
        """
        # Save to disk
        fig_path = os.path.join(self.log_dir, "images", f"{tag}_step{step}.png")
        figure.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        
        # Log to TensorBoard
        if self.tensorboard_writer:
            self.tensorboard_writer.add_figure(tag, figure, step)
    
    def log_model_weights(self, model: torch.nn.Module, step: int) -> None:
        """
        Log model weights and gradients as histograms to TensorBoard.

        Args:
            model (torch.nn.Module): PyTorch model.
            step (int): Step number for TensorBoard logging.
        """
        if self.tensorboard_writer:
            for name, param in model.named_parameters():
                self.tensorboard_writer.add_histogram(name, param, step)
                if param.grad is not None:
                    self.tensorboard_writer.add_histogram(f"{name}.grad", param.grad, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None) -> None:
        """
        Log a string of text to both console/file and TensorBoard (if step is provided).

        Args:
            tag (str): Category label for the text.
            text (str): The message or content to log.
            step (Optional[int]): Step number for TensorBoard (optional).
        """
        if step is not None:
            text = f"[Step {step}] {text}"
        self.logger.info(text)
        if self.tensorboard_writer and step is not None:
            self.tensorboard_writer.add_text(tag, text, step)
    
    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """
        Log a set of hyperparameters and associated final metrics.

        Args:
            hparams (Dict[str, Any]): Dictionary of hyperparameter names and values.
            metrics (Dict[str, float]): Dictionary of final performance metrics.
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, metrics)
        
        # Also log to file
        self.logger.info("Hyperparameters:")
        for k, v in hparams.items():
            self.logger.info(f"  {k}: {v}")
        self.logger.info("Metrics:")
        for k, v in metrics.items():
            self.logger.info(f"  {k}: {v:.4f}")
    
    def close(self) -> None:
        """
        Close the TensorBoard writer and clean up all log handlers.
        """
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Remove handlers to avoid duplicate logging
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def __del__(self):
        """
        Destructor to ensure that loggers and writers are properly closed when the object is deleted.
        """
        self.close()