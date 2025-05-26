import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from model_util import init_weights

class Mlp(nn.Module):
    """
    A simple feedforward multilayer perceptron (MLP) used within transformer blocks.

    Consists of two linear layers with an activation function in between, and optional dropout.

    Args:
        in_features (int): Dimensionality of input features.
        hidden_features (int, optional): Number of hidden units. If None, defaults to in_features.
        out_features (int, optional): Dimensionality of output features. If None, defaults to in_features.
        act_layer (nn.Module, optional): Activation function. Default: nn.GELU.
        drop (float, optional): Dropout probability applied after each linear layer. Default: 0.0.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        # Set default hidden and output features if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # First linear transformation
        self.fc1 = nn.Linear(in_features, hidden_features) 
        # Non-linear activation
        self.act = act_layer()
        # Second linear transformation
        self.fc2 = nn.Linear(hidden_features, out_features) 
        # Dropout for regularization
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Partition an input image into non-overlapping local windows.

    Args:
        x (Tensor): Input tensor of shape (B, H, W, C)
        window_size (int): Size of the square window

    Returns:
        Tensor: Windows of shape (num_windows * B, window_size, window_size, C)
    """
    B, H, W, C = x.shape # Batch size, Height, Width, Channels

    assert H% window_size == 0 and W% window_size == 0, "H and W must be divisible by window_size"

    # Reshape to separate windows along height and width
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # Permute dimensions to bring window elements together
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Merge windows into batch dimension
    windows = x.view(-1, window_size, window_size, C)

    return windows

def window_reverse(windows, window_size, H, W):
    """
    Reconstruct the original image from non-overlapping windows.

    Args:
        windows (Tensor): Windows tensor of shape (num_windows * B, window_size, window_size, C)
        window_size (int): Size of the square window
        H (int): Original image height
        W (int): Original image width

    Returns:
        Tensor: Reconstructed image tensor of shape (B, H, W, C)
    """
    # Calculate original batch size
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    # Reshape to grid of windows
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)

    # Reorder and reconstruct the image
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()

    # Merge windows back into full image
    x = x.view(B, H, W, -1)

    return x

class WindowAttention(nn.Module):
    """
    Window-based multi-head self-attention (W-MSA) module with continuous relative position bias.

    Implements scaled cosine attention and supports shifted windows (SW-MSA).

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Height and width of local attention window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, enables learnable bias for Q and V. Default: True.
        attn_drop (float, optional): Dropout rate applied to attention weights. Default: 0.0.
        proj_drop (float, optional): Dropout rate after output projection. Default: 0.0.
        pretrained_window_size (tuple[int]): Window size used during pretraining for CRPB normalization.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0.0, proj_drop=0.0, pretrained_window_size=[0,0]):
        super().__init__()
        self.dim = dim # Number of channels in input features
        self.window_size = window_size # Window size (height,width)
        self.pretrained_window_size = to_2tuple(pretrained_window_size) # For pre-trained relative bias
        self.num_heads = num_heads # Number of attention heads

        # Learnable logit scaling per head, initialized as log(10)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True) 

        # MLP for continuous relative position bias (CRPB)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )

        # Coordinate grid for relative position (height and width offsets)
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)

        # Create 2D grid and normalize
        relative_coords_table = torch.stack(
            torch.meshgrid(relative_coords_h, relative_coords_w, indexing='ij'), dim=-1
        ).unsqueeze(0)

        # Normalize coordinates by pretrained or current window size
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)

        relative_coords_table *= 8  # Scale to roughly [-8, 8]

        # Non-linear log scaling to emphasize small distances
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0)

        # Normalize to [-1, 1]
        relative_coords_table /= np.log2(8)

        # Register buffer so it moves with model but is not trained
        self.register_buffer("relative_coords_table", relative_coords_table)

        # Compute pair-wise relative position index for tokens inside window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])

        # Compute scalar indices for all relative positions
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)

        relative_coords[:, :, 0] += self.window_size[0] - 1 # Shift to positive
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Flatten to scalar indices
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV projections with optional bias
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop) # Dropout on attention weights
        self.proj = nn.Linear(dim, dim) # Output projection
        self.proj_drop = nn.Dropout(proj_drop) # Dropout after projection
        self.softmax = nn.Softmax(dim=-1) # Softmax for attention scores

    def forward(self, x, mask=None):
        B_, N, C = x.shape

        # QKV Projection with selective bias
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias
            ))

        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Cosine attention with learnable logit scaling
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        # Continuous relative position bias (CRPB)
        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1
        )  # (Wh*Ww, Wh*Ww, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, Wh*Ww, Wh*Ww)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)

        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply mask if provided (for shifted windows)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Attention output projection
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'
    
    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
class SwinTransformerBlock(nn.Module):
    """ 
    Swin Transformer Block.

    This block applies window-based multi-head self-attention (W-MSA) or shifted window-based MSA (SW-MSA),
    followed by a feed-forward network (MLP), with residual connections and LayerNorm.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Spatial resolution of the input feature map (H, W).
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        shift_size (int): Shift size for SW-MSA (used to enable cross-window connections).
        mlp_ratio (float): Ratio of hidden dimension to embedding dimension in MLP.
        qkv_bias (bool, optional): If True, adds learnable bias to query, key, and value. Default: True.
        drop (float, optional): Dropout rate after linear projections in MLP. Default: 0.0.
        attn_drop (float, optional): Dropout rate within the attention mechanism. Default: 0.0.
        drop_path (float, optional): Drop path (stochastic depth) rate. Default: 0.0.
        act_layer (nn.Module, optional): Activation function. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
        pretrained_window_size (int): Window size used during pretraining (for relative positional encoding).
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=0.0,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (H, W)
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            # If the input is smaller than the window size, don't apply window partitioning
            self.window_size = min(self.input_resolution)
            self.shift_size = 0
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        self.norm1 = norm_layer(dim) # First normalisation layer before attention

        # Window-based Self Attention
        self.attn = WindowAttention(
            dim=dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size)
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity() # DropPath is stochastic depth (acts like residual dropout)

        self.norm2 = norm_layer(dim) # Second normalisation layer before MLP
        
        # MLP Block
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Attention mask for shifted windows (prevents attention across window boundaries)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 x H x W x 1

            # Define slices for 3 regions along each dimension
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))

            # Assign unique values to each region to build attention mask
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # Partition the mask into windows and compute attention mask
            mask_windows = window_partition(img_mask, self.window_size)  # (num_windows, ws, ws, 1)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        # Register attention mask as buffe (not a parameter, but saved in state_dict)
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "Input feature has wrong size"

        shortcut = x # Residual Connection
        x = self.norm1(x)
        x = x.view(B, H, W, C) # Normalise and reshape to 2D spatial format

        # Apply cyclic shift if shift_size > 0
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition input into non-overlapping windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # Window-based multi-head self-attention with optional mask
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # Merge windows back to the full feature map
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # (B, H, W, C)

        # Reverse cyclic shift to restore original positions
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C) # Flatten back

        # Add residual connection and apply drop path
        x = shortcut + self.drop_path(x)

        # MLP Block with second residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / (self.window_size * self.window_size)
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * int(self.dim * self.mlp_ratio)
        # norm2
        flops += self.dim * H * W
        return flops
    
class PatchMerging(nn.Module):
    """ 
    Patch Merging Layer.

    This layer downsamples the spatial resolution by a factor of 2 and increases the channel dimension.

    Args:
        input_resolution (tuple[int]): Spatial resolution of the input feature map (H, W).
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer applied after linear projection. Default: nn.LayerNorm.
    """
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # Tuple of (H, W)
        self.dim = dim  # Input channel dimension

        # Linear projection from concatenated 4 neighboring patches (4C) to 2C
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

        # Normalization after linear projection
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        # Extract non-overlapping 2x2 patches: top-left, bottom-left, top-right, bottom-right
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        # Concatenate along channel dimension → B H/2 W/2 4C
        x = torch.cat([x0, x1, x2, x3], -1)

        # Flatten spatial dimensions → B H/2*W/2 4C
        x = x.view(B, -1, 4 * C)

        # Apply linear projection to reduce dimensionality → 2C
        x = self.reduction(x)

        # Apply layer normalization
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        num_patches = (H // 2) * (W // 2)
        flops_linear = num_patches * (4 * self.dim) * (2 * self.dim)
        flops_norm = num_patches * (2 * self.dim)

        total_flops = flops_linear + flops_norm
        return total_flops

class BasicLayer(nn.Module):
    """ 
    A basic Swin Transformer layer for one stage of the network.

    This module stacks several Swin Transformer blocks with optional downsampling via PatchMerging.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Spatial resolution of the input feature map (H, W).
        depth (int): Number of Swin Transformer blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local attention window size.
        mlp_ratio (float): Ratio of hidden dimension to embedding dimension in MLP. Default: 4.0.
        qkv_bias (bool, optional): If True, adds learnable bias to query, key, and value. Default: True.
        drop (float, optional): Dropout rate after linear projections in MLP. Default: 0.0.
        attn_drop (float, optional): Dropout rate within the attention mechanism. Default: 0.0.
        drop_path (float | list[float], optional): Stochastic depth rate or list of rates. Default: 0.0.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm.
        downsample (nn.Module | None, optional): Optional downsampling layer (e.g., PatchMerging). Default: None.
        use_checkpoint (bool): If True, uses gradient checkpointing for memory efficiency. Default: False.
        pretrained_window_size (int): Window size used during pretraining (for relative positional encoding).
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0.0, attn_drop=0.0,
                 drop_path=0.0, norm_layer=nn.LayerNorm, downsample=None, 
                 use_checkpoint=False, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Ensure drop_path is a list of length = depth
        if isinstance(drop_path, list):
            if len(drop_path) != depth:
                raise ValueError(f"drop_path list length ({len(drop_path)}) does not match depth ({depth})")
        else:
            drop_path = [drop_path] * depth

        # Build a sequence of SwinTransformerBlocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,  # Alternate shift for SW-MSA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                pretrained_window_size=pretrained_window_size
            )
            for i in range(depth)
        ])

        # PatchMerging Layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        # Pass input through each Swin Transformer block, optionally using gradient checkpointing to save memory
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x) 

        # Apply downsampling layer if it exists
        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, "
            f"input_resolution={self.input_resolution}, "
            f"depth={self.depth}"
        )

    def flops(self) -> int:
        total_flops = 0
        for block in self.blocks:
            total_flops += block.flops()
        if self.downsample is not None:
            total_flops += self.downsample.flops()
        return total_flops

    def _init_respostnorm(self):
        for block in self.blocks:
            # Check if block has norm1 and norm2 attributes to avoid attribute errors
            if hasattr(block, 'norm1'):
                nn.init.constant_(block.norm1.bias, 0)
                nn.init.constant_(block.norm1.weight, 0)
            if hasattr(block, 'norm2'):
                nn.init.constant_(block.norm2.bias, 0)
                nn.init.constant_(block.norm2.weight, 0)

class PatchEmbed(nn.Module):
    """ 
    Converts input images into patch embeddings using a convolutional projection.

    Args:
        img_size (int or tuple): Input image spatial size. Default: 224.
        patch_size (int or tuple): Size of each image patch. Default: 4.
        in_chans (int): Number of input channels (e.g., 1 for grayscale). Default: 1.
        embed_dim (int): Dimensionality of patch embeddings. Default: 96.
        norm_layer (nn.Module, optional): Optional normalization layer applied after embedding. Default: None.
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None):
        super().__init__()

        # Convert img_size and patch_size to 2D tuples for height and width
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # Compute resolution after patching (number of patches along H and W)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        # Store number of input channels and embedding dimension per patch
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # Convolutional projection layer slices image into non-overlapping patches,
        # each patch embedded into embed_dim-dimensional space
        self.proj = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

        # Optional normalization layer applied after patch embedding
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        B, C, H, W = x.shape

        # Assert input spatial dimensions match expected image size
        assert H == self.img_size[0] and W == self.img_size[1], (
            f"Input size ({H}x{W}) does not match expected ({self.img_size[0]}x{self.img_size[1]})."
        )

        # Apply patch embedding convolution
        # Output shape: [B, embed_dim, H/patch_size, W/patch_size]
        x = self.proj(x)

        # Flatten spatial dimensions (H', W') into one dimension and transpose
        # to shape: [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Apply normalization layer if provided
        if self.norm is not None:
            x = self.norm(x)

        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        patch_area = self.patch_size[0] * self.patch_size[1]

        # FLOPs for convolution:
        # - Each output element does (in_chans * patch_area) multiplications,
        #   each counted as 2 operations (multiply + add)
        conv_flops = 2 * Ho * Wo * self.embed_dim * self.in_chans * patch_area

        # FLOPs for bias addition: one add per output element
        bias_flops = Ho * Wo * self.embed_dim

        flops = conv_flops + bias_flops

        # Approximate FLOPs for LayerNorm (mean, variance, normalization)
        if self.norm is not None:
            norm_flops = Ho * Wo * 4 * self.embed_dim
            flops += norm_flops

        return flops

class PatchUnmerging(nn.Module):
    """
    Reverses patch merging to upsample feature maps by rearranging and projecting tokens.

    Args:
        output_resolution (tuple of int): Target spatial resolution (H, W) after unmerging.
        dim (int): Number of input channels per token. Must be divisible by 2.
        norm_layer (nn.Module, optional): Normalization layer to apply after unmerging. Default: nn.LayerNorm.
    """
    def __init__(self, output_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.output_resolution = output_resolution
        self.input_dim = dim
        self.output_dim = dim // 2
        assert dim % 2 == 0, "Input dim must be divisible by 2"
        
        self.expand = nn.Linear(dim, 4 * self.output_dim, bias=False)
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        H, W = self.output_resolution
        B, L, C = x.shape

        assert C == self.input_dim, "Channel dimension mismatch"
        assert L == (H // 2) * (W // 2), "Input token length does not match expected spatial size"

        # Expand channels: (B, L, 2C) → (B, L, 4C)
        x = self.expand(x)

        # Reshape to (B, H//2, W//2, 4*C)
        x = x.view(B, H // 2, W // 2, 4 * self.output_dim)

        # Split along channel dimension into 4 spatial patches
        x0, x1, x2, x3 = x.chunk(4, dim=-1)  # Each is (B, H//2, W//2, C)

        # Initialize upsampled output grid: (B, H, W, C)
        out = torch.zeros(B, H, W, self.output_dim, device=x.device, dtype=x.dtype)

        # Fill in 2x2 blocks (reversing patch merging)
        out[:, 0::2, 0::2, :] = x0  # top-left
        out[:, 1::2, 0::2, :] = x1  # bottom-left
        out[:, 0::2, 1::2, :] = x2  # top-right
        out[:, 1::2, 1::2, :] = x3  # bottom-right

        # Flatten spatial dimensions and normalize: (B, H*W, C)
        out = out.view(B, H * W, self.output_dim)
        out = self.norm(out)

        return out

    def extra_repr(self):
        return (
            f"output_resolution={self.output_resolution}, "
            f"input_dim={self.input_dim}, "
            f"output_dim={self.output_dim}"
        )

    def flops(self):
        # FLOPs for the linear projection: 2 * tokens * input_dim * output_dim_expanded
        H, W = self.output_resolution
        num_tokens = (H // 2) * (W // 2)
        # Expand output channels: 4 * output_dim
        expanded_dim = 4 * self.output_dim
        # Multiply-add counts = 2 * multiply + add operations
        flops_expand = 2 * num_tokens * self.input_dim * expanded_dim
        
        # Normalization FLOPs (LayerNorm approx 5 * tokens * channels)
        flops_norm = 5 * H * W * self.output_dim
        
        return flops_expand + flops_norm

class PatchUnEmbed(nn.Module):
    """
    Projects token embeddings back into pixel space by reshaping and linearly projecting patches.

    Args:
        img_size (int or tuple): Original input image size. Default: 224.
        patch_size (int or tuple): Spatial size of each patch. Default: 4.
        embed_dim (int): Dimensionality of token embeddings. Default: 96.
        out_chans (int): Number of output image channels (e.g., 1 for grayscale). Default: 1.
        norm_layer (nn.Module or None): Optional normalization layer before projection. Default: None.
    """
    def __init__(self, img_size=224, patch_size=4, embed_dim=96, out_chans=1, norm_layer=None):
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)  # Store as tuple
        self.embed_dim = embed_dim
        self.out_chans = out_chans
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # Calculate total elements per patch
        patch_elems = self.patch_size[0] * self.patch_size[1]
        self.proj = nn.Linear(embed_dim, out_chans * patch_elems)

    def forward(self, x, patch_resolution):
        B, N, C = x.shape
        H, W = patch_resolution
        assert N == H * W, f"Expected {H * W} tokens, got {N}"

        x = self.norm(x)
        x = self.proj(x)  # [B, N, out_chans * ph * pw]
        
        # Get patch dimensions as separate integers
        ph, pw = self.patch_size[0], self.patch_size[1]
        
        # Reshape with explicit integers
        x = x.view(B, H, W, self.out_chans, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # [B, C, H, ph, W, pw]
        x = x.view(B, self.out_chans, H * ph, W * pw)
        return x

    def flops(self):
        H, W = self.img_size
        Ph, Pw = self.patch_size
        N = (H // Ph) * (W // Pw)
        return 2 * N * self.embed_dim * (self.out_chans * Ph * Pw)

class BasicLayerUp(nn.Module):
    """
    A Swin Transformer layer block followed by optional upsampling and skip connection integration.

    Args:
        dim (int): Number of feature channels.
        input_resolution (tuple[int, int]): Spatial resolution of input features (H, W).
        output_resolution (tuple[int, int]): Target resolution after upsampling (H, W).
        depth (int): Number of Swin Transformer blocks.
        num_heads (int): Number of attention heads in each block.
        window_size (int): Local attention window size.
        mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension. Default: 4.0.
        qkv_bias (bool): Whether to use bias in QKV projections. Default: True.
        drop (float): Dropout probability after each block. Default: 0.0.
        attn_drop (float): Dropout rate on attention weights. Default: 0.0.
        drop_path (float or list[float]): Stochastic depth drop rate. Default: None.
        norm_layer (nn.Module): Normalization layer used throughout. Default: nn.LayerNorm.
        upsample (nn.Module or None): Optional upsampling module (e.g., PatchUnmerging). Default: None.
        use_checkpoint (bool): Enable gradient checkpointing for memory efficiency. Default: False.
        skip_channels (int or None): Optional number of channels from skip connections. Default: None.
    """
    def __init__(self, dim, input_resolution, output_resolution, depth, num_heads, window_size, 
                 mlp_ratio=4.0, qkv_bias=True, drop=0.0, attn_drop=0.0, drop_path=None, 
                 norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False, skip_channels=None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        if isinstance(drop_path, list):
            assert len(drop_path) == depth
        else:
            drop_path = [drop_path] * depth

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
            )
            for i in range(depth)
        ])

        self.upsample = upsample(output_resolution, dim, norm_layer=norm_layer) if upsample else None

        # Modified skip connection handling
        if skip_channels is not None:
            # Project skip to match current dim
            self.skip_proj = nn.Sequential(
                nn.Linear(skip_channels, dim),
                norm_layer(dim)
            )
            # Project concatenated features (dim + dim) to dim
            self.concat_proj = nn.Sequential(
                nn.Linear(2 * dim, dim),
                norm_layer(dim)
            )
        else:
            self.skip_proj = None
            self.concat_proj = None

    def forward(self, x, skip=None):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)

        if self.upsample is not None:
            x = self.upsample(x)

        if skip is not None and self.skip_proj is not None:
            # Project skip to correct dimension
            skip = self.skip_proj(skip)
            
            # Ensure spatial dimensions match
            B, L_x, C_x = x.shape
            B, L_skip, C_skip = skip.shape
            
            if L_skip != L_x:
                # Calculate spatial dimensions
                H_skip = int(L_skip**0.5)
                W_skip = H_skip
                H_x = int(L_x**0.5)
                W_x = H_x
                
                # Reshape and interpolate
                skip = skip.transpose(1, 2).view(B, C_skip, H_skip, W_skip)
                skip = F.interpolate(skip, size=(H_x, W_x), mode='bilinear', align_corners=False)
                skip = skip.view(B, C_skip, -1).transpose(1, 2)
            
            # Concatenate and project
            x = torch.cat([x, skip], dim=-1)
            x = self.concat_proj(x)

        return x

    def extra_repr(self) -> str:
        """String representation for debugging."""
        return (
            f"dim={self.dim}, "
            f"input_resolution={self.input_resolution}, "
            f"output_resolution={self.output_resolution}, "
            f"depth={self.depth}"
        )

    def flops(self) -> int:
        total_flops = 0
        for block in self.blocks:
            total_flops += block.flops()
        if self.upsample is not None and hasattr(self.upsample, "flops"):
            total_flops += self.upsample.flops()
        return total_flops

    def _init_respostnorm(self):
        for block in self.blocks:
            if hasattr(block, 'norm1'):
                nn.init.constant_(block.norm1.bias, 0)
                nn.init.constant_(block.norm1.weight, 0)
            if hasattr(block, 'norm2'):
                nn.init.constant_(block.norm2.bias, 0)
                nn.init.constant_(block.norm2.weight, 0)

class SwinUNetGenerator(nn.Module):
    """
    Swin Transformer based U-Net generator.

    Args:
        img_size (int): Input image size (assumed square).
        patch_size (int): Size of patches for patch embedding.
        in_chans (int): Number of input image channels.
        out_chans (int): Number of output image channels.
        embed_dim (int): Embedding dimension for the transformer.
        depths (list[int]): Number of transformer blocks at each stage.
        num_heads (list[int]): Number of attention heads per stage.
        window_size (int): Window size for local attention.
        mlp_ratio (float): Expansion ratio for MLP hidden dim.
        qkv_bias (bool): Whether to use bias in QKV projections.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate.
        drop_path_rate (float): Stochastic depth rate.
        norm_layer (nn.Module): Normalization layer type.
        ape (bool): Use absolute positional embedding.
        patch_norm (bool): Use normalization after patch embedding.
        use_checkpoint (bool): Use checkpointing for memory savings.
        **kwargs: Additional arguments.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=1, out_chans=1, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7, mlp_ratio=4.0, qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False, **kwargs,):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm

        # Patch embedding: splits input image into patches and embeds them
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        self.patches_resolution = self.patch_embed.patches_resolution
        num_patches = self.patch_embed.num_patches

        # Absolute positional embedding parameter initialization if used
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rates linearly spaced over all blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder layers: each layer includes multiple transformer blocks and optional downsampling
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i),
                input_resolution=(
                    self.patches_resolution[0] // (2 ** i),
                    self.patches_resolution[1] // (2 ** i),
                ),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if i < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        # Decoder layers with upsampling, mirror the encoder in reverse order
        self.layers_up = nn.ModuleList()
        for i in reversed(range(self.num_layers - 1)):
            input_res = (
                self.patches_resolution[0] // (2 ** (i + 1)),
                self.patches_resolution[1] // (2 ** (i + 1)),
            )
            output_res = (
                self.patches_resolution[0] // (2 ** i),
                self.patches_resolution[1] // (2 ** i),
            )
            layer_up = BasicLayerUp(
                dim=int(embed_dim * 2 ** (i + 1)),
                input_resolution=input_res,
                output_resolution=output_res,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                norm_layer=norm_layer,
                upsample=PatchUnmerging,
                use_checkpoint=use_checkpoint,
            )
            self.layers_up.append(layer_up)

        # Normalization before patch unembedding to reconstruct the image
        self.norm_up = norm_layer(embed_dim)

        # Patch unembedding layer to reconstruct image from transformer tokens
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        # Initialize weights using model_util's function
        init_weights(self, init_type='normal', init_gain=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Embed input image patches
        x = self.patch_embed(x)

        # Add absolute positional embedding if enabled
        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        # Encoder forward pass: extract features at each stage for skip connections
        enc_features = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            enc_features.append(x)

        # Decoder forward pass with skip connections
        for i, layer_up in enumerate(self.layers_up):
            skip = enc_features[self.num_layers - 2 - i]  # Corresponding encoder features
            x = layer_up(x, skip)

        # Normalize before patch unembedding
        x = self.norm_up(x)

        # Validate token length matches expected patches resolution
        B, L, C = x.shape
        H, W = self.patches_resolution
        assert L == H * W, f"Token length {L} does not match patches resolution {H}x{W}"

        # Reconstruct image from patches
        x = self.patch_unembed(x, (H, W))

        return x

    def flops(self):
        flops = 0
        
        # Add patch embedding FLOPs
        if hasattr(self.patch_embed, 'flops'):
            flops += self.patch_embed.flops()
        
        # Add encoder layers FLOPs
        for layer in self.layers:
            if hasattr(layer, 'flops'):
                flops += layer.flops()
        
        # Add decoder layers FLOPs
        for layer_up in self.layers_up:
            if hasattr(layer_up, 'flops'):
                flops += layer_up.flops()
        
        # Add patch unembedding FLOPs
        if hasattr(self.patch_unembed, 'flops'):
            flops += self.patch_unembed.flops()
        
        return int(flops)  # Explicitly cast to integer