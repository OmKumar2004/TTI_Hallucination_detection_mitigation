import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm

########################################
# Attention Utilities
########################################

# Global variable to store attention maps
_last_cross_attn = []

def save_cross_attn_hook(module, input, output):
    """
    Hook to save cross-attention maps during a forward pass.
    We average over batch and, if applicable, head dimensions.
    """
    global _last_cross_attn
    if hasattr(output, 'shape') and output.dim() >= 3 and output.numel() > 0:
        if output.dim() >= 4:
            # For example, output shape: [B, num_heads, tokens, tokens]
            attn_map = output.detach().cpu().mean(dim=0).mean(dim=0)  # now shape [tokens, tokens]
        else:
            attn_map = output.detach().cpu().mean(dim=0)
        _last_cross_attn.append(attn_map)

def register_attention_hooks(unet_model):
    """
    Register forward hooks on cross-attention layers. 
    Adjust the module name filter ('attn2') as needed.
    """
    global _last_cross_attn
    _last_cross_attn = []  # reset storage
    for name, module in unet_model.named_modules():
        # In many Stable Diffusion implementations, cross-attention is in modules named "attn2".
        if "attn2" in name and isinstance(module, nn.Module):
            if hasattr(module, 'to_q'):
                module.register_forward_hook(save_cross_attn_hook)

def get_last_cross_attention_list():
    """
    Returns a list of the collected attention maps without stacking.
    """
    global _last_cross_attn
    if len(_last_cross_attn) == 0:
        raise ValueError("No cross-attention maps collected. Did you run inference?")
    return _last_cross_attn

def get_last_cross_attention_resized(target_shape: Tuple[int, int]=(32,32)):
    """
    Returns the collected attention maps, resized to target_shape.
    Each map is individually resized using bilinear interpolation.
    """
    attn_list = get_last_cross_attention_list()  # List of tensors
    resized_maps = []
    for attn in attn_list:
        if attn.numel() == 0:
            continue
        if attn.dim() == 2:
            attn = attn.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        elif attn.dim() == 3:
            attn = attn.unsqueeze(0)  # [1, C, H, W]
        attn_resized = F.interpolate(attn.float(), size=target_shape, mode='bilinear', align_corners=False)
        resized_maps.append(attn_resized.squeeze(0))  # remains [C, target_H, target_W]
    if len(resized_maps) == 0:
        raise ValueError("No valid cross-attention maps available for resizing.")
    return torch.stack(resized_maps)  # shape [N_layers, C, target_H, target_W]

########################################
# Custom Hallucination Penalty Loss
########################################

def compute_hallucination_penalty(attention_maps: torch.Tensor, x_t_hat: torch.Tensor) -> torch.Tensor:
    """
    Computes the hallucination penalty loss as follows:
    - attention_maps: a tensor of shape [N_layers, C, H, W] (after resizing)
    - x_t_hat: predicted features (from the UNet output) of shape [B, ..., H, W]
    
    Steps:
      1. Average attention_maps over channels and layers to produce a 2D map A_sum.
      2. Normalize A_sum via clipping between [0,1] and compute its complement:
         M_halluc = 1 - clip(A_sum, 0, 1)
      3. Multiply M_halluc elementwise with x_t_hat and compute an L1 loss.
    
    (You can swap L1 with L2 as desired.)
    """
    # Average attention maps over channel dimension: [N_layers, H, W]
    attn_avg = attention_maps.mean(dim=1)
    # Average over layers to obtain A_sum: [H, W]
    A_sum = attn_avg.mean(dim=0)
    A_sum_clipped = torch.clamp(A_sum, 0.0, 1.0)
    M_halluc = 1.0 - A_sum_clipped  # higher values indicate hallucination regions
    
    # Ensure M_halluc is 4D: [1, 1, H, W]
    if M_halluc.dim() == 2:
        M_halluc = M_halluc.unsqueeze(0).unsqueeze(0)
    
    # Expand M_halluc to match x_t_hat if needed
    if M_halluc.shape[-2:] != x_t_hat.shape[-2:]:
        M_halluc = F.interpolate(M_halluc, size=x_t_hat.shape[-2:], mode='bilinear', align_corners=False)
    
    # Move M_halluc to same device as x_t_hat
    M_halluc = M_halluc.to(x_t_hat.device)
    
    # Compute L1 loss: penalize non-zero activations in hallucinated regions.
    loss = F.l1_loss(M_halluc * x_t_hat, torch.zeros_like(x_t_hat))
    return loss

########################################
# Visualization Helpers (Optional)
########################################

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int]=(0,0,0)):
    h, w, c = image.shape
    offset = int(h * 0.2)
    new_img = np.ones((h+offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (w - textsize[0]) // 2
    text_y = h + offset - textsize[1] // 2
    cv2.putText(new_img, text, (text_x, text_y), font, 1, text_color, 2)
    return new_img

def view_images(images, num_rows=1, offset_ratio=0.02):
    if isinstance(images, list):
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    empty_img = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [img.astype(np.uint8) for img in images] + [empty_img] * num_empty
    num_items = len(images)
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    grid = np.ones((h * num_rows + offset * (num_rows - 1),
                    w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            grid[i*(h+offset):i*(h+offset)+h, j*(w+offset):j*(w+offset)+w] = images[i*num_cols + j]
    display(Image.fromarray(grid))
