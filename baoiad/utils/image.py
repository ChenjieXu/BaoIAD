"""Image utility functions."""
import torch
from PIL import Image


def save_tensor_image(tensor: torch.Tensor, path: str) -> None:
    """Save a CHW or HW tensor as an image file.

    Equivalent to torchvision.utils.save_image for a single image.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3:
        # CHW -> HWC
        arr = tensor.detach().cpu().clamp(0, 1).mul(255).byte().permute(1, 2, 0).numpy()
    elif tensor.dim() == 2:
        arr = tensor.detach().cpu().clamp(0, 1).mul(255).byte().numpy()
    else:
        raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

    Image.fromarray(arr).save(path)
