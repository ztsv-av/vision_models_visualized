from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from pathlib import Path
from typing import List, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torch import nn
from torchvision import datasets

from data.datasets import ImageNetV2IndexFolder
from models.models import get_preprocess
from utils.labels import get_labels


ModelFamily = Literal["convnext", "vit", "mixer"]


def infer_model_family(model_id: str) -> ModelFamily:
    """
    Infer high-level model family from identifier.

    Parameters
    ----------
    model_id : str
        Identifier such as "convnext_t", "convnext_b",
        "deit_s", "deit_b", "mlpmixer_b16", etc.

    Returns
    -------
    {"convnext", "vit", "mixer"}
        Model family name.

    Raises
    ------
    ValueError
        If the family cannot be inferred.
    """
    mid = model_id.lower()
    if mid.startswith("convnext"):
        return "convnext"
    if mid.startswith("deit") or mid.startswith("vit"):
        return "vit"
    if mid.startswith("mlpmixer") or mid.startswith("mixer"):
        return "mixer"
    raise ValueError(f"Cannot infer model family from id '{model_id}'.")


def load_raw_dataset(
    dataset_name: str,
    dataset_root: str,
    split: str,
):
    """
    Load a raw dataset (without transforms) to access original images.

    Parameters
    ----------
    dataset_name : str
        Dataset identifier: "imagenetv2" or "imagefolder".
    dataset_root : str
        Root directory for the dataset.
    split : str
        Split name. For "imagenetv2" this is ignored, for "imagefolder"
        it is the directory passed to :class:`ImageFolder`.

    Returns
    -------
    torch.utils.data.Dataset
        Dataset returning (PIL.Image.Image, label).
    """
    root = Path(dataset_root).expanduser()

    if dataset_name == "imagenetv2":
        ds = ImageNetV2IndexFolder(
            root=str(root),
            transform=None,
        )
    elif dataset_name == "imagefolder":
        ds = datasets.ImageFolder(
            root=str(root),
            transform=None,
        )
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    return ds


def get_image_and_gt(
    dataset_name: str,
    dataset,
    index: int,
) -> Tuple[Image.Image, int, str]:
    """
    Fetch a sample and map its label to an ImageNet class index.

    For ``imagenetv2`` the labels are already 0..999.
    For ``imagefolder`` we assume subdirectory names are ImageNet
    class indices as strings (e.g. "9", "22", "207", "250").

    Parameters
    ----------
    dataset_name : str
        Dataset identifier.
    dataset : Dataset
        Raw dataset.
    index : int
        Sample index.

    Returns
    -------
    pil_image : PIL.Image.Image
        Image for the sample.
    gt_idx : int
        Ground-truth ImageNet class index.
    gt_name : str
        Human-readable label from ``utils.labels``.
    """
    pil_image, label = dataset[index]
    local_idx = int(label)

    if dataset_name == "imagefolder" and hasattr(dataset, "class_to_idx"):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        imagenet_idx_str = idx_to_class[local_idx]
        gt_idx = int(imagenet_idx_str)
    else:
        gt_idx = local_idx

    gt_name = get_labels(gt_idx)
    return pil_image, gt_idx, gt_name


def prepare_input_for_model(
    model_id: str,
    pil_image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """
    Apply model-specific preprocessing to a PIL image.

    Parameters
    ----------
    model_id : str
        Model identifier.
    pil_image : PIL.Image.Image
        Raw input image.
    device : torch.device
        Target device.

    Returns
    -------
    torch.Tensor
        Input batch tensor of shape (1, C, H, W).
    """
    transform = get_preprocess(model_id=model_id, is_training=False)
    tensor = transform(pil_image).unsqueeze(0).to(device)
    return tensor


def get_convnext_target_layer(model: nn.Module) -> nn.Module:
    """
    Select a ConvNeXt feature layer for Grad-CAM.

    Parameters
    ----------
    model : torch.nn.Module
        ConvNeXt model.

    Returns
    -------
    torch.nn.Module
        Target layer with spatial feature maps (N, C, H, W).

    Raises
    ------
    ValueError
        If the layer cannot be found.
    """
    if hasattr(model, "stages"):
        return model.stages[-1]
    raise ValueError("ConvNeXt target layer heuristic failed: no 'stages' attribute.")


def extract_vit_self_attention(
    model: nn.Module,
    inputs: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Extract self-attention matrices from a ViT/DeiT model via hooks.

    Parameters
    ----------
    model : torch.nn.Module
        Vision Transformer model.
    inputs : torch.Tensor
        Input batch of shape (N, C, H, W).

    Returns
    -------
    list of torch.Tensor
        List of attention tensors, one per layer, each with shape
        (B, num_heads, T, T).
    """
    attn_list: List[torch.Tensor] = []
    hooks = []

    def make_hook():
        def hook(module, module_input, module_output):
            x = module_input[0]
            B, N, C = x.shape
            num_heads = module.num_heads
            head_dim = C // num_heads
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
            q, k, _ = qkv[0], qkv[1], qkv[2]
            scale = getattr(module, "scale", head_dim**-0.5)
            attn = (q @ k.transpose(-2, -1)) * scale
            attn = attn.softmax(dim=-1)
            attn_list.append(attn.detach())

        return hook

    if not hasattr(model, "blocks"):
        raise ValueError("ViT/DeiT model has no 'blocks' attribute.")

    for block in model.blocks:
        if hasattr(block, "attn"):
            h = block.attn.register_forward_hook(make_hook())
            hooks.append(h)

    with torch.no_grad():
        _ = model(inputs)

    for h in hooks:
        h.remove()

    if not attn_list:
        raise RuntimeError("No attention matrices captured from ViT/DeiT model.")

    return attn_list


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Overlay a heatmap on top of an RGB image.

    Parameters
    ----------
    image : np.ndarray
        Base RGB image, shape (H, W, 3), float32 in [0, 1].
    heatmap : np.ndarray
        Heatmap, shape (H, W), float32 in [0, 1].
    alpha : float, optional
        Blending factor in [0, 1]. Default: 0.5.

    Returns
    -------
    np.ndarray
        Blended image, shape (H, W, 3), float32 in [0, 1].
    """
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap)[:, :, :3]
    overlay = (1 - alpha) * image + alpha * heatmap_rgb
    overlay = np.clip(overlay, 0.0, 1.0)
    return overlay


def resize_heatmap_to_image(
    image: np.ndarray,
    heatmap: np.ndarray,
) -> np.ndarray:
    """
    Resize a heatmap to match an image spatial resolution.

    Parameters
    ----------
    image : np.ndarray
        RGB image of shape (H, W, 3).
    heatmap : np.ndarray
        Heatmap of shape (h, w).

    Returns
    -------
    np.ndarray
        Resized heatmap of shape (H, W), float32 in [0, 1].
    """
    H, W = image.shape[:2]

    if heatmap.shape == (H, W):
        return heatmap

    hm_resized = (
        np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((W, H), resample=Image.BILINEAR)).astype(
            np.float32
        )
        / 255.0
    )
    return hm_resized
