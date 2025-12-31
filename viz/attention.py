from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class VitAttentionRolloutConfig:
    """
    Configuration for attention rollout.

    Parameters
    ----------
    head_fusion : {"mean", "max"}, optional
        Strategy for aggregating attention across heads. Default ``"mean"``.
    add_identity : bool, optional
        If True, add identity to each attention matrix and renormalize.
        This follows the common practice from ViT interpretability work.
        Default ``True``.
    """

    head_fusion: Literal["mean", "max"] = "mean"
    add_identity: bool = True


class VitAttentionRollout:
    """
    Compute attention rollout over a list of attention matrices.

    Parameters
    ----------
    config : VitAttentionRolloutConfig, optional
        Rollout configuration.
    """

    def __init__(self, config: Optional[VitAttentionRolloutConfig] = None) -> None:
        self.config = config or VitAttentionRolloutConfig()

    def _fuse_heads(self, attn: Tensor) -> Tensor:
        """
        Fuse attention heads.

        Parameters
        ----------
        attn : torch.Tensor
            Attention tensor of shape (B, H, T, T).

        Returns
        -------
        torch.Tensor
            Fused attention tensor of shape (B, T, T).
        """
        if self.config.head_fusion == "mean":
            return attn.mean(dim=1)
        if self.config.head_fusion == "max":
            return attn.max(dim=1)[0]
        raise ValueError(f"Unsupported head_fusion: {self.config.head_fusion}")

    def __call__(self, attn_list: List[Tensor]) -> Tensor:
        """
        Compute attention rollout.

        Parameters
        ----------
        attn_list : list of torch.Tensor
            List of attention tensors, each of shape (B, H, T, T).

        Returns
        -------
        torch.Tensor
            Rolled-out attention of shape (B, T, T).
        """
        if not attn_list:
            raise ValueError("attn_list is empty in VitAttentionRollout.")

        result: Optional[Tensor] = None

        for attn in attn_list:
            # attn: (B, H, T, T)
            fused = self._fuse_heads(attn)  # (B, T, T)

            if self.config.add_identity:
                B, T, _ = fused.shape
                eye = torch.eye(T, device=fused.device).unsqueeze(0).expand(B, -1, -1)
                fused = fused + eye
                fused = fused / fused.sum(dim=-1, keepdim=True)

            result = fused if result is None else fused @ result

        return result  # (B, T, T)


@dataclass
class VitLastLayerAttentionConfig:
    """
    Configuration for last-layer attention visualization.

    Parameters
    ----------
    head_fusion : {"mean", "max"}, optional
        Strategy for aggregating attention across heads. Default ``"mean"``.
    add_identity : bool, optional
        If True, add identity and renormalize. Default ``False``.
    """

    head_fusion: Literal["mean", "max"] = "mean"
    add_identity: bool = False


class VitLastLayerAttention:
    """
    Use only the last layer CLS-to-patch attention for visualization,
    similar to the original ViT paper.

    Parameters
    ----------
    config : VitLastLayerAttentionConfig, optional
        Configuration for last-layer attention.
    """

    def __init__(
        self,
        config: Optional[VitLastLayerAttentionConfig] = None,
    ) -> None:
        self.config = config or VitLastLayerAttentionConfig()

    def _fuse_heads(self, attn: Tensor) -> Tensor:
        if self.config.head_fusion == "mean":
            return attn.mean(dim=1)
        if self.config.head_fusion == "max":
            return attn.max(dim=1)[0]
        raise ValueError(f"Unsupported head_fusion: {self.config.head_fusion}")

    def __call__(self, attn_list: List[Tensor]) -> Tensor:
        """
        Compute last-layer attention.

        Parameters
        ----------
        attn_list : list of torch.Tensor
            List of attention tensors, each of shape (B, H, T, T).

        Returns
        -------
        torch.Tensor
            Last-layer attention of shape (B, T, T).
        """
        if not attn_list:
            raise ValueError("attn_list is empty in VitLastLayerAttention.")

        last = attn_list[-1]  # (B, H, T, T)
        fused = self._fuse_heads(last)  # (B, T, T)

        if self.config.add_identity:
            B, T, _ = fused.shape
            eye = torch.eye(T, device=fused.device).unsqueeze(0).expand(B, -1, -1)
            fused = fused + eye
            fused = fused / fused.sum(dim=-1, keepdim=True)

        return fused


def class_token_attention_to_grid(
    attn: Tensor,
    token_index: int,
    grid_size: int,
) -> Tensor:
    """
    Convert class-token attention to a spatial grid.

    Parameters
    ----------
    attn : torch.Tensor
        Attention tensor of shape (B, T, T) where T = 1 + H*W.
    token_index : int
        Index of the class token (usually 0).
    grid_size : int
        Spatial grid size (H == W == grid_size).

    Returns
    -------
    torch.Tensor
        Attention grid of shape (B, 1, H, W).
    """
    B, T, _ = attn.shape
    num_patches = T - 1
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"grid_size^2 ({grid_size ** 2}) != num_patches ({num_patches}). "
            "Check the input attention or grid_size."
        )

    cls_attn = attn[:, token_index, 1:]  # (B, num_patches)
    grid = cls_attn.reshape(B, 1, grid_size, grid_size)  # (B, 1, H, W)

    # normalize each map to [0, 1]
    B, C, H, W = grid.shape
    grid_flat = grid.view(B, -1)
    min_vals = grid_flat.min(dim=1, keepdim=True)[0]
    max_vals = grid_flat.max(dim=1, keepdim=True)[0]
    denom = (max_vals - min_vals).clamp(min=1e-8)
    grid_norm = (grid_flat - min_vals) / denom
    grid_norm = grid_norm.view(B, C, H, W)

    return grid_norm


def upsample_attention_to_image(
    attn_grid: Tensor,
    image_size: Tuple[int, int],
) -> Tensor:
    """
    Upsample an attention grid to image size.

    Parameters
    ----------
    attn_grid : torch.Tensor
        Attention grid of shape (B, 1, H, W).
    image_size : tuple of int
        Target image size (height, width).

    Returns
    -------
    torch.Tensor
        Upsampled attention heatmap of shape (B, 1, H_img, W_img).
    """
    attn_upsampled = F.interpolate(
        attn_grid,
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return attn_upsampled


def to_numpy(t: Tensor) -> np.ndarray:
    """
    Convert a tensor to a NumPy array.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor.

    Returns
    -------
    np.ndarray
        NumPy array.
    """
    return t.detach().cpu().numpy()
