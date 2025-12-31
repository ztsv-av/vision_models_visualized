from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


@dataclass
class SaliencyConfig:
    """
    Configuration for gradient-based saliency.

    Parameters
    ----------
    use_absolute : bool, optional
        If True, use absolute value of gradients. Default ``True``.
    channel_aggregation : {"l2", "mean"}, optional
        Strategy for aggregating gradients across channels.
        Default ``"l2"`` (Euclidean norm).
    normalize : bool, optional
        If True, normalize each saliency map to [0, 1]. Default ``True``.
    """

    use_input_gradient: bool = True
    use_absolute: bool = False
    channel_aggregation: str = "l2"
    normalize: bool = True


class Saliency:
    """
    Compute gradient-based saliency maps for a model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to analyze. It should output class logits.
    device : str or torch.device
        Device used for inference and gradient computation.
    config : SaliencyConfig, optional
        Saliency configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        config: Optional[SaliencyConfig] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.config = config or SaliencyConfig()

    def __call__(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute saliency maps for a batch of images.

        Parameters
        ----------
        inputs : torch.Tensor
            Input images of shape ``(N, C, H, W)``.
        target_class : int or None, optional
            Class index for which to compute saliency.
            If ``None``, uses the predicted class for each sample.

        Returns
        -------
        torch.Tensor
            Saliency maps of shape ``(N, 1, H, W)`` with values in [0, 1]
            if normalization is enabled.
        """
        self.model.zero_grad()

        x = inputs.to(self.device)
        x.requires_grad_(True)

        logits = self.model(x)  # (N, C)

        if target_class is None:
            target_indices = logits.argmax(dim=1)
        else:
            target_indices = torch.full(
                (logits.size(0),),
                fill_value=target_class,
                dtype=torch.long,
                device=logits.device,
            )

        selected = logits.gather(1, target_indices.unsqueeze(1)).sum()
        selected.backward()

        grads = x.grad  # (N, C, H, W)
        if grads is None:
            raise RuntimeError("No gradients found on inputs. " "Ensure requires_grad is set correctly.")

        # 1. absolute gradient
        if self.config.use_absolute:
            grads = grads.abs()

        if getattr(self.config, "use_input_gradient", False):
            grads = grads * x

        # 2. Aggregate across channels
        if self.config.channel_aggregation == "l2":
            saliency = torch.sqrt((grads**2).sum(dim=1, keepdim=True))
        elif self.config.channel_aggregation == "mean":
            saliency = grads.mean(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unsupported channel_aggregation: {self.config.channel_aggregation}")

        if self.config.normalize:
            saliency = self._normalize(saliency)

        return saliency.detach().cpu()

    @staticmethod
    def _normalize(sal: torch.Tensor) -> torch.Tensor:
        """
        Normalize saliency maps to [0, 1] per sample, with optional
        percentile clipping and slight smoothing to improve visualization.

        Parameters
        ----------
        sal : torch.Tensor
            Saliency maps of shape (N, 1, H, W).

        Returns
        -------
        torch.Tensor
            Normalized (and lightly smoothed) saliency maps.
        """
        N, C, H, W = sal.shape
        sal_flat = sal.view(N, -1)

        # percentile clipping: keep only top p% of values
        p = 95.0
        k = int((100.0 - p) / 100.0 * sal_flat.size(1))
        if k > 0:
            topk_vals, _ = torch.topk(sal_flat, k=sal_flat.size(1) - k, dim=1)
            thresh = topk_vals.min(dim=1, keepdim=True)[0]
            sal_flat = torch.clamp(sal_flat - thresh, min=0.0)

        # normalize to [0, 1]
        min_vals = sal_flat.min(dim=1, keepdim=True)[0]
        max_vals = sal_flat.max(dim=1, keepdim=True)[0]
        denom = (max_vals - min_vals).clamp(min=1e-8)
        sal_norm = (sal_flat - min_vals) / denom
        sal_norm = sal_norm.view(N, C, H, W)

        # cheap smoothing to reduce checkerboard artifacts
        # 3x3 average filter
        sal_smooth = torch.nn.functional.avg_pool2d(
            sal_norm,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        return sal_smooth

    @staticmethod
    def to_numpy(sal: torch.Tensor) -> np.ndarray:
        """
        Convert saliency maps to NumPy arrays.

        Parameters
        ----------
        sal : torch.Tensor
            Saliency maps of shape ``(N, 1, H, W)``.

        Returns
        -------
        np.ndarray
            Saliency maps as NumPy arrays of shape ``(N, H, W)``.
        """
        return sal.squeeze(1).cpu().numpy()
