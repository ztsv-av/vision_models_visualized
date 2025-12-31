from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class GradCAMConfig:
    """
    Configuration for Grad-CAM computation.

    Parameters
    ----------
    normalize : bool, optional
        Whether to normalize heatmaps to [0, 1]. Default ``True``.
    """

    normalize: bool = True


class GradCAM:
    """
    Grad-CAM implementation for visualizing model attention.

    Parameters
    ----------
    model : torch.nn.Module
        Model to analyze. It should output class logits.
    target_layer : torch.nn.Module
        Layer whose activations are used for Grad-CAM.
        This should have a spatial output (e.g. ``(N, C, H, W)``).
    device : str or torch.device, optional
        Device used for inference and gradient computation.
    config : GradCAMConfig, optional
        Configuration for Grad-CAM behavior.

    Notes
    -----
    Grad-CAM is computed as:

    .. math::

        L^c_{GradCAM} = ReLU\\left( \\sum_k w_k^c A^k \\right),

    where :math:`A^k` are feature maps and :math:`w_k^c` are channel-wise
    weights obtained by global average pooling of gradients.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: torch.device | str,
        config: Optional[GradCAMConfig] = None,
    ) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = torch.device(device)
        self.config = config or GradCAMConfig()

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_hook = self.target_layer.register_forward_hook(self._forward_hook_fn)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook_fn)

    def _forward_hook_fn(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """
        Store activations from the target layer during the forward pass.
        """
        self.activations = output.detach()

    def _backward_hook_fn(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """
        Store gradients from the target layer during the backward pass.
        """
        self.gradients = grad_output[0].detach()

    def __call__(
        self,
        inputs: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Grad-CAM heatmaps for a batch of images.

        Parameters
        ----------
        inputs : torch.Tensor
            Input images of shape ``(N, C, H, W)``.
        target_class : int or None, optional
            Class index to compute Grad-CAM for.
            If ``None``, uses the predicted class for each sample.

        Returns
        -------
        torch.Tensor
            Heatmaps of shape ``(N, 1, H, W)`` in the input resolution.
            Values are in [0, 1] if ``normalize=True`` in config.
        """
        self.model.zero_grad()
        inputs = inputs.to(self.device)

        # Forward
        logits = self.model(inputs)  # shape (N, C)
        if target_class is None:
            target_indices = logits.argmax(dim=1)
        else:
            target_indices = torch.full(
                (logits.size(0),),
                fill_value=target_class,
                dtype=torch.long,
                device=logits.device,
            )

        # Clear any stale gradients
        if self.activations is None:
            raise RuntimeError(
                "Target layer activations not captured. " "Check that the target_layer is used in forward()."
            )

        # Backward for each sample
        loss = logits.gather(1, target_indices.unsqueeze(1)).sum()
        loss.backward(retain_graph=True)

        if self.gradients is None:
            raise RuntimeError(
                "Target layer gradients not captured. " "Check that the target_layer is used in forward()."
            )

        # activations: (N, C, H, W), gradients: (N, C, H, W)
        activations = self.activations
        gradients = self.gradients

        # Compute channel-wise weights via global average pooling over spatial dims
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (N, 1, H, W)
        cam = torch.relu(cam)

        # Normalize per sample
        if self.config.normalize:
            cam = self._normalize(cam)

        # Resize to input size
        cam = F.interpolate(
            cam,
            size=(inputs.shape[2], inputs.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return cam.detach().cpu()

    @staticmethod
    def _normalize(cam: torch.Tensor) -> torch.Tensor:
        """
        Normalize heatmaps to [0, 1] per sample.

        Parameters
        ----------
        cam : torch.Tensor
            Heatmaps of shape ``(N, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Normalized heatmaps of the same shape.
        """
        N = cam.size(0)
        cam_reshaped = cam.view(N, -1)
        min_vals = cam_reshaped.min(dim=1, keepdim=True)[0]
        max_vals = cam_reshaped.max(dim=1, keepdim=True)[0]
        denom = (max_vals - min_vals).clamp(min=1e-8)
        cam_norm = (cam_reshaped - min_vals) / denom
        return cam_norm.view_as(cam)

    def release(self) -> None:
        """
        Remove hooks and release references.

        This should be called when GradCAM is no longer needed to avoid
        memory leaks.
        """
        self._forward_hook.remove()
        self._backward_hook.remove()
        self.activations = None
        self.gradients = None

    def to_numpy(self, cam: torch.Tensor) -> np.ndarray:
        """
        Convert heatmaps to NumPy array for further processing.

        Parameters
        ----------
        cam : torch.Tensor
            Heatmaps of shape ``(N, 1, H, W)``.

        Returns
        -------
        np.ndarray
            Heatmaps as NumPy array with shape ``(N, H, W)``.
        """
        cam_np = cam.squeeze(1).cpu().numpy()
        return cam_np
