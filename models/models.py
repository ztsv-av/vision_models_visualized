from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import nn

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


@dataclass
class ModelSpec:
    """
    Specification for a vision model.

    Parameters
    ----------
    id : str
        Short identifier, e.g. "convnext_t".
    timm_name : str
        timm model name.
    pretrained : bool, optional
        Whether to load pretrained weights.
        Default: ``True``.
    num_classes : int, optional
        Number of classes in the dataset.
        Default: ``1000``.
    label : str | None, optional
        Human-readable name.
        Default ``None``.
    """

    id: str
    timm_name: str
    pretrained: bool = True
    num_classes: int = 1000
    label: str | None = None

    def display_name(self) -> str:
        return self.label or self.id


_MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "convnext_t": ModelSpec(
        id="convnext_t",
        timm_name="convnext_tiny",
        pretrained=True,
        num_classes=1000,
        label="ConvNeXt-Tiny",
    ),
    "convnext_b": ModelSpec(
        id="convnext_b",
        timm_name="convnext_base",
        pretrained=True,
        num_classes=1000,
        label="ConvNeXt-B",
    ),
    "deit_s": ModelSpec(
        id="deit_s",
        timm_name="deit_small_patch16_224",
        pretrained=True,
        num_classes=1000,
        label="DeiT-Small/16",
    ),
    "deit_b": ModelSpec(
        id="deit_b",
        timm_name="deit_base_patch16_224",
        pretrained=True,
        num_classes=1000,
        label="DeiT-B",
    ),
    "mlpmixer_b16": ModelSpec(
        id="mlpmixer_b16",
        timm_name="mixer_b16_224",
        pretrained=True,
        num_classes=1000,
        label="MLP-Mixer B/16",
    ),
    "mlpmixer_l16": ModelSpec(
        id="mlpmixer_l16",
        timm_name="mixer_l16_224",
        pretrained=True,
        num_classes=1000,
        label="MLP-Mixer-L/16",
    ),
}


def list_models() -> Dict[str, ModelSpec]:
    """
    Return a dictionary of all registered models.

    Returns
    -------
    Dict[str, ModelSpec]
        Mapping from model_id to ModelSpec.
    """
    return dict(_MODEL_REGISTRY)


def get_model_spec(model_id: str) -> ModelSpec:
    """
    Retrieve the ModelSpec for a given identifier.

    Parameters
    ----------
    model_id : str
        Key in the model registry.

    Returns
    -------
    ModelSpec

    Raises
    ------
    KeyError
        If model_id is unknown.
    """
    if model_id not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model_id '{model_id}'. " f"Available: {', '.join(sorted(_MODEL_REGISTRY))}")
    return _MODEL_REGISTRY[model_id]


def create_model(
    model_id: str,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
) -> nn.Module:
    """
    Instantiate a model with pretrained weights.

    Parameters
    ----------
    model_id : str
        Identifier in the registry.
    device : str or torch.device, optional
        Device to move the model to.
        Default chooses GPU if available, else CPU.
    eval_mode : bool, optional
        If True, sets the model to eval() mode.
        Default ``True``.

    Returns
    -------
    nn.Module
        The initialized model.
    """
    spec = get_model_spec(model_id)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = timm.create_model(
        spec.timm_name,
        pretrained=spec.pretrained,
    )

    model.to(device)

    if eval_mode:
        model.eval()

    return model


def get_preprocess(model_id: str, is_training: bool = False):
    """
    Build preprocessing transforms for a given model.

    Parameters
    ----------
    model_id : str
        Identifier in the registry.
    is_training : bool, optional
        If True, use training-time augmentations.
        Default ``False``.

    Returns
    -------
    torchvision.transforms.Compose
        Preprocessing pipeline for images.
    """
    spec = get_model_spec(model_id)

    tmp = timm.create_model(spec.timm_name, pretrained=spec.pretrained)
    config = resolve_data_config({}, model=tmp)

    transform = create_transform(
        **config,
        is_training=is_training,
    )
    return transform


def get_input_size(model_id: str) -> Tuple[int, int, int]:
    """
    Get (C, H, W) input size of the model.

    Parameters
    ----------
    model_id : str
        Model identifier.

    Returns
    -------
    tuple of int
        Expected input size (C, H, W).
    """
    spec = get_model_spec(model_id)
    tmp = timm.create_model(spec.timm_name, pretrained=spec.pretrained)
    config = resolve_data_config({}, model=tmp)
    return tuple(config["input_size"])
