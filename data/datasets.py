from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Dict, List

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS

from models.models import get_preprocess


DatasetName = Literal["imagenetv2", "imagefolder"]
SplitName = Literal["train", "val", "test"]


@dataclass
class DatasetConfig:
    """
    Configuration for building a dataset and dataloader.

    Parameters
    ----------
    name : {"imagenetv2", "imagefolder"}
        Dataset type identifier.
    root : str
        Root directory for the dataset.
        For ``"imagenetv2"`` and ``"imagefolder"``, this is the directory
        that contains class subfolders.
    split : {"train", "val", "test"}, optional
        Data split. Default ``"train"``.
    download : bool, optional
        Whether to download the dataset if supported and not present.
        Default ``False``.
    batch_size : int, optional
        Batch size for the dataloader. Default ``32``.
    shuffle : bool, optional
        Whether to shuffle data in the dataloader.
        Default ``False`` (internally overridden to ``True`` for train split).
    num_workers : int, optional
        Number of worker processes for data loading. Default ``4``.
    pin_memory : bool, optional
        Whether to pin memory in the dataloader. Default ``True``.
    drop_last : bool, optional
        Whether to drop the last incomplete batch. Default ``False``.
    """

    name: DatasetName
    root: str
    split: SplitName = "train"
    download: bool = False
    batch_size: int = 32
    shuffle: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False


def _create_imagefolder(
    cfg: DatasetConfig,
    transform,
):
    """
    Build a generic ImageFolder dataset.

    Parameters
    ----------
    cfg : DatasetConfig
        Dataset configuration.
    transform : callable
        Transform pipeline applied to each sample.

    Returns
    -------
    torchvision.datasets.ImageFolder
        Configured ImageFolder dataset.

    Notes
    -----
    The expected directory structure for ``cfg.root`` is:

        root/
          class_a/
            img1.png
            img2.png
          class_b/
            img3.png
            ...

    If your dataset is structured as::

        dataset_root/
          train/
          val/
          test/

    then pass, for example, ``root="dataset_root/train"`` with ``split="train"``.
    """
    root = str(Path(cfg.root).expanduser())

    ds = datasets.ImageFolder(
        root=root,
        transform=transform,
    )
    return ds


class ImageNetV2IndexFolder(datasets.DatasetFolder):
    """
    ImageNetV2 dataset where class folders are named by numeric indices.

    This dataset expects a directory layout of the form:

        root/
          0/
            img1.jpeg
            ...
          1/
            img2.jpeg
            ...
          ...
          999/
            imgX.jpeg

    Class labels are assigned as ``int(folder_name)``, so that folder
    ``"0"`` corresponds to class index 0, folder ``"1"`` to class index 1,
    and so on.

    Parameters
    ----------
    root : str
        Root directory of the dataset.
    transform : callable, optional
        Transform pipeline applied to each sample.
    target_transform : callable, optional
        Transform applied to the target labels.
    loader : callable, optional
        Function to load an image given its path. Default uses
        ``torchvision.datasets.folder.default_loader``.
    """

    def __init__(
        self,
        root: str,
        transform=None,
        target_transform=None,
        loader=default_loader,
    ) -> None:
        super().__init__(
            root=root,
            loader=loader,
            extensions=IMG_EXTENSIONS,
            transform=transform,
            target_transform=target_transform,
        )

        # DatasetFolder.__init__ calls self.find_classes,
        # so self.classes and self.class_to_idx are already set

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Find class folders and map them to numeric labels.

        Parameters
        ----------
        directory : str
            Root directory.

        Returns
        -------
        classes : list of str
            Sorted list of class folder names (still as strings).
        class_to_idx : dict
            Mapping from folder name to numeric label (int(folder_name)).
        """
        # all immediate subdirectories are classes
        classes = [d.name for d in os.scandir(directory) if d.is_dir()]

        # sort numerically, assuming folder names are "0", "1", ..., "999"
        try:
            classes_sorted = sorted(classes, key=lambda x: int(x))
        except ValueError as e:
            raise ValueError(
                "Non-numeric folder name encountered in ImageNetV2IndexFolder. "
                "Expected subfolders named '0', '1', ..., '999'."
            ) from e

        class_to_idx = {cls_name: int(cls_name) for cls_name in classes_sorted}
        return classes_sorted, class_to_idx


def _create_imagenetv2(
    cfg: DatasetConfig,
    transform,
):
    """
    Build an ImageNetV2 dataset with numeric folder names.

    Parameters
    ----------
    cfg : DatasetConfig
        Dataset configuration.
    transform : callable
        Transform pipeline applied to each sample.

    Returns
    -------
    ImageNetV2IndexFolder
        Configured ImageNetV2 dataset.
    """
    root = str(Path(cfg.root).expanduser())

    ds = ImageNetV2IndexFolder(
        root=root,
        transform=transform,
    )
    return ds


def create_dataset(
    cfg: DatasetConfig,
    model_id: str,
    is_training: Optional[bool] = None,
):
    """
    Create a dataset instance for a given configuration and model.

    Parameters
    ----------
    cfg : DatasetConfig
        Dataset configuration.
    model_id : str
        Model identifier used to select preprocessing.
    is_training : bool, optional
        Override for training/evaluation preprocessing.
        If ``None``, ``split == "train"`` implies training mode.

    Returns
    -------
    torch.utils.data.Dataset
        The initialized dataset.

    Raises
    ------
    ValueError
        If the dataset name is unsupported.
    """
    if is_training is None:
        is_training = cfg.split == "train"

    transform = get_preprocess(model_id=model_id, is_training=is_training)

    if cfg.name == "imagenetv2":
        ds = _create_imagenetv2(cfg, transform)
    elif cfg.name == "imagefolder":
        ds = _create_imagefolder(cfg, transform)
    else:
        raise ValueError(f"Unsupported dataset name: {cfg.name}")

    return ds


def build_dataloader(
    cfg: DatasetConfig,
    model_id: str,
    is_training: Optional[bool] = None,
) -> Tuple[DataLoader, torch.utils.data.Dataset]:
    """
    Create a dataset and corresponding dataloader for a given model.

    Parameters
    ----------
    cfg : DatasetConfig
        Dataset configuration.
    model_id : str
        Model identifier used to select preprocessing.
    is_training : bool, optional
        Override for training/evaluation preprocessing.

    Returns
    -------
    tuple
        A tuple ``(dataloader, dataset)`` where:

        * dataloader : torch.utils.data.DataLoader
        * dataset : torch.utils.data.Dataset
    """
    ds = create_dataset(cfg, model_id=model_id, is_training=is_training)

    shuffle = cfg.shuffle
    if cfg.shuffle is False and cfg.split == "train":
        shuffle = True

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )

    return loader, ds


def infer_num_classes(dataset) -> int:
    """
    Infer the number of classes from a dataset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset instance, typically from torchvision.

    Returns
    -------
    int
        Number of classes.

    Raises
    ------
    AttributeError
        If the number of classes cannot be inferred.

    Notes
    -----
    This function supports datasets that expose either a
    ``classes`` attribute (e.g. ``ImageFolder``) or integer targets
    via a ``targets`` attribute.
    """
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        return len(classes)

    targets = getattr(dataset, "targets", None)
    if targets is not None and len(targets) > 0:
        return int(max(targets)) + 1

    raise AttributeError(
        "Unable to infer number of classes from dataset. " "Specify the number of classes manually instead."
    )
