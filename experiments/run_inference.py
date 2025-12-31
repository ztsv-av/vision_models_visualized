from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from tqdm import tqdm

from config.seed import set_seed
from config.vars import (
    DEVICE,
    DEFAULT_BATCH_SIZE,
    NUM_WORKERS,
    RESULTS_DIR,
)
from models.models import create_model
from data.datasets import DatasetConfig, build_dataloader


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference with pretrained model(s) and save outputs.")

    parser.add_argument(
        "--model-id",
        type=str,
        nargs="+",
        required=True,
        choices=["convnext_t", "deit_s", "mlpmixer_b16", "convnext_b", "deit_b", "mlpmixer_l16"],
        help="One or more model identifiers.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        choices=["imagenetv2", "imagefolder"],
        help="Dataset type.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to use. Default 'test'.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for inference. Default {DEFAULT_BATCH_SIZE}.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of DataLoader workers. Default {NUM_WORKERS}.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=("Optional cap on the number of samples to process. If None, run over the entire dataset."),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Optional path for outputs. "
            "For a single model, this may be a file or a directory. "
            "For multiple models, this is treated as a directory; "
            "one .npz per model will be written inside."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Compute device, e.g. 'cuda' or 'cpu'. Default '{DEVICE}'.",
    )

    return parser.parse_args()


def build_dataset_config(
    args: argparse.Namespace,
) -> DatasetConfig:
    """
    Build a DatasetConfig from parsed arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.

    Returns
    -------
    DatasetConfig
        Dataset configuration instance.
    """
    cfg = DatasetConfig(
        name=args.dataset_name,  # type: ignore[arg-type]
        root=args.dataset_root,
        split=args.split,  # type: ignore[arg-type]
        download=False,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return cfg


def infer_output_path_for_model(
    args: argparse.Namespace,
    model_id: str,
    multi_model: bool,
) -> Path:
    """
    Construct an output .npz path for a given model.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    model_id : str
        Model identifier.
    multi_model : bool
        True if more than one model is being processed in this run.

    Returns
    -------
    pathlib.Path
        Path to the output .npz file for this model.
    """
    if args.output_path is not None:
        base = Path(args.output_path).expanduser()

        if multi_model:
            # treat as directory when running multiple models
            base.mkdir(parents=True, exist_ok=True)
            return base / f"{args.dataset_name}_{args.split}_{model_id}.npz"

        # single-model case: if output_path has a suffix, use it as file
        if base.suffix:
            base.parent.mkdir(parents=True, exist_ok=True)
            return base

        # otherwise treat as directory
        base.mkdir(parents=True, exist_ok=True)
        return base / f"{args.dataset_name}_{args.split}_{model_id}.npz"

    # default: use RESULTS_DIR
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{args.dataset_name}_{args.split}_{model_id}.npz"
    return results_dir / filename


def run_inference(
    model_id: str,
    dataset_cfg: DatasetConfig,
    device: str = DEVICE,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Run inference on a dataset and collect predictions.

    Parameters
    ----------
    model_id : str
        Identifier of the model to use.
    dataset_cfg : DatasetConfig
        Dataset configuration.
    device : str, optional
        Device string, e.g. ``"cuda"`` or ``"cpu"``.
        Default uses value from ``config.vars.DEVICE``.
    max_samples : int or None, optional
        Maximum number of samples to process.
        If ``None``, use the entire dataset.

    Returns
    -------
    dict
        A dictionary containing:

        * "logits" : np.ndarray, shape (N, C)
        * "preds" : np.ndarray, shape (N,)
        * "labels" : np.ndarray or None, shape (N,)
        * "indices" : np.ndarray, shape (N,)
        * "meta" : dict with metadata fields
    """
    torch_device = torch.device(device)

    # model
    print("\nLoading model...")
    model = create_model(model_id=model_id, device=torch_device, eval_mode=True)
    print(f"Model {model_id} built.")

    # data
    print(f"\nBuilding DataLoader for {dataset_cfg.name}...")
    loader, raw_ds = build_dataloader(
        dataset_cfg,
        model_id=model_id,
        is_training=False,
    )
    print("DataLoader built.")

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_indices: list[int] = []

    idx_to_class = None
    if dataset_cfg.name == "imagefolder":
        idx_to_class = {v: k for k, v in raw_ds.class_to_idx.items()}

    model.eval()
    processed = 0
    total = len(loader.dataset)

    if max_samples is not None:
        total = min(total, max_samples)

    print("Starting inference loop...")
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f"Inference [{model_id}]")):
            images = images.to(torch_device)

            logits = model(images)
            all_logits.append(logits.cpu())
            if dataset_cfg.name == "imagefolder":
                mapped = labels.clone()
                for i in range(mapped.size(0)):
                    mapped[i] = int(idx_to_class[int(mapped[i])])
                all_labels.append(mapped)
            else:
                all_labels.append(labels.clone())

            batch_size = images.size(0)
            start_index = batch_idx * loader.batch_size
            indices = list(range(start_index, start_index + batch_size))
            all_indices.extend(indices)

            processed += batch_size
            if max_samples is not None and processed >= max_samples:
                break
    print("Inference finished.")

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    indices_array = np.array(all_indices, dtype=np.int64)

    if max_samples is not None:
        logits_tensor = logits_tensor[:max_samples]
        labels_tensor = labels_tensor[:max_samples]
        indices_array = indices_array[:max_samples]

    preds_tensor = torch.argmax(logits_tensor, dim=1)

    outputs: Dict[str, Any] = {
        "logits": logits_tensor.numpy(),
        "preds": preds_tensor.numpy(),
        "labels": labels_tensor.numpy(),
        "indices": indices_array,
        "meta": {
            "model_id": model_id,
            "dataset_name": dataset_cfg.name,
            "split": dataset_cfg.split,
            "num_samples": int(logits_tensor.shape[0]),
            "num_classes": int(logits_tensor.shape[1]),
        },
    }

    return outputs


def main() -> None:
    """
    Parse arguments, run inference for one or more models, save results.
    """
    args = parse_args()
    set_seed()

    model_ids: List[str] = args.model_id
    multi_model = len(model_ids) > 1

    print("Starting run_inference")
    print("Args received:")
    print("  model-id(s):", model_ids)
    print("  dataset-name:", args.dataset_name)
    print("  dataset-root:", args.dataset_root)
    print("  split:", args.split)
    print("  batch-size:", args.batch_size)

    print("\nCreating dataset config...")
    dataset_cfg = build_dataset_config(args)
    print(f"DatasetConfig:\n {dataset_cfg}")

    for mid in model_ids:
        print("\n" + "=" * 60)
        print(f"Running inference for model: {mid}")
        print("=" * 60)

        outputs = run_inference(
            model_id=mid,
            dataset_cfg=dataset_cfg,
            device=args.device,
            max_samples=args.max_samples,
        )

        output_path = infer_output_path_for_model(args, model_id=mid, multi_model=multi_model)

        print("\nSaving results.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            logits=outputs["logits"],
            preds=outputs["preds"],
            labels=outputs["labels"],
            indices=outputs["indices"],
            meta=np.array([outputs["meta"]], dtype=object),
        )
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
