from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import torch
from PIL import Image

from config.seed import set_seed
from config.vars import DEVICE, FIGURES_DIR
from models.models import create_model
from viz.grad_cam import GradCAM, GradCAMConfig
from viz.saliency import Saliency, SaliencyConfig
from viz.attention import (
    VitAttentionRollout,
    VitAttentionRolloutConfig,
    VitLastLayerAttention,
    VitLastLayerAttentionConfig,
    class_token_attention_to_grid,
    upsample_attention_to_image,
    to_numpy as vit_to_numpy,
)
from viz.common import (
    infer_model_family,
    load_raw_dataset,
    get_image_and_gt,
    prepare_input_for_model,
    get_convnext_target_layer,
    extract_vit_self_attention,
    overlay_heatmap_on_image,
    resize_heatmap_to_image,
)
from utils.labels import get_labels


def make_figure(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    method_name: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Create and save a side-by-side figure with original and explanation.
    """
    img = np.array(pil_image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    heatmap_resized = resize_heatmap_to_image(img, heatmap)
    overlay = overlay_heatmap_on_image(img, heatmap_resized, alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(method_name)
    axes[1].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a single model on a single image.")

    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="Model identifier, e.g. 'convnext_t', 'convnext_b', 'deit_s', 'deit_b', 'mlpmixer_b16', 'mlpmixer_l16'.",
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
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split.",
    )
    parser.add_argument(
        "--index",
        type=int,
        required=True,
        help="Index of the sample in the dataset to visualize.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save the output figure (PNG).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEVICE,
        help=f"Device for inference, e.g. 'cuda' or 'cpu'. Default '{DEVICE}'.",
    )
    parser.add_argument(
        "--vit-attn-method",
        type=str,
        default="last_layer",
        choices=["rollout", "last_layer"],
        help="Attention visualization method for ViT/DeiT models.",
    )
    parser.add_argument(
        "--mixer-method",
        type=str,
        default="saliency",
        choices=["saliency", "token-cam"],
        help="Visualization method for MLP-Mixer models.",
    )

    return parser.parse_args()


def infer_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return Path(args.output_path).expanduser()

    fig_dir = Path(FIGURES_DIR)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fname = f"viz_{args.dataset_name}_{args.split}_{args.model_id}_idx{args.index}.png"
    return fig_dir / fname


def main() -> None:
    args = parse_args()
    set_seed()

    device = torch.device(args.device)

    # dataset + GT
    raw_ds = load_raw_dataset(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        split=args.split,
    )

    if args.index < 0 or args.index >= len(raw_ds):
        raise IndexError(f"Index {args.index} is out of bounds for dataset of size {len(raw_ds)}.")

    pil_image, gt_idx, gt_name = get_image_and_gt(
        dataset_name=args.dataset_name,
        dataset=raw_ds,
        index=args.index,
    )

    # model + input
    family = infer_model_family(args.model_id)
    model = create_model(args.model_id, device=device, eval_mode=True)
    x = prepare_input_for_model(args.model_id, pil_image, device)

    # prediction
    with torch.no_grad():
        logits = model(x)
    pred_idx = int(logits.argmax(dim=1).item())
    pred_name = get_labels(pred_idx)

    # explanation per family
    if family == "convnext":
        method_name = "Grad-CAM"
        target_layer = get_convnext_target_layer(model)
        explainer = GradCAM(
            model=model,
            target_layer=target_layer,
            device=device,
            config=GradCAMConfig(normalize=True),
        )
        heatmaps = explainer(x, target_class=pred_idx)
        heatmap = heatmaps[0, 0].numpy()

    elif family == "vit":
        if args.vit_attn_method == "rollout":
            method_name = "Attention rollout"
        else:
            method_name = "Last-layer attention"

        attn_list = extract_vit_self_attention(model, x)
        if args.vit_attn_method == "rollout":
            attn_maps = VitAttentionRollout(VitAttentionRolloutConfig())(attn_list)
        else:
            attn_maps = VitLastLayerAttention(VitLastLayerAttentionConfig())(attn_list)

        B, T, _ = attn_maps.shape
        assert B == 1
        grid_size = int(np.sqrt(T - 1))
        attn_grid = class_token_attention_to_grid(
            attn_maps,
            token_index=0,
            grid_size=grid_size,
        )
        attn_img = upsample_attention_to_image(
            attn_grid,
            image_size=(x.shape[2], x.shape[3]),
        )
        heatmap = vit_to_numpy(attn_img)[0, 0]

    elif family == "mixer":
        if args.mixer_method != "saliency":
            raise NotImplementedError(f"mixer-method '{args.mixer_method}' is not implemented.")
        method_name = "Saliency"
        explainer = Saliency(
            model=model,
            device=device,
            config=SaliencyConfig(
                use_input_gradient=True,
                use_absolute=False,
                channel_aggregation="l2",
                normalize=True,
            ),
        )
        sal_maps = explainer(x, target_class=pred_idx)
        heatmap = sal_maps[0, 0].numpy()

    else:
        raise NotImplementedError(f"No visualization implemented for model family '{family}'.")

    # title + save
    output_path = infer_output_path(args)
    title = f"{args.model_id} | idx={args.index} | " f"GT={gt_idx} ({gt_name}) | pred={pred_idx} ({pred_name})"
    make_figure(
        pil_image=pil_image,
        heatmap=heatmap,
        method_name=method_name,
        title=title,
        output_path=output_path,
    )

    print(f"Saved visualization to: {output_path}")


if __name__ == "__main__":
    main()
