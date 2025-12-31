from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
from pathlib import Path
from typing import List

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
    load_raw_dataset,
    get_image_and_gt,
    prepare_input_for_model,
    get_convnext_target_layer,
    extract_vit_self_attention,
    overlay_heatmap_on_image,
    resize_heatmap_to_image,
)
from utils.labels import get_labels


def create_comparison_figure(
    pil_image: Image.Image,
    heat_convnext: np.ndarray,
    heat_deit: np.ndarray,
    heat_mixer: np.ndarray,
    title: str,
    output_path: Path,
    mixer_method: str,
) -> None:
    """
    Create and save a 4-panel comparison figure.
    """
    img = np.array(pil_image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)

    heat_convnext = resize_heatmap_to_image(img, heat_convnext)
    heat_deit = resize_heatmap_to_image(img, heat_deit)
    heat_mixer = resize_heatmap_to_image(img, heat_mixer)

    overlay_convnext = overlay_heatmap_on_image(img, heat_convnext)
    overlay_deit = overlay_heatmap_on_image(img, heat_deit)
    overlay_mixer = overlay_heatmap_on_image(img, heat_mixer)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay_convnext)
    axes[1].set_title("ConvNeXt Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay_deit)
    axes[2].set_title("DeiT Attention")
    axes[2].axis("off")

    axes[3].imshow(overlay_mixer)
    mixer_title = "Mixer Saliency" if mixer_method == "saliency" else "Mixer Token-CAM"
    axes[3].set_title(mixer_title)
    axes[3].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def visualize_index(
    idx: int,
    dataset,
    device: torch.device,
    output_dir: Path,
    dataset_name: str,
    vit_attn_method: str,
    mixer_method: str,
    convnext_id: str,
    deit_id: str,
    mixer_id: str,
) -> None:
    """
    Visualize all three models for a single dataset index.
    """
    pil_image, gt_idx, gt_name = get_image_and_gt(
        dataset_name=dataset_name,
        dataset=dataset,
        index=idx,
    )

    # models
    model_conv = create_model(convnext_id, device=device, eval_mode=True)
    model_deit = create_model(deit_id, device=device, eval_mode=True)
    model_mixer = create_model(mixer_id, device=device, eval_mode=True)

    # inputs
    x_conv = prepare_input_for_model(convnext_id, pil_image, device)
    x_deit = prepare_input_for_model(deit_id, pil_image, device)
    x_mixer = prepare_input_for_model(mixer_id, pil_image, device)

    # predictions
    with torch.no_grad():
        logits_conv = model_conv(x_conv)
        logits_deit = model_deit(x_deit)
        logits_mixer = model_mixer(x_mixer)

    pred_conv_idx = int(logits_conv.argmax(dim=1).item())
    pred_deit_idx = int(logits_deit.argmax(dim=1).item())
    pred_mixer_idx = int(logits_mixer.argmax(dim=1).item())

    pred_conv_name = get_labels(pred_conv_idx)
    pred_deit_name = get_labels(pred_deit_idx)
    pred_mixer_name = get_labels(pred_mixer_idx)

    # ConvNeXt Grad-CAM
    target_layer = get_convnext_target_layer(model_conv)
    cam = GradCAM(
        model=model_conv,
        target_layer=target_layer,
        device=device,
        config=GradCAMConfig(normalize=True),
    )
    cam_maps = cam(x_conv, target_class=pred_conv_idx)
    heat_convnext = cam_maps[0, 0].numpy()

    # DeiT attention (rollout or last-layer)
    attn_list = extract_vit_self_attention(model_deit, x_deit)
    if vit_attn_method == "rollout":
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
        image_size=(x_deit.shape[2], x_deit.shape[3]),
    )
    heat_deit = vit_to_numpy(attn_img)[0, 0]

    # MLP-Mixer visualization
    if mixer_method == "saliency":
        sal = Saliency(
            model=model_mixer,
            device=device,
            config=SaliencyConfig(
                use_input_gradient=True,
                use_absolute=False,
                channel_aggregation="l2",
                normalize=True,
            ),
        )
        sal_maps = sal(x_mixer, target_class=pred_mixer_idx)
        heat_mixer = sal_maps[0, 0].numpy()
    else:
        raise NotImplementedError(f"mixer-method '{mixer_method}' is not implemented.")

    title = (
        f"idx={idx} | GT={gt_idx} ({gt_name}) | "
        f"{convnext_id}: {pred_conv_idx} ({pred_conv_name}) | "
        f"{deit_id}: {pred_deit_idx} ({pred_deit_name}) | "
        f"{mixer_id}: {pred_mixer_idx} ({pred_mixer_name})"
    )

    output_path = output_dir / f"comparison_{dataset_name}_idx{idx}.png"
    create_comparison_figure(
        pil_image=pil_image,
        heat_convnext=heat_convnext,
        heat_deit=heat_deit,
        heat_mixer=heat_mixer,
        title=title,
        output_path=output_path,
        mixer_method=mixer_method,
    )

    print(f"Saved comparison figure for idx={idx} to: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ConvNeXt vs DeiT vs MLP-Mixer on selected indices.")

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
        "--indices",
        type=int,
        nargs="*",
        default=None,
        help="Explicit dataset indices to visualize. If not provided, use analysis JSON.",
    )
    parser.add_argument(
        "--analysis-path",
        type=str,
        default=None,
        help="Path to analysis JSON from run_analysis.py (required if indices are not given).",
    )
    parser.add_argument(
        "--case",
        type=str,
        default="mixed",
        choices=["all_correct", "all_wrong", "mixed"],
        help="Agreement set to sample from if using analysis JSON.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Number of samples to draw from the selected case when using analysis JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=FIGURES_DIR,
        help=f"Directory to save figures. Default '{FIGURES_DIR}'.",
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
        help="Attention visualization method for DeiT.",
    )
    parser.add_argument(
        "--mixer-method",
        type=str,
        default="saliency",
        choices=["saliency", "token-cam"],
        help="Visualization method for MLP-Mixer.",
    )
    parser.add_argument(
        "--convnext-id",
        type=str,
        default="convnext_t",
        help="ConvNeXt model identifier, e.g. 'convnext_t' or 'convnext_b'.",
    )
    parser.add_argument(
        "--deit-id",
        type=str,
        default="deit_s",
        help="DeiT model identifier, e.g. 'deit_s' or 'deit_b'.",
    )
    parser.add_argument(
        "--mixer-id",
        type=str,
        default="mlpmixer_b16",
        help="MLP-Mixer model identifier, e.g. 'mlpmixer_b16' or 'mlpmixer_l16'.",
    )

    return parser.parse_args()


def choose_indices_from_analysis(
    analysis_path: str,
    case: str,
    num_samples: int,
) -> List[int]:
    path = Path(analysis_path).expanduser()
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "agreement_sets" not in data or case not in data["agreement_sets"]:
        raise KeyError(f"Case '{case}' not found in agreement_sets of analysis file {analysis_path}.")

    indices = data["agreement_sets"][case]
    if not indices:
        raise ValueError(f"No indices found for case '{case}' in analysis file.")

    rng = np.random.default_rng()
    chosen = rng.choice(indices, size=min(num_samples, len(indices)), replace=False)
    return [int(i) for i in chosen]


def main() -> None:
    args = parse_args()
    set_seed()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir).expanduser()

    dataset = load_raw_dataset(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        split=args.split,
    )

    # determine indices to visualize
    if args.indices is not None and len(args.indices) > 0:
        indices = args.indices
    else:
        if args.analysis_path is None:
            raise ValueError("Either --indices or --analysis-path must be provided.")
        indices = choose_indices_from_analysis(
            analysis_path=args.analysis_path,
            case=args.case,
            num_samples=args.num_samples,
        )

    print(f"Selected indices: {indices}")

    for idx in indices:
        if idx < 0 or idx >= len(dataset):
            print(f"Skipping idx={idx}: out of bounds for dataset length {len(dataset)}.")
            continue
        visualize_index(
            idx=idx,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            dataset_name=args.dataset_name,
            vit_attn_method=args.vit_attn_method,
            mixer_method=args.mixer_method,
            convnext_id=args.convnext_id,
            deit_id=args.deit_id,
            mixer_id=args.mixer_id,
        )


if __name__ == "__main__":
    main()
