from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

from config.seed import set_seed
from config.vars import RESULTS_DIR
from utils.results_loader import (
    InferenceResults,
    load_multiple_results,
    align_results_on_common_indices,
    compute_agreement_sets,
)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze multiple model result files and compute agreement sets.")

    parser.add_argument(
        "--results",
        type=str,
        nargs="+",
        required=True,
        help=(
            "Paths to `.npz` result files from run_inference.py. " "Each file should correspond to a different model."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help=(
            "Optional path to output JSON summary. " "If not provided, a default name is constructed under results/."
        ),
    )

    return parser.parse_args()


def infer_output_path(
    result_paths: List[str],
    output_path: str | None,
) -> Path:
    """
    Construct an output JSON path based on inputs if none is given.

    Parameters
    ----------
    result_paths : list of str
        List of input `.npz` result file paths.
    output_path : str or None
        Optional user-provided output path.

    Returns
    -------
    pathlib.Path
        Path to the output JSON file.
    """
    if output_path is not None:
        return Path(output_path).expanduser()

    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)

    # use model stems joined by "+" as part of the filename
    stems = [Path(p).stem for p in result_paths]
    joined = "+".join(stems)
    filename = f"analysis_{joined}.json"

    return results_dir / filename


def compute_per_model_sets(
    aligned_results: Dict[str, InferenceResults],
) -> Dict[str, Dict[str, list[int]]]:
    """
    Compute per-model correctness sets.

    Parameters
    ----------
    aligned_results : dict
        Mapping ``model_id -> InferenceResults`` aligned on common indices.

    Returns
    -------
    dict
        Nested mapping ``model_id -> { "correct": [...], "wrong": [...] }``
        where inner lists store sample indices.
    """
    per_model: Dict[str, Dict[str, list[int]]] = {}

    # all models share the same `indices` array after alignment
    for model_id, res in aligned_results.items():
        idx = res.indices
        correct_mask = res.preds == res.labels
        wrong_mask = ~correct_mask

        per_model[model_id] = {
            "correct": idx[correct_mask].tolist(),
            "wrong": idx[wrong_mask].tolist(),
        }

    return per_model


def compute_pairwise_sets(
    aligned_results: Dict[str, InferenceResults],
) -> Dict[str, Dict[str, list[int]]]:
    """
    Compute pairwise correctness patterns for models.

    Parameters
    ----------
    aligned_results : dict
        Mapping ``model_id -> InferenceResults`` aligned on common indices.

    Returns
    -------
    dict
        Nested mapping::

            pairwise[model_i][model_j] = indices_where_i_correct_and_j_wrong

        where indices are stored as Python lists.
    """
    model_ids = list(aligned_results.keys())
    if len(model_ids) < 2:
        return {}

    # all share the same indices
    reference_indices = aligned_results[model_ids[0]].indices

    pairwise: Dict[str, Dict[str, list[int]]] = {}

    # precompute correctness masks per model
    correct_masks: Dict[str, np.ndarray] = {}
    for mid, res in aligned_results.items():
        if not np.array_equal(res.indices, reference_indices):
            raise ValueError("Indices mismatch between models. Make sure results are aligned.")
        correct_masks[mid] = res.preds == res.labels

    for i, mid_i in enumerate(model_ids):
        pairwise[mid_i] = {}
        for j, mid_j in enumerate(model_ids):
            if i == j:
                continue
            correct_i = correct_masks[mid_i]
            correct_j = correct_masks[mid_j]
            # i correct, j wrong
            mask = correct_i & (~correct_j)
            pairwise[mid_i][mid_j] = reference_indices[mask].tolist()

    return pairwise


def results_summary(
    aligned_results: Dict[str, InferenceResults],
    agreement_sets: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """
    Build a JSON-serializable summary of the analysis.

    Parameters
    ----------
    aligned_results : dict
        Mapping ``model_id -> InferenceResults`` aligned on common indices.
    agreement_sets : dict
        Global agreement sets as returned by ``compute_agreement_sets``.

    Returns
    -------
    dict
        Summary dictionary containing metrics, agreement sets and model ids.
    """
    model_metrics = {}
    for mid, res in aligned_results.items():
        model_metrics[mid] = {
            "accuracy": res.accuracy(),
            "num_samples": int(res.logits.shape[0]),
            "num_classes": int(res.logits.shape[1]),
        }

    summary: Dict[str, Any] = {
        "models": list(aligned_results.keys()),
        "metrics": model_metrics,
        "agreement_sets": {name: indices.tolist() for name, indices in agreement_sets.items()},
    }
    return summary


def main() -> None:
    """
    Load results, align them, compute agreement and save summary.
    """
    args = parse_args()
    set_seed()

    # load all result files
    results = load_multiple_results(args.results)

    # align on common indices
    aligned_results, common_indices = align_results_on_common_indices(results)

    print(f"Loaded {len(aligned_results)} models.")
    print(f"Common samples: {len(common_indices)}")

    # global agreement sets
    agreement_sets = compute_agreement_sets(aligned_results)

    # per-model correctness
    per_model_sets = compute_per_model_sets(aligned_results)

    # pairwise patterns (i correct, j wrong)
    pairwise_sets = compute_pairwise_sets(aligned_results)

    # build summary
    summary = results_summary(aligned_results, agreement_sets)
    summary["per_model_sets"] = per_model_sets
    summary["pairwise_sets"] = pairwise_sets

    # decide output path
    output_path = infer_output_path(args.results, args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON serialize (convert any NumPy types to Python)
    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary_serializable = json.loads(json.dumps(summary, default=_convert))

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary_serializable, f, indent=2)

    print(f"Saved analysis summary to: {output_path}")

    print("\nPer-model accuracy:")
    for mid, metrics in summary["metrics"].items():
        acc = metrics["accuracy"]
        print(f"  {mid}: {acc:.4f}")

    print("\nGlobal sets (sizes):")
    for name, idx_list in summary["agreement_sets"].items():
        print(f"  {name}: {len(idx_list)} samples")


if __name__ == "__main__":
    main()
