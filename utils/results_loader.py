from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass
class InferenceResults:
    """
    Container for inference outputs of a single model.

    Parameters
    ----------
    model_id : str
        Identifier of the model (e.g. ``"convnext_t"``).
    logits : np.ndarray
        Array of raw model outputs, shape ``(N, C)``.
    preds : np.ndarray
        Predicted class indices, shape ``(N,)``.
    labels : np.ndarray
        Ground-truth class indices, shape ``(N,)``.
    indices : np.ndarray
        Sample indices, shape ``(N,)``.
    meta : dict
        Additional metadata (dataset, split, etc.).
    """

    model_id: str
    logits: np.ndarray
    preds: np.ndarray
    labels: np.ndarray
    indices: np.ndarray
    meta: dict

    def accuracy(self) -> float:
        """
        Compute top-1 accuracy for this result set.

        Returns
        -------
        float
            Fraction of correctly classified samples.
        """
        if self.labels is None:
            return np.nan
        correct = (self.preds == self.labels).astype(np.float32)
        return float(correct.mean())


def load_npz_results(path: str | Path) -> InferenceResults:
    """
    Load inference results from a compressed `.npz` file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the saved `.npz` file.

    Returns
    -------
    InferenceResults
        Loaded inference results.
    """
    path = Path(path).expanduser()
    data = np.load(path, allow_pickle=True)

    logits = data["logits"]
    preds = data["preds"]
    labels = data["labels"]
    indices = data["indices"]
    meta_raw = data["meta"]

    # parsing
    if isinstance(meta_raw, dict):
        # directly stored as dict
        meta = meta_raw
    elif isinstance(meta_raw, np.ndarray):
        # could be array([dict], dtype=object)
        if meta_raw.size == 1:
            item = meta_raw[0]
            meta = item if isinstance(item, dict) else item.item()
        else:
            raise ValueError(
                f"Unexpected meta array shape: {meta_raw.shape}. " "Expected a single-element object array."
            )
    else:
        # fallback: try item()
        try:
            meta = meta_raw.item()
        except Exception as e:
            raise ValueError(f"Unable to parse meta field from {path}: {meta_raw}") from e

    if not isinstance(meta, dict):
        raise ValueError(f"Meta field did not decode to a dict: {meta}")

    model_id = meta.get("model_id", path.stem)

    return InferenceResults(
        model_id=model_id,
        logits=logits,
        preds=preds,
        labels=labels,
        indices=indices,
        meta=meta,
    )


def load_multiple_results(
    paths: Iterable[str | Path],
) -> Dict[str, InferenceResults]:
    """
    Load multiple `.npz` result files.

    Parameters
    ----------
    paths : iterable of str or pathlib.Path
        Paths to `.npz` files. Each file should correspond to one model.

    Returns
    -------
    dict
        Mapping ``model_id -> InferenceResults``.
    """
    results: Dict[str, InferenceResults] = {}
    for p in paths:
        res = load_npz_results(p)
        if res.model_id in results:
            raise ValueError(
                f"Duplicate model_id '{res.model_id}' when loading results. " f"Paths: {p} and previously loaded."
            )
        results[res.model_id] = res
    return results


def align_results_on_common_indices(
    results: Dict[str, InferenceResults],
) -> Tuple[Dict[str, InferenceResults], np.ndarray]:
    """
    Align multiple models' results on common sample indices.

    Parameters
    ----------
    results : dict
        Mapping ``model_id -> InferenceResults``.

    Returns
    -------
    tuple
        A tuple ``(aligned_results, common_indices)`` where:

        * aligned_results : dict
            Mapping ``model_id -> InferenceResults`` restricted to the
            intersection of indices across all models.
        * common_indices : np.ndarray
            Sorted array of common indices.

    Notes
    -----
    This is useful when models were run on the same dataset but potentially with different max_samples or ordering.
    """
    if not results:
        raise ValueError("No results provided for alignment.")

    # compute intersection of indices
    index_sets = [set(r.indices.tolist()) for r in results.values()]
    common = set.intersection(*index_sets)
    if not common:
        raise ValueError("No common indices found across models.")

    common_indices = np.array(sorted(common), dtype=np.int64)

    aligned: Dict[str, InferenceResults] = {}
    for model_id, r in results.items():
        # map index -> position in original arrays
        index_to_pos = {idx: pos for pos, idx in enumerate(r.indices.tolist())}

        positions = [index_to_pos[idx] for idx in common_indices]
        positions_arr = np.array(positions, dtype=np.int64)

        aligned_logits = r.logits[positions_arr]
        aligned_preds = r.preds[positions_arr]
        aligned_labels = r.labels[positions_arr]
        aligned_indices = r.indices[positions_arr]

        aligned[model_id] = InferenceResults(
            model_id=model_id,
            logits=aligned_logits,
            preds=aligned_preds,
            labels=aligned_labels,
            indices=aligned_indices,
            meta=r.meta,
        )

    return aligned, common_indices


def compute_agreement_sets(
    aligned_results: Dict[str, InferenceResults],
) -> Dict[str, np.ndarray]:
    """
    Compute sets of sample indices for different agreement patterns.

    Parameters
    ----------
    aligned_results : dict
        Mapping ``model_id -> InferenceResults`` aligned on common indices.
        All results must have the same ``indices`` array.

    Returns
    -------
    dict
        Dictionary containing arrays of indices for:

        * ``"all_correct"`` : correct for all models
        * ``"all_wrong"`` : wrong for all models
        * ``"mixed"`` : at least one model correct and at least one wrong

    Notes
    -----
    This function assumes that all models:
        * were evaluated on the same dataset/split
        * have ground-truth labels aligned on the same indices
    """
    if not aligned_results:
        raise ValueError("No aligned results provided.")

    model_ids: List[str] = list(aligned_results.keys())
    first = aligned_results[model_ids[0]]
    indices = first.indices

    # Collect correctness masks for each model
    correct_masks: List[np.ndarray] = []
    for mid in model_ids:
        r = aligned_results[mid]
        if not np.array_equal(r.indices, indices):
            raise ValueError(
                f"Indices mismatch for model '{mid}'. "
                "Ensure results are aligned before calling compute_agreement_sets."
            )
        correct = r.preds == r.labels
        correct_masks.append(correct.astype(bool))

    correct_stack = np.stack(correct_masks, axis=0)  # shape (M, N)
    all_correct_mask = np.all(correct_stack, axis=0)
    all_wrong_mask = ~np.any(correct_stack, axis=0)
    mixed_mask = ~(all_correct_mask | all_wrong_mask)

    return {
        "all_correct": indices[all_correct_mask],
        "all_wrong": indices[all_wrong_mask],
        "mixed": indices[mixed_mask],
    }
