from __future__ import annotations

import json
from functools import lru_cache
from typing import Dict
from pathlib import Path


_JSON_PATH = Path("./data/imagenet_class_info.json")


@lru_cache()
def _load_mapping() -> Dict[int, str]:
    """
    Load mapping from class index to human-readable label.

    Returns
    -------
    dict
        Mapping from int class index (cid) to a short label string.
    """
    if not _JSON_PATH.exists():
        raise FileNotFoundError(f"Expected class info JSON at '{_JSON_PATH}'.")

    with _JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cid_to_name: Dict[int, str] = {}
    for entry in data:
        cid = int(entry["cid"])
        synset = entry.get("synset") or []
        if synset:
            name = synset[0]
        else:
            name = entry.get("wnid", f"class_{cid}")
        cid_to_name[cid] = name

    return cid_to_name


def get_labels(cid: int) -> str:
    """
    Get the human-readable label for an ImageNet-v2 class index.

    Parameters
    ----------
    cid : int
        Class index in [0, 999].

    Returns
    -------
    str
        Human-readable label string.
    """
    mapping = _load_mapping()
    return mapping.get(cid, f"class_{cid}")
