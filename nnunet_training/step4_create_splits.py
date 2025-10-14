import json
import math
import random
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def save_json(p: Path, data) -> None:
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def stratified_split(case_ids: list[str], val_fraction: float) -> tuple[list[str], list[str]]:
    # prefix is like D031_, D032_, ...
    buckets: dict[str, list[str]] = {}
    for cid in case_ids:
        prefix = cid.split("_")[0]
        buckets.setdefault(prefix, []).append(cid)
    train, val = [], []
    for _, ids in buckets.items():
        ids_sorted = sorted(ids)
        n_val = max(1, math.floor(len(ids_sorted) * val_fraction))
        random.shuffle(ids_sorted)
        val.extend(ids_sorted[:n_val])
        train.extend(ids_sorted[n_val:])
    return train, val


def main() -> None:
    cfg = load_json(CONFIG_PATH)
    random.seed(42)

    preproc_root = Path(cfg["preprocessed_root"])  # shared persistent
    dataset_id = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    val_fraction = float(cfg.get("validation_fraction", 0.02))

    ds_folder = preproc_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    # list case ids from labelsTr in raw to be robust
    raw_root = Path(cfg["raw_root"])  # shared persistent
    raw_labels = (raw_root / f"Dataset{dataset_id:03d}_{dataset_name}" / "labelsTr").glob("*.nii.gz")
    case_ids = [p.stem for p in raw_labels]
    case_ids = sorted(case_ids)
    if not case_ids:
        raise RuntimeError("No cases found for split generation")

    train, val = stratified_split(case_ids, val_fraction)
    splits = [{"train": train, "val": val}]
    save_json(ds_folder / "splits_final.json", splits)
    print(f"Created splits_final.json with {len(train)} train / {len(val)} val cases")


if __name__ == "__main__":
    main()


