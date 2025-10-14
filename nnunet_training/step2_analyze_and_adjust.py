import json
import math
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.json")


def load_json(p: Path) -> dict:
    with open(p, "r") as f:
        return json.load(f)


def save_json(p: Path, data: dict) -> None:
    with open(p, "w") as f:
        json.dump(data, f, indent=2)


def ceil_divisible(x: int, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def main() -> None:
    cfg = load_json(CONFIG_PATH)
    preproc_root = Path(cfg["preprocessed_root"])  # shared persistent
    logs_root = Path(cfg["logs_root"])  # shared persistent
    dataset_id = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    conf_name = cfg["configuration_name"]
    plans_name = cfg["plans_name"]
    safety_margin_percent = int(cfg.get("safety_margin_percent", 5))

    ds_folder = preproc_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    fingerprint_fp = ds_folder / "dataset_fingerprint.json"
    plans_fp = ds_folder / f"{plans_name}.json"

    logs_root.mkdir(parents=True, exist_ok=True)

    fp = load_json(fingerprint_fp)
    plans = load_json(plans_fp)

    # Use 3d_fullres as base
    base = plans["configurations"]["3d_fullres"]
    target_spacing = base["spacing"]
    num_pool = base["num_pool_per_axis"]

    # dataset_fingerprint stores median sizes/spacings; here we use a conservative approach:
    # fall back to original_median_shape_after_transp if available, else require manual verification
    median_shape = plans.get("original_median_shape_after_transp")
    if median_shape is None:
        # If not available, ask user to verify or set manually; here we abort with helpful info
        report = {
            "hint": "original_median_shape_after_transp not found in plans; please compute max resampled size manually or re-run planner.",
            "target_spacing": target_spacing,
            "num_pool_per_axis": num_pool
        }
        save_json(logs_root / "patch_size_report.json", report)
        print("Missing original_median_shape_after_transp in plans; wrote hint to patch_size_report.json")
        return

    # Safety: inflate median by 1.5x to approximate worst-case, then add safety margin
    inflated = [int(math.ceil(d * 1.5)) for d in median_shape]
    # Add safety margin
    inflated = [int(math.ceil(d * (1 + safety_margin_percent / 100.0))) for d in inflated]

    # Round up to be divisible by pooling
    patch_size = []
    for d, pools in zip(inflated, num_pool):
        patch_size.append(ceil_divisible(d, 2 ** pools))

    # Create/overwrite single-pass config
    plans["configurations"][conf_name] = {
        "inherits_from": "3d_fullres",
        "patch_size": patch_size,
        "batch_size": 1,
        "data_identifier": conf_name
    }

    # Persist
    save_json(plans_fp, plans)
    save_json(logs_root / "patch_size_report.json", {
        "target_spacing": target_spacing,
        "num_pool_per_axis": num_pool,
        "median_shape_after_transp": median_shape,
        "computed_patch_size": patch_size,
        "safety_margin_percent": safety_margin_percent
    })
    print(f"Updated {plans_fp} with configuration {conf_name} and patch_size {patch_size}")


if __name__ == "__main__":
    main()


