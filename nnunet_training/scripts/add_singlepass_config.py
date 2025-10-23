import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    tmp.replace(path)


def product_strides(strides: List[List[int]]) -> Tuple[int, int, int]:
    dz = dy = dx = 1
    for s in strides:
        z, y, x = s
        dz *= z
        dy *= y
        dx *= x
    return dz, dy, dx


def ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m if m > 1 else x


def compute_fullimage_patch_size(
    fingerprint_path: Path, strides: List[List[int]]
) -> Tuple[int, int, int]:
    fp = load_json(fingerprint_path)
    shapes = fp.get("shapes_after_crop")
    if not shapes:
        raise RuntimeError(f"No shapes_after_crop in {fingerprint_path}")
    max_z = max(s[0] for s in shapes)
    max_y = max(s[1] for s in shapes)
    max_x = max(s[2] for s in shapes)
    dz, dy, dx = product_strides(strides)
    # Ensure divisibility by total downsampling
    max_z = ceil_to_multiple(max_z, dz)
    max_y = ceil_to_multiple(max_y, dy)
    max_x = ceil_to_multiple(max_x, dx)
    return max_z, max_y, max_x


def scale_batch_size(old_bs: int, old_ps: List[int], new_ps: Tuple[int, int, int]) -> int:
    old_vox = int(old_ps[0]) * int(old_ps[1]) * int(old_ps[2])
    new_vox = int(new_ps[0]) * int(new_ps[1]) * int(new_ps[2])
    if new_vox <= 0 or old_vox <= 0:
        return max(1, old_bs)
    scaled = max(1, int(old_bs * (old_vox / new_vox)))
    return scaled


def add_singlepass_config(
    plans_path: Path,
    fingerprint_path: Path,
    base_config_name: str,
    new_config_name: str,
    new_data_identifier: str,
) -> bool:
    plans = load_json(plans_path)
    configs: Dict[str, dict] = plans.get("configurations", {})
    if base_config_name not in configs:
        raise RuntimeError(f"Base config {base_config_name} not found in {plans_path}")
    base_cfg = configs[base_config_name]
    arch = base_cfg.get("architecture", {})
    arch_kwargs = arch.get("arch_kwargs", {})
    strides = arch_kwargs.get("strides")
    if not strides or not isinstance(strides, list):
        raise RuntimeError(f"No strides found in architecture for {plans_path}")
    # compute target full-image patch size
    new_ps = compute_fullimage_patch_size(fingerprint_path, strides)
    old_ps = base_cfg.get("patch_size")
    old_bs = int(base_cfg.get("batch_size", 1))
    new_bs = scale_batch_size(old_bs, old_ps, new_ps) if old_ps else max(1, old_bs)
    # build new config by inheriting
    new_cfg = {
        "inherits_from": base_config_name,
        "data_identifier": new_data_identifier,
        "patch_size": [int(new_ps[0]), int(new_ps[1]), int(new_ps[2])],
        "batch_size": int(new_bs),
    }
    # insert or update
    changed = False
    prev = configs.get(new_config_name)
    if prev != new_cfg:
        configs[new_config_name] = new_cfg
        plans["configurations"] = configs
        write_json(plans_path, plans)
        changed = True
    return changed


def find_dataset_dirs(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("Dataset")])


def find_plans_file(dataset_dir: Path) -> Path:
    # Prefer ResEncL plans if present
    cand = dataset_dir / "nnUNetResEncUNetLPlans.json"
    if cand.exists():
        return cand
    # fallback to any nnUNetPlans*.json
    for p in dataset_dir.glob("*Plans*.json"):
        return p
    raise FileNotFoundError(f"No plans file in {dataset_dir}")


def main():
    parser = argparse.ArgumentParser(description="Add single-pass 3d_fullres config to plans")
    parser.add_argument(
        "--preprocessed-root",
        default=os.environ.get("nnUNet_preprocessed", ""),
        help="Path to nnUNet_preprocessed root",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        type=int,
        help="Optional dataset IDs to update (e.g., 31 32 33). If omitted, all under root are used.",
    )
    parser.add_argument(
        "--base-config",
        default="3d_fullres",
        help="Base config to inherit from",
    )
    parser.add_argument(
        "--new-config",
        default="3d_fullres_singlepass",
        help="Name of the new configuration",
    )
    parser.add_argument(
        "--data-identifier",
        default="nnUNetPlans_3d_singlepass",
        help="data_identifier for the new preprocessed cache",
    )
    args = parser.parse_args()

    root = Path(args.preprocessed_root).resolve() if args.preprocessed_root else None
    if root is None or not root.exists():
        raise FileNotFoundError("--preprocessed-root not provided or does not exist")

    ds_dirs = find_dataset_dirs(root)
    if args.datasets:
        ids = {f"Dataset{str(i).zfill(3)}" for i in args.datasets}
        ds_dirs = [d for d in ds_dirs if d.name.split("_")[0] in ids]

    changed_any = False
    for d in ds_dirs:
        try:
            plans_path = find_plans_file(d)
            fingerprint_path = d / "dataset_fingerprint.json"
            if not fingerprint_path.exists():
                print(f"Skipping {d.name}: no dataset_fingerprint.json")
                continue
            changed = add_singlepass_config(
                plans_path,
                fingerprint_path,
                args.base_config,
                args.new_config,
                args.data_identifier,
            )
            if changed:
                changed_any = True
                print(f"Updated {plans_path} with {args.new_config}")
            else:
                print(f"No change for {plans_path} (already up to date)")
        except Exception as e:
            print(f"ERROR processing {d}: {e}")

    if not changed_any:
        print("No files changed.")


if __name__ == "__main__":
    main()


