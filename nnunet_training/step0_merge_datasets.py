import json
import os
import re
import shutil
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.json")


def ensure_dirs(paths: list[str]) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def find_cases(folder: Path, with_labels: bool) -> list[tuple[Path, Path | None]]:
    images_tr = folder / "imagesTr"
    labels_tr = folder / "labelsTr"
    if not images_tr.exists() or not labels_tr.exists():
        return []
    pairs: list[tuple[Path, Path | None]] = []
    by_case: dict[str, list[Path]] = {}
    for f in images_tr.glob("*.nii.gz"):
        # caseId_0000.nii.gz -> caseId
        m = re.match(r"(.+)_\d{4}\.nii\.gz$", f.name)
        if not m:
            continue
        cid = m.group(1)
        by_case.setdefault(cid, []).append(f)
    for cid, chans in by_case.items():
        lbl = labels_tr / f"{cid}.nii.gz"
        pairs.append((Path(cid), lbl if lbl.exists() else None))
    return pairs


def read_dataset_json(fp: Path) -> dict:
    with open(fp, "r") as f:
        return json.load(f)


def merge_labels(label_dicts: list[dict]) -> dict:
    merged: dict[str, int] = {"background": 0}
    next_id = 1
    for ld in label_dicts:
        for name, lid in ld.items():
            if name == "background":
                continue
            if name not in merged:
                merged[name] = next_id
                next_id += 1
    return merged


def main() -> None:
    cfg = load_config()
    raw_root = Path(cfg["raw_root"])  # shared persistent
    logs_root = Path(cfg["logs_root"])  # shared persistent
    dataset_id = int(cfg["dataset_id"])
    dataset_name = cfg["dataset_name"]
    source_ids = list(cfg["source_dataset_ids"])  # 31..50

    target_ds = raw_root / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr_out = target_ds / "imagesTr"
    labels_tr_out = target_ds / "labelsTr"
    ensure_dirs([images_tr_out.as_posix(), labels_tr_out.as_posix(), logs_root.as_posix()])

    # Collect source metadata
    label_dicts: list[dict] = []
    total_images = 0
    total_labels = 0
    mapping_log: list[dict] = []

    for sid in source_ids:
        src = raw_root / f"Dataset{sid:03d}_*"
        # Resolve the exact folder (first glob match)
        matches = list(raw_root.glob(f"Dataset{sid:03d}_*"))
        if not matches:
            continue
        ds_folder = matches[0]
        ds_json = ds_folder / "dataset.json"
        if ds_json.exists():
            meta = read_dataset_json(ds_json)
            if "labels" in meta:
                label_dicts.append(meta["labels"])
        # enumerate cases
        pairs = find_cases(ds_folder, with_labels=True)
        for cid, lbl in pairs:
            # symlink all channels
            chans = sorted((ds_folder / "imagesTr").glob(f"{cid}_*.nii.gz"))
            for ch in chans:
                out = images_tr_out / f"D{sid:03d}_{ch.name}"
                if not out.exists():
                    os.symlink(ch.as_posix(), out.as_posix())
                    total_images += 1
            # label
            if lbl is not None:
                out_lbl = labels_tr_out / f"D{sid:03d}_{lbl.name}"
                if not out_lbl.exists():
                    os.symlink(lbl.as_posix(), out_lbl.as_posix())
                    total_labels += 1
            mapping_log.append({
                "source_dataset": ds_folder.name,
                "case": cid.as_posix(),
                "images": [p.name for p in chans],
                "label": lbl.name if lbl is not None else None
            })

    # Build unified dataset.json
    merged_labels = merge_labels(label_dicts)
    # Infer file ending from first image
    file_ending = ".nii.gz"
    channel_names = {"0": "CT"}
    dataset_json = {
        "channel_names": channel_names,
        "labels": merged_labels,
        "numTraining": total_labels,
        "file_ending": file_ending,
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    with open((target_ds / "dataset.json").as_posix(), "w") as f:
        json.dump(dataset_json, f, indent=2)

    # Write report
    report = {
        "target_dataset": target_ds.as_posix(),
        "num_images": total_images,
        "num_labels": total_labels,
        "labels": merged_labels,
        "mapping_samples": mapping_log[:50]
    }
    with open((logs_root / "step0_report.json").as_posix(), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Merged into: {target_ds}")
    print(f"Images: {total_images}, Labels: {total_labels}")


if __name__ == "__main__":
    main()


