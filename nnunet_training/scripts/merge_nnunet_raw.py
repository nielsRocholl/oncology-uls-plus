import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple


def read_json(path: Path) -> Dict:
    with path.open("r") as f:
        return json.load(f)


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=False)


def find_dataset_dir(raw_root: Path, dataset_id: int) -> Path:
    pattern = f"Dataset{dataset_id:03d}_*"
    matches = sorted([p for p in raw_root.glob(pattern) if p.is_dir()])
    if len(matches) == 0:
        raise FileNotFoundError(f"No dataset directory found for id {dataset_id:03d} under {raw_root}")
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple dataset directories found for id {dataset_id:03d} under {raw_root}: {[m.name for m in matches]}"
        )
    return matches[0]


def strip_file_ending(filename: str, file_ending: str) -> str:
    if not filename.endswith(file_ending):
        raise ValueError(f"File {filename} does not end with expected file ending {file_ending}")
    return filename[: -len(file_ending)]


def _index_images(images_tr: Path, file_ending: str) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    # Use os.scandir for faster directory iteration on large folders
    with os.scandir(images_tr) as it:
        for entry in it:
            name = entry.name
            if not name.endswith(file_ending):
                continue
            base = name[: -len(file_ending)]
            if "_" not in base:
                continue
            case_id, chan = base.rsplit("_", 1)
            if len(chan) != 4 or not chan.isdigit():
                continue
            index.setdefault(case_id, []).append(images_tr / name)
    # sort channel lists by channel index
    for case_id, paths in index.items():
        def channel_index(path: Path) -> int:
            base = strip_file_ending(path.name, file_ending)
            chan = base.split("_")[-1]
            return int(chan)

        paths.sort(key=channel_index)
    return index


def collect_cases(dataset_dir: Path, file_ending: str) -> List[Tuple[str, List[Path], Path]]:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    if not images_tr.is_dir() or not labels_tr.is_dir():
        raise FileNotFoundError(f"Expected imagesTr and labelsTr in {dataset_dir}")

    # Build a one-pass index of images to avoid O(N_cases * N_images) scans
    img_index = _index_images(images_tr, file_ending)

    label_files = sorted([p for p in labels_tr.iterdir() if p.name.endswith(file_ending)])
    cases: List[Tuple[str, List[Path], Path]] = []
    for lab in label_files:
        case_id = strip_file_ending(lab.name, file_ending)
        imgs = img_index.get(case_id, [])
        if len(imgs) == 0:
            raise FileNotFoundError(f"No image channels found for case {case_id} in {images_tr}")
        cases.append((case_id, imgs, lab))
    return cases


def ensure_consistent_metadata(ref_meta: Dict, meta: Dict) -> None:
    keys = ["channel_names", "labels", "file_ending"]
    for k in keys:
        if ref_meta.get(k) != meta.get(k):
            raise ValueError(f"Inconsistent '{k}' across datasets. Cannot merge.")
    # optional key: overwrite_image_reader_writer must also match if present anywhere
    ref_oirw = ref_meta.get("overwrite_image_reader_writer")
    oirw = meta.get("overwrite_image_reader_writer")
    if (ref_oirw or oirw) and ref_oirw != oirw:
        raise ValueError("Inconsistent 'overwrite_image_reader_writer' across datasets. Cannot merge.")


def safe_link_or_copy(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError("mode must be 'link' or 'copy'")


def compute_total_ops(raw_root: Path, dataset_ids: List[int]) -> Tuple[int, Dict]:
    ref_meta = None
    total = 0
    for ds_id in dataset_ids:
        ds_dir = find_dataset_dir(raw_root, ds_id)
        meta = read_json(ds_dir / "dataset.json")
        if ref_meta is None:
            ref_meta = meta
        else:
            ensure_consistent_metadata(ref_meta, meta)
        file_ending = meta["file_ending"]
        labels_tr = ds_dir / "labelsTr"
        num_cases = len([p for p in labels_tr.iterdir() if p.name.endswith(file_ending)])
        num_channels = len(meta["channel_names"]) if isinstance(meta.get("channel_names"), dict) else 1
        total += num_cases * (num_channels + 1)
    return total, (ref_meta or {})


def render_progress(done: int, total: int, force: bool = False) -> None:
    width = 40
    ratio = 0 if total == 0 else done / total
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = int(ratio * 100)
    sys.stdout.write(f"\rProgress: [{bar}] {done}/{total} ({percent}%)")
    sys.stdout.flush()
    if done >= total or force:
        sys.stdout.write("\n")
        sys.stdout.flush()


def merge_datasets(
    raw_root: Path,
    dataset_ids: List[int],
    dest_id: int,
    dest_name: str,
    mode: str,
    force: bool,
    always_prefix: bool,
    manifest_path: Path,
) -> None:
    dest_dir = raw_root / f"Dataset{dest_id:03d}_{dest_name}"
    images_out = dest_dir / "imagesTr"
    labels_out = dest_dir / "labelsTr"

    if dest_dir.exists() and force:
        shutil.rmtree(images_out, ignore_errors=True)
        shutil.rmtree(labels_out, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    images_out.mkdir(exist_ok=True)
    labels_out.mkdir(exist_ok=True)

    # progress + gather and validate metadata
    total_ops, meta_summary = compute_total_ops(raw_root, dataset_ids)
    done_ops = 0
    render_progress(done_ops, total_ops)

    ref_meta = None
    file_ending = None
    merged_manifest: Dict[str, Dict] = {"datasets": {}, "cases": {}}
    total_cases = 0
    existing_case_ids: set = set()
    update_interval = max(1, total_ops // 100)

    for ds_id in dataset_ids:
        ds_dir = find_dataset_dir(raw_root, ds_id)
        meta = read_json(ds_dir / "dataset.json")
        if ref_meta is None:
            ref_meta = meta
            file_ending = ref_meta["file_ending"]
        else:
            ensure_consistent_metadata(ref_meta, meta)
        merged_manifest["datasets"][f"{ds_id:03d}"] = ds_dir.name

        print(f"\nProcessing dataset {ds_id:03d}...", flush=True)
        cases = collect_cases(ds_dir, file_ending)  # type: ignore[arg-type]
        for case_id, img_paths, lab_path in cases:
            new_case_id = case_id
            if always_prefix or new_case_id in existing_case_ids:
                new_case_id = f"D{ds_id:03d}__{case_id}"
            # avoid collisions even after prefixing (paranoia)
            while new_case_id in existing_case_ids:
                new_case_id = f"D{ds_id:03d}__{new_case_id}"
            existing_case_ids.add(new_case_id)

            # link/copy images
            for img in img_paths:
                base = strip_file_ending(img.name, file_ending)
                chan = base.split("_")[-1]
                dst = images_out / f"{new_case_id}_{chan}{file_ending}"
                safe_link_or_copy(img, dst, mode)
                done_ops += 1
                if done_ops % update_interval == 0 or done_ops == total_ops:
                    render_progress(done_ops, total_ops)

            # link/copy label
            dst_lab = labels_out / f"{new_case_id}{file_ending}"
            safe_link_or_copy(lab_path, dst_lab, mode)
            done_ops += 1
            if done_ops % update_interval == 0 or done_ops == total_ops:
                render_progress(done_ops, total_ops)

            merged_manifest["cases"][new_case_id] = {
                "origin_dataset_id": f"{ds_id:03d}",
                "origin_dataset_dirname": ds_dir.name,
                "origin_case_id": case_id,
            }
            total_cases += 1

    # ensure final progress update
    render_progress(done_ops, total_ops, force=True)

    # write combined dataset.json
    assert ref_meta is not None and file_ending is not None
    out_meta: Dict = {
        "channel_names": ref_meta["channel_names"],
        "labels": ref_meta["labels"],
        "numTraining": total_cases,
        "file_ending": file_ending,
    }
    if "overwrite_image_reader_writer" in ref_meta:
        out_meta["overwrite_image_reader_writer"] = ref_meta["overwrite_image_reader_writer"]
    write_json(dest_dir / "dataset.json", out_meta)

    # manifest for traceability
    write_json(manifest_path, merged_manifest)

    print(
        f"Merged {len(dataset_ids)} datasets into {dest_dir.name}. Total cases: {total_cases}. Mode: {mode}."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge multiple nnUNet_raw datasets into a single combined dataset."
    )
    p.add_argument(
        "dataset_ids",
        type=int,
        nargs="+",
        help="Dataset IDs to merge (e.g., 31 32 ... 50)",
    )
    p.add_argument(
        "--raw-root",
        type=Path,
        default=Path(os.environ.get("nnUNet_raw", "")) if os.environ.get("nnUNet_raw") else None,
        help="Path to nnUNet_raw. Defaults to env nnUNet_raw.",
    )
    p.add_argument("--dest-id", type=int, required=True, help="Destination dataset id (e.g., 100)")
    p.add_argument(
        "--dest-name",
        type=str,
        required=True,
        help="Destination dataset name (e.g., ULS23_Combined)",
    )
    p.add_argument(
        "--mode",
        choices=["link", "copy"],
        default="link",
        help="Use symlinks (link) or copy files (copy).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="If set, clears imagesTr/labelsTr in destination before merging.",
    )
    p.add_argument(
        "--always-prefix",
        action="store_true",
        help="Always prefix new case ids with origin dataset id (DXXX__).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to write a manifest JSON mapping merged cases to their origins. Defaults to DEST/dataset_merged_manifest.json",
    )
    args = p.parse_args()
    if args.raw_root is None:
        raise RuntimeError("--raw-root is required if env nnUNet_raw is not set")
    return args


def main() -> None:
    args = parse_args()
    raw_root: Path = args.raw_root
    dest_dir = raw_root / f"Dataset{args.dest_id:03d}_{args.dest_name}"
    manifest_path = args.manifest or (dest_dir / "dataset_merged_manifest.json")
    merge_datasets(
        raw_root=raw_root,
        dataset_ids=args.dataset_ids,
        dest_id=args.dest_id,
        dest_name=args.dest_name,
        mode=args.mode,
        force=args.force,
        always_prefix=args.always_prefix,
        manifest_path=manifest_path,
    )


if __name__ == "__main__":
    main()


