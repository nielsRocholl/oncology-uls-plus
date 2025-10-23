import argparse
import csv
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation


STRUCT = np.ones((3, 3, 3), dtype=bool)


def load_seg_bool(p: Path) -> np.ndarray:
    return nib.load(str(p)).get_fdata() > 0.5


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    sa = a.sum()
    sb = b.sum()
    if sa == 0 and sb == 0:
        return 1.0
    den = sa + sb
    return float(2.0 * inter / den) if den > 0 else 0.0


def bmask(m: np.ndarray) -> np.ndarray:
    return np.logical_xor(m, binary_erosion(m, structure=STRUCT, iterations=1))


def biou(a: np.ndarray, b: np.ndarray) -> float:
    ba = binary_dilation(bmask(a), structure=STRUCT, iterations=1)
    bb = binary_dilation(bmask(b), structure=STRUCT, iterations=1)
    uni = np.logical_or(ba, bb)
    u = uni.sum()
    if u == 0:
        return 1.0
    inter = np.logical_and(ba, bb).sum()
    return float(inter / u)


def lesion_type(name: str) -> str:
    m = re.search(r"_type-([^_]+)", name)
    return m.group(1) if m else "unknown"


def triad_key(name: str) -> str:
    return re.sub(r"_aug[12](?=\\.nii\\.gz$)", "", name)


def role(name: str) -> str:
    if re.search(r"_aug1(?=\\.nii\\.gz$)", name):
        return "aug1"
    if re.search(r"_aug2(?=\\.nii\\.gz$)", name):
        return "aug2"
    return "normal"


def stats(vals: list[float]) -> tuple[float, float, int]:
    if not vals:
        return 0.0, 0.0, 0
    arr = np.asarray(vals, float)
    return float(arr.mean()), float(arr.std(ddof=0)), int(arr.size)


def write_rows(rows: list[dict], out_csv: Path) -> None:
    cols = [
        "scope","lesion_type","n_cases","dsc_mean","dsc_std","biou_mean","biou_std",
        "n_triplets","agree_dsc_mean","agree_dsc_std","agree_biou_mean","agree_biou_std",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})


def evaluate(dataset_root: Path, preds_dir: Path, out_csv: Path) -> None:
    labels_dir = dataset_root / "labelsTr"
    pred_files = sorted(preds_dir.glob("*.nii.gz"))
    eval_rec: list[tuple[str, float, float]] = []
    groups: dict[str, dict[str, Path]] = {}

    for pf in pred_files:
        lf = labels_dir / pf.name
        if not lf.exists():
            continue
        p = load_seg_bool(pf); g = load_seg_bool(lf)
        eval_rec.append((lesion_type(pf.name), dice(g, p), biou(g, p)))
        k = triad_key(pf.name); r = role(pf.name)
        groups.setdefault(k, {})[r] = pf

    rows: list[dict] = []
    if eval_rec:
        by_t: dict[str, list[tuple[float, float]]] = {}
        for t, d, b in eval_rec:
            by_t.setdefault(t, []).append((d, b))
        all_d = [d for _, d, _ in eval_rec]; all_b = [b for _, _, b in eval_rec]
        md, sd, n = stats(all_d); mb, sb, _ = stats(all_b)
        rows.append({"scope":"evaluation","lesion_type":"ALL","n_cases":n,
                     "dsc_mean":md,"dsc_std":sd,"biou_mean":mb,"biou_std":sb})
        for t in sorted(by_t):
            dvals = [x[0] for x in by_t[t]]; bvals = [x[1] for x in by_t[t]]
            md, sd, n = stats(dvals); mb, sb, _ = stats(bvals)
            rows.append({"scope":"evaluation","lesion_type":t,"n_cases":n,
                        "dsc_mean":md,"dsc_std":sd,"biou_mean":mb,"biou_std":sb})

    triad: list[tuple[str, float, float]] = []
    for k, rs in groups.items():
        if not {"normal","aug1","aug2"}.issubset(rs):
            continue
        pn = load_seg_bool(rs["normal"]); p1 = load_seg_bool(rs["aug1"]); p2 = load_seg_bool(rs["aug2"])
        d = (dice(pn, p1) + dice(pn, p2) + dice(p1, p2)) / 3.0
        b = (biou(pn, p1) + biou(pn, p2) + biou(p1, p2)) / 3.0
        triad.append((lesion_type(k), float(d), float(b)))

    if triad:
        by_t2: dict[str, list[tuple[float, float]]] = {}
        for t, d, b in triad:
            by_t2.setdefault(t, []).append((d, b))
        all_d = [d for _, d, _ in triad]; all_b = [b for _, _, b in triad]
        md, sd, n = stats(all_d); mb, sb, _ = stats(all_b)
        rows.append({"scope":"agreement","lesion_type":"ALL","n_triplets":n,
                     "agree_dsc_mean":md,"agree_dsc_std":sd,
                     "agree_biou_mean":mb,"agree_biou_std":sb})
        for t in sorted(by_t2):
            dvals = [x[0] for x in by_t2[t]]; bvals = [x[1] for x in by_t2[t]]
            md, sd, n = stats(dvals); mb, sb, _ = stats(bvals)
            rows.append({"scope":"agreement","lesion_type":t,"n_triplets":n,
                        "agree_dsc_mean":md,"agree_dsc_std":sd,
                        "agree_biou_mean":mb,"agree_biou_std":sb})

    write_rows(rows, out_csv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128"))
    p.add_argument("--preds", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/preds"))
    p.add_argument("--out", type=Path, default=Path("/data/bodyct/experiments/nielsrocholl/ULS+/nnUNet_raw/Dataset401_Longitudinal_CT_Test_128/uls_metrics.csv"))
    args = p.parse_args()
    evaluate(args.dataset_root, args.preds, args.out)


if __name__ == "__main__":
    main()
