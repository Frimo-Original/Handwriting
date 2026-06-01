import json
import sys
from pathlib import Path

import numpy as np
import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import config  # noqa: E402


def convert(json_path, txt_path, out_npz):
    with open(json_path, "r", encoding="utf-8") as f:
        trajectory = json.load(f)
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()

    pts = np.asarray(trajectory, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"{json_path} must contain points shaped as (T, 3)")
    pts[:, 2] = (pts[:, 2] == 1).astype(np.float32)
    pts[-1, 2] = 1.0

    text_indices = np.asarray(
        config.encode_text(
            text,
            append_eos=getattr(config, "append_eos_to_dataset", True),
            normalize=True,
        ),
        dtype=np.int64,
    )
    np.savez_compressed(out_npz, points=pts, text_indices=text_indices)


def merge_npz(npz_files, out_npz):
    all_points = []
    all_texts = []

    for file_path in tqdm.tqdm(npz_files, desc="Merging"):
        data = np.load(file_path, allow_pickle=True)
        pts = data["points"]
        text_indices = data["text_indices"]
        if pts.dtype == np.object_:
            pts = pts[0]
        if text_indices.dtype == np.object_:
            text_indices = text_indices[0]
        all_points.append(np.asarray(pts, dtype=np.float32))
        all_texts.append(np.asarray(text_indices, dtype=np.int64).reshape(-1))

    np.savez_compressed(
        out_npz,
        points=np.asarray(all_points, dtype=object),
        text_indices=np.asarray(all_texts, dtype=object),
    )


def main():
    (DATASET_DIR / "npzs").mkdir(parents=True, exist_ok=True)
    npz_files = []
    json_files = sorted(
        (DATASET_DIR / "jsons").glob("trajectory_*.json"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    for json_path in tqdm.tqdm(json_files, desc="Converting json"):
        idx = json_path.stem.split("_")[-1]
        txt_path = DATASET_DIR / "texts" / f"trajectory_{idx}.txt"
        if not txt_path.exists():
            print(f"Skip {json_path.name}: missing {txt_path.name}")
            continue
        out_npz = DATASET_DIR / "npzs" / f"handwriting_trajectory_{idx}.npz"
        convert(json_path, txt_path, out_npz)
        npz_files.append(out_npz)

    merge_npz(npz_files, DATASET_DIR / "all_trajectories.npz")


if __name__ == "__main__":
    main()
