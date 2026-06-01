import argparse
import json
import re
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def trajectory_id(path):
    match = re.fullmatch(r"trajectory_(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def sorted_trajectory_paths(directory):
    paths = []
    for path in Path(directory).glob("trajectory_*.json"):
        item_id = trajectory_id(path)
        if item_id is not None:
            paths.append((item_id, path))
    return [path for _, path in sorted(paths)]


def load_points(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pts = np.asarray(raw, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError(f"{path} must contain points shaped as (T, 2+) or (T, 3)")
    return pts


def load_meta(path):
    if not path:
        return {}
    meta_path = Path(path)
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_meta_path(json_path):
    json_path = Path(json_path)
    candidates = [
        json_path.with_suffix(".meta"),
        json_path.with_suffix(".meta.json"),
        json_path.parent / f"{json_path.stem}.meta",
        json_path.parent / f"{json_path.stem}.meta.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def robust_height(y):
    return float(np.percentile(y, 95) - np.percentile(y, 5))


def summarize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return None
    return {
        "count": int(values.size),
        "p10": float(np.percentile(values, 10)),
        "p25": float(np.percentile(values, 25)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
    }


def kmeans_1d(values, k, iterations=50):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size < k:
        raise ValueError("Not enough points for requested line count")

    percentiles = np.linspace(10, 90, k)
    centers = np.percentile(values, percentiles).astype(np.float32)

    for _ in range(iterations):
        distances = np.abs(values[:, None] - centers[None, :])
        labels = distances.argmin(axis=1)
        next_centers = centers.copy()
        for idx in range(k):
            cluster = values[labels == idx]
            if cluster.size:
                next_centers[idx] = cluster.mean()
        if np.allclose(next_centers, centers):
            break
        centers = next_centers

    order = np.argsort(centers)
    remap = np.empty_like(order)
    remap[order] = np.arange(k)
    return centers[order], remap[labels]


def line_stats(points, expected_lines):
    y = points[:, 1]
    y_low, y_high = np.percentile(y, [1, 99])
    clipped = points[(y >= y_low) & (y <= y_high)]
    if expected_lines <= 1:
        line_y = clipped[:, 1]
        return [
            {
                "center_px": float(np.median(line_y)),
                "height_px": robust_height(line_y),
                "points": int(line_y.size),
            }
        ]

    centers, labels = kmeans_1d(clipped[:, 1], expected_lines)
    stats = []
    for idx, center in enumerate(centers):
        cluster = clipped[labels == idx, 1]
        stats.append(
            {
                "center_px": float(center),
                "height_px": robust_height(cluster),
                "points": int(cluster.size),
            }
        )
    return stats


def dataset_heights(json_dir):
    heights = []
    for path in sorted_trajectory_paths(json_dir):
        try:
            pts = load_points(path)
        except Exception:
            continue
        heights.append(robust_height(pts[:, 1]))
    return heights


def text_line_count(text_path):
    if not text_path.exists():
        return 1
    text = text_path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    if text.endswith("\n"):
        text = text[:-1]
    return max(1, text.count("\n") + 1)


def dataset_line_measurements(json_dir, text_dir):
    line_heights = []
    center_gaps = []
    samples_with_gaps = 0

    for path in sorted_trajectory_paths(json_dir):
        item_id = trajectory_id(path)
        text_path = Path(text_dir) / f"trajectory_{item_id}.txt"
        expected_lines = text_line_count(text_path)
        try:
            stats = line_stats(load_points(path), expected_lines)
        except Exception:
            continue

        line_heights.extend(item["height_px"] for item in stats)
        centers = [item["center_px"] for item in stats]
        gaps = [centers[idx + 1] - centers[idx] for idx in range(len(centers) - 1)]
        if gaps:
            samples_with_gaps += 1
            center_gaps.extend(gaps)

    return line_heights, center_gaps, samples_with_gaps


def format_px(value):
    return f"{value:.1f}px"


def print_summary(title, summary):
    if summary is None:
        print(title)
        print("- no data")
        return

    print(title)
    print(f"- samples: {summary['count']}")
    print(
        "- robust height px: "
        f"p10={format_px(summary['p10'])}, "
        f"p25={format_px(summary['p25'])}, "
        f"p50={format_px(summary['p50'])}, "
        f"p75={format_px(summary['p75'])}, "
        f"p90={format_px(summary['p90'])}, "
        f"p95={format_px(summary['p95'])}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare vertical spacing of a new trajectory with the current dataset."
    )
    parser.add_argument("--sample", required=True, help="Path to the new trajectory JSON.")
    parser.add_argument("--meta", default="", help="Optional path to the sidecar .meta file.")
    parser.add_argument("--json-dir", default=str(PROJECT_ROOT / "dataset" / "jsons"))
    parser.add_argument("--text-dir", default=str(PROJECT_ROOT / "dataset" / "texts"))
    parser.add_argument("--expected-lines", type=int, default=2)
    parser.add_argument("--try-mm", default="8,9,10,11,12")
    args = parser.parse_args()

    sample_path = Path(args.sample)
    meta_path = Path(args.meta) if args.meta else auto_meta_path(sample_path)
    meta = load_meta(meta_path)

    heights = dataset_heights(args.json_dir)
    if not heights:
        raise ValueError(f"No trajectory_*.json files found in {args.json_dir}")
    dataset_summary = summarize(heights)
    dataset_line_heights, dataset_center_gaps, dataset_gap_samples = dataset_line_measurements(
        args.json_dir, args.text_dir
    )
    dataset_line_summary = summarize(dataset_line_heights)
    dataset_gap_summary = summarize(dataset_center_gaps)

    points = load_points(sample_path)
    sample_lines = line_stats(points, max(1, args.expected_lines))
    sample_heights = [item["height_px"] for item in sample_lines]
    sample_height_summary = summarize(sample_heights)
    centers = [item["center_px"] for item in sample_lines]
    center_gaps = [centers[idx + 1] - centers[idx] for idx in range(len(centers) - 1)]

    print_summary("Current dataset", dataset_summary)
    print()
    print_summary("Current dataset per detected text line", dataset_line_summary)
    if dataset_gap_summary:
        print()
        print("Current dataset line gaps")
        print(f"- multi-line samples: {dataset_gap_samples}")
        print(
            "- center gap px: "
            f"p10={format_px(dataset_gap_summary['p10'])}, "
            f"p25={format_px(dataset_gap_summary['p25'])}, "
            f"p50={format_px(dataset_gap_summary['p50'])}, "
            f"p75={format_px(dataset_gap_summary['p75'])}, "
            f"p90={format_px(dataset_gap_summary['p90'])}"
        )
    print()
    print(f"Sample: {sample_path}")
    if meta_path:
        print(f"Meta: {meta_path}")
    else:
        print("Meta: not found")
    print(f"- points: {len(points)}")
    print(f"- detected/expected lines: {len(sample_lines)}")
    for idx, item in enumerate(sample_lines, start=1):
        print(
            f"- line {idx}: center={format_px(item['center_px'])}, "
            f"height={format_px(item['height_px'])}, points={item['points']}"
        )

    if center_gaps:
        print("- center gaps: " + ", ".join(format_px(value) for value in center_gaps))
        print(f"- avg center gap: {format_px(float(np.mean(center_gaps)))}")

    print(
        "- sample line heights: "
        f"p50={format_px(sample_height_summary['p50'])}, "
        f"dataset-line p50 ratio={sample_height_summary['p50'] / dataset_line_summary['p50']:.2f}, "
        f"dataset-line p75 ratio={sample_height_summary['p50'] / dataset_line_summary['p75']:.2f}"
    )
    if center_gaps and dataset_gap_summary:
        avg_gap = float(np.mean(center_gaps))
        print(
            "- sample center gap: "
            f"dataset-gap p50 ratio={avg_gap / dataset_gap_summary['p50']:.2f}, "
            f"dataset-gap p75 ratio={avg_gap / dataset_gap_summary['p75']:.2f}"
        )

    grid_step_px = meta.get("gridStepPx")
    grid_step_mm = meta.get("gridStepMm")
    ydpi = meta.get("ydpi")
    if grid_step_px:
        print()
        print("Recorder spacing")
        print(f"- grid step: {format_px(float(grid_step_px))}")
        if grid_step_mm:
            print(f"- grid step mm: {float(grid_step_mm):.2f}mm")
        if center_gaps:
            print(f"- avg center gap / grid step: {float(np.mean(center_gaps)) / float(grid_step_px):.2f}")

    if ydpi:
        print()
        print("Candidate gridStepMm values on this device")
        for raw in args.try_mm.split(","):
            raw = raw.strip()
            if not raw:
                continue
            mm = float(raw)
            px = mm * float(ydpi) / 25.4
            scale = px / float(grid_step_px) if grid_step_px else 1.0
            print(f"- {mm:g}mm -> {format_px(px)} ({scale:.2f}x current)")


if __name__ == "__main__":
    main()
