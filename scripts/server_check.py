import sys
from pathlib import Path

import torch


def main():
    print("Python:", sys.version.replace("\n", " "))
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            memory_gb = props.total_memory / 1024**3
            print(f"GPU {idx}: {torch.cuda.get_device_name(idx)} ({memory_gb:.2f} GB)")

    paths = [
        Path("src/config.py"),
        Path("src/run_training.py"),
        Path("dataset/jsons"),
        Path("dataset/texts"),
        Path("dataset/all_trajectories.npz"),
    ]
    for path in paths:
        status = "ok" if path.exists() else "missing"
        print(f"{status:7} {path}")

    print("json files:", len(list(Path("dataset/jsons").glob("trajectory_*.json"))))
    print("txt files: ", len(list(Path("dataset/texts").glob("trajectory_*.txt"))))


if __name__ == "__main__":
    main()
