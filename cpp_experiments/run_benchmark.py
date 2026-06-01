# -*- coding: utf-8 -*-
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from torch.utils import cmake_prefix_path


ROOT = Path(__file__).resolve().parents[1]
CPP_DIR = ROOT / "cpp_experiments"
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import config  # noqa: E402


def run(command, cwd=ROOT):
    print("+", " ".join(str(part) for part in command))
    subprocess.run([str(part) for part in command], cwd=cwd, check=True)


def python_package_cmake():
    try:
        import cmake  # noqa: WPS433
    except ImportError:
        return None

    cmake_bin_dir = Path(getattr(cmake, "CMAKE_BIN_DIR", ""))
    exe_name = "cmake.exe" if platform.system() == "Windows" else "cmake"
    candidate = cmake_bin_dir / exe_name
    return candidate if candidate.exists() else None


def find_cmake(explicit=""):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))

    env_cmake = os.environ.get("CMAKE_EXE", "").strip()
    if env_cmake:
        candidates.append(Path(env_cmake))

    path_cmake = shutil.which("cmake")
    if path_cmake:
        candidates.append(Path(path_cmake))

    package_cmake = python_package_cmake()
    if package_cmake:
        candidates.append(package_cmake)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    install_hint = (
        "CMake executable was not found. Install it into the active venv with:\n"
        "  python -m pip install cmake\n"
        "or install CMake/Visual Studio Build Tools and add cmake.exe to PATH.\n"
        "You can also pass --cmake C:\\\\path\\\\to\\\\cmake.exe or set CMAKE_EXE."
    )
    raise FileNotFoundError(install_hint)


def executable_path(build_dir):
    names = ["bench_train_step.exe"] if platform.system() == "Windows" else ["bench_train_step"]
    candidates = []
    for name in names:
        candidates.extend(
            [
                build_dir / name,
                build_dir / "Release" / name,
                build_dir / "Debug" / name,
                build_dir / "RelWithDebInfo" / name,
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Cannot find bench_train_step executable in {build_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Build and run the C++ handwriting train-step benchmark.")
    parser.add_argument("--build-dir", default=str(CPP_DIR / "build"))
    parser.add_argument("--batch", default=str(CPP_DIR / "batches" / "batch.bin"))
    parser.add_argument("--batch-size", type=int, default=getattr(config, "batch_size", 24))
    parser.add_argument("--batch-mode", choices=["first", "median", "longest", "shortest"], default="median")
    parser.add_argument("--max-seq-len", type=int, default=getattr(config, "max_seq_len", 3000))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--cmake", default="", help="Path to cmake executable. Defaults to PATH or Python cmake package.")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    build_dir = Path(args.build_dir).resolve()
    batch_path = Path(args.batch).resolve()

    if not args.skip_prepare:
        run(
            [
                sys.executable,
                CPP_DIR / "prepare_batch.py",
                "--output",
                batch_path,
                "--batch-size",
                args.batch_size,
                "--max-seq-len",
                args.max_seq_len,
                "--mode",
                args.batch_mode,
            ]
        )

    if not args.skip_build:
        cmake_exe = find_cmake(args.cmake)
        configure = [
            cmake_exe,
            "-S",
            CPP_DIR,
            "-B",
            build_dir,
            f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}",
        ]
        if platform.system() != "Windows":
            configure.append("-DCMAKE_BUILD_TYPE=Release")
        run(configure)
        run([cmake_exe, "--build", build_dir, "--config", "Release", "--parallel", str(os.cpu_count() or 2)])

    exe = executable_path(build_dir)
    run(
        [
            exe,
            "--batch",
            batch_path,
            "--device",
            args.device,
            "--warmup",
            args.warmup,
            "--iters",
            args.iters,
            "--vocab-size",
            config.vocab_size,
            "--lstm-size",
            config.lstm_size,
            "--K",
            config.K,
            "--n-mixtures",
            config.n_mixtures,
            "--kappa-initial-bias",
            config.kappa_initial_bias,
            "--lr",
            config.learning_rate,
            "--attention-loss-weight",
            config.attention_loss_weight,
            "--grad-clip",
            config.grad_clip,
            "--max-pen-up-pos-weight",
            config.max_pen_up_pos_weight,
        ]
    )


if __name__ == "__main__":
    main()
