#
# Copyright (C) 2026, SNU
# SNU VGI lab
# All rights reserved.
#
# Shared helpers for multi-scene, multi-GPU parallel training/evaluation.
#

import argparse
import concurrent.futures
import os
import shlex
import subprocess
import sys
from pathlib import Path
from queue import Queue


DEFAULT_DATA_PATH = "/path/to/TLC-Calib"
DEFAULT_TRAIN_ARGS = [
    "--eval",
    "--from_lidar",
    "--use_rig",
    "--opt_pose",
    "--pose_scheduler",
    "--adaptive_voxel",
]
DATASET_ALIASES = {
    "kitti-360": "kitti-360",
    "fast-livo2": "fast-livo2",
    "waymo": "waymo",
}
PARALLEL_CONTROL_ARGS = {
    "--parallel",
    "-p",
    "--data_path",
    "-d",
    "--output_path",
    "-o",
    "--datasets",
    "--scenes",
    "--gpus",
    "--repeat",
    "--skip_nvs",
    "-s",
    "--source_path",
    "-m",
    "--model_path",
}
PARALLEL_CONTROL_VALUE_ARGS = {
    "--data_path",
    "-d",
    "--output_path",
    "-o",
    "--datasets",
    "--scenes",
    "--gpus",
    "--repeat",
    "-s",
    "--source_path",
    "-m",
    "--model_path",
}


PARALLEL_MULTI_VALUE_ARGS = {
    "--datasets",
    "--scenes",
    "--gpus",
}


def filter_subprocess_train_argv(argv):
    filtered = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in PARALLEL_CONTROL_ARGS:
            i += 1
            if arg in PARALLEL_CONTROL_VALUE_ARGS:
                if arg in PARALLEL_MULTI_VALUE_ARGS:
                    while i < len(argv) and not argv[i].startswith("-"):
                        i += 1
                else:
                    i += 1
            continue
        filtered.append(arg)
        i += 1
    return filtered


def normalize_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name.lower(), dataset_name.lower())


def discover_scenes(data_path, datasets=None, scenes=None):
    data_path = Path(data_path)
    requested_datasets = {normalize_dataset_name(name) for name in datasets or []}
    requested_scenes = set(scenes or [])

    discovered = []
    for dataset_dir in sorted(path for path in data_path.iterdir() if path.is_dir()):
        dataset_key = normalize_dataset_name(dataset_dir.name)
        if requested_datasets and dataset_key not in requested_datasets and dataset_dir.name not in requested_datasets:
            continue

        for scene_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            if requested_scenes and scene_dir.name not in requested_scenes:
                continue
            discovered.append((dataset_key, dataset_dir.name, scene_dir.name, scene_dir))

    return discovered


def run_command(command, cwd, env=None, log_prefix=""):
    printable = " ".join(shlex.quote(str(part)) for part in command)
    prefix = f"{log_prefix} " if log_prefix else ""
    print(f"\n{prefix}\033[1m[CMD]\033[0m {printable}", flush=True)
    return subprocess.run(command, cwd=cwd, env=env, check=True).returncode


def detect_gpu_ids():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return [0]


def resolve_gpu_ids(gpu_args):
    if gpu_args is None:
        return detect_gpu_ids()
    if len(gpu_args) == 1 and gpu_args[0].lower() == "all":
        return detect_gpu_ids()
    return [int(gpu_id) for gpu_id in gpu_args]


def make_run_name(repeat_idx, repeat_count):
    if repeat_count == 1:
        return "eval"
    return f"eval_{repeat_idx + 1:02d}"


def build_jobs(scenes, output_root, repeat):
    jobs = []
    for dataset_key, dataset_dir_name, scene_name, source_path in scenes:
        scene_output_root = output_root / dataset_key / scene_name
        for repeat_idx in range(repeat):
            run_name = make_run_name(repeat_idx, repeat)
            jobs.append(
                {
                    "dataset_key": dataset_key,
                    "dataset_dir_name": dataset_dir_name,
                    "scene_name": scene_name,
                    "source_path": source_path,
                    "model_path": scene_output_root / run_name,
                    "scene_output_root": scene_output_root,
                }
            )
    return jobs


def make_env(gpu_id=None):
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return env


def build_train_command(repo_root, source_path, model_path, dataset_key, train_extra_args=None, use_default_train_args=True):
    extra_args = list(train_extra_args or [])
    if use_default_train_args:
        extra_args = [*DEFAULT_TRAIN_ARGS, *extra_args]

    return [
        sys.executable,
        str(Path(repo_root) / "train.py"),
        "-s",
        str(source_path),
        "-m",
        str(model_path),
        "--dataset",
        dataset_key,
        *extra_args,
    ]


def run_single_train(repo_root, job, gpu_id, train_extra_args, use_default_train_args=True):
    log_prefix = f"[GPU {gpu_id}][{job['dataset_dir_name']}/{job['scene_name']}]"
    env = make_env(gpu_id)
    model_path = Path(job["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    command = build_train_command(
        repo_root=repo_root,
        source_path=job["source_path"],
        model_path=model_path,
        dataset_key=job["dataset_key"],
        train_extra_args=train_extra_args,
        use_default_train_args=use_default_train_args,
    )
    run_command(command, cwd=repo_root, env=env, log_prefix=log_prefix)
    return log_prefix


def run_single_eval(repo_root, job, gpu_id, model_iter, train_extra_args, skip_nvs=False, use_default_train_args=True):
    log_prefix = f"[GPU {gpu_id}][{job['dataset_dir_name']}/{job['scene_name']}]"
    env = make_env(gpu_id)
    model_path = Path(job["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)

    train_command = build_train_command(
        repo_root=repo_root,
        source_path=job["source_path"],
        model_path=model_path,
        dataset_key=job["dataset_key"],
        train_extra_args=train_extra_args,
        use_default_train_args=use_default_train_args,
    )
    run_command(train_command, cwd=repo_root, env=env, log_prefix=log_prefix)

    pose_command = [
        sys.executable,
        str(Path(repo_root) / "metrics_pose.py"),
        "-m",
        str(model_path),
    ]
    run_command(pose_command, cwd=repo_root, env=env, log_prefix=log_prefix)

    if not skip_nvs:
        nvs_command = [
            sys.executable,
            str(Path(repo_root) / "metrics_nvs.py"),
            "-m",
            str(model_path),
            "--model_iter",
            str(model_iter),
        ]
        run_command(nvs_command, cwd=repo_root, env=env, log_prefix=log_prefix)

    return log_prefix


def run_parallel_jobs(jobs, gpu_ids, worker_fn, label="Parallel"):
    if not gpu_ids:
        raise RuntimeError("No GPUs available for parallel execution.")

    gpu_queue = Queue()
    for gpu_id in gpu_ids:
        gpu_queue.put(gpu_id)

    print(
        f"\n\033[1;34m[{label}]\033[0m "
        f"{len(jobs)} job(s), {len(gpu_ids)} GPU(s): {gpu_ids}",
        flush=True,
    )

    failed_jobs = []

    def run_job(job):
        gpu_id = gpu_queue.get()
        log_prefix = f"[GPU {gpu_id}][{job['dataset_dir_name']}/{job['scene_name']}]"
        try:
            print(f"\n{log_prefix} started", flush=True)
            worker_fn(job, gpu_id)
            print(f"\n{log_prefix} finished", flush=True)
            return job, None
        except subprocess.CalledProcessError as exc:
            print(f"\n\033[91m{log_prefix} failed with exit code {exc.returncode}\033[0m", flush=True)
            return job, exc
        finally:
            gpu_queue.put(gpu_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        results = list(executor.map(run_job, jobs))

    for job, error in results:
        if error is not None:
            failed_jobs.append((job, error))

    if failed_jobs:
        details = ", ".join(
            f"{job['dataset_dir_name']}/{job['scene_name']}" for job, _ in failed_jobs
        )
        raise RuntimeError(f"Parallel execution failed for scene(s): {details}")


def run_parallel_train(
    repo_root,
    data_path,
    output_root,
    datasets=None,
    scenes=None,
    repeat=1,
    gpu_ids=None,
    train_extra_args=None,
    use_default_train_args=True,
):
    repo_root = Path(repo_root)
    data_path = Path(data_path)
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    discovered = discover_scenes(data_path, datasets, scenes)
    if not discovered:
        raise RuntimeError(f"No scenes found under {data_path}")

    jobs = build_jobs(discovered, output_root, repeat)
    gpu_ids = resolve_gpu_ids(gpu_ids)
    train_extra_args = train_extra_args or []

    def worker(job, gpu_id):
        run_single_train(repo_root, job, gpu_id, train_extra_args, use_default_train_args=use_default_train_args)

    run_parallel_jobs(jobs, gpu_ids, worker, label="Parallel Train")


def run_parallel_eval(
    repo_root,
    data_path,
    output_root,
    datasets=None,
    scenes=None,
    repeat=1,
    model_iter=30000,
    gpu_ids=None,
    train_extra_args=None,
    skip_nvs=False,
):
    repo_root = Path(repo_root)
    data_path = Path(data_path)
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = repo_root / output_root

    discovered = discover_scenes(data_path, datasets, scenes)
    if not discovered:
        raise RuntimeError(f"No scenes found under {data_path}")

    jobs = build_jobs(discovered, output_root, repeat)
    gpu_ids = resolve_gpu_ids(gpu_ids)
    train_extra_args = train_extra_args or []

    def worker(job, gpu_id):
        run_single_eval(
            repo_root,
            job,
            gpu_id,
            model_iter=model_iter,
            train_extra_args=train_extra_args,
            skip_nvs=skip_nvs,
        )

    run_parallel_jobs(jobs, gpu_ids, worker, label="Parallel Eval")
    return jobs, output_root


def split_train_extra_args(argv):
    return filter_subprocess_train_argv(argv)


def add_parallel_train_args(parser):
    parser.add_argument(
        "--parallel",
        "-p",
        action="store_true",
        help="Run multiple scenes in parallel across GPUs. Requires --data_path and -m/--output_path.",
    )
    parser.add_argument(
        "--data_path",
        "-d",
        type=str,
        default=None,
        help="Dataset root for parallel mode, e.g. data/TLC-Calib.",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        type=str,
        default=None,
        help="Output root for parallel mode. Alias of -m when --parallel is set.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Dataset names to run in parallel mode, e.g. kitti-360 fast-livo2.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help="Scene names to run in parallel mode.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=None,
        help="GPU ids for parallel mode, e.g. --gpus 0 1 2 3 or --gpus all.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of runs per scene in parallel mode.",
    )


def run_parallel_train_cli():
    parser = argparse.ArgumentParser(
        description="TLC-Calib multi-scene multi-GPU parallel training launcher.",
        add_help=True,
    )
    add_parallel_train_args(parser)
    args, train_extra_args = parser.parse_known_args()

    if not args.parallel:
        parser.error("Parallel mode requires --parallel.")

    output_root = args.output_path
    if output_root is None:
        parser.error("Parallel mode requires --output_path or -m.")

    data_path = args.data_path or DEFAULT_DATA_PATH
    repo_root = Path(__file__).resolve().parent.parent

    if args.gpus is not None and not args.parallel:
        args.parallel = True

    run_parallel_train(
        repo_root=repo_root,
        data_path=data_path,
        output_root=output_root,
        datasets=args.datasets,
        scenes=args.scenes,
        repeat=args.repeat,
        gpu_ids=args.gpus,
        train_extra_args=train_extra_args,
    )


def launch_parallel_train_from_train_args(args, train_extra_args):
    repo_root = Path(__file__).resolve().parent.parent
    output_root = args.output_path or args.model_path
    if not output_root:
        raise RuntimeError("Parallel training requires --output_path or -m as the output root.")

    data_path = args.data_path or DEFAULT_DATA_PATH
    passthrough_args = filter_subprocess_train_argv(sys.argv[1:] + train_extra_args)
    run_parallel_train(
        repo_root=repo_root,
        data_path=data_path,
        output_root=output_root,
        datasets=args.datasets,
        scenes=args.scenes,
        repeat=args.repeat,
        gpu_ids=args.gpus,
        train_extra_args=passthrough_args,
        use_default_train_args=False,
    )
