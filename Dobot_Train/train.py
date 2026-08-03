#!/usr/bin/env python3
"""Train a selected LeRobot policy using the editable settings below."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORWARDED_SECTIONS = {"dataset", "policy", "optimizer", "scheduler", "eval"}
ROOT_TRAIN_SECTION = "train"

# =============================================================================
# USER SETTINGS: edit this section, then run `python Dobot_Train/train.py`.
# =============================================================================

# 选择 LeRobot 训练策略，例如 act、pi0、pi05。
# 必须是本脚本 SUPPORTED_POLICY_TYPES 中已注册且环境依赖完整的策略名称。
SELECTED_POLICY = "pi05"

# 实验名称，用于生成训练任务名和输出目录；建议包含任务、模型或数据版本信息。
EXPERIMENT_NAME = "test_pi05_euler_aligned"

# LeRobot 数据集的逻辑标识，必须与转换脚本中的 REPO_ID 及数据集元数据一致。
# 它用于 LeRobot 识别数据集，不是磁盘路径。
DATASET_REPO_ID = "local/dobot_cr5a_test123"

# 转换后的 LeRobot 数据集本地目录，其中应包含 data、meta、videos/images 等内容。
DATASET_ROOT = "datasets/test123/test123_lerobot"

# 训练计算设备：使用 NVIDIA GPU 时设为 "cuda"，仅用 CPU 时设为 "cpu"。
DEVICE = "cuda"

# 指定当前进程可见的 GPU 编号；"0" 表示只使用第 1 块 GPU。
# 多个编号可写为 "0,1"，但是否支持多卡还取决于 LeRobot 训练实现。
CUDA_VISIBLE_DEVICES = "0"

# 优化器更新总步数，不是数据集遍历轮数；数值越大，训练时间通常越长。
STEPS = 3000

# 每个训练步骤使用的样本数。增大可提高吞吐量，但会明显增加显存占用。
BATCH_SIZE = 1

# DataLoader 并行读取、解码和预处理数据的 CPU 子进程数量。
# 数值过小可能让 GPU 等待数据，过大则会增加内存占用和进程竞争。
NUM_WORKERS = 4

# 每个 DataLoader worker 提前准备的 batch 数；当前最多预取约 4 x 2 个 batch。
PREFETCH_FACTOR = 2

# 每隔多少个训练步骤输出一次 loss、学习率、数据耗时和显存等日志。
LOG_FREQUENCY = 50

# 每隔多少个训练步骤保存一次 checkpoint；设为 3000 表示本次只在结束时保存。
# 保存过于频繁会占用较多磁盘空间。
SAVE_FREQUENCY = 3000

# Leave blank to generate:
# outputs/train/<policy>_<experiment>_<device>_bs<batch>_w<workers>_<steps>
OUTPUT_DIR = "outputs/train/test123_pi05"
JOB_NAME = ""

WANDB_MODE = "disabled"
ENABLE_WANDB = False
PUSH_TO_HUB = False

# 已完整下载基础模型和 tokenizer 时设为 True，训练将只使用本地 Hugging Face
# 缓存，不再因网络断开而在模型或 tokenizer 初始化阶段失败。
HUGGINGFACE_OFFLINE = True

# 所有策略共用的最终覆盖项。这里只能填写每一种可选策略都支持的字段；
# 通常保持为空，避免切换模型时把某个模型的专用参数传给另一个模型。
POLICY_OVERRIDES: dict[str, Any] = {}

# 各策略独立的覆盖项。脚本只会读取当前 SELECTED_POLICY 对应的配置，
# 因此切换 ACT、Pi0、Pi0.5 等策略时不会发生参数串用。
POLICY_OVERRIDES_BY_TYPE: dict[str, dict[str, Any]] = {
    "pi05": {
        # 从官方 Pi0.5 基础模型开始微调。
        "pretrained_path": "lerobot/pi05_base",
        # RTX 4090 支持 BF16，可显著降低显存占用。
        "dtype": "bfloat16",
        # 用额外计算换取更低的训练显存。
        "gradient_checkpointing": True,
        # 第一次训练先关闭，便于排查加载和训练问题。
        "compile_model": False,
        # 冻结视觉语言模型，只训练动作专家和投影层。
        "train_expert_only": True,
        # train_expert_only 已经会冻结整个 VLM。
        "freeze_vision_encoder": False,
        # 每次预测未来 30 帧动作，30 FPS 下约等于 1 秒。
        "chunk_size": 30,
        # 推理时执行 10 帧后重新观察相机并重新预测。
        "n_action_steps": 10,
        # 当前数据是绝对 TCP 位姿，第一轮不要切换为相对动作。
        "use_relative_actions": False,
    },
}

# Set either value to a complete LeRobot optimizer/scheduler table to replace
# the selected policy's training preset. None keeps the policy preset.
CUSTOM_OPTIMIZER: dict[str, Any] | None = None
CUSTOM_SCHEDULER: dict[str, Any] | None = None

# Optional final overrides for common dataset/train fields.
DATASET_OVERRIDES: dict[str, Any] = {}
TRAIN_OVERRIDES: dict[str, Any] = {}

# =============================================================================
# POLICY PRESETS: these names are registered by this repository's LeRobot.
# Empty policy tables intentionally use LeRobot's own defaults.
# Large VLA policies can require model downloads, extra dependencies, a
# language task prompt in the dataset, and substantially more GPU memory.
# =============================================================================

SUPPORTED_POLICY_TYPES = (
    "act",
    "diffusion",
    "eo1",
    "evo1",
    "fastwam",
    "gaussian_actor",
    "groot",
    "lingbot_va",
    "molmoact2",
    "multi_task_dit",
    "pi0",
    "pi05",
    "pi0_fast",
    "smolvla",
    "tdmpc",
    "vla_jepa",
    "vqbet",
    "wall_x",
    "xvla",
)

LANGUAGE_CONDITIONED_POLICIES = {
    "eo1",
    "evo1",
    "fastwam",
    "groot",
    "lingbot_va",
    "molmoact2",
    "pi0",
    "pi05",
    "pi0_fast",
    "smolvla",
    "vla_jepa",
    "wall_x",
    "xvla",
}

POLICY_PRESETS: dict[str, dict[str, Any]] = {
    name: {"policy": {}} for name in SUPPORTED_POLICY_TYPES
}
POLICY_PRESETS["act"] = {
    "policy": {
        "chunk_size": 30,
        "n_action_steps": 10,
        "vision_backbone": "resnet18",
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        "use_vae": True,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "dropout": 0.1,
        "kl_weight": 10.0,
    },
    "optimizer": {
        "type": "adamw",
        "lr": 0.00001,
        "weight_decay": 0.0001,
        "grad_clip_norm": 10.0,
    },
}
POLICY_PRESETS["diffusion"] = {
    "policy": {
        "n_obs_steps": 2,
        "horizon": 64,
        "n_action_steps": 32,
        "vision_backbone": "resnet18",
        "use_separate_rgb_encoder_per_camera": True,
    }
}

# =============================================================================
# END USER SETTINGS
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start LeRobot training from the editable settings in this file."
    )
    parser.add_argument(
        "--policy",
        choices=SUPPORTED_POLICY_TYPES,
        default=None,
        help="Temporarily override SELECTED_POLICY for this run.",
    )
    parser.add_argument(
        "--list-policies",
        action="store_true",
        help="Print policy names supported by this LeRobot source tree and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and print the generated command without training.",
    )
    return parser.parse_args()


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def generated_run_name(policy_type: str) -> str:
    device_label = "gpu" if DEVICE.startswith("cuda") else DEVICE.replace(":", "")
    return (
        f"{policy_type}_{EXPERIMENT_NAME}_{device_label}_"
        f"bs{BATCH_SIZE}_w{NUM_WORKERS}_{STEPS}"
    )


def build_inline_config(policy_type: str | None = None) -> dict[str, Any]:
    selected = (policy_type or SELECTED_POLICY).strip().lower()
    if selected not in POLICY_PRESETS:
        supported = ", ".join(SUPPORTED_POLICY_TYPES)
        raise ValueError(
            f"Unsupported SELECTED_POLICY={selected!r}. Available policies: {supported}"
        )

    run_name = generated_run_name(selected)
    policy = {
        "type": selected,
        "device": DEVICE,
        "push_to_hub": PUSH_TO_HUB,
    }
    policy = deep_merge(policy, POLICY_PRESETS[selected].get("policy", {}))
    policy = deep_merge(policy, POLICY_OVERRIDES_BY_TYPE.get(selected, {}))
    policy = deep_merge(policy, POLICY_OVERRIDES)

    dataset = {
        "repo_id": DATASET_REPO_ID,
        "root": DATASET_ROOT,
        "return_uint8": True,
    }
    dataset = deep_merge(dataset, DATASET_OVERRIDES)

    train = {
        "output_dir": OUTPUT_DIR.strip() or f"outputs/train/{run_name}",
        "job_name": JOB_NAME.strip() or run_name,
        "steps": STEPS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "log_freq": LOG_FREQUENCY,
        "save_freq": SAVE_FREQUENCY,
        "save_checkpoint": True,
        "env_eval_freq": 0,
        "wandb": {"enable": ENABLE_WANDB},
    }
    if NUM_WORKERS > 0:
        train["prefetch_factor"] = PREFETCH_FACTOR
        train["persistent_workers"] = True
    train = deep_merge(train, TRAIN_OVERRIDES)

    config: dict[str, Any] = {
        "runtime": {
            "wandb_mode": WANDB_MODE,
            "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
        },
        "dataset": dataset,
        "policy": policy,
        "train": train,
    }

    optimizer = (
        deepcopy(CUSTOM_OPTIMIZER)
        if CUSTOM_OPTIMIZER is not None
        else deepcopy(POLICY_PRESETS[selected].get("optimizer"))
    )
    scheduler = (
        deepcopy(CUSTOM_SCHEDULER)
        if CUSTOM_SCHEDULER is not None
        else deepcopy(POLICY_PRESETS[selected].get("scheduler"))
    )
    if optimizer:
        config["optimizer"] = optimizer
    if scheduler:
        config["scheduler"] = scheduler
    return config


def cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        return json.dumps(value, separators=(",", ","))
    raise TypeError(f"Unsupported configuration value: {value!r}")


def flatten_options(prefix: str, values: dict[str, Any]) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"Invalid empty configuration key under '{prefix or 'train'}'.")
        option_name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            options.extend(flatten_options(option_name, value))
        else:
            options.append((option_name, cli_value(value)))
    return options


def require_table(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a dictionary.")
    return value


def validate_dataset_for_policy(dataset_root: Path, policy_type: str) -> None:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot metadata not found: {info_path}")
    with info_path.open("r", encoding="utf-8") as handle:
        info = json.load(handle)
    features = info.get("features", {})
    required = {
        "observation.state",
        "action",
        "observation.images.top",
        "observation.images.wrist",
    }
    missing = sorted(required - set(features))
    if missing:
        raise ValueError(
            "Dataset is missing required Dobot dual-camera features: "
            + ", ".join(missing)
        )

    report_path = dataset_root / "conversion_report.json"
    report: dict[str, Any] = {}
    if report_path.is_file():
        with report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        quality_gate = report.get("quality_gate")
        if isinstance(quality_gate, dict) and not bool(quality_gate.get("passed", False)):
            violations = quality_gate.get("violations", [])
            raise ValueError(
                "Dataset conversion quality gate did not pass: "
                + ", ".join(str(item) for item in violations)
            )

    if policy_type in LANGUAGE_CONDITIONED_POLICIES:
        task_prompt = report.get("task_prompt")
        if not isinstance(task_prompt, str) or not task_prompt.strip():
            raise ValueError(
                f"Policy '{policy_type}' is language-conditioned, but "
                "conversion_report.json has no task_prompt. Reconvert with "
                "--task-prompt before training this policy."
            )


def validate_policy_configuration(config: dict[str, Any]) -> None:
    if importlib.util.find_spec("lerobot") is None:
        raise RuntimeError(
            "LeRobot is not installed in this Python environment. "
            "Run: conda activate dobot-lerobot"
        )
    try:
        from lerobot.policies import make_policy_config
    except ImportError as exc:
        raise RuntimeError(f"Cannot import LeRobot policy factory: {exc}") from exc

    policy = deepcopy(require_table(config, "policy"))
    policy_type = str(policy.pop("type"))
    # This pass validates field names and optional dependencies only. Device
    # availability is checked separately immediately before real training.
    policy["device"] = "cpu"
    try:
        make_policy_config(policy_type, **policy)
    except Exception as exc:
        raise ValueError(
            f"Policy '{policy_type}' configuration or dependencies are invalid: {exc}"
        ) from exc


def policy_configuration_status(policy_type: str) -> tuple[bool, str]:
    if importlib.util.find_spec("lerobot") is None:
        return False, "LeRobot is not installed"
    try:
        from lerobot.policies import make_policy_config

        make_policy_config(policy_type, device="cpu", push_to_hub=False)
    except Exception as exc:
        return False, str(exc)
    return True, "configuration available"


def validate_requested_device(config: dict[str, Any]) -> None:
    device = str(require_table(config, "policy").get("device", "")).strip()
    if not device.startswith("cuda"):
        return
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is not installed in this environment") from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Requested policy.device={device}, but torch.cuda.is_available() is False. "
            "Training was not started to prevent an unintended CPU fallback."
        )


def build_command(config: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    allowed_sections = {"runtime", ROOT_TRAIN_SECTION, *FORWARDED_SECTIONS}
    unknown_sections = set(config) - allowed_sections
    if unknown_sections:
        names = ", ".join(sorted(unknown_sections))
        raise ValueError(
            f"Unsupported top-level section(s): {names}. "
            "Use [train], [dataset], [policy], [optimizer], [scheduler], [eval], or [runtime]."
        )

    runtime = require_table(config, "runtime")
    train = require_table(config, ROOT_TRAIN_SECTION)
    dataset = require_table(config, "dataset")
    policy = require_table(config, "policy")

    if not str(dataset.get("root", "")).strip():
        raise ValueError("[dataset].root must point to a converted LeRobot dataset.")
    dataset_root = Path(str(dataset["root"]))
    if not dataset_root.is_absolute():
        dataset_root = PROJECT_ROOT / dataset_root
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Converted LeRobot dataset not found: {dataset_root}")
    dataset["root"] = str(dataset_root)

    if not str(policy.get("type", "")).strip():
        raise ValueError("[policy].type is required, for example 'act'.")
    if not str(policy.get("device", "")).strip():
        raise ValueError("[policy].device is required, for example 'cpu' or 'cuda'.")
    policy_type = str(policy["type"]).strip()
    validate_dataset_for_policy(dataset_root, policy_type)
    if not str(train.get("output_dir", "")).strip():
        raise ValueError("[train].output_dir is required.")
    if int(train.get("steps", 0)) <= 0:
        raise ValueError("[train].steps must be greater than zero.")

    command = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    for name, value in flatten_options("", train):
        command.append(f"--{name}={value}")
    for section in ("dataset", "policy", "optimizer", "scheduler", "eval"):
        values = config.get(section)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section '{section}' must be a dictionary.")
        for name, value in flatten_options(section, values):
            command.append(f"--{name}={value}")

    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    if HUGGINGFACE_OFFLINE:
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
    wandb_mode = runtime.get("wandb_mode")
    if wandb_mode is not None:
        environment["WANDB_MODE"] = cli_value(wandb_mode)
    cuda_visible_devices = str(runtime.get("cuda_visible_devices", "")).strip()
    if cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return command, environment


def main() -> int:
    args = parse_args()
    if args.list_policies:
        print("Available LeRobot policies:")
        for name in SUPPORTED_POLICY_TYPES:
            available, detail = policy_configuration_status(name)
            if name == "act" and available:
                detail = "validated Dobot preset"
            status = "ready" if available else "unavailable"
            print(f"- {name}: {status} ({detail})")
        return 0

    try:
        config = build_inline_config(args.policy)
        config_source = "Dobot_Train/train.py USER SETTINGS"
        validate_policy_configuration(config)
        command, environment = build_command(config)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    print(f"Configuration source: {config_source}")
    print(f"Selected policy: {config['policy']['type']}")
    print(f"Dataset: {config['dataset']['root']}")
    print(f"Output: {config['train']['output_dir']}")
    print("Generated LeRobot command:")
    print(" \\\n  ".join(command))
    if args.dry_run:
        return 0
    try:
        validate_requested_device(config)
    except RuntimeError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    print("\nStarting training. Press Ctrl+C to stop it.")
    return subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
