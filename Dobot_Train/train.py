#!/usr/bin/env python3
"""Train a selected LeRobot policy using the editable settings below."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
import importlib.util
import json
import math
import os
import re
import signal
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

# 本次训练读取的数据集任务名。该模块保持独立，拆分使用时可单独修改。
TRAIN_TASK_NAME = "cosmetics_quality"

# 实验名称，用于生成训练任务名；默认沿用本脚本的训练任务名。
EXPERIMENT_NAME = TRAIN_TASK_NAME

# LeRobot 数据集的逻辑标识，必须与转换脚本中的 REPO_ID 及数据集元数据一致。
# 它用于 LeRobot 识别数据集，不是磁盘路径。
DATASET_REPO_ID = f"local/dobot_cr5a_{TRAIN_TASK_NAME}"

# 转换后的 LeRobot 数据集本地目录，其中应包含 data、meta、videos/images 等内容。
DATASET_ROOT = f"datasets/{TRAIN_TASK_NAME}/{TRAIN_TASK_NAME}_lerobot"

# False 保留转换数据集自身的图像统计，不用 ImageNet 统计覆盖。
# 该设置与已验证的服务器训练命令保持一致。
DATASET_USE_IMAGENET_STATS = False

# 训练计算设备：使用 NVIDIA GPU 时设为 "cuda"，仅用 CPU 时设为 "cpu"。
DEVICE = "cuda"

# 指定当前进程可见的 GPU 编号；"0" 表示只使用第 1 块 GPU。
# 多个编号可写为 "0,1"，但是否支持多卡还取决于 LeRobot 训练实现。
CUDA_VISIBLE_DEVICES = "0"

# 优化器更新总步数，不是数据集遍历轮数；数值越大，训练时间通常越长。
STEPS = 30000

# 每个训练步骤使用的样本数。1 是单张 24 GB RTX 4090 的保守默认值；
# 确认显存余量后可尝试增大到 2。
BATCH_SIZE = 1

# DataLoader 并行读取、解码和预处理数据的 CPU 子进程数量。
# 数值过小可能让 GPU 等待数据，过大则会增加内存占用和进程竞争。
NUM_WORKERS = 4

# 每个 DataLoader worker 提前准备的 batch 数；当前最多预取约 4 x 2 个 batch。
PREFETCH_FACTOR = 2

# 每隔多少个训练步骤输出一次 loss、学习率、数据耗时和显存等日志。
LOG_FREQUENCY = 200

# 每隔多少个训练步骤保存一次 checkpoint；30000 步时会生成 030000 最终结果。
# 保存过于频繁会占用较多磁盘空间。
SAVE_FREQUENCY = 5000

# 固定为 outputs/train/<采集任务名>，让推理脚本可直接定位最终 checkpoint。
# 再次训练同名任务前，需要清理旧输出或修改 TRAIN_TASK_NAME。
OUTPUT_DIR = f"outputs/train/{TRAIN_TASK_NAME}"
JOB_NAME = TRAIN_TASK_NAME

WANDB_MODE = "disabled"
ENABLE_WANDB = False
PUSH_TO_HUB = False

# 首次使用官方模型 ID 时保持 False，允许自动下载基础模型和 tokenizer。
# 下载完成后可改为 True，仅使用本地 Hugging Face 缓存进行训练。
HUGGINGFACE_OFFLINE = False

# 单张 RTX 4090 的 Pi0.5 配置，只在 SELECTED_POLICY="pi05" 时生效。
# 可填写 Hugging Face 模型 ID，或改为本机完整的 Pi0.5 基础模型目录。
PI05_PRETRAINED_PATH = "lerobot/pi05_base"
# RTX 4090 支持 BF16，可显著降低训练显存占用。
PI05_DTYPE = "bfloat16"
# 每次预测 50 帧动作；项目 30 Hz 控制周期下约覆盖 1.67 秒。
PI05_CHUNK_SIZE = 50
# 非 RTC 模式最多连续执行的动作数；必须小于或等于 CHUNK_SIZE。
PI05_N_ACTION_STEPS = 50
# 数据集记录绝对 TCP 目标，False 表示不转换为相对动作。
PI05_USE_RELATIVE_ACTIONS = False
# 通过重算中间激活降低显存占用，单张 4090 建议保持开启。
PI05_GRADIENT_CHECKPOINTING = True
# 冻结视觉语言模型，只微调动作专家和投影层，适配 24 GB 显存。
PI05_TRAIN_EXPERT_ONLY = True
# train_expert_only 已冻结整个 VLM，无需额外冻结视觉编码器。
PI05_FREEZE_VISION_ENCODER = False
# 首次训练关闭 torch.compile，减少启动时间和排障复杂度。
PI05_COMPILE_MODEL = False

# 训练时实时解析 LeRobot 日志并绘制指标曲线。
ENABLE_TRAINING_METRICS = True
# 指标输出根目录；每次训练会创建 <METRICS_OUTPUT_ROOT>/<job_name>/。
METRICS_OUTPUT_ROOT = "outputs/Loss"
# 每收到多少条 step 指标日志重绘一次 PNG；1 表示每次日志都更新。
PLOT_EVERY_METRIC_ROWS = 1

# 所有策略共用的最终覆盖项。这里只能填写每一种可选策略都支持的字段；
# 通常保持为空，避免切换模型时把某个模型的专用参数传给另一个模型。
POLICY_OVERRIDES: dict[str, Any] = {}

# 各策略独立的覆盖项。脚本只会读取当前 SELECTED_POLICY 对应的配置，
# 因此切换 ACT、Pi0、Pi0.5 等策略时不会发生参数串用。
POLICY_OVERRIDES_BY_TYPE: dict[str, dict[str, Any]] = {
    "act": {
        # ACT 从头训练策略网络；视觉骨干使用 ImageNet 预训练 ResNet18。
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
        # 每次预测未来 30 帧动作，30 FPS 下约等于 1 秒。
        "chunk_size": 30,
        # 推理执行 10 帧后重新观察双路相机并重新预测。
        "n_action_steps": 10,
        # ACT 当前只支持单帧 observation。
        "n_obs_steps": 1,
        # Transformer 主干规模，适合 RTX 4090 和当前双路 640x480 数据。
        "dim_model": 512,
        "n_heads": 8,
        "dim_feedforward": 3200,
        "n_encoder_layers": 4,
        "n_decoder_layers": 1,
        # 使用 ACT 的 VAE 训练目标。
        "use_vae": True,
        "latent_dim": 32,
        "n_vae_encoder_layers": 4,
        "dropout": 0.1,
        "kl_weight": 10.0,
    },
    "pi0": {
        # 从官方 Pi0 基础模型开始微调。
        "pretrained_path": "lerobot/pi0_base",
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
        # 当前数据是绝对 TCP 位姿，不切换为相对动作。
        "use_relative_actions": False,
    },
    "pi05": {
        # 从官方 Pi0.5 基础模型开始微调。
        "pretrained_path": PI05_PRETRAINED_PATH,
        # RTX 4090 支持 BF16，可显著降低显存占用。
        "dtype": PI05_DTYPE,
        # 用额外计算换取更低的训练显存。
        "gradient_checkpointing": PI05_GRADIENT_CHECKPOINTING,
        # 第一次训练先关闭，便于排查加载和训练问题。
        "compile_model": PI05_COMPILE_MODEL,
        # 冻结视觉语言模型，只训练动作专家和投影层。
        "train_expert_only": PI05_TRAIN_EXPERT_ONLY,
        # train_expert_only 已经会冻结整个 VLM。
        "freeze_vision_encoder": PI05_FREEZE_VISION_ENCODER,
        # 与 scripts/Dobot_inference.py 的 RTC 动作块保持一致。
        "chunk_size": PI05_CHUNK_SIZE,
        "n_action_steps": PI05_N_ACTION_STEPS,
        # 当前数据是绝对 TCP 位姿，第一轮不要切换为相对动作。
        "use_relative_actions": PI05_USE_RELATIVE_ACTIONS,
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
    "policy": {},
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
        "use_imagenet_stats": DATASET_USE_IMAGENET_STATS,
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


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
METRIC_RE = re.compile(
    r"(?<!\S)([A-Za-z][A-Za-z0-9_/]*)\s*[:=]\s*"
    r"(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?[KMG]?)"
)


def parse_training_metrics(line: str) -> dict[str, float] | None:
    """Parse one LeRobot `step:... loss:...` training log line."""
    clean = ANSI_ESCAPE_RE.sub("", line).strip()
    step_match = re.search(r"(?:^|\s)step\s*[:=]\s*\d+", clean)
    if step_match is None:
        return None
    values: dict[str, float] = {}
    for key, raw_value in METRIC_RE.findall(clean[step_match.start() :]):
        multiplier = 1.0
        if raw_value[-1:] in {"K", "M", "G"}:
            multiplier = {"K": 1e3, "M": 1e6, "G": 1e9}[raw_value[-1]]
            raw_value = raw_value[:-1]
        values[key] = float(raw_value) * multiplier
    return values if "step" in values else None


class TrainingMetricsRecorder:
    """Save training logs, structured metrics, and live PNG line charts."""

    def __init__(self, config: dict[str, Any]) -> None:
        train = require_table(config, "train")
        job_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(train["job_name"]))
        root = Path(METRICS_OUTPUT_ROOT).expanduser()
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        self.output_dir = root / job_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, float]] = []
        self.rows_since_plot = 0
        self.log_handle = (self.output_dir / "training.log").open(
            "w", encoding="utf-8", buffering=1
        )
        (self.output_dir / "run_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def ingest(self, text: str) -> None:
        clean = ANSI_ESCAPE_RE.sub("", text).replace("\r", "\n")
        self.log_handle.write(clean)
        for line in clean.splitlines():
            metrics = parse_training_metrics(line)
            if metrics is None:
                continue
            if self.rows and self.rows[-1].get("step") == metrics["step"]:
                self.rows[-1].update(metrics)
            else:
                self.rows.append(metrics)
            self.rows_since_plot += 1
            self.save_tables()
            if self.rows_since_plot >= max(PLOT_EVERY_METRIC_ROWS, 1):
                self.save_plots()
                self.rows_since_plot = 0

    def close(self) -> None:
        try:
            if self.rows:
                self.save_tables()
                self.save_plots()
        finally:
            self.log_handle.close()

    def save_tables(self) -> None:
        preferred = [
            "step", "loss", "l1_loss", "kld_loss", "grdn", "lr",
            "data_s", "updt_s", "smp/s", "mem_gb", "smpl", "ep", "epch",
        ]
        all_fields = {key for row in self.rows for key in row}
        fields = [key for key in preferred if key in all_fields]
        fields.extend(sorted(all_fields - set(fields)))
        with (self.output_dir / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.rows)
        (self.output_dir / "metrics.json").write_text(
            json.dumps(self.rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def save_plots(self) -> None:
        available = {key for row in self.rows for key in row}
        loss_keys = sorted(key for key in available if "loss" in key.lower())
        charts = [
            ("loss_curves.png", "Training Loss", loss_keys),
            ("gradient_norm.png", "Gradient Norm", ["grdn"]),
            ("learning_rate.png", "Learning Rate", ["lr"]),
            ("timing.png", "Step Timing", ["data_s", "updt_s"]),
            ("throughput.png", "Samples Per Second", ["smp/s"]),
            ("gpu_memory.png", "GPU Memory (GB)", ["mem_gb"]),
        ]
        for filename, title, keys in charts:
            present = [key for key in keys if key in available]
            if present:
                self.draw_chart(self.output_dir / filename, title, present)

    def draw_chart(self, path: Path, title: str, keys: list[str]) -> None:
        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as exc:
            print(f"[metrics warning] Cannot create PNG charts without Pillow: {exc}")
            return

        width, height = 1400, 800
        left, top, right, bottom = 110, 85, 45, 90
        plot_right, plot_bottom = width - right, height - bottom
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        colors = ["#1565C0", "#D32F2F", "#2E7D32", "#6A1B9A", "#EF6C00"]
        series: dict[str, list[tuple[float, float]]] = {}
        for key in keys:
            points = [
                (float(row["step"]), float(row[key]))
                for row in self.rows
                if "step" in row and key in row and math.isfinite(float(row[key]))
            ]
            if points:
                series[key] = points
        if not series:
            return

        x_values = [x for points in series.values() for x, _ in points]
        y_values = [y for points in series.values() for _, y in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        if x_min == x_max:
            x_max = x_min + 1.0
        if y_min == y_max:
            padding = max(abs(y_min) * 0.05, 1e-9)
        else:
            padding = (y_max - y_min) * 0.08
        y_min, y_max = y_min - padding, y_max + padding

        def px(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * (plot_right - left)

        def py(value: float) -> float:
            return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - top)

        draw.text((left, 28), title, fill="#111111", font=font)
        for index in range(6):
            ratio = index / 5
            y = top + ratio * (plot_bottom - top)
            value = y_max - ratio * (y_max - y_min)
            draw.line((left, y, plot_right, y), fill="#E0E0E0", width=1)
            draw.text((8, y - 7), f"{value:.4g}", fill="#424242", font=font)
            x = left + ratio * (plot_right - left)
            step = x_min + ratio * (x_max - x_min)
            draw.line((x, top, x, plot_bottom), fill="#F0F0F0", width=1)
            draw.text((x - 18, plot_bottom + 12), f"{step:.0f}", fill="#424242", font=font)
        draw.line((left, top, left, plot_bottom), fill="#333333", width=2)
        draw.line((left, plot_bottom, plot_right, plot_bottom), fill="#333333", width=2)
        draw.text(((left + plot_right) / 2 - 15, height - 35), "Step", fill="#222222", font=font)

        legend_x = plot_right - 180
        for index, (key, points) in enumerate(series.items()):
            color = colors[index % len(colors)]
            coordinates = [(px(x), py(y)) for x, y in points]
            if len(coordinates) == 1:
                x, y = coordinates[0]
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color)
            else:
                draw.line(coordinates, fill=color, width=3, joint="curve")
            legend_y = 28 + index * 18
            draw.line((legend_x, legend_y + 5, legend_x + 24, legend_y + 5), fill=color, width=3)
            draw.text((legend_x + 32, legend_y), key, fill="#222222", font=font)

        temporary = path.with_suffix(path.suffix + ".tmp")
        image.save(temporary, format="PNG")
        temporary.replace(path)


def run_training_with_metrics(
    command: list[str], environment: dict[str, str], config: dict[str, Any]
) -> int:
    recorder = TrainingMetricsRecorder(config) if ENABLE_TRAINING_METRICS else None
    if recorder is not None:
        print(f"Training metrics: {recorder.output_dir}")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if recorder is not None:
                try:
                    recorder.ingest(line)
                except Exception as exc:
                    print(f"[metrics warning] Could not update metrics: {exc}")
        return process.wait()
    except KeyboardInterrupt:
        print("\nStopping training and saving final metric plots...", flush=True)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        return 130
    finally:
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:
                print(f"[metrics warning] Could not save final metrics: {exc}")


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
    return run_training_with_metrics(command, environment, config)


if __name__ == "__main__":
    raise SystemExit(main())
