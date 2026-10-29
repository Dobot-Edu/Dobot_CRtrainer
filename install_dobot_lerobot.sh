#!/usr/bin/env bash
#
# Create or repair the CUDA-enabled LeRobot training/inference environment.
# Usage:
#   bash install_dobot_lerobot.sh
#   bash install_dobot_lerobot.sh --dry-run

set -Eeuo pipefail

ENV_NAME="dobot-lerobot"
PYTHON_VERSION="3.12.13"
DEFAULT_CONDA_ROOT="/home/dobot/miniforge3"
DRY_RUN=false

usage() {
    cat <<'EOF'
用法：bash install_dobot_lerobot.sh [--dry-run] [--help]

创建或修复：
  /home/dobot/miniforge3/envs/dobot-lerobot

安装内容包括 CUDA 版 PyTorch、本项目本地 LeRobot 源码、数据转换、
ACT/Pi0/Pi0.5 训练与推理依赖，以及 RealSense D405+D435 双相机依赖。

选项：
  --dry-run  只显示将执行的命令，不修改环境
  --help     显示帮助

可以通过 CONDA_ROOT 覆盖 Miniforge 根目录，例如：
  CONDA_ROOT="$HOME/miniforge3" bash install_dobot_lerobot.sh
EOF
}

for argument in "$@"; do
    case "$argument" in
        --dry-run)
            DRY_RUN=true
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            printf '未知参数：%s\n\n' "$argument" >&2
            usage >&2
            exit 2
            ;;
    esac
done

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$DEFAULT_CONDA_ROOT}"
CONDA_BIN="$CONDA_ROOT/bin/conda"
ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
ENV_PYTHON="$ENV_PREFIX/bin/python"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
LEROBOT_SOURCE="$PROJECT_ROOT/lerobot-main"

if [[ ! -x "$CONDA_BIN" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BIN="$(command -v conda)"
        CONDA_ROOT="$(cd -- "$(dirname -- "$CONDA_BIN")/.." && pwd)"
        ENV_PREFIX="$CONDA_ROOT/envs/$ENV_NAME"
        ENV_PYTHON="$ENV_PREFIX/bin/python"
    else
        printf '错误：找不到 Conda。请先安装 Miniforge，或设置 CONDA_ROOT。\n' >&2
        exit 1
    fi
fi

if [[ ! -f "$REQUIREMENTS_FILE" ]]; then
    printf '错误：缺少依赖文件：%s\n' "$REQUIREMENTS_FILE" >&2
    exit 1
fi
if [[ ! -f "$LEROBOT_SOURCE/pyproject.toml" ]]; then
    printf '错误：缺少本地 LeRobot 源码：%s\n' "$LEROBOT_SOURCE" >&2
    exit 1
fi

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

run() {
    print_command "$@"
    if [[ "$DRY_RUN" == false ]]; then
        "$@"
    fi
}

export CONDA_REMOTE_CONNECT_TIMEOUT_SECS="${CONDA_REMOTE_CONNECT_TIMEOUT_SECS:-60}"
export CONDA_REMOTE_READ_TIMEOUT_SECS="${CONDA_REMOTE_READ_TIMEOUT_SECS:-3600}"
export CONDA_REMOTE_MAX_RETRIES="${CONDA_REMOTE_MAX_RETRIES:-20}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-3600}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-600}"

cd "$PROJECT_ROOT"

printf '项目目录：%s\n' "$PROJECT_ROOT"
printf '环境名称：%s\n' "$ENV_NAME"
printf '安装路径：%s\n' "$ENV_PREFIX"
printf 'Conda：%s\n' "$CONDA_BIN"

if [[ -f "$ENV_PREFIX/conda-meta/history" ]]; then
    printf '检测到已有环境，将补齐并校正依赖，不会删除现有环境。\n'
    run "$CONDA_BIN" install --yes --prefix "$ENV_PREFIX" \
        "python=$PYTHON_VERSION" pip
else
    run "$CONDA_BIN" create --yes --prefix "$ENV_PREFIX" \
        "python=$PYTHON_VERSION" pip
fi

PIP_OPTIONS=(--timeout 3600 --retries 100 --prefer-binary)

run "$ENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" --upgrade \
    pip setuptools wheel

# requirements.txt contains the validated CUDA 11.8 PyTorch versions and the
# editable local installation of ./lerobot-main[training,pi].
run "$ENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" \
    --requirement "$REQUIREMENTS_FILE"

run "$ENV_PYTHON" -m pip check

if [[ "$DRY_RUN" == true ]]; then
    printf '\nDry-run 完成：没有修改 Conda 环境。\n'
    exit 0
fi

(
    cd "$PROJECT_ROOT"
    "$ENV_PYTHON" - <<'PY'
from importlib.metadata import version

import cv2
import datasets
import dynamixel_sdk
import lerobot
import numpy
import pyarrow
import pyrealsense2
import requests
import serial
import sentencepiece
import torch
import torchvision
import transformers
import yaml

print("\n依赖验证通过：")
for package in (
    "torch",
    "torchvision",
    "lerobot",
    "transformers",
    "datasets",
    "pyarrow",
    "numpy",
    "opencv-python",
    "pyrealsense2",
    "pyserial",
    "dynamixel-sdk",
):
    print(f"  {package}=={version(package)}")
print(f"  PyTorch CUDA build: {torch.version.cuda}")
print(f"  CUDA currently available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
else:
    print("  警告：PyTorch 已安装 CUDA 版，但当前进程未检测到可用 GPU。")
PY
)

printf '\n安装完成。打开新终端后运行：\n'
printf '  source %q\n' "$CONDA_ROOT/etc/profile.d/conda.sh"
printf '  conda activate %q\n' "$ENV_NAME"
printf '  cd %q\n' "$PROJECT_ROOT"
printf '  python Dobot_Train/train.py --dry-run\n'
printf '\n真机推理前请先检查 scripts/Dobot_inference.py 中的模型、权重、\n'
printf '任务提示词与 EXECUTE 安全开关，不要在安装后直接启动真机。\n'
