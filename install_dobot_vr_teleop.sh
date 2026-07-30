#!/usr/bin/env bash
#
# Create or repair the lightweight Dobot VR teleoperation environment.
# Usage:
#   bash install_dobot_vr_teleop.sh
#   bash install_dobot_vr_teleop.sh --dry-run

set -Eeuo pipefail

ENV_NAME="dobot-vr-teleop"
PYTHON_VERSION="3.11.15"
DEFAULT_CONDA_ROOT="/home/dobot/miniforge3"
DRY_RUN=false

usage() {
    cat <<'EOF'
用法：bash install_dobot_vr_teleop.sh [--dry-run] [--help]

创建或修复：
  /home/dobot/miniforge3/envs/dobot-vr-teleop

选项：
  --dry-run  只显示将执行的命令，不修改环境
  --help     显示帮助

可以通过 CONDA_ROOT 覆盖 Miniforge 根目录，例如：
  CONDA_ROOT="$HOME/miniforge3" bash install_dobot_vr_teleop.sh
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

run "$ENV_PYTHON" -m pip install "${PIP_OPTIONS[@]}" \
    "numpy==2.4.6" \
    "PyYAML==6.0.3" \
    "requests==2.34.2" \
    "opencv-python==5.0.0.93" \
    "pyrealsense2==2.58.2.10647"

if [[ "$DRY_RUN" == true ]]; then
    printf '\nDry-run 完成：没有修改 Conda 环境。\n'
    exit 0
fi

"$ENV_PYTHON" - <<'PY'
from importlib.metadata import version

import cv2
import numpy
import pyrealsense2
import requests
import yaml

print("\n依赖验证通过：")
for package in ("numpy", "PyYAML", "requests", "opencv-python", "pyrealsense2"):
    print(f"  {package}=={version(package)}")
print(f"  OpenCV GUI backend available: {'GUI:' in cv2.getBuildInformation()}")
PY

printf '\n安装完成。打开新终端后运行：\n'
printf '  source %q\n' "$CONDA_ROOT/etc/profile.d/conda.sh"
printf '  conda activate %q\n' "$ENV_NAME"
printf '  cd %q\n' "$PROJECT_ROOT"
printf '  python Data_Collections/vr_teleop/collection/start_vr_teleop.py --help\n'
