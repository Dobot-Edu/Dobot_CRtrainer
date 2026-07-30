# Dobot CR5A VR Teleoperation and LeRobot Training

基于 **Meta Quest 3 + Dobot CR5A + 双路相机** 的机器人数据采集、LeRobot
数据转换、策略训练与真机推理项目。

项目默认硬件：

- 机械臂：Dobot CR5A
- VR 设备：Meta Quest 3
- 顶部相机：Logitech C930e
- 腕部相机：Intel RealSense D435
- 夹爪：DH-Robotics PGE-50-26
- 图像规格：双路 `640x480 @ 30 FPS`
- 训练策略：ACT、Pi0、Pi0.5，以及当前 LeRobot 环境支持的其他策略

> [!CAUTION]
> 真机运行前请确认急停可用、工作空间内无人，并先使用模拟模式或关闭
> `EXECUTE` 检查相机画面和模型输出。

## 功能

- Quest 3 VR 实时遥操作 CR5A
- 顶部和腕部双相机同步采集
- 保存机械臂 TCP、夹爪状态和自然语言任务
- 原始数据转换为 LeRobot 数据集
- 数据完整性与时序对齐诊断
- ACT、Pi0、Pi0.5 训练
- CUDA 真机推理、实时画面和推理视频录制

## 项目结构

```text
Dobot_CRtrainer/
├── Data_Collections/            # VR 遥操作与原始数据采集
├── Dobot_Init/                  # 机械臂状态控制和初始位姿恢复
├── Dobot_Train/                 # 数据检查、诊断和模型训练
├── config/                      # 相机、机械臂和采集配置
├── scripts/                     # 数据转换与模型推理
├── quest_unity_app/             # Quest APK
├── install_dobot_vr_teleop.sh   # 遥操作环境安装脚本
├── install_dobot_lerobot.sh     # 训练推理环境安装脚本
└── requirements.txt
```

数据集、模型权重、推理视频和缓存默认不包含在 GitHub 精简版本中。

## 环境要求

- Ubuntu 22.04 或兼容 Linux 系统
- Miniforge/Conda：`/home/dobot/miniforge3`
- NVIDIA GPU 与可用的 CUDA 驱动
- Python 3.11：VR 遥操作环境
- Python 3.12：LeRobot 训练与推理环境

## 1. 安装环境

进入项目目录：

```bash
cd /home/dobot/桌面/Dobot_CRtrainer
```

安装 VR 遥操作环境：

```bash
bash install_dobot_vr_teleop.sh
```

安装 LeRobot 训练推理环境：

```bash
bash install_dobot_lerobot.sh
```

安装脚本会创建：

```text
/home/dobot/miniforge3/envs/dobot-vr-teleop
/home/dobot/miniforge3/envs/dobot-lerobot
```

只检查安装步骤、不实际安装：

```bash
bash install_dobot_vr_teleop.sh --dry-run
bash install_dobot_lerobot.sh --dry-run
```

## 2. 机械臂初始化

激活遥操作环境：

```bash
conda activate dobot-vr-teleop
```

查看机械臂状态：

```bash
python Dobot_Init/robot_state_control.py status
```

使能机械臂：

```bash
python Dobot_Init/robot_state_control.py
```

先检查初始位姿恢复流程：

```bash
python Dobot_Init/restore_initial_pose.py --dry-run
```

确认安全后恢复初始位姿：

```bash
python Dobot_Init/restore_initial_pose.py
```

默认机械臂地址为 `192.168.5.100`，初始 TCP 位姿为：

```text
[141, -473, 469, 180, 0, -90]
```

## 3. 相机与遥操作测试

先使用模拟机械臂验证双路相机：

```bash
python Data_Collections/vr_teleop/collection/start_vr_teleop.py \
  --robot-mode mock \
  --dry-run \
  --no-gripper \
  --camera-preview
```

正常情况下会显示顶部相机和腕部相机两个画面。相机设备、序列号、分辨率和
帧率在 `config/camera.yaml` 中配置。

Quest 端安装包：

```text
quest_unity_app/Builds/DobotVrTeleop.apk
```

默认 VR 网络端口：

- Quest 数据端口：`5005/UDP`
- 主机发现端口：`5006/UDP`

## 4. 采集原始数据

```bash
conda activate dobot-vr-teleop

python Data_Collections/vr_teleop/collection/start_vr_teleop.py \
  --record-data \
  --dataset-dir datasets \
  --task test123
```

Quest 控制器操作：

| 按键 | 功能 |
|---|---|
| `A` | 校准 VR 控制器与机械臂 |
| `Grip` | 按住后允许机械臂跟随 |
| `B` | 开始或结束一个 episode |
| `Trigger` | 控制夹爪开合 |

原始数据保存结构：

```text
datasets/<task>/episodes/episode_<timestamp>/
├── metadata.json
├── frames.jsonl
├── data.npz
└── images/
    ├── top/rgb/
    └── wrist/rgb/
```

检查原始数据：

```bash
python Dobot_Train/inspect_dataset.py \
  --dataset-dir datasets \
  --task test123 \
  --validate-npz
```

## 5. 转换为 LeRobot 数据集

激活训练环境：

```bash
conda activate dobot-lerobot
```

打开 `scripts/convert_to_lerobot.py`，修改文件顶部的用户配置：

```python
DATASET_DIR = "datasets"
TASK_NAME = "test123"
OUTPUT_DIR = "datasets/test123/test123_lerobot"
REPO_ID = "local/dobot_cr5a_test123"
TASK_PROMPT = "Pick up the gear and place it at the target position."
OVERWRITE_EXISTING = True
```

然后直接运行：

```bash
python scripts/convert_to_lerobot.py
```

这些参数必须保持对应：

| 参数 | 含义 |
|---|---|
| `TASK_NAME` | 原始数据所在的任务目录名 |
| `OUTPUT_DIR` | 转换后的本地 LeRobot 数据集目录 |
| `REPO_ID` | 写入数据集元数据的唯一标识 |
| `TASK_PROMPT` | 每一帧保存的自然语言任务 |
| `OVERWRITE_EXISTING` | 是否覆盖已有转换结果 |

转换后的主要字段：

```text
observation.images.top
observation.images.wrist
observation.state
action
task
```

其中 `observation.state` 和 `action` 均为：

```text
[x, y, z, rx, ry, rz, gripper]
```

夹爪约定：`0 = 闭合`，`1 = 打开`。

## 6. 数据诊断

在 `Dobot_Train/diagnose_training_data.py` 顶部设置原始数据目录和 LeRobot
数据目录，然后运行：

```bash
python Dobot_Train/diagnose_training_data.py
```

正式训练前应确认：

- 顶部和腕部图像都存在且帧数一致
- 图像分辨率为 `640x480`
- state、action、时间戳与图像正确对齐
- TCP 欧拉角已转换到连续且一致的等价角度分支
- 夹爪 observation 和 action 包含真实开合变化
- 任务提示词与训练配置一致

## 7. 模型训练

打开 `Dobot_Train/train.py`，修改顶部训练配置：

```python
SELECTED_POLICY = "act"  # 也可选择 pi0 或 pi05
EXPERIMENT_NAME = "test_act"
DATASET_REPO_ID = "local/dobot_cr5a_test123"
DATASET_ROOT = "datasets/test123/test123_lerobot"
DEVICE = "cuda"

STEPS = 5000
BATCH_SIZE = 8
NUM_WORKERS = 8
SAVE_FREQUENCY = 5000
```

检查最终训练命令：

```bash
python Dobot_Train/train.py --dry-run
```

开始训练：

```bash
python Dobot_Train/train.py
```

Pi0 和 Pi0.5 通常需要更大的显存，并可能需要提前登录 Hugging Face：

```bash
hf auth login
hf auth whoami
```

训练结果默认保存在：

```text
outputs/train/<experiment_name>/
```

## 8. 模型推理

打开 `scripts/Dobot_inference.py`：

1. 在 `MODEL_REGISTRY` 中登记 checkpoint、策略类型和相机映射。
2. 将 `SELECTED_MODEL` 设置为对应注册名称。
3. 确认 `TASK_PROMPT` 与训练数据中的任务提示一致。

第一次测试请关闭真机执行：

```python
SELECTED_MODEL = "your_model_name"
EXECUTE = False
ENABLE_ROBOT = False
EXECUTE_GRIPPER = False
SHOW_CAMERA = True
RECORD_VIDEO = True
```

运行：

```bash
conda activate dobot-lerobot
python scripts/Dobot_inference.py
```

检查两路画面、模型输出和录制视频正常后，再启用真机：

```python
EXECUTE = True
ENABLE_ROBOT = True
EXECUTE_GRIPPER = True
```

推理视频默认保存在：

```text
outputs/inference_videos/
```

本机运行会直接显示相机窗口。通过 SSH 运行时，脚本可按
`PREVIEW_TARGET = "host"` 将窗口显示在机器人主机桌面；也可以使用 X11
转发并将 `PREVIEW_TARGET` 设置为 `"session"`。

## 常见问题

| 现象 | 检查项 |
|---|---|
| 相机窗口无法显示 | OpenCV 是否带 GTK；`DISPLAY` 是否正确；可改用 `ffplay` |
| D435 无图像 | USB 3.0 连接、设备序列号、RealSense 占用情况 |
| VR 操作出现 timeout | Quest Wi-Fi、UDP 端口、防火墙、主机负载和采集写盘速度 |
| 训练 GPU 利用率低 | `data_s`、DataLoader worker 数、图像解码和磁盘速度 |
| 推理画面正常但机械臂不动 | `EXECUTE`、`ENABLE_ROBOT`、robot mode 和安全限制 |
| 腕部持续旋转 | 欧拉角分支、训练数据对齐和 `ORIENTATION_LIMIT_DEG` |
| 每次重新下载权重 | 检查 Hugging Face 缓存目录与离线配置 |

## 安全说明

- 第一次使用新数据集或新模型时必须先执行无动作推理。
- 真机测试从低速和较小的单步位移限制开始。
- 不要绕过碰撞检测、工作空间限制和 watchdog。
- 推理期间保持急停按钮可触及。
- 模型异常、相机断流或网络超时时立即停止机械臂。

## License

项目中的 LeRobot、Pi 系列模型、RealSense SDK、Dobot SDK 和其他第三方组件
分别遵循其原始许可证。发布或商用前请核对各依赖及模型权重的许可要求。
