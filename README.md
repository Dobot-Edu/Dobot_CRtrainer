# Dobot CRtrainer

Dobot CRtrainer 是面向 Dobot CR5A 的机器人学习项目，覆盖硬件初始化、双目视觉、主从手或 Quest 3 遥操作、数据采集、LeRobot 转换、Pi0.5 训练和真实机械臂推理。

当前标准链路使用单臂 Dynamixel 主手、Dobot CR5A、PGE-50-26 夹爪和两台 Intel RealSense D405。训练与推理针对单张 NVIDIA RTX 4090 配置。策略输出为 7 维绝对动作：

```text
[TCP x, TCP y, TCP z, TCP Rx, TCP Ry, TCP Rz, gripper]
```

TCP 位置单位为毫米，姿态单位为度；夹爪范围为 `[0, 1]`，其中 `0` 表示关闭，`1` 表示打开。

> [!WARNING]
> 本项目可以向真实 CR5A 连续发送运动指令。首次运行、修改标定、切换模型或更换相机后，必须先执行只读检查和 dry-run。真机运行时保持急停按钮可触达，不要同时启动遥操作、采集和模型推理进程。

## 1. 项目结构

```text
Dobot_CRtrainer/
├── config/                         # 机械臂、相机、夹爪、网络和安全配置
├── datasets/                       # 原始和转换后的数据集，默认不提交 Git
├── Data_Collections/
│   ├── master_teleop/              # 单主手采集主流程
│   └── vr_teleop/                  # Quest 3 VR 遥操作及共用采集模块
├── Dobot_Init/                     # 机械臂状态控制和初始位姿恢复
├── Dobot_Train/                    # 数据诊断和统一训练入口
├── lerobot-main/                   # 项目内置 LeRobot 源码
├── outputs/                        # checkpoint、指标和推理视频
├── quest_unity_app/                # Quest 3 Unity 工程与 APK
├── scripts/                        # 数据转换、Dobot API 和推理入口
├── tests/                          # 单元测试和双 D405 手动预览测试
├── install_dobot_vr_teleop.sh      # 采集环境安装脚本
├── install_dobot_lerobot.sh        # 训练/推理环境安装脚本
└── requirements.txt                # RTX 4090 训练/推理依赖
```

所有命令均从项目根目录执行：

```bash
cd /path/to/Dobot_CRtrainer
```

## 2. 硬件与系统要求

### 2.1 硬件

| 设备 | 当前适配方式 |
| --- | --- |
| Dobot CR5A | 控制器 TCP 接口，默认地址 `192.168.5.1` |
| 单臂主手 | Dynamixel/X-trainer 串口协议，默认 2 Mbps |
| 夹爪 | DH-Robotics PGE-50-26，经 Dobot 工具端 RS485 控制 |
| 顶部相机 | Intel RealSense D405，按序列号绑定为 `top` |
| 腕部相机 | Intel RealSense D405，按序列号绑定为 `wrist` |
| VR 设备 | Meta Quest 3，可选的 VR 遥操作输入 |
| 训练 GPU | 单张 NVIDIA RTX 4090，默认使用 CUDA 设备 0 |

两台 D405 默认采集 `640 x 480 @ 30 FPS` RGB 图像。建议使用稳定的 USB 3.x 端口；出现掉帧或启动超时时，优先检查 USB 带宽、供电和线缆。

### 2.2 软件环境

- Ubuntu Linux 机器人控制主机。
- Miniforge 或 Conda。
- NVIDIA 驱动可正常识别 RTX 4090。
- 控制主机和 CR5A 控制器位于同一网段。
- 使用 Quest 3 时，Quest 与主机处于可互通的局域网。

项目使用两个隔离环境：

| 环境 | Python | 用途 |
| --- | --- | --- |
| `dobot-vr-teleop` | 3.11.15 | 初始化、主从手/VR 遥操作、双 D405 采集 |
| `dobot-lerobot` | 3.12.13 | LeRobot 转换、诊断、GPU 训练和推理 |

## 3. 安装环境

```bash
# 采集和遥操作环境
bash install_dobot_vr_teleop.sh

# LeRobot、CUDA 训练和推理环境
bash install_dobot_lerobot.sh
```

脚本会创建或修复对应环境，不会删除已有环境。训练环境通过 `requirements.txt` 安装 CUDA 版 PyTorch，并以 editable 模式安装项目内的 `lerobot-main`。

验证 GPU：

```bash
conda activate dobot-lerobot
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 4. 硬件配置

### 4.1 CR5A 网络

编辑 `config/network.yaml`：

```yaml
robot_network:
  ip: "192.168.5.1"
  dashboard_port: 29999
  move_port: 29999
  feedback_port: 30004
```

当前配置假定主机直连网卡为 `192.168.5.1/24`。更换地址时修改该配置，或用脚本的 `--robot-ip` 临时覆盖。

### 4.2 双 D405 相机

枚举相机：

```bash
conda activate dobot-vr-teleop
python tests/test_dual_d405_preview.py --list
```

将序列号写入 `config/camera.yaml`：

```yaml
camera:
  name: "top"
  driver: "realsense"
  serial: "<TOP_D405_SERIAL>"
  width: 640
  height: 480
  fps: 30

cameras:
  wrist:
    name: "wrist"
    driver: "realsense"
    serial: "<WRIST_D405_SERIAL>"
    width: 640
    height: 480
    fps: 30
```

打开两路独立预览：

```bash
python tests/test_dual_d405_preview.py
```

存在两台以上 D405 时显式指定目标设备：

```bash
python tests/test_dual_d405_preview.py \
  --serial <TOP_D405_SERIAL> <WRIST_D405_SERIAL>
```

在窗口中按 `q` 或 `Esc` 退出，也可以使用 `Ctrl+C`。

### 4.3 夹爪

PGE-50-26 的 RS485 寄存器、速度、力和开合位置位于 `config/gripper.yaml`。默认使用 Dobot 工具端口 `60000`，位置范围为：

```text
0      完全关闭
1000   完全打开
```

首次使用前确认工具电源、RS485 波特率、从站 ID 和寄存器地址与现场设备一致。

### 4.4 主手

主手基础配置位于 `Data_Collections/master_teleop/config/master_hand.yaml`。探测脚本会把实际串口、左右手类型和标定结果保存到：

```text
Data_Collections/master_teleop/config/master_runtime.yaml
```

Linux 用户需要串口权限：

```bash
sudo usermod -aG dialout "$USER"
```

执行后注销桌面或退出 SSH 并重新登录，再用 `groups` 确认包含 `dialout`。

### 4.5 Quest 3（可选）

将 `quest_unity_app/Builds/DobotVrTeleop.apk` 安装到 Quest 3。主机端配置位于 `config/host_network.yaml`：

```yaml
host_network:
  listen_ip: "0.0.0.0"
  listen_port: 5005
  discovery_enabled: true
  discovery_port: 5006
```

`local_ip` 只是自动发现失败时的回退地址，部署到其他主机时应改为该主机在 Quest 局域网中的地址。

## 5. 任务名配置规范

为便于后续拆分模块，采集、转换、训练和推理分别维护自己的任务名，不互相导入配置。当前四个模块均默认使用 `cosmetics_quality`；运行同一条完整链路时，应手动保持以下配置一致：

| 模块 | 配置文件 | 配置项 |
| --- | --- | --- |
| 主从手采集 | `Data_Collections/master_teleop/collection/launch_nodes.py` | `DATASET_TASK_NAME = "cosmetics_quality"` |
| 数据转换 | `scripts/convert_to_lerobot.py` | `TASK_NAME = "cosmetics_quality"` |
| 模型训练 | `Dobot_Train/train.py` | `TRAIN_TASK_NAME = "cosmetics_quality"` |
| 模型推理 | `scripts/Dobot_inference.py` | `INFERENCE_TASK_NAME = "cosmetics_quality"` |

任务名建议只使用小写英文、数字和下划线。各模块由自己的任务名生成或读取以下路径：

| 阶段 | 自动生成或读取的位置 |
| --- | --- |
| 原始采集 | `datasets/<task>/episodes/` |
| LeRobot 数据集 | `datasets/<task>/<task>_lerobot/` |
| LeRobot repo ID | `local/dobot_cr5a_<task>` |
| 训练输出 | `outputs/train/<task>/` |
| 30000 步模型 | `outputs/train/<task>/checkpoints/030000/pretrained_model/` |
| Pi0.5 推理 | 自动读取上述 30000 步模型 |

任务目录名与自然语言提示词用途不同：

- 各模块的任务名决定该模块读取或生成的文件夹、repo ID 和 checkpoint 路径。
- `TASK_PROMPT` 是写入训练样本并传给语言条件模型的英文任务描述。
- Pi0/Pi0.5 的训练提示词和推理提示词必须语义一致。

## 6. 机械臂初始化

```bash
conda activate dobot-vr-teleop

# 查看状态
python Dobot_Init/robot_state_control.py status

# 排除物理原因后清除报警并使能
python Dobot_Init/robot_state_control.py clear-error --confirm
python Dobot_Init/robot_state_control.py enable --confirm

# 只读检查初始位姿
python Dobot_Init/restore_initial_pose.py --dry-run

# 使用 MovJ 恢复绝对初始关节位姿
python Dobot_Init/restore_initial_pose.py
```

默认初始关节为：

```text
[90, 0, 90, 0, -90, -90] deg
```

恢复脚本默认同时初始化夹爪；只调试机械臂时可追加 `--no-gripper`。

## 7. 主从手数据采集

主从手模式通过关节空间 `ServoJ` 控制 CR5A，是当前任务名自动贯通训练和推理的标准入口。

### 7.1 探测主手串口

```bash
conda activate dobot-vr-teleop
python Data_Collections/master_teleop/collection/1_find_port.py
```

### 7.2 调整并锁定初始姿态

```bash
python Data_Collections/master_teleop/collection/3_just_buttonA.py
```

1. 脚本启动后主手处于锁定状态。
2. 短按 A 解锁主手。
3. 将主手移动到配置的初始姿态，并完全打开主手夹爪。
4. 再次短按 A 锁定。
5. 使用 `Ctrl+C` 退出；脚本保留主手力矩，供下一步读取同一姿态。

### 7.3 检查或重新标定零偏

只读检查：

```bash
python Data_Collections/master_teleop/collection/2_get_offset.py
```

只有更换主手、维修电机或确认零偏丢失时才重新标定：

```bash
python Data_Collections/master_teleop/collection/2_get_offset.py \
  --calibrate --confirm
```

### 7.4 CR5A、夹爪和相机预检

正式初始化：

```bash
python Data_Collections/master_teleop/collection/launch_nodes.py 
```

正式流程会检查主手和 CR5A，使能机械臂，以限速关节轨迹经过抬升点回到初始位姿，然后执行夹爪打开、关闭测试。跳过夹爪测试可追加 `--no-gripper`。

### 7.5 真机采集

```bash
python Data_Collections/master_teleop/collection/run_control.py
```

裸命令默认连接并使能真实 CR5A、设置速度因子 100%、启用夹爪、启动 `top` 和 `wrist` 两台 D405、显示两路预览，并启用 episode 记录器。

常用覆盖：

```bash
# 采集双路图像但不显示窗口
python Data_Collections/master_teleop/collection/run_control.py --no-camera-preview

# 临时禁用相机
python Data_Collections/master_teleop/collection/run_control.py --no-camera

# 临时禁用数据记录
python Data_Collections/master_teleop/collection/run_control.py --no-record-data
```

| 操作 | 行为 |
| --- | --- |
| A 短按 | 切换主手锁定/解锁 |
| A 长按并松开 | 开启或停止 Servo；首次开启前执行主从动态对齐 |
| B 短按 | 开始或停止一个 episode |
| `Ctrl+C` | 停止采集并关闭连接 |

长按 A 启动 Servo 时，程序先冻结主手六关节目标，并以约 `11.5 deg/s` 对齐 CR5A。只有机械臂已使能、无报警、无碰撞，且当前位置、目标位置和关节均满足安全约束时才允许运动。

### 7.6 原始数据结构

```text
datasets/<task>/episodes/episode_<timestamp>/
├── metadata.json
├── frames.jsonl
├── data.npz
└── images/
    ├── top/rgb/*.jpg
    └── wrist/rgb/*.jpg
```

`metadata.json` 中只有状态完成且 `valid=true` 的 episode 默认进入转换。

## 8. Quest 3 VR 采集（可选）

VR 与主从手是两套独立输入，不得同时运行：

```bash
conda activate dobot-vr-teleop

# 使用模拟机器人验证通信、按钮和双相机，不连接真实 CR5A
python Data_Collections/vr_teleop/collection/start_vr_teleop.py \
  --robot-mode mock --dry-run --no-enable-robot

# 真机 VR 遥操作和数据采集
python Data_Collections/vr_teleop/collection/start_vr_teleop.py
```

| Quest 输入 | 行为 |
| --- | --- |
| A | 标定控制器与机械臂的相对映射 |
| Grip | 按住时允许机械臂跟随 |
| B | 开始或停止 episode |

VR 默认任务名由 `Data_Collections/vr_teleop/collection/vr_teleop.py` 的 `DEFAULT_TASK` 管理，也可用 `--task` 临时覆盖。要让 VR 数据进入标准训练链路，应确保它最终写入与 `DATASET_TASK_NAME` 相同的任务目录。

## 9. 检查原始数据

```bash
conda activate dobot-lerobot
python Dobot_Train/inspect_dataset.py
```

该脚本默认读取采集模块的任务名，检查 episode 数量、`metadata.json`、`frames.jsonl` 和 `data.npz`。发现无效 episode、缺帧或损坏数据时，不要直接训练。

## 10. 转换为 LeRobot 数据集

编辑 `scripts/convert_to_lerobot.py` 顶部配置区。任务、输出目录和 repo ID 已自动派生，通常只需要检查自然语言提示词：

```python
DATASET_DIR = "datasets"
TASK_NAME = "cosmetics_quality"
OUTPUT_DIR = f"datasets/{TASK_NAME}/{TASK_NAME}_lerobot"
REPO_ID = f"local/dobot_cr5a_{TASK_NAME}"
ENABLE_COSMETICS_STAGE_LABELS = True
OVERWRITE_EXISTING = True
```

先验证计划，再执行转换：

```bash
conda activate dobot-lerobot
python scripts/convert_to_lerobot.py
```

默认转换规则：

- 30 FPS，图像尺寸 `640 x 480`。
- 输出 `observation.images.top` 和 `observation.images.wrist`。
- 不保留 VR 控制器输入，因为自主推理没有 VR 输入。
- 只使用 `valid=true` 的 episode。
- 检查双相机时间差、图像年龄、重复帧率、动作反馈和机器人安全状态。
- 质量门禁失败时停止；`--no-quality-gate` 仅用于诊断。

核心特征：

```text
observation.state        = 6-D 实际 TCP + 1-D 实际夹爪
action                   = 6-D 目标 TCP + 1-D 目标夹爪
observation.images.top   = 顶部 D405 RGB
observation.images.wrist = 腕部 D405 RGB
task                     = 自然语言任务提示词
```

输出结构：

```text
datasets/<task>/<task>_lerobot/
├── data/
├── images/ 或 videos/
├── meta/
└── conversion_report.json
```

默认保存图像；需要视频格式时使用 `--use-videos`。`OVERWRITE_EXISTING=True` 会删除并重建旧输出，单次使用 `--no-overwrite` 可保护已有结果。

### 10.1 分阶段语言标签

当前 `cosmetics_quality` 默认设置 `ENABLE_COSMETICS_STAGE_LABELS=True`。直接运行转换脚本时，它会根据去抖后的夹爪开合事件划分“右侧料盘到秤、按绿灯从秤取料、按黄灯从秤取料、绿灯放入左下料盘、黄灯放入左上料盘”五类提示词，并根据最终放置位置聚类判断绿灯或黄灯轨迹。五条英文提示词位于脚本顶部配置区，必须与推理脚本逐字一致。

转换完成后检查 `conversion_report.json`、`meta/segment_label_stats.json`、`meta/train_eval_split.json` 和 `meta/segment_label_audits/`。临时转换普通单提示词任务时可在配置区设为 `False`，或使用 `--no-cosmetics-stage-labels`。

## 11. 训练前质量诊断

```bash
conda activate dobot-lerobot
python Dobot_Train/diagnose_training_data.py
```

诊断内容包括 TCP 欧拉角分支、夹爪 observation/action、双相机同步和重复帧、采样误差、反馈覆盖率、动作执行率与机器人安全状态。只有以下结果才进入训练：

```text
Result: PASS
```

`--no-strict` 只应用于分析，不应作为跳过质量问题的正式入口。

## 12. RTX 4090 本机训练

统一入口为 `Dobot_Train/train.py`。日常参数集中在文件顶部的 `USER SETTINGS`，直接运行 Python 脚本，不需要 `accelerate launch`。

当前 Pi0.5 配置：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `TRAIN_TASK_NAME` | `cosmetics_quality` | 独立设置训练数据和输出任务名 |
| `SELECTED_POLICY` | `pi05` | Pi0.5 策略 |
| `DEVICE` | `cuda` | CUDA 训练 |
| `CUDA_VISIBLE_DEVICES` | `0` | 单张 GPU 0 |
| `STEPS` | `30000` | 优化更新步数 |
| `BATCH_SIZE` | `1` | 24 GB RTX 4090 的保守默认 batch 大小 |
| `NUM_WORKERS` | `4` | DataLoader 子进程数 |
| `PREFETCH_FACTOR` | `2` | 每个 worker 预取 batch 数 |
| `LOG_FREQUENCY` | `200` | 每 200 步输出指标 |
| `SAVE_FREQUENCY` | `5000` | 每 5000 步保存 |
| `DATASET_USE_IMAGENET_STATS` | `False` | 使用数据集自身统计量 |
| `PI05_DTYPE` | `bfloat16` | 4090 低精度类型 |
| `PI05_CHUNK_SIZE` | `50` | 每次预测 50 帧动作 |
| `PI05_N_ACTION_STEPS` | `50` | 非 RTC 最大连续动作数 |
| `PI05_USE_RELATIVE_ACTIONS` | `False` | 绝对 TCP 动作 |
| `PI05_GRADIENT_CHECKPOINTING` | `True` | 降低显存占用 |
| `PI05_TRAIN_EXPERT_ONLY` | `True` | 冻结 VLM，微调动作专家和投影层 |
| `PI05_COMPILE_MODEL` | `False` | 便于首次启动和排障 |

基础模型和联网模式：

```python
PI05_PRETRAINED_PATH = "lerobot/pi05_base"
HUGGINGFACE_OFFLINE = False
```

首次运行会下载基础模型和 tokenizer。下载完成后可将 `HUGGINGFACE_OFFLINE` 改为 `True`；也可把 `PI05_PRETRAINED_PATH` 改为完整本地模型目录。

```bash
conda activate dobot-lerobot

# 查看可用策略
python Dobot_Train/train.py --list-policies

# 验证数据集、策略配置和生成命令
python Dobot_Train/train.py --dry-run

# 开始训练
python Dobot_Train/train.py
```

训练脚本会拒绝缺少双相机特征、语言提示词、转换质量门禁未通过或 CUDA 不可用的数据。再次训练同名任务前，应备份或清理旧的 `outputs/train/<task>`。

checkpoint：

```text
outputs/train/<task>/checkpoints/005000/pretrained_model/
outputs/train/<task>/checkpoints/010000/pretrained_model/
...
outputs/train/<task>/checkpoints/030000/pretrained_model/
```

指标：

```text
outputs/Loss/<task>/
├── training.log
├── run_config.json
├── metrics.csv
├── metrics.json
└── *.png
```

## 13. 模型推理

推理入口是 `scripts/Dobot_inference.py`。Pi0.5 自动读取：

```text
outputs/train/<INFERENCE_TASK_NAME>/checkpoints/030000/pretrained_model
```

标准任务不需要复制或重命名模型。

### 13.1 选择任务模式

普通单提示词任务：

```python
SELECTED_MODEL = "PI0.5"
INFERENCE_TASK_NAME = "cosmetics_quality"
TASK_MODE = "fixed"
TASK_PROMPT = "与训练数据一致的英文提示词"
```

化妆品称重分拣模型可使用：

```python
TASK_MODE = "segmented_cosmetics"
```

该模式使用五条阶段提示词、双相机指示灯颜色检测和任务状态机。只有使用相同提示词训练的模型才能启用，并且必须保持 `RTC_ENABLED=True`。

### 13.2 首次推理禁止真机运动

脚本当前默认允许真机执行。首次加载新 checkpoint 前先改为：

```python
EXECUTE = False
ENABLE_ROBOT = False
EXECUTE_GRIPPER = False
SHOW_CAMERA = True
RECORD_VIDEO = True
```

```bash
conda activate dobot-lerobot
python scripts/Dobot_inference.py
```

该模式使用模拟机器人状态，不发送运动指令，但会加载模型和双 D405，用于检查 checkpoint、7 维动作、相机映射、提示词、RTC 队列、预览和视频录制。视频默认写入 `outputs/inference_videos/`。

### 13.3 真机推理

完成离线检查、工作区确认和初始位姿确认后再设置：

```python
EXECUTE = True
ENABLE_ROBOT = True
EXECUTE_GRIPPER = True
```

```bash
python scripts/Dobot_inference.py
```

真机流程会加载模型、启动双 D405、连接 CR5A、初始化夹爪，并在配置允许时通过 `MovJ` 前往 `INFERENCE_START_JOINTS_DEG`。随后 RTC 后台线程持续预测动作块，主线程按 `config/robot.yaml` 的 ServoP 周期执行。

| 配置 | 当前值 | 作用 |
| --- | --- | --- |
| `STEPS` | `15000` | 非连续模式最大推理步数 |
| `CONTINUOUS_TASKS` | `True` | 完成一罐后复位并继续 |
| `CUDA_VISIBLE_DEVICES` | `0` | 单张 GPU 0 |
| `INFERENCE_DEVICE` | `cuda` | 使用 CUDA 加载和执行模型 |
| `SPEED_FACTOR_PERCENT` | `80` | 真机全局速度百分比 |
| `MOVE_TO_INFERENCE_START` | `True` | 先移动到训练数据代表性起点 |
| `RTC_ENABLED` | `True` | 实时动作块队列 |
| `SHOW_CAMERA` | `True` | 显示拼接相机画面 |
| `RECORD_VIDEO` | `True` | 保存推理视频 |

模型输出同时受 `config/safety.yaml` 的工作空间、单步位移和速度约束，以及姿态范围、报警、碰撞、反馈超时、使能状态、夹爪迟滞和 RTC 队列超时检查。

## 14. 安全停止

异常运动时优先按机械臂急停。终端正常停止使用 `Ctrl+C`。必要时从单独终端发送：

```bash
conda activate dobot-vr-teleop
python Dobot_Init/robot_state_control.py stop --confirm
```

停止后先排除物理原因，再清除报警并恢复初始位姿。不要在碰撞、急停或报警原因不明确时直接执行 `clear-error`。

## 15. 常见问题

### 找不到两台 D405

运行 `python tests/test_dual_d405_preview.py --list`，确认检测到两台 D405，序列号与 `config/camera.yaml` 一致。检查 USB 3.x 连接、线缆和供电；连接超过两台时用 `--serial` 指定。

### SSH 启动后没有相机窗口

主机预览由 `config/host_network.yaml` 的 `preview_target`、`host_desktop_display` 和 `host_xauthority` 控制。只采集不显示时使用 `--no-camera-preview`；推理时可将 `PREVIEW_MODE` 设为 `off`，视频录制仍可保留。

### 主手串口 Permission denied

```bash
groups
ls -l /dev/serial/by-id/
```

确认当前用户属于 `dialout`，然后重新插拔 USB。

### 转换被质量门禁阻止

不要直接用 `--no-quality-gate` 训练。先查看 violation，再运行 `python Dobot_Train/diagnose_training_data.py`。常见原因包括双相机不同步、重复帧、反馈缺失、episode 中存在碰撞/报警，以及动作与状态的 TCP 差值过大。

### Pi0.5 基础模型无法下载

联网时保持 `HUGGINGFACE_OFFLINE=False` 和 `PI05_PRETRAINED_PATH="lerobot/pi05_base"`。离线训练必须事先完整下载模型和 tokenizer，再设置本地路径和 `HUGGINGFACE_OFFLINE=True`。

### CUDA 不可用或显存不足

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

默认 `BATCH_SIZE=1` 已按 24 GB RTX 4090 保守设置。仍出现显存不足时，保持 `PI05_GRADIENT_CHECKPOINTING=True` 和 `PI05_TRAIN_EXPERT_ONLY=True`，关闭其他 GPU 进程，并确认推理与训练没有同时占用显存。

## 16. 数据和 Git 规范

根目录 `.gitignore` 默认排除数据集、checkpoint、指标、日志、推理视频、Hugging Face/W&B/Python 缓存、模型权重和 Unity 生成目录。数据集、模型和视频不应直接提交源码仓库；需要共享时使用独立数据存储、模型仓库或 GitHub Release。

公开提交前还应检查相机序列号、主手 USB ID 和局域网 IP 是否需要替换为示例值。

## 17. 许可证

项目许可证见根目录 `LICENSE`。项目内置 LeRobot 源码保留其自身的 `lerobot-main/LICENSE`。
