# Dobot_CRtrainer 快速运行
Dobot CR5A 的机械臂初始化、VR 双相机数据采集、LeRobot 格式转换、模型训练和真机推理。

## GitHub 精简版说明

仓库包含源码、配置、Unity 工程源文件，以及已经构建好的 Quest 安装包：

```text
quest_unity_app/Builds/DobotVrTeleop.apk
```

可以使用 SideQuest 或 ADB 将该 APK 安装到 Quest 3。仓库不包含以下本地运行产物：

- `datasets/` 中的原始采集数据和 LeRobot 转换数据；
- `outputs/` 中的训练权重、优化器状态、日志和推理视频；
- Hugging Face、PyTorch、pip 等下载缓存；

数据采集和训练时会按配置重新创建 `datasets/` 与 `outputs/` 内容。模型推理前，
需要先完成训练，并把 `scripts/Dobot_inference.py` 中注册的 checkpoint 路径指向
本机实际生成的 `pretrained_model`。需要修改 VR 应用时，可以使用 Unity 打开
`quest_unity_app/`，重新构建并替换上述 APK。

所有命令先进入项目：
```bash
cd Dobot_CRtrainer
```
环境分工：

```text
conda activate dobot-vr-teleop 初始化、VR 遥操作、数据采集
dobot-lerobot    格式转换、训练、真机推理
```
## 1. 机械臂初始化

```bash
conda activate dobot-vr-teleop

# 使能机械臂
python Dobot_Init/robot_state_control.py

# 查看状态
python Dobot_Init/robot_state_control.py status

# 恢复初始位姿
python Dobot_Init/restore_initial_pose.py
```

## 2. VR 数据采集

验证VR遥操作：
python Data_Collections/vr_teleop/collection/start_vr_teleop.py

```bash
# 在Data_Collections/vr_teleop/collection/vr_teleop.py脚本中更改数据保存路径和任务名称
# 相对路径以 project_root（Dobot_CRtrainer 项目根目录）为基准；
DEFAULT_DATASET_DIR: str | Path = "datasets"        //数据集路径
DEFAULT_TASK = "test"                               //数据集任务名称

# 运行下面命令开始遥操作采集数据
conda activate dobot-vr-teleop
python Data_Collections/vr_teleop/collection/start_vr_teleop.py --record-data

```
Quest 操作：

```text
A       校准 VR 控制器与机械臂
Grip    按住时允许遥操作
B       开始或停止一个 episode
```

# 检查采集的原生数据集的合法性

python Dobot_Train/inspect_dataset.py 

## 3. 转换为 LeRobot 格式

编辑 `scripts/convert_to_lerobot.py` 顶部的 `USER SETTINGS`：

```python
# 原始采集数据的根目录。脚本会在其中查找
# <DATASET_DIR>/<TASK_NAME>/episodes/episode_xxx 等采集回合。
DATASET_DIR = "datasets"

# 本次转换的任务目录名，必须与 DATASET_DIR 下的实际文件夹名称一致。例如：
TASK_NAME = "test"

# 转换后的 LeRobot 数据集保存目录，包含 data、meta、videos 等内容。例如：
OUTPUT_DIR = "datasets/test/test_lerobot"

# 写入 LeRobot 元数据的数据集标识。使用本地数据时可采用
# local/<名称>；训练脚本中的 dataset.repo_id 必须与这里保持一致。例如：
REPO_ID = "local/dobot_cr5a_test_aligned"

# 每一帧关联的自然语言任务描述，供 Pi0、Pi0.5、VLA 等语言条件策略使用。
# 同一任务的转换、训练和推理阶段应使用语义一致的提示词。例如：
TASK_PROMPT = "Pick up the gear and place it at the target position."

# True：若 OUTPUT_DIR 已存在，则允许删除并重新生成；旧结果会被覆盖。
# False：若输出目录已存在则停止转换，可用于防止误删已有数据。
OVERWRITE_EXISTING = True
```
```bash
conda activate dobot-lerobot
python scripts/convert_to_lerobot.py
```

运行 `Dobot_Train/diagnose_training_data.py` 严格检查转换结果：
python Dobot_Train/diagnose_training_data.py
```
只有输出 `Result: PASS` 的数据才用于训练。

## 4. 模型训练

编辑 `Dobot_Train/train.py` 顶部的 `USER SETTINGS`：

```python
# 选择 LeRobot 训练策略，例如 act、pi0、pi05。
# 必须是本脚本 SUPPORTED_POLICY_TYPES 中已注册且环境依赖完整的策略名称。
SELECTED_POLICY = "pi05"
# 实验名称，用于生成训练任务名和输出目录；建议包含任务、模型或数据版本信息。
EXPERIMENT_NAME = "test_pi05_euler_aligned"
# LeRobot 数据集的逻辑标识，必须与转换脚本中的 REPO_ID 及数据集元数据一致。
# 它用于 LeRobot 识别数据集，不是磁盘路径。
DATASET_REPO_ID = "local/dobot_cr5a_test_aligned"
# 转换后的 LeRobot 数据集本地目录，其中应包含 data、meta、videos/images 等内容。
DATASET_ROOT = "datasets/test/test_lerobot"
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
```
开始训练：
```bash
conda activate dobot-lerobot
python Dobot_Train/train.py
```
查看可选策略：
```bash
python Dobot_Train/train.py --list-policies
```
模型保存在：
```text
outputs/train/<实验目录>/checkpoints/<步数>/pretrained_model/
```
## 5. 模型推理

编辑 `scripts/Dobot_inference.py`：

```python
# 选择 MODEL_REGISTRY 中的模型
SELECTED_MODEL = "ACT"
# 完整真机推理
EXECUTE = True
ENABLE_ROBOT = True
# 在推理过程中控制夹爪是否能够开合，默认true是开。
EXECUTE_GRIPPER = True
# 是否实时显示顶部和腕部相机的拼接推理画面。
SHOW_CAMERA = True
# 是否把相机画面和当前预测动作录制为 MP4，输出到 outputs/inference_videos。
RECORD_VIDEO = True
```
运行推理：
```bash
conda activate dobot-lerobot
python scripts/Dobot_inference.py
```
## 6. 安全停止

异常运动时优先按机械臂急停。终端正常停止使用 `Ctrl+C`。

控制器停止命令：

```bash
conda activate dobot-vr-teleop
python Dobot_Init/robot_state_control.py stop --confirm
```
注意:不要同时运行 VR 遥操作和模型推理，也不要让多个程序同时占用相机或 Dobot TCP 端口。
