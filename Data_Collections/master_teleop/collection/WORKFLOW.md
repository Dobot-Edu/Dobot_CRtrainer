# CR5A 单主手采集流程

所有命令都在项目根目录运行。

首次使用先授予当前用户串口权限，然后注销桌面或退出 SSH 并重新登录：

```bash
sudo usermod -aG dialout "$USER"
```

重新登录后运行 `groups`，确认包含 `dialout`。如果 `/dev/serial/by-id/` 中仍没有
主手设备，请重新插拔 USB 并检查线缆和供电。

## 1. 查找并保存主手端口

```bash
python Data_Collections/master_teleop/collection/1_find_port.py
```

脚本探测串口、波特率和左右手 ID，并把结果写入
`Data_Collections/master_teleop/config/master_runtime.yaml`。

## 2. 先锁定主手初始化姿态（运行 `3_just_buttonA.py`）

```bash
python Data_Collections/master_teleop/collection/3_just_buttonA.py
```

脚本启动后主手立即上力矩并保持刚性。操作顺序必须是：

1. A 短按一次，主手解锁；
2. 移动主手到合适的初始化姿态，并把主手夹爪完全打开；
3. A 再短按一次，主手锁定；
4. 按 `Ctrl+C` 退出。

退出时脚本有意不释放主手力矩，保证下一步读取的是完全相同的刚性姿态。
如果主手仍处于解锁状态，第一次 `Ctrl+C` 只会提示先锁定，不会退出。

## 3. 检查主手关节角（运行 `2_get_offset.py`）

在主手锁定状态下运行：

```bash
python Data_Collections/master_teleop/collection/2_get_offset.py
```

默认只读取当前主手六关节角，计算它们相对
`[90, 0, 90, 0, -90, -90]` 的偏移，并生成诊断记录；不会修改标定配置。
仅在更换主手、维修电机或确认零偏已失效时，才进行显式标定：

```bash
python Data_Collections/master_teleop/collection/2_get_offset.py --calibrate --confirm
```

标定前必须把主手准确摆到配置的标准关节姿态并锁定。只有当主手已经
和机械臂当前六关节姿态逐关节对应时，才使用 `--use-current-robot-pose`。

## 4. CR5A 初始位姿与夹爪测试

```bash
python Data_Collections/master_teleop/collection/launch_nodes.py
```

如需同时检查相机：

```bash
python Data_Collections/master_teleop/collection/launch_nodes.py --check-camera
```

默认流程会清除可恢复报警、上使能 CR5A，先抬起机械臂再恢复初始位姿。
位姿恢复后，脚本等待长按 A；按住超过 1 秒并松开后，夹爪会以
100% 速度快速张开、等待约 0.3 秒、再闭合。短按 A 不触发夹爪测试。

如果只需读取状态且禁止机械臂和夹爪动作，使用：

```bash
python Data_Collections/master_teleop/collection/launch_nodes.py --skip-pose-init
```

## 5. 遥操作与采集

数据集任务名在 `launch_nodes.py` 顶部的下面这一行修改：

```python
DATASET_TASK_NAME = "master_teleoperation"
```

`launch_nodes.py` 启动时会输出任务的数据目录。`run_control.py` 会自动读取这个
任务名；第一次短按 B 开始 episode，第二次短按 B 结束并保存到：

```text
datasets/<DATASET_TASK_NAME>/episodes/
```

先执行 dry-run：

```bash
python Data_Collections/master_teleop/collection/run_control.py --dry-run --no-gripper
```

dry-run 默认启用数据记录器，但不向 CR5A 发送 `ServoJ`。B 短按开始/结束 episode。
按 `Ctrl+C` 停止时，程序会先锁定主手当前位置，再关闭串口但保持主手力矩。
数据默认写入当前项目的 `datasets/<task>/episodes/`。
完成六关节方向、初始对齐和限位检查后，直接运行实机入口：

```bash
python Data_Collections/master_teleop/collection/run_control.py
```

裸命令默认执行 CR5A、速度因子 100%、启用 PGE-50-26、启用数据记录器，
并启动顶部和腕部双路相机及预览窗口。无需相机时添加 `--no-camera`；需要
采集图像但不显示时添加 `--no-camera-preview`；需要显式 dry-run 时添加
`--dry-run`。本地和 SSH 启动都将窗口显示在机械臂主机桌面 `:1`，显示目标
可在 `config/host_network.yaml` 中修改。确保急停可触达。

`run_control.py` 启动时不连接、不初始化也不移动 follower 夹爪。第一次长按 A
才初始化夹爪，并把 follower 夹爪移动到主手夹爪当时的开合状态；与此同时，
CR5A 六关节开始向主手六关节对应姿态动态对齐。

按钮逻辑执行：启动时主手锁定；短按 A 随时切换主手锁定和解锁，
便于在 Servo 停止时先调整主手姿态。长按 A 开启 Servo 时，程序会物理锁定主手，
冻结长按松开瞬间的主手六关节目标，让 follower 夹爪对齐主手夹爪状态，同时以约
`11.5 deg/s` 执行 CR5A 六关节动态对齐。机械臂反馈到达冻结目标前不会进入实时跟随。
出现 `[master aligned]` 后，主手
根据短按 A 的逻辑状态保持锁定或自动解锁；锁定时再短按 A 即进入快速跟随。
再次长按 A 停止 Servo。

长按 A 前必须确认 CR5A 为已使能状态且 `collision_state=0`。程序还会在任何
follower 夹爪动作前检查报警、当前与冻结目标工作区、六关节反馈和关节限位。默认
允许任意大小的主从初始关节差，并以对齐速度逐步执行；不满足安全条件时会拒绝本次
同步并打印具体原因。
