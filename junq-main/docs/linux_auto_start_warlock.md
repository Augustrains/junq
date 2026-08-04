# 从 Linux 启动 Windows Warlock 并进行多轮训练

本文记录已经验证成功的连接方式：

- Windows 用户：yang
- Linux 训练机：yangguobin@10.184.17.177
- Linux UDP 监听：0.0.0.0:50050
- Windows OpenSSH Server：TCP 22
- Linux 到 Windows 的反向隧道入口：127.0.0.1:2222
- Windows 计划任务：AFSIM-Warlock
- Linux 项目：/hard_data1/user/yangguobin/LLM/junq/dppo

## 工作原理

Windows 先建立到 Linux 的 SSH 连接，并通过 -R 把 Linux 的
127.0.0.1:2222 转发到 Windows 的 127.0.0.1:22。Linux 训练脚本随后
通过该端口登录 Windows，调用计划任务启动或停止 Warlock。

数据链路与控制链路相互独立：

- 控制链路：Linux 127.0.0.1:2222 -> Windows OpenSSH -> 计划任务。
- 仿真链路：Warlock udpnet -> Linux IP 10.184.17.177:50050。
- 训练链路：Linux Python 接收 state、生成 action、接收新 state 和 reward。

## 一次性配置

### 1. Windows 启用 OpenSSH Server

在 Windows 管理员 PowerShell 中执行：

~~~powershell
Get-Service sshd
Set-Service sshd -StartupType Automatic
Start-Service sshd
Get-NetTCPConnection -LocalPort 22 -State Listen
~~~

预期看到 sshd Running Automatic，并且 TCP 22 正在监听。

Linux 公钥需要写入 Windows 的管理员授权文件：

~~~text
C:\ProgramData\ssh\administrators_authorized_keys
~~~

完成后确认文件权限正确，并重启 sshd。

### 2. Windows 创建 Warlock 计划任务

计划任务名称固定为 AFSIM-Warlock。它必须在已登录的 Windows 用户 yang
的交互会话中运行，否则 Warlock GUI 可能启动在不可见会话中。

当前使用的程序和想定：

~~~text
程序：
D:\junq\afsim_work\afsim-2.9.0-win64_bin\bin_release\warlock.exe

工作目录：
D:\junq\afsim_work\afsim-2.9.0-win64_bin\demos\air_to_air

想定：
scenarios/island_assault_min.txt
~~~

验证计划任务：

~~~powershell
Get-ScheduledTask -TaskName "AFSIM-Warlock"
Start-ScheduledTask -TaskName "AFSIM-Warlock"
Start-Sleep -Seconds 5
Get-Process warlock | Select-Object Id,CPU,StartTime
Get-ScheduledTaskInfo -TaskName "AFSIM-Warlock"
~~~

停止验证：

~~~powershell
Stop-ScheduledTask -TaskName "AFSIM-Warlock"
Get-Process warlock,wizard -ErrorAction SilentlyContinue | Stop-Process -Force
~~~

### 3. 确认 Warlock udpnet 地址

Windows 想定中的 udpnet 目标必须指向 Linux，不能使用 127.0.0.1：

~~~text
10.184.17.177:50050
~~~

Linux 训练脚本监听：

~~~text
0.0.0.0:50050
~~~

### 4. Windows 建立反向 SSH 隧道

在 Windows PowerShell 中执行以下单行命令：

~~~powershell
ssh -N -T -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -R 2222:127.0.0.1:22 yang177
~~~

yang177 对应 Windows SSH 配置：

~~~sshconfig
Host yang177
    HostName 10.184.17.177
    User yangguobin
    Port 22
    IdentityFile C:/Users/yang/.ssh/177_yangguobin
~~~

该 PowerShell 窗口需要保持运行。关闭窗口或 SSH 断开后，Linux 将无法控制
Windows。

## 每次训练前验证

### 1. Linux 验证反向端口

在 Linux 执行：

~~~bash
ssh -p 2222 \
  -i ~/.ssh/id_ed25519 \
  -o BatchMode=yes \
  -o ConnectTimeout=10 \
  yang@127.0.0.1 whoami
~~~

成功输出：

~~~text
yangguobin\yang
~~~

如果连接到端口 22 而不是 2222，实际连接的是 Linux 自己的 SSH 服务，通常会
出现：

~~~text
yang@127.0.0.1: Permission denied (publickey)
~~~

### 2. Linux 验证计划任务控制

启动：

~~~bash
ssh -p 2222 -i ~/.ssh/id_ed25519 yang@127.0.0.1 \
  'powershell -NoProfile -Command "Start-ScheduledTask -TaskName AFSIM-Warlock"'
~~~

检查：

~~~bash
sleep 5
ssh -p 2222 -i ~/.ssh/id_ed25519 yang@127.0.0.1 \
  'powershell -NoProfile -Command "Get-Process warlock | Select-Object Id,CPU,StartTime"'
~~~

停止：

~~~bash
ssh -p 2222 -i ~/.ssh/id_ed25519 yang@127.0.0.1 \
  'powershell -NoProfile -Command "Stop-ScheduledTask -TaskName AFSIM-Warlock"'
~~~

## 自动启动的短程 live rollout

确认反向隧道保持运行后，在 Linux 执行：

~~~bash
cd /hard_data1/user/yangguobin/LLM/junq/dppo
conda activate rl

WINDOWS_AUTO_WARLOCK=1 \
WINDOWS_SSH_TARGET=yang@127.0.0.1 \
WINDOWS_SSH_PORT=2222 \
WINDOWS_SSH_KEY="$HOME/.ssh/id_ed25519" \
WINDOWS_WARLOCK_START_CMD='powershell -NoProfile -Command "Start-ScheduledTask -TaskName AFSIM-Warlock"' \
WINDOWS_WARLOCK_STOP_CMD='powershell -NoProfile -Command "Stop-ScheduledTask -TaskName AFSIM-Warlock; Get-Process warlock,wizard -ErrorAction SilentlyContinue | Stop-Process -Force"' \
WINDOWS_WARLOCK_START_DELAY=5 \
AUTO_EPISODES=1 \
ALGORITHM=happo \
UPDATES=1 \
ROLLOUT_STEPS=64 \
DECISION_SECONDS=0.05 \
CHECKPOINT_DIR=checkpoints/happo_live_smoke \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/linux_train_bottom_mappo.sh
~~~

预期关键输出：

~~~text
windows_auto_warlock=1 target=yang@127.0.0.1
Windows start Warlock: yang@127.0.0.1:2222
live_platforms_ready True known_count 73
bottom_algorithm happo
update 1 steps 64
saved .../latest.pt
~~~

脚本正常退出、收到 Ctrl+C 或发生训练终局时，会调用 Windows 停止命令。
当 AUTO_EPISODES=1 且想定提前终止时，脚本保存 latest.pt、重启 Warlock，
再从检查点继续，直到达到目标 update。

## 常见故障

### Permission denied (publickey)

首先检查日志中的端口，必须是 yang@127.0.0.1:2222。然后手动复测带密钥的
SSH 命令。若仍失败，检查
C:\ProgramData\ssh\administrators_authorized_keys 是否包含 Linux 公钥。

### Connection refused

反向隧道没有运行，或 Windows 到 Linux 的 SSH 会话已经断开。重新执行 Windows
侧的 ssh -N -T -R 命令。

### live_platforms_ready False

控制链路已经启动 Warlock，但仿真 UDP 没有到达 Linux。检查：

1. Warlock 是否真正开始运行想定。
2. udpnet 目标是否为 10.184.17.177:50050。
3. Linux 防火墙是否允许 UDP 50050。
4. 是否已有其他训练进程占用 UDP 50050。

### Warlock 进程存在但界面不可见

计划任务运行身份或登录类型不正确。将任务配置为用户 yang，并仅在该用户已
登录时运行；不要让 SSH 直接用后台 Start-Process 替代交互式计划任务。
