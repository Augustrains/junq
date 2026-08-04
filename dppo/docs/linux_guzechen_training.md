# guzechen@amax 跨机训练

当前链路：

- Linux：`guzechen@amax`，项目目录 `~/junq`，PyTorch 环境 `smacv2`。
- Linux UDP：`0.0.0.0:50050`。
- Windows：`10.67.93.225`，运行 Warlock。
- Warlock 场景：`scenarios/island_assault_linux_train.txt`，UDP 目标为 `10.184.17.133:50050`。
- 控制通道：Linux `127.0.0.1:2222` 经反向 SSH 隧道到 Windows OpenSSH 22。
- Windows 计划任务：`AFSIM-Warlock`。

Windows 重启或隧道中断后，在 Windows PowerShell 执行：

```powershell
powershell -ExecutionPolicy Bypass -File D:\junq\dppo\scripts\windows_start_linux_control_tunnel.ps1
```

验证 Linux 能控制 Windows：

```bash
ssh -p 2222 -i ~/.ssh/133_guzechen -o BatchMode=yes yang@127.0.0.1 whoami
```

预期输出：

```text
yangguobin\yang
```

开始第一阶段侦察+打击 HAPPO 训练：

```bash
ssh gzc133
cd ~/junq
conda activate smacv2

CUDA_VISIBLE_DEVICES=3 \
TRAIN_EPISODES=100 \
CHECKPOINT_DIR="$HOME/junq/checkpoints/happo_recon_attack" \
bash scripts/linux_train_recon_attack_happo.sh
```

脚本会自动完成：启动 Windows Warlock、等待 73 个平台注册、运行一回合、保存
`latest.pt`、回合间重启 Warlock，并在退出时关闭 Warlock。若
`CHECKPOINT_DIR/latest.pt` 已存在，脚本会自动续训。

真实输入诊断：

```bash
cd ~/junq
conda activate smacv2
python train/network_input_diagnostic.py --bind --full --platform-timeout 120
```

诊断时必须使用 `island_assault_linux_train.txt` 启动 Warlock，并保证没有其他
进程占用 Linux UDP 50050。
