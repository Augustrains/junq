# 复用攻击网络并重新训练侦察网络

训练脚本 `train/train_recon_attack_parallel_eval.py` 支持从旧 checkpoint 仅导入攻击 Actor。

## 加载范围

使用 `--attack-init-checkpoint` 时：

- 导入所有攻击机 Actor 权重；
- 侦察机 Actor 随机初始化；
- 所有 Critic 随机初始化；
- 所有优化器重新创建；
- rollout buffer、update 编号、训练步数和 episode 计数从零开始；
- 攻击 Actor 继续训练，以学习新返航动作和返航奖励。

该参数与完整续训参数 `--resume` 互斥。

## 学习率

- `--lr`：侦察 Actor 和所有 Critic 的学习率，默认 `3e-4`。
- `--attack-lr`：攻击 Actor 学习率。
- 使用攻击网络 warm start 且未指定 `--attack-lr` 时，自动使用 `0.1 * --lr`，即默认 `3e-5`。

## 本机训练命令

```powershell
cd D:\junq\junq-main

python .\train\train_recon_attack_parallel_eval.py `
  --workers 4 `
  --base-port 50050 `
  --warlock-control local `
  --warlock-task-prefix AFSIM-Warlock- `
  --attack-init-checkpoint "D:\junq\junq-main\checkpoints\happo_reward_fixed_production\bottom_happo_recon_attack_parallel_eval_update_000004.pt" `
  --lr 3e-4 `
  --attack-lr 3e-5 `
  --checkpoint-dir "D:\junq\junq-main\checkpoints\happo_recon_fresh_attack_warm_start" `
  --updates 100 `
  --episodes-per-worker 100 `
  --simulation-clock-rate 60 `
  --bottom-decisions-per-hour 50 `
  --eval-episodes 1
```

启动日志应包含：

```text
attack_actor_warm_start ... policies 10 attack_lr 3e-05 recon_lr 0.0003 critics fresh optimizers fresh
```

新 checkpoint 的 `initialization` 字段会记录旧攻击网络路径、导入的攻击策略名称和两类 Actor 的学习率。
