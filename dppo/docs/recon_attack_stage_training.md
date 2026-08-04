# 侦察与打击阶段 HAPPO 训练

## 阶段范围

该阶段从主想定开始，只允许高层课程指示器下发 `RECON`、`ATTACK` 和 `WAIT`。
只有侦察与攻击网络更新参数，登陆和陆军网络不参与本阶段更新。

每场 episode 在以下高质量登陆条件全部成立后结束：

- 已满足主想定登陆窗口：蓝方攻击机击毁数不少于 5，存活 SAM 不超过 1。
- 至少 3 架红方侦察机存活。
- 至少 3 架红方攻击机存活。
- 至少 1 艘存活运输船仍携带尚未卸载的陆军。

触发边界后停止下发新动作，等待原生武器和探测事件结算，随后把结算奖励与阶段终局奖励加入轨迹，设置 `done=1`，并自动保存一个登陆快照。

## 轨迹与更新

缓冲区保存每个实体从 episode 开始到终止的连续轨迹，包括未被任务激活时的 HOLD 状态。

更新顺序为：

1. 在完整实体轨迹上计算 GAE 和 return。
2. 让延迟探测、导弹命中和阶段终局奖励穿过 HOLD 区间向前传播。
3. GAE 计算完成后，筛选 `assigned=1` 的动作更新 actor。
4. 两个网络都达到完整 episode 数和受控动作步数要求后，执行一次顺序 HAPPO 更新。
5. 更新后清空旧策略缓冲区，后续样本由新策略重新采集。

默认门槛：

- 侦察：至少 2048 个受控动作步。
- 攻击：至少 2048 个受控动作步。
- 两类网络各自至少出现在 8 条完整 episode 中。

## Linux 启动

Windows 计划任务 `AFSIM-Warlock` 应指向主想定。Linux 使用已经验证过的反向 SSH 端口 `2222`：

```bash
cd ~/LLM/junq/dppo

TARGET_UPDATES=100 \
RECON_MIN_SAMPLES=2048 \
ATTACK_MIN_SAMPLES=2048 \
MAX_EPISODE_STEPS=5000 \
DECISION_SECONDS=0.05 \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/linux_train_recon_attack_happo.sh
```

从已有侦察阶段 checkpoint 初始化：

```bash
RESUME=checkpoints/recon_only/latest.pt \
CHECKPOINT_DIR=checkpoints/happo_recon_attack \
TARGET_UPDATES=100 \
bash scripts/linux_train_recon_attack_happo.sh
```

训练 checkpoint 保存在 `CHECKPOINT_DIR`，未达到更新门槛的完整轨迹缓冲区也保存在 `latest.pt` 中。每场结束后 Linux 启动器自动关闭并重新启动 Windows Warlock，再从 `latest.pt` 接着收集同一策略版本的数据。

登陆快照保存在：

```text
stage_snapshots/landing/*.json
```

快照池默认最多保留 50 个，超过容量后淘汰最早的快照。

## 关键日志

```text
episode_progress ... samples ... landing ...
happo_waiting_for_samples ... required_samples ...
landing_snapshot_saved ... settlement_reward ... terminal_bonus ...
happo_update ... samples ... metrics ...
episode_complete ... high_quality_landing_ready ...
```

`happo_waiting_for_samples` 表示完整轨迹已经保存，但至少一个网络尚未同时满足动作步数和完整 episode 数要求，此时参数不会更新。
