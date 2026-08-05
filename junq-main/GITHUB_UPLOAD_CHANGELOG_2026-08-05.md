# GitHub 代码上传记录

- 仓库：`Augustrains/junq`
- Pull Request：`#1 Add scripted recon demo and update 4 policy`
- PR 地址：<https://github.com/Augustrains/junq/pull/1>
- 合并日期：2026-08-05
- 合并提交：`caaddbe8531db795880b612a5a2f98f531c98928`
- 功能提交：
  - `fc24e3a6d87f6ae9235aac99b65dc94073b063bd`：新增脚本侦察演示、正式场景和 update 4 权重
  - `262c1220a77673a533782108ed534fd309ed0b6c`：同步最新环境与训练修改

## 已更新的代码和配置路径

### AFSIM 场景

- `afsim_work/afsim-2.9.0-win64_bin/demos/air_to_air/scenarios/island_assault_happo_demo.txt`
  - 主演示场景更新命中概率、远程对地导弹、导弹命中结算、蓝方空战重选目标和手动终止逻辑。
- `afsim_work/afsim-2.9.0-win64_bin/demos/air_to_air/scenarios/island_assault_min.txt`
  - 同步正式空战重选目标及导弹相关逻辑。
- `afsim_work/afsim-2.9.0-win64_bin/demos/air_to_air/scenarios/island_assault_min_p1.txt`
  - 同步并行场景 1 的正式逻辑。
- `afsim_work/afsim-2.9.0-win64_bin/demos/air_to_air/scenarios/island_assault_min_p2.txt`
  - 同步并行场景 2 的正式逻辑。

### 环境代码和配置

- `junq-main/envs/afsim_env.py`
  - 更新对地导弹 45 km 发射判定、远程武器范围、攻击状态清理、返航回退、新回合状态重置和原生暂停恢复。
- `junq-main/envs/rl_interface.py`
  - 更新侦察距离奖励、攻击动作失败后的返航回退及锁定状态返航支持。
- `junq-main/envs/afsim_units.json`
  - 默认运行场景改为 `island_assault_happo_demo.txt`。
- `junq-main/envs/attack_state_fields.json`
  - 更新攻击机 AGM 武器射程和 45 km 水平发射距离配置。
- `junq-main/envs/recon_state_fields.json`
  - 更新侦察状态中的 AGM 射程归一化配置。
- `junq-main/envs/combat_model.json`
  - 删除旧的 Python 命中概率配置，命中概率统一由 AFSIM 场景结算。

### 训练代码

- `junq-main/train/train_recon_attack_parallel.py`
  - 更新并行训练的远程 Warlock 启停、UDP 清理、平台注册重试、原生决策暂停验证及按类型共享策略选项。
- `junq-main/train/train_recon_attack_parallel_eval.py`
  - 支持评估复用已经暂停的运行场景，减少不必要的 Warlock 重启。

## 新上传的文件

- `.gitattributes`
  - 为当前 update 4 权重配置 Git LFS 跟踪规则。
- `junq-main/show/demo_happo_scripted_recon_warlock.py`
  - 新增“攻击机使用 HAPPO 网络、侦察机使用周期诱导脚本”的演示入口。
- `junq-main/test/test_demo_happo_scripted_recon.py`
  - 新增混合策略演示测试，验证侦察机脚本动作和攻击机网络调用。
- `junq-main/checkpoints/happo_reward_fixed_production/bottom_happo_recon_attack_parallel_eval_update_000004.pt`
  - 新增当前演示和评估使用的 update 4 模型权重，通过 Git LFS 保存。

## 当前使用的权重

- 仓库相对路径：
  - `junq-main/checkpoints/happo_reward_fixed_production/bottom_happo_recon_attack_parallel_eval_update_000004.pt`
- 本机完整路径：
  - `D:\junq\junq-main\checkpoints\happo_reward_fixed_production\bottom_happo_recon_attack_parallel_eval_update_000004.pt`
- Checkpoint update：`4`
- 策略数量：`20`
- 文件大小：`94,247,946` 字节
- Git LFS SHA-256：`0c91f6c77b0be45c4c63754dae8ee26b5e48b785b790e5578d6fc5524229f8fc`

## 已完成验证

- Python 语法检查通过。
- 环境 JSON 配置解析通过。
- 脚本侦察/网络攻击混合策略测试通过。
- update 4 checkpoint 加载及模型兼容性检查通过。
- `envs`、`train` 与合并前当前运行目录的有效源码差异为 0。
- 合并后 `origin/main` 与发布分支的目标文件树差异为 0。

