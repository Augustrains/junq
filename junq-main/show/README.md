# Warlock HAPPO 演示

`demo_happo_warlock.py` 使用以下权重进行单回合、纯推理演示，不会训练或改写
checkpoint：

`checkpoints/happo_reward_fixed_production/bottom_happo_recon_attack_parallel_eval_update_000004.pt`

在 PowerShell 中运行：

```powershell
cd D:\junq\junq-main
python .\show\demo_happo_warlock.py
```

脚本默认依据 `show/afsim_demo_units.json` 自动启动 Warlock。该配置继承
`envs/afsim_units.json`，但改用专用的 `island_assault_happo_demo.txt` 想定；
该想定已经开启蓝方战斗机、SAM 和地面部队的主动攻击。脚本监听 UDP 50050，等待
全部平台上线，加载 checkpoint 中的 10 个侦察机策略和 10 个攻击机策略，然后用确定性
动作执行演示。运行期间会打印 `DEMO_STEP`；结束摘要写入
`show/last_demo_result.json`。脚本退出时会关闭由它启动的 Warlock。

如果已经手工打开了 Warlock 场景：

```powershell
python .\show\demo_happo_warlock.py --external-warlock
```

只检查权重、网络和当前环境配置是否匹配，不启动 Warlock：

```powershell
python .\show\demo_happo_warlock.py --check-only
```

常用选项：

- `--device cpu` 或 `--device cuda`
- `--port 50050`
- `--max-steps 5000`
- `--log-every 10`
- `--native-decision-pause`：想定包含 `DecisionReady` 原生暂停点时启用
- `--sample-actions`：按策略分布采样；默认使用确定性动作，便于重复演示
- `--config-path <json>`：使用另一份本机 Warlock/想定配置

若平台等待超时，先确认 `envs/afsim_units.json` 中的 `warlock_path`、
`scenario_dir`、`scenario_file` 均指向本机，并确认想定 UDP 目标端口为 50050。
