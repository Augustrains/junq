# 测试目录

这里集中存放不属于正式训练与主场景运行链路的代码和数据。

- `live/`：需要启动或连接 AFSim/Warlock 的联机测试。
- `diagnostics/`：状态查看器、任务链路诊断与问题定位脚本。
- `tools/`：手工仿真、输入检查、时钟基准等辅助工具。
- `configs/`：仅供测试使用的场景配置。
- `artifacts/`：测试日志和临时输出；该目录中的日志由 `.gitignore` 忽略。

请在项目根目录 `dppo` 下以模块方式运行脚本，例如：

```powershell
python -m test.diagnostics.show_attack_state
python -m test.live.test_attack_action3_air
python -m test.tools.network_input_diagnostic
```

正式项目代码仍位于 `envs/`、`train/`、`agents/` 和 `scripts/`。