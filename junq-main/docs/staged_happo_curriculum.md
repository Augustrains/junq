# HAPPO 阶段化课程训练计划

## 总体目标

使用同一套 HAPPO 优化框架，通过逐步增加可训练网络和高层任务集合，缓解侦察、
攻击、登陆、地面任务之间有效样本数量差异过大的问题。

所有网络在每个阶段都保留动作推理能力，但只有当前阶段列出的网络允许梯度更新。
高层课程指示器只能下发当前阶段允许的任务。

## 三类真实想定模板

1. 主想定：scenarios/island_assault_min.txt
   - 初始完整部署。
   - 用于侦察、侦察攻击协同和最终全流程联合微调。
2. 登陆模板：scenarios/island_assault_stage_landing.txt
   - 待创建。
   - 从已完成必要侦察和压制、运输船尚未抵岸的状态开始。
3. 陆军模板：scenarios/island_assault_stage_ground.txt
   - 待创建。
   - 从运输船刚完成卸载、陆军尚未向据点推进的状态开始。

登陆和陆军模板仍保留侦察机、攻击机与蓝方防御力量，以支持跨域协同训练。

## 五个课程阶段

| 阶段 | 想定 | 可训练网络 | 允许任务 | 状态 |
|---|---|---|---|---|
| S1 recon_only | 主想定 | RECON | RECON、WAIT | 已实现 |
| S2 recon_attack | 主想定 | RECON、ATTACK | RECON、ATTACK、WAIT | 配置已登记 |
| S3 landing | 登陆模板 | RECON、ATTACK、LANDING | RECON、ATTACK、LANDING、WAIT | 模板待创建 |
| S4 ground | 陆军模板 | 全部 | 全部任务 | 模板待创建 |
| S5 full | 主想定 | 全部 | 全部任务 | 联合微调待实现 |

## 实施任务

### A. 阶段配置与网络冻结

状态：已实现。

- 统一阶段注册表。
- 训练命令增加 --curriculum-stage。
- 冻结网络继续 act，但不执行 optimizer.step。
- 冻结参数 requires_grad=False。
- HAPPO 中冻结策略的 correction ratio 固定为 1。
- 检查点和 metrics 记录阶段与可训练网络集合。
- 高层指示器过滤当前阶段未允许的任务。

### B. 阶段想定模板

状态：待实现。

- 创建登陆模板。
- 创建陆军模板。
- 验证平台位置、存活状态、弹药、已知目标和任务条件。
- 重置任务组、目标 reservation、奖励事件和终止基线。
- 分别执行真实 Warlock 启动测试。

### C. 分任务 On-Policy Buffer

状态：待实现。

- 为 RECON、ATTACK、LANDING、GROUND 建立独立轨迹缓冲区。
- 只保存 task-assigned 有效样本。
- 按实体和 episode 保存连续轨迹。
- 当前激活网络全部达到 min_samples 后统一执行 HAPPO 更新。
- 设置 max_collect_steps，防止缺少合法任务时无限等待。
- 更新后清空旧策略产生的样本。

### D. 阶段奖励 Profile

状态：待实现。

- 保持团队奖励和事件语义不变。
- 为各阶段配置局部 potential shaping 权重。
- 阶段切换不能让同一事件奖励符号反转。
- 完整流程阶段逐步降低局部 shaping，增强最终团队结果权重。

### E. 阶段晋级和完整训练

状态：待实现。

- 根据成功率、覆盖率、有效打击率、登陆率和占领率晋级。
- 从前一阶段检查点初始化下一阶段已有网络。
- 新加入网络随机初始化或使用启发式行为克隆预热。
- 最终回到主想定进行所有网络联合微调。

## 第一阶段启动方式

Linux 启动参数：

~~~bash
CURRICULUM_STAGE=recon_only \
ALGORITHM=happo \
bash scripts/linux_train_bottom_mappo.sh
~~~

第一阶段预期日志：

~~~text
curriculum_stage recon_only
trainable ['recon']
tasks ['RECON', 'WAIT']
~~~

ATTACK、LANDING、GROUND 网络仍可进行前向推理，但参数保持冻结，高层也不会为它们
创建新任务组。
