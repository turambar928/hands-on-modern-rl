# RL for LLM：六个算法，六个小实验

这个目录不按教材章节组织，只回答一个问题：**某个 RL 算法到底在做什么，它和 LLM 训练有什么关系？**

所有实验都很小，不下载大模型。我们用小型 PyTorch 策略模拟 LLM 的动作概率，用 SwanLab 展示训练曲线。先理解训练机制，再把同一套概念迁移到真实 LLM。

## 一张表看懂它们的位置

| 顺序 | 算法 | 它解决的问题 | 在 LLM / Agent 中的位置 |
| --- | --- | --- | --- |
| 1 | DQN | 从历史经验中学习离散动作价值 | 可用于小型工具路由器；通常不直接训练 LLM token |
| 2 | REINFORCE | 用整段回答的奖励直接更新策略 | 序列级策略梯度基础，也是很多 RLHF 公式的起点 |
| 3 | Actor-Critic | 用 Critic 降低策略梯度方差 | PPO/RLHF 中的 value head 和 advantage 来源 |
| 4 | PPO | 限制在线策略每次更新的幅度 | 经典 RLHF：SFT → Reward Model → PPO |
| 5 | DPO | 直接从 chosen/rejected 偏好对训练 | SFT 后的离线偏好对齐，不需要在线 rollout |
| 6 | GRPO/RLVR | 用同题多回答的相对奖励训练推理 | 数学、代码等可验证任务的在线推理强化 |

最常见的 LLM 后训练关系是：

```text
预训练模型
   ↓
SFT 模型
   ├─ 离线偏好对 chosen/rejected ─────────→ DPO
   ├─ Reward Model 打分 + 在线 rollout ───→ PPO
   └─ 自动 verifier + 同题多次采样 ───────→ GRPO / RLVR
```

REINFORCE 和 Actor-Critic 是理解 PPO/GRPO 的基础结构；DQN、SAC、TD3 等经典控制算法不是当前 LLM token 后训练的主线。本目录先保留与 LLM 最相关的算法，连续控制算法可在需要机器人或具身任务时再补。

## 环境

仓库根目录已经有 `uv` 管理的 `.venv`。安装本目录的最小依赖：

```bash
cd /Users/turambar928/Documents/GitHub/hands-on-modern-rl/rl_for_llm
uv pip install -r requirements.txt
```

每次只运行一个实验，例如：

```bash
uv run python 04_ppo_rlhf/train.py
```

启动本地 SwanLab：

```bash
uv run swanlab watch swanlog
```

然后打开 <http://127.0.0.1:5092>。每个脚本也支持 `--mode disabled`，用于不记录曲线的快速测试。

## 推荐顺序

如果目标是理解 LLM 后训练，推荐：

```text
REINFORCE → Actor-Critic → PPO → DPO → GRPO
```

DQN 可以单独阅读。它对理解 value-based RL 很有帮助，但不是当前训练生成式 LLM 的主流方法。

不要一次运行全部脚本。跑一个、看懂曲线、修改奖励或超参数，再进入下一个。
