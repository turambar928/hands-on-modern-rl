# PPO：经典在线 RLHF 更新

## 它干嘛

PPO 使用当前策略采样回答、获得奖励、计算 advantage，再重复更新策略。它保存 rollout 时的 old probability，并用 probability ratio clipping 防止一次更新过猛。

核心是：

```text
ratio = new_probability / old_probability
ratio 被限制在大约 [0.8, 1.2]
```

## 这个实验

一个模拟 SFT 模型为四类 prompt 生成 4-token 回答。规则奖励模拟 Reward Model，冻结的初始策略模拟 reference/SFT model，Critic 模拟 value head。

## 在 LLM 训练中的位置

```text
预训练 → SFT → 偏好数据 → Reward Model → PPO rollout/update → 对齐模型
```

PPO 是在线方法：训练时必须不断用最新模型生成回答。它能直接优化任意奖励，但工程复杂、成本高，而且需要控制对 reference model 的 KL 偏移。

```bash
uv run python 04_ppo_rlhf/train.py
```

SwanLab 先看：

- `eval/exact_accuracy`：最终回答质量。
- `train/task_reward`：规则/奖励模型给出的原始分数。
- `train/shaped_reward`：扣除 reference KL 惩罚后的奖励。
- `train/clip_fraction`：触发 PPO 安全阀的样本比例。
- `train/approx_kl`：一次 PPO 更新中新旧策略的偏移。
- `train/reference_kl`：当前模型相对初始 SFT 模型的偏移。

