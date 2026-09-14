# DPO：直接学习人类偏好对

## 它干嘛

DPO 输入的是离线偏好数据 `(prompt, chosen, rejected)`。它让模型相对 reference policy 更偏向 chosen，不需要训练 Reward Model，也不需要边训练边生成 rollout。

## 这个实验

每类 prompt 有四个候选回答，数据标记哪个回答优于其他回答。策略从一个模拟 SFT/reference 模型出发，直接优化 chosen/rejected 的相对 log probability。

## 在 LLM 训练中的位置

```text
预训练 → SFT → 收集 chosen/rejected → DPO → 对齐模型
```

DPO 严格来说是偏好优化方法，不是“智能体和环境交互”的在线 RL。它简单、稳定、便宜，适合已经有高质量偏好对的数据；但不能像 PPO/GRPO 那样在线探索新回答。

```bash
uv run python 05_dpo_preference/train.py
```

SwanLab 先看：

- `train/loss`：DPO logistic loss。
- `train/preference_accuracy`：chosen 隐式奖励是否高于 rejected。
- `train/reward_margin`：chosen 与 rejected 的隐式奖励差。
- `eval/chosen_probability`：模型给首选回答的平均概率。
- `train/reference_kl`：策略相对 SFT/reference 模型偏移了多少。

