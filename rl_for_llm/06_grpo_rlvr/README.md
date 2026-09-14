# GRPO + RLVR：同一道题生成一组答案再比较

## 它干嘛

GRPO 对同一个 prompt 一次采样多个回答，用组内平均值和标准差把奖励转换成相对 advantage。高于组平均的回答被提高概率，低于平均的回答被降低概率，因此不需要单独训练 Critic。

RLVR 表示 Reinforcement Learning with Verifiable Rewards：奖励由答案检查器、代码测试或形式校验器给出，而不是依赖主观的 Reward Model。

## 这个实验

每道算术题有多个候选答案。策略每次为同一道题采样 8 个回答，答案检查器给正确答案 `1`、错误答案 `0`，然后计算组相对 advantage 并进行 clipped policy update。

## 在 LLM 训练中的位置

```text
预训练/SFT 模型 → 每题采样一组推理 → verifier 打分 → GRPO 更新
```

它特别适合数学、代码、结构化工具调用等可以自动验证的任务。若一组回答全对或全错，组内标准差为零，这组数据几乎无法提供相对学习信号。

```bash
uv run python 06_grpo_rlvr/train.py
```

SwanLab 先看：

- `train/group_reward`：采样回答的平均可验证奖励。
- `eval/answer_accuracy`：贪心回答准确率。
- `train/zero_variance_groups`：整组全对或全错的比例。
- `train/advantage_std`：组相对学习信号。
- `train/clip_fraction`：PPO-style clipping 触发比例。
- `train/reference_kl`：相对初始模型的漂移。

