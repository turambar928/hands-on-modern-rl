# REINFORCE：用整段回答的奖励训练策略

## 它干嘛

REINFORCE 是最直接的策略梯度：采样一个动作，得到奖励，再用 `-log π(a|s) × reward` 更新策略。高奖励动作概率上升，低奖励动作概率下降。

## 这个实验

模型看到“事实问答、代码解释、安全咨询”三类 prompt，选择简洁、详细、安全三种回答风格。奖励只在完整选择结束后给出，模拟 LLM 的序列级 reward。

## 和 LLM 的关系

真实 LLM 中，动作是整段 token 序列，reward model 给整段回答打分。REINFORCE 是理解 RLHF 策略梯度最短的入口，但方差大，所以工程中通常会加入 baseline、Critic 或组内基线。

```bash
uv run python 02_reinforce_response_style/train.py
```

SwanLab 先看：

- `eval/style_accuracy`：是否学会不同 prompt 对应的回答风格。
- `train/mean_reward`：策略采样所得平均奖励。
- `train/entropy`：策略还保留多少随机探索。

