# Actor-Critic：给策略配一个价值评估器

## 它干嘛

Actor 负责选择动作，Critic 预测当前状态未来能得到多少回报。训练 Actor 时使用 `advantage = return - value`，比直接使用原始 reward 的 REINFORCE 更稳定。

## 这个实验

模型根据四类 prompt 生成一段 4-token 的短回答，完成后才获得整体正确率奖励。Actor 学生成，Critic 学着预测这段回答最终能得多少分。

## 和 LLM 的关系

经典 PPO-RLHF 会在语言模型旁边加入 value head。Reward Model 给完整回答打分，value head 把这个分数转成各 token 的 advantage，帮助信用分配。

```bash
uv run python 03_actor_critic_value_head/train.py
```

SwanLab 先看：

- `eval/token_accuracy`：生成 token 的正确率。
- `eval/exact_accuracy`：整段回答完全正确的比例。
- `train/value_loss`：Critic 的预测误差。
- `train/advantage_std`：策略梯度学习信号的波动。

