# DQN：给 Agent 选择工具

## 它干嘛

DQN 学的是每个状态下各动作的价值 `Q(s, a)`，然后选择价值最大的动作。它通过 replay buffer 复用旧经验，因此是 off-policy 算法。

## 这个实验

输入是四类复合请求，动作是计算、天气、搜索、写作四个工具。每个请求需要按顺序调用两个工具；第一步没有最终奖励，DQN 必须通过下一状态的 Q 值学习哪个工具能带来后续成功。

## 和 LLM 的关系

DQN 通常不直接训练 LLM 的逐 token 生成，因为词表和状态空间太大。但它可以训练独立的离散工具路由器，或者帮助理解后续算法中的 value/Q 概念。

```bash
uv run python 01_dqn_tool_router/train.py
```

SwanLab 先看：

- `eval/pipeline_accuracy`：两步工具链完全正确的比例，应接近 1。
- `train/loss`：Q 网络的 TD 误差。
- `train/epsilon`：随机探索概率，逐渐降低。
