"""DQN learns a tiny discrete tool router for an agent."""

from __future__ import annotations

import argparse
import random
import sys
from collections import deque
from pathlib import Path

import swanlab
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_args, finish_run, seed_everything, start_run

TOOLS = ["calculator", "weather", "search", "writer"]
TASKS = [
    "比较营收并写摘要",
    "查询天气并计算温差",
    "解方程并写解释",
    "检索目的地并查天气",
]
PIPELINES = [[2, 3], [1, 0], [0, 3], [2, 1]]


class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


def encode_state(task_ids: torch.Tensor, stages: torch.Tensor) -> torch.Tensor:
    tasks = F.one_hot(task_ids, num_classes=len(TASKS)).float()
    stage_vectors = F.one_hot(stages, num_classes=2).float()
    return torch.cat([tasks, stage_vectors], dim=-1)


@torch.no_grad()
def evaluate(model: QNetwork) -> float:
    correct = 0
    for task, pipeline in enumerate(PIPELINES):
        predicted = []
        for stage in range(2):
            state = encode_state(torch.tensor([task]), torch.tensor([stage]))
            predicted.append(int(model(state).argmax()))
        correct += int(predicted == pipeline)
    return correct / len(TASKS)


def train(steps: int, seed: int, mode: str) -> None:
    seed_everything(seed)
    online = QNetwork()
    target = QNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=3e-3)
    replay: deque[tuple[int, int, int, float, int, bool]] = deque(maxlen=2_000)

    start_run(
        "01-dqn-tool-router",
        mode,
        {"algorithm": "DQN", "steps": steps, "seed": seed, "task": "tool routing"},
    )

    for step in range(steps):
        task = random.randrange(len(TASKS))
        epsilon = max(0.05, 1.0 - step / max(1, steps * 0.7))
        episode_reward = 0.0
        loss_value = 0.0
        for stage in range(2):
            state = encode_state(torch.tensor([task]), torch.tensor([stage]))
            if random.random() < epsilon:
                action = random.randrange(len(TOOLS))
            else:
                with torch.no_grad():
                    action = int(online(state).argmax())

            correct = action == PIPELINES[task][stage]
            done = (not correct) or stage == 1
            reward = -1.0 if not correct else (1.0 if done else 0.0)
            next_stage = min(stage + 1, 1)
            replay.append((task, stage, action, reward, next_stage, done))
            episode_reward += reward

            if len(replay) >= 64:
                batch = random.sample(replay, 64)
                task_ids = torch.tensor([x[0] for x in batch])
                stages = torch.tensor([x[1] for x in batch])
                actions = torch.tensor([x[2] for x in batch])
                rewards = torch.tensor([x[3] for x in batch])
                next_stages = torch.tensor([x[4] for x in batch])
                dones = torch.tensor([x[5] for x in batch], dtype=torch.float32)

                predicted_q = (
                    online(encode_state(task_ids, stages))
                    .gather(1, actions[:, None])
                    .squeeze(1)
                )
                with torch.no_grad():
                    next_q = (
                        target(encode_state(task_ids, next_stages)).max(dim=1).values
                    )
                    td_target = rewards + 0.95 * (1.0 - dones) * next_q
                loss = F.smooth_l1_loss(predicted_q, td_target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_value = loss.item()

            if done:
                break

        if step % 50 == 0:
            target.load_state_dict(online.state_dict())

        swanlab_metrics = {
            "train/episode_reward": episode_reward,
            "train/loss": loss_value,
            "train/epsilon": epsilon,
            "train/replay_size": len(replay),
            "eval/pipeline_accuracy": evaluate(online),
        }
        swanlab.log(swanlab_metrics, step=step)

    print("\n学习到的两步工具链：")
    with torch.no_grad():
        for task_id, task in enumerate(TASKS):
            actions = []
            for stage in range(2):
                state = encode_state(torch.tensor([task_id]), torch.tensor([stage]))
                actions.append(int(online(state).argmax()))
            print(f"  {task} -> {TOOLS[actions[0]]} -> {TOOLS[actions[1]]}")
    print(f"最终完整工具链准确率: {evaluate(online):.0%}")
    finish_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="DQN tool-routing demo")
    add_common_args(parser, default_steps=800)
    args = parser.parse_args()
    train(args.steps, args.seed, args.mode)


if __name__ == "__main__":
    main()
