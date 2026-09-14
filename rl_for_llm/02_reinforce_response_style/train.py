"""REINFORCE learns which response style fits each prompt type."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import swanlab
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_args, finish_run, seed_everything, start_run

PROMPTS = ["事实问答", "代码解释", "安全咨询"]
STYLES = ["简洁回答", "详细推导", "安全拒答"]


class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 24), nn.Tanh(), nn.Linear(24, 3))

    def forward(self, prompt: torch.Tensor) -> torch.Tensor:
        return self.net(prompt)


@torch.no_grad()
def accuracy(policy: Policy) -> float:
    prompts = F.one_hot(torch.arange(3), num_classes=3).float()
    actions = policy(prompts).argmax(dim=-1)
    return (actions == torch.arange(3)).float().mean().item()


def train(steps: int, seed: int, mode: str) -> None:
    seed_everything(seed)
    policy = Policy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)
    batch_size = 64

    start_run(
        "02-reinforce-response-style",
        mode,
        {
            "algorithm": "REINFORCE",
            "steps": steps,
            "seed": seed,
            "batch_size": batch_size,
        },
    )

    for step in range(steps):
        prompt_ids = torch.randint(0, len(PROMPTS), (batch_size,))
        prompt_vectors = F.one_hot(prompt_ids, num_classes=len(PROMPTS)).float()
        distribution = Categorical(logits=policy(prompt_vectors))
        actions = distribution.sample()
        rewards = torch.where(actions == prompt_ids, 1.0, -0.2)

        # REINFORCE: reward is treated as a constant weight on log probability.
        loss = -(distribution.log_prob(actions) * rewards).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        swanlab.log(
            {
                "train/mean_reward": rewards.mean().item(),
                "train/policy_loss": loss.item(),
                "train/entropy": distribution.entropy().mean().item(),
                "eval/style_accuracy": accuracy(policy),
            },
            step=step,
        )

    print("\n学习到的回答风格：")
    with torch.no_grad():
        prompts = F.one_hot(torch.arange(3), num_classes=3).float()
        actions = policy(prompts).argmax(dim=-1)
    for prompt, action in zip(PROMPTS, actions.tolist()):
        print(f"  {prompt} -> {STYLES[action]}")
    print(f"最终准确率: {accuracy(policy):.0%}")
    finish_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="REINFORCE response-style demo")
    add_common_args(parser, default_steps=250)
    args = parser.parse_args()
    train(args.steps, args.seed, args.mode)


if __name__ == "__main__":
    main()
