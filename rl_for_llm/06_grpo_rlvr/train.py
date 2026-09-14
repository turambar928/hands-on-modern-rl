"""GRPO learns math answers from group-relative verifiable rewards."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import swanlab
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_args, finish_run, seed_everything, start_run

QUESTIONS = ["2+3", "7-4", "3×3", "8÷2", "5+6", "12-5"]
CORRECT_ANSWERS = torch.tensor([1, 3, 4, 0, 2, 3])
N_ANSWERS = 5


class MathPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(0.15 * torch.randn(len(QUESTIONS), N_ANSWERS))

    def distributions(self, question_ids: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits[question_ids])


def exact_reference_kl(policy: MathPolicy, reference: MathPolicy) -> torch.Tensor:
    log_p = F.log_softmax(policy.logits, dim=-1)
    log_q = F.log_softmax(reference.logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()


@torch.no_grad()
def accuracy(policy: MathPolicy) -> float:
    return (policy.logits.argmax(dim=-1) == CORRECT_ANSWERS).float().mean().item()


def train(steps: int, seed: int, mode: str) -> None:
    seed_everything(seed)
    policy = MathPolicy()
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-2)
    group_size = 8
    clip_epsilon = 0.2
    kl_coefficient = 0.02
    question_ids = torch.arange(len(QUESTIONS))[:, None].expand(-1, group_size)

    start_run(
        "06-grpo-rlvr",
        mode,
        {
            "algorithm": "GRPO",
            "reward": "verifiable correctness",
            "steps": steps,
            "seed": seed,
            "group_size": group_size,
            "clip_epsilon": clip_epsilon,
        },
    )

    for step in range(steps):
        with torch.no_grad():
            old_distribution = policy.distributions(question_ids)
            answers = old_distribution.sample()
            old_log_probs = old_distribution.log_prob(answers)
            rewards = (answers == CORRECT_ANSWERS[:, None]).float()
            group_mean = rewards.mean(dim=1, keepdim=True)
            group_std = rewards.std(dim=1, keepdim=True, unbiased=False)
            advantages = (rewards - group_mean) / (group_std + 1e-4)

        clip_fractions = []
        policy_losses = []
        for _ in range(3):
            distribution = policy.distributions(question_ids)
            new_log_probs = distribution.log_prob(answers)
            ratios = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratios * advantages
            clipped = ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            reference_kl = exact_reference_kl(policy, reference)
            loss = policy_loss + kl_coefficient * reference_kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                clip_fractions.append(
                    ((ratios - 1).abs() > clip_epsilon).float().mean()
                )
                policy_losses.append(policy_loss.detach())

        zero_variance = (group_std.squeeze(1) < 1e-6).float().mean()
        swanlab.log(
            {
                "train/group_reward": rewards.mean().item(),
                "train/policy_loss": torch.stack(policy_losses).mean().item(),
                "train/advantage_std": advantages.std(unbiased=False).item(),
                "train/zero_variance_groups": zero_variance.item(),
                "train/clip_fraction": torch.stack(clip_fractions).mean().item(),
                "train/reference_kl": exact_reference_kl(policy, reference).item(),
                "train/entropy": old_distribution.entropy().mean().item(),
                "eval/answer_accuracy": accuracy(policy),
            },
            step=step,
        )

    print("\n学习到的答案索引：")
    predicted = policy.logits.detach().argmax(dim=-1)
    for question, answer, target in zip(
        QUESTIONS, predicted.tolist(), CORRECT_ANSWERS.tolist()
    ):
        print(f"  {question}: 预测 {answer}, 正确索引 {target}")
    print(f"最终准确率: {accuracy(policy):.0%}")
    finish_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy GRPO/RLVR demo")
    add_common_args(parser, default_steps=180)
    args = parser.parse_args()
    train(args.steps, args.seed, args.mode)


if __name__ == "__main__":
    main()
