"""A tiny but structurally faithful PPO-RLHF loop."""

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

TARGETS = torch.tensor(
    [
        [0, 1, 2, 0],
        [1, 2, 1, 0],
        [2, 2, 0, 1],
        [0, 2, 1, 2],
    ]
)
N_PROMPTS, SEQUENCE_LENGTH = TARGETS.shape
VOCAB_SIZE = 3


class TinyLanguagePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Direct logits keep the demo focused on PPO rather than architecture.
        self.logits = nn.Parameter(torch.zeros(N_PROMPTS, SEQUENCE_LENGTH, VOCAB_SIZE))
        self.values = nn.Parameter(torch.zeros(N_PROMPTS))
        # Simulate a partially helpful SFT starting point.
        with torch.no_grad():
            for prompt in range(N_PROMPTS):
                for position in range(SEQUENCE_LENGTH - 1):
                    self.logits[prompt, position, TARGETS[prompt, position]] = 0.7

    def distributions(self, prompt_ids: torch.Tensor) -> Categorical:
        return Categorical(logits=self.logits[prompt_ids])


def response_reward(actions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    token_score = (actions == targets).float().mean(dim=1)
    exact_bonus = 0.5 * (actions == targets).all(dim=1).float()
    return token_score + exact_bonus


@torch.no_grad()
def evaluate(policy: TinyLanguagePolicy) -> tuple[float, float]:
    generated = policy.logits.argmax(dim=-1)
    token_accuracy = (generated == TARGETS).float().mean().item()
    exact_accuracy = (generated == TARGETS).all(dim=1).float().mean().item()
    return token_accuracy, exact_accuracy


def train(steps: int, seed: int, mode: str) -> None:
    seed_everything(seed)
    policy = TinyLanguagePolicy()
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=2e-2)
    batch_size = 128
    clip_epsilon = 0.2
    kl_coefficient = 0.03

    start_run(
        "04-ppo-rlhf",
        mode,
        {
            "algorithm": "PPO",
            "steps": steps,
            "seed": seed,
            "batch_size": batch_size,
            "clip_epsilon": clip_epsilon,
            "kl_coefficient": kl_coefficient,
        },
    )

    for step in range(steps):
        # Rollout: sample responses with the current (old) policy.
        prompt_ids = torch.randint(0, N_PROMPTS, (batch_size,))
        with torch.no_grad():
            old_distribution = policy.distributions(prompt_ids)
            actions = old_distribution.sample()
            old_log_probs = old_distribution.log_prob(actions)
            old_values = policy.values[prompt_ids]
            reference_log_probs = reference.distributions(prompt_ids).log_prob(actions)
            task_rewards = response_reward(actions, TARGETS[prompt_ids])
            sampled_reference_kl = (old_log_probs - reference_log_probs).sum(dim=1)
            shaped_rewards = task_rewards - kl_coefficient * sampled_reference_kl
            advantages = shaped_rewards - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        clip_fractions = []
        approximate_kls = []
        policy_losses = []
        value_losses = []
        entropies = []

        # Reuse the rollout for a few epochs, while clipping large policy changes.
        for _ in range(4):
            distribution = policy.distributions(prompt_ids)
            new_log_probs = distribution.log_prob(actions)
            token_advantages = advantages[:, None].expand_as(new_log_probs)
            ratios = torch.exp(new_log_probs - old_log_probs)
            unclipped = ratios * token_advantages
            clipped = (
                ratios.clamp(1 - clip_epsilon, 1 + clip_epsilon) * token_advantages
            )
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = F.mse_loss(policy.values[prompt_ids], shaped_rewards)
            entropy = distribution.entropy().mean()
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                clip_fractions.append(
                    ((ratios - 1).abs() > clip_epsilon).float().mean()
                )
                approximate_kls.append((old_log_probs - new_log_probs).mean())
                policy_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropies.append(entropy.detach())

        token_accuracy, exact_accuracy = evaluate(policy)
        swanlab.log(
            {
                "train/task_reward": task_rewards.mean().item(),
                "train/shaped_reward": shaped_rewards.mean().item(),
                "train/policy_loss": torch.stack(policy_losses).mean().item(),
                "train/value_loss": torch.stack(value_losses).mean().item(),
                "train/entropy": torch.stack(entropies).mean().item(),
                "train/clip_fraction": torch.stack(clip_fractions).mean().item(),
                "train/approx_kl": torch.stack(approximate_kls).mean().item(),
                "train/reference_kl": sampled_reference_kl.mean().item(),
                "eval/token_accuracy": token_accuracy,
                "eval/exact_accuracy": exact_accuracy,
            },
            step=step,
        )

    token_accuracy, exact_accuracy = evaluate(policy)
    print(f"最终 token 准确率: {token_accuracy:.0%}")
    print(f"最终整段准确率: {exact_accuracy:.0%}")
    finish_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy PPO-RLHF demo")
    add_common_args(parser, default_steps=120)
    args = parser.parse_args()
    train(args.steps, args.seed, args.mode)


if __name__ == "__main__":
    main()
