"""Actor-Critic learns short responses from delayed sequence rewards."""

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

TARGETS = torch.tensor(
    [
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
    ]
)
SEQUENCE_LENGTH = TARGETS.shape[1]


class ActorCritic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # A tiny tabular policy/value head keeps attention on the algorithm.
        # Real LLMs produce these logits and values with a Transformer backbone.
        self.actor_logits = nn.Parameter(0.05 * torch.randn(4, SEQUENCE_LENGTH, 2))
        self.state_values = nn.Parameter(torch.zeros(4, SEQUENCE_LENGTH))

    def forward(
        self, prompt_ids: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_logits[prompt_ids, positions], self.state_values[
            prompt_ids, positions
        ]


@torch.no_grad()
def evaluate(model: ActorCritic) -> tuple[float, float]:
    prompt_ids = torch.arange(4).repeat_interleave(SEQUENCE_LENGTH)
    positions = torch.arange(SEQUENCE_LENGTH).repeat(4)
    logits, _ = model(prompt_ids, positions)
    generated = logits.argmax(dim=-1).view(4, SEQUENCE_LENGTH)
    token_accuracy = (generated == TARGETS).float().mean().item()
    exact_accuracy = (generated == TARGETS).all(dim=1).float().mean().item()
    return token_accuracy, exact_accuracy


def train(steps: int, seed: int, mode: str) -> None:
    seed_everything(seed)
    model = ActorCritic()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch_size = 256

    start_run(
        "03-actor-critic-value-head",
        mode,
        {
            "algorithm": "Actor-Critic",
            "steps": steps,
            "seed": seed,
            "reward": "sequence-level",
        },
    )

    for step in range(steps):
        prompt_ids = torch.randint(0, 4, (batch_size,))
        flat_prompts = prompt_ids.repeat_interleave(SEQUENCE_LENGTH)
        positions = torch.arange(SEQUENCE_LENGTH).repeat(batch_size)
        logits, values = model(flat_prompts, positions)
        distribution = Categorical(logits=logits)
        actions = distribution.sample().view(batch_size, SEQUENCE_LENGTH)

        targets = TARGETS[prompt_ids]
        rewards = (actions == targets).float().mean(dim=1)
        # The same delayed sequence score supervises every generated position.
        returns = rewards.repeat_interleave(SEQUENCE_LENGTH)
        advantages = returns - values.detach()
        normalized_advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-8
        )

        log_probs = distribution.log_prob(actions.flatten())
        actor_loss = -(log_probs * normalized_advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = distribution.entropy().mean()
        loss = actor_loss + 0.5 * value_loss - 0.01 * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        token_accuracy, exact_accuracy = evaluate(model)
        swanlab.log(
            {
                "train/mean_reward": rewards.mean().item(),
                "train/actor_loss": actor_loss.item(),
                "train/value_loss": value_loss.item(),
                "train/entropy": entropy.item(),
                "train/advantage_std": advantages.std().item(),
                "eval/token_accuracy": token_accuracy,
                "eval/exact_accuracy": exact_accuracy,
            },
            step=step,
        )

    token_accuracy, exact_accuracy = evaluate(model)
    print(f"最终 token 准确率: {token_accuracy:.0%}")
    print(f"最终整段准确率: {exact_accuracy:.0%}")
    finish_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Actor-Critic value-head demo")
    add_common_args(parser, default_steps=250)
    args = parser.parse_args()
    train(args.steps, args.seed, args.mode)


if __name__ == "__main__":
    main()
