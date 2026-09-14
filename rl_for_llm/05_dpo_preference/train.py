"""DPO learns from offline chosen/rejected preference pairs."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import swanlab
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import add_common_args, finish_run, seed_everything, start_run

N_PROMPTS = 4
N_RESPONSES = 4


class ResponsePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(N_PROMPTS, N_RESPONSES))
        with torch.no_grad():
            # A mildly useful SFT model, but far from fully aligned.
            self.logits.diagonal().fill_(0.2)

    def log_probs(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.logits[prompt_ids], dim=-1)


@torch.no_grad()
def exact_reference_kl(policy: ResponsePolicy, reference: ResponsePolicy) -> float:
    log_p = F.log_softmax(policy.logits, dim=-1)
    log_q = F.log_softmax(reference.logits, dim=-1)
    p = log_p.exp()
    return (p * (log_p - log_q)).sum(dim=-1).mean().item()


def train(steps: int, seed: int, mode: str) -> None:
    seed_everything(seed)
    policy = ResponsePolicy()
    reference = copy.deepcopy(policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-2)
    beta = 0.1

    # For each prompt, response with the same id is chosen over all three negatives.
    prompt_ids = torch.arange(N_PROMPTS).repeat_interleave(N_RESPONSES - 1)
    chosen_ids = prompt_ids.clone()
    rejected_ids = torch.tensor(
        [
            response
            for prompt in range(N_PROMPTS)
            for response in range(N_RESPONSES)
            if response != prompt
        ]
    )

    start_run(
        "05-dpo-preference",
        mode,
        {
            "algorithm": "DPO",
            "steps": steps,
            "seed": seed,
            "beta": beta,
            "pairs": len(prompt_ids),
        },
    )

    for step in range(steps):
        policy_log_probs = policy.log_probs(prompt_ids)
        with torch.no_grad():
            reference_log_probs = reference.log_probs(prompt_ids)

        policy_ratio = policy_log_probs.gather(1, chosen_ids[:, None]).squeeze(
            1
        ) - policy_log_probs.gather(1, rejected_ids[:, None]).squeeze(1)
        reference_ratio = reference_log_probs.gather(1, chosen_ids[:, None]).squeeze(
            1
        ) - reference_log_probs.gather(1, rejected_ids[:, None]).squeeze(1)
        reward_margin = beta * (policy_ratio - reference_ratio)
        loss = -F.logsigmoid(reward_margin).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            chosen_probability = F.softmax(policy.logits, dim=-1).diagonal().mean()
            preference_accuracy = (reward_margin > 0).float().mean()
            greedy_accuracy = (
                (policy.logits.argmax(dim=-1) == torch.arange(N_PROMPTS)).float().mean()
            )
        swanlab.log(
            {
                "train/loss": loss.item(),
                "train/preference_accuracy": preference_accuracy.item(),
                "train/reward_margin": reward_margin.mean().item(),
                "train/reference_kl": exact_reference_kl(policy, reference),
                "eval/chosen_probability": chosen_probability.item(),
                "eval/greedy_accuracy": greedy_accuracy.item(),
            },
            step=step,
        )

    probabilities = F.softmax(policy.logits.detach(), dim=-1)
    print("\n每类 prompt 对首选回答的概率：")
    for prompt in range(N_PROMPTS):
        print(f"  prompt {prompt}: {probabilities[prompt, prompt]:.1%}")
    finish_run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy DPO preference demo")
    add_common_args(parser, default_steps=180)
    args = parser.parse_args()
    train(args.steps, args.seed, args.mode)


if __name__ == "__main__":
    main()
