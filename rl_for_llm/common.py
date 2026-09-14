"""Shared utilities for the compact RL-for-LLM experiments."""

from __future__ import annotations

import argparse
import random

import numpy as np
import swanlab
import torch

SWANLAB_MODES = ("local", "online", "offline", "disabled")


def add_common_args(parser: argparse.ArgumentParser, default_steps: int) -> None:
    parser.add_argument("--steps", type=int, default=default_steps)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=SWANLAB_MODES, default="local")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def start_run(name: str, mode: str, config: dict) -> None:
    swanlab.init(
        project="rl-for-llm",
        name=name,
        mode=mode,
        config=config,
    )


def finish_run() -> None:
    swanlab.finish()
