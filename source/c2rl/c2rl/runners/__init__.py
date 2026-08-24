"""Unified runner for c2rl: routes PPO/SAC → skrl Runner, contraction algorithms (C3M/LQR/SDLQR/C2RL) → native skrl Agent subclasses."""

from c2rl.runners.contraction_runner import ContractionRunner

__all__ = ["ContractionRunner"]
