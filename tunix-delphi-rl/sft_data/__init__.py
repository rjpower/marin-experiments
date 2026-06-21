# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data loaders for the tunix-delphi-rl experiments.

Currently exposes :mod:`sft_data.instruction_datasets` — a self-contained loader
for HuggingFace instruction/chat-SFT datasets (tulu et al.) that yields OpenAI
chat messages with **no dependency on ``marin.*`` at runtime**.
"""
