# Beyond Structural Agnosticism: Stable-Rank-Guided LoRA for Structure-Aware Fine-Tuning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

This repository contains the official implementation of **SA-LORA** (Structure-Aware LoRA).

## 📖 Abstract

Current parameter-efficient fine-tuning (PEFT) methods like Low-Rank Adaptation (LoRA) are structurally agnostic, applying uniform configurations across all layers, which overlooks their vast functional heterogeneity. We propose Structure-Aware LoRA (SA-LoRA), a new framework that automatically tailors fine-tuning intensity to each layer's intrinsic complexity. Specifically, we leverage the Stable Rank as a spectral metric to align the adaptation magnitude of each layer with its pre-trained spectral structure, enabling an automated, principled allocation of learning capacity that directly addresses the structural agnosticism of existing methods.
To enhance adaptability and robustness, we introduce a hybrid calibration mechanism that fuses the task-agnostic prior with task-specific gradient feedback, underpinned by a budget-conservation principle to ensure stability. Extensive experiments demonstrate that SA-LoRA consistently outperforms strong PEFT baselines, often achieving state-of-the-art performance with enhanced stability.



## 🚀 Key Features

- **Spectrally-Guided Adaptation**: Uses Stable Rank ($srank$) to determine the optimal adaptation magnitude for each layer.
- **Hybrid Calibration**: Combines task-agnostic structural priors with task-specific gradient signals.
- **Budget Conservation**: Ensures the total trainable parameter budget remains comparable to standard LoRA while redistributing capacity effectively.
- **Plug-and-Play**: Compatible with RoBERTa, GPT-2, Llama, and other Transformer-based architectures.
