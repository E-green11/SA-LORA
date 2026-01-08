# SA-LORA: From Structural Agnosticism to Spectrally-Guided Adaptation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

This repository contains the official implementation of **SA-LORA** (Structure-Aware LoRA).

## 📖 Abstract

Current parameter-efficient fine-tuning (PEFT) methods like LoRA apply uniform configurations across all layers, overlooking the vast functional heterogeneity of pre-trained models. 

**SA-LORA** is a novel framework that automatically tailors fine-tuning intensity to each layer's intrinsic complexity. We leverage **Stable Rank** as a spectral metric to measure the effective dimensionality of weight matrices. Furthermore, we introduce a hybrid calibration mechanism that fuses this structural prior with task-specific gradient feedback, all governed by a budget-conservation principle.



## 🚀 Key Features

- **Spectrally-Guided Adaptation**: Uses Stable Rank ($srank$) to determine the optimal adaptation magnitude for each layer.
- **Hybrid Calibration**: Combines task-agnostic structural priors with task-specific gradient signals.
- **Budget Conservation**: Ensures the total trainable parameter budget remains comparable to standard LoRA while redistributing capacity effectively.
- **Plug-and-Play**: Compatible with RoBERTa, GPT-2, Llama, and other Transformer-based architectures.
