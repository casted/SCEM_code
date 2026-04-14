# DiffusionLLIE / SCEM for Low-Light Image Enhancement

PyTorch implementation for low-light image enhancement based on the paper:

**Diffusion-Based Low-Light Image Enhancement with Color and Luminance Priors**  
Xuanshuo Fu, Lei Kang, Javier Vazquez-Corral  
ICRA 2026

Project page: https://casted.github.io/scem/
Arxiv: https://arxiv.org/pdf/2603.00337

---

## Overview

This repository implements a conditional diffusion framework for low-light image enhancement with a **Structured Control Embedding Module (SCEM)**.

Given a low-light input image, SCEM extracts structured priors to guide the denoising process of a U-Net-based diffusion model. According to the paper webpage, the method uses the following control signals:

- **Illumination**
- **Illumination-invariant features**
- **Shadow priors**
- **Color-invariant cues**

These physically meaningful priors are concatenated and injected into the diffusion model to improve enhancement quality and cross-dataset generalization.

The method is trained on **LOLv1** and evaluated on:

- LOLv1
- LOLv2-real
- LSRW
- DICM
- MEF
- LIME

---

## Repository Status

This README is written based on the paper webpage and the current training scripts in this repository.

At the moment, the codebase contains two related training/evaluation scripts:

- `train_try.py`


So if you reproduce results, please make sure you are using the intended script and matching checkpoint/configuration.

---

## Features

- Conditional diffusion model for low-light image enhancement
- SCEM-inspired structured priors
- Training on paired low/normal-light images
- Evaluation with PSNR / SSIM
- Support for checkpoint saving and DDIM sampling
- Multi-GPU wrapper support (`DataParallel` / optional `DDP`)
