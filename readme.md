# Auto-Finetune Pipeline

Professional Neural Network Training Interface for Fine-tuning Stable Diffusion Models

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/gradio-4.0%2B-orange.svg)](https://gradio.app)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow.svg)](https://huggingface.co)

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [System Requirements](#system-requirements)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Usage Guide](#usage-guide)
7. [Project Structure](#project-structure)
8. [Configuration](#configuration)
9. [Training Methods](#training-methods)
10. [Google Colab Integration](#google-colab-integration)
11. [Troubleshooting](#troubleshooting)
12. [FAQ](#faq)
13. [Contributing](#contributing)
14. [License](#license)
15. [Acknowledgments](#acknowledgments)

---

## Overview

Auto-Finetune Pipeline is a professional-grade tool for fine-tuning Stable Diffusion models on custom datasets. It provides an intuitive web interface, automatic hardware detection, and support for both local and cloud training.

Target Users:
- Machine learning researchers and engineers
- Digital artists and designers
- Creative agencies and studios
- Educational institutions
- Businesses requiring custom image generation

---

## Features

### Training Platforms
- Local Training: Use your own GPU or CPU infrastructure
- Cloud Training: Export to Google Colab for free GPU access

### Language Support
- English interface
- Russian interface
- Extensible architecture for additional languages

### User Interface Components
- Web-based graphical interface requiring no coding
- Real-time training monitoring with loss metrics
- Dataset management with image preview and caption editing
- Built-in inference for testing trained models

### Technical Configuration
- Multiple training methods: LoRA, DreamBooth, Textual Inversion
- Memory optimization features: FP16 precision, xFormers, attention slicing
- Full hyperparameter control
- Automatic hardware detection and optimization

### Export Capabilities
- Dataset export as ZIP archive
- Configuration export as JSON
- Colab notebook generation for cloud training
- Model checkpoint saving and export

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| Python | 3.10 or higher |
| RAM | 8 GB |
| Storage | 10 GB free space |
| OS | Windows 10/11, Linux, macOS |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX 3060/4060 or better |
| VRAM | 8 GB or more |
| RAM | 16 GB |
| Storage | 20 GB free space |
| OS | Windows 11 or Ubuntu 20.04+ |

### Supported GPUs
- NVIDIA GPUs with CUDA support (RTX 20xx, 30xx, 40xx series)
- Apple Silicon (M1/M2/M3) with MPS backend
- CPU-only mode for testing (slow)

---

## Installation

### Method 1: Automatic Installer (Windows)

1. Download or clone the repository
2. Run `install.bat` as administrator
3. Wait for installation to complete
4. Run `start.bat` to launch the application

### Method 2: Automatic Installer (Linux/macOS)
chmod +x install.sh
./install.sh
./start.sh

text

### Method 3: Manual Installation
Clone repository
git clone https://github.com/yourusername/auto-finetune-pipeline.git
cd auto-finetune-pipeline

Create virtual environment
python -m venv venv

Activate virtual environment
Windows:
venv\Scripts\activate

Linux/macOS:
source venv/bin/activate

Install dependencies
pip install -r requirements.txt

For NVIDIA GPU (optional but recommended)
pip install -r requirements-cuda.txt

Create required directories
mkdir datasets outputs logs checkpoints exports

Launch application
python webui_platform.py

---

## Quick Start

### Step 1: Launch the Application
python webui.py

The interface will open at `http://127.0.0.1:7860`

### Step 2: Prepare Dataset

1. Navigate to "Dataset Management" tab
2. Upload your images (JPG, PNG, WEBP formats)
3. Add captions/descriptions for each image
4. Click "Save Dataset"

### Step 3: Configure Training

1. Go to "Training Configuration" tab
2. Select training method (LoRA recommended for beginners)
3. Set number of epochs (50-100 for good results)
4. Adjust batch size based on your GPU memory

### Step 4: Start Training

1. Go to "Training Control" tab
2. Click "START TRAINING"
3. Monitor loss values in real-time
4. Checkpoints are saved automatically

### Step 5: Generate Images

1. Go to "Inference" tab
2. Select your trained model
3. Enter a text prompt
4. Click "Generate Image"

---

## Usage Guide

### Dataset Preparation

The dataset folder should have the following structure:
datasets/
└── your_dataset_name/
├── image1.jpg
├── image2.png
├── image3.jpg
└── captions.csv

CSV file format:
```csv
image,caption
image1,description of first image
image2,description of second image
image3,description of third image
Best Practices:

Use 20-100 images for LoRA training

Use 100-500 images for DreamBooth

Images should be 512x512 pixels (square)

Captions should be descriptive (10-30 words)

Maintain consistent style across dataset

Training Parameters
Parameter	LoRA	DreamBooth	Textual Inversion
Epochs	50-100	30-50	100-200
Learning Rate	1e-4	5e-6	5e-4
Batch Size	1-2	1	4-8
Memory Usage	Low	High	Very Low
Memory Optimization
For GPUs with limited VRAM (8GB or less):

Enable FP16 (Mixed Precision)

Enable Attention Slicing

Enable VAE Slicing

Enable xFormers if available

Set batch_size=1

Increase gradient_accumulation_steps

Configuration
config.yaml Structure
# Dataset Configuration
dataset_path: "./datasets/my_dataset"
image_size: 512
validation_split: 0.1

# Training Configuration
training:
  method: "lora"           # lora, dreambooth, textual_inversion
  num_epochs: 50
  batch_size: 1
  gradient_accumulation_steps: 2
  learning_rate: 0.0001
  lr_scheduler: "cosine"
  warmup_steps: 100
  weight_decay: 0.01
  max_grad_norm: 1.0

# Model Configuration
model:
  base_model: "runwayml/stable-diffusion-v1-5"
  use_fp16: true
  enable_xformers: true
  enable_attention_slicing: true
  enable_vae_slicing: true

# LoRA Configuration (when method = lora)
lora:
  rank: 4
  alpha: 32
  target_modules: ["to_q", "to_k", "to_v", "to_out.0"]

# Export Configuration
export:
  save_checkpoints: true
  save_checkpoints_every: 5
  save_full_model: true
Training Methods
LoRA (Low-Rank Adaptation)
Best for: Small datasets (20-200 images), style transfer, character training

Advantages:

Low memory usage (4-8 GB VRAM)

Fast training (30-60 minutes)

Small model files (10-50 MB)

Easy to combine multiple LoRAs

Configuration:

Rank: 4-16 (higher = more capacity)

Learning rate: 1e-4 to 5e-4

Epochs: 50-100

DreamBooth
Best for: Large datasets (100-1000+ images), object training, concept learning

Advantages:

High quality results

Good for specific objects/people

Handles diverse poses and angles

Disadvantages:

High memory usage (16+ GB VRAM)

Slow training (2-4 hours)

Large model files (2+ GB)

Configuration:

Learning rate: 5e-6 to 2e-6

Epochs: 30-50

Prior preservation: recommended

Textual Inversion
Best for: Small datasets (3-10 images), embedding specific concepts

Advantages:

Very low memory usage (4 GB VRAM)

Tiny file size (50-100 KB)

Fast training (10-30 minutes)

Disadvantages:

Limited to single concept

Lower quality than LoRA/DreamBooth

Requires good captions

Configuration:

Learning rate: 5e-4 to 1e-3

Epochs: 100-200

No additional parameters

Google Colab Integration
Exporting to Colab
Select "Google Colab" platform in the interface

Go to "Google Colab Export" tab

Click "Export Dataset as ZIP"

Click "Export Config as JSON"

Click "Generate Colab Notebook"

Download all three files

Running in Colab
Open Google Colab (colab.research.google.com)

Upload the generated .ipynb file

Enable GPU: Runtime -> Change runtime type -> GPU

Upload dataset ZIP and config JSON when prompted

Run cells in order from top to bottom

Download trained model from the output

Colab Limitations
Session timeout after 12 hours

Limited to free tier GPU (Tesla T4, sometimes V100)

Storage resets after session ends

Download model before disconnection

Troubleshooting
CUDA Out of Memory
Symptoms:

"CUDA out of memory" error

Training crashes during first epoch

Solutions:

Reduce batch_size to 1

Enable FP16 in config

Enable attention_slicing

Enable vae_slicing

Increase gradient_accumulation_steps

Reduce image_size to 384

CUDA Not Available
Symptoms:

"CUDA available: False" in logs

Training uses CPU (very slow)

Solutions (Windows):

nvidia-smi
If command fails, install NVIDIA drivers from nvidia.com

Then reinstall PyTorch:

pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Model Won't Load
Symptoms:

Error downloading from Hugging Face

Connection timeout

Solutions:

Check internet connection

Try alternative model in config:

CompVis/stable-diffusion-v1-4

stabilityai/stable-diffusion-2-1

Download model manually:

from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipe.save_pretrained("./models/sd-v1-5")
Then set base_model to "./models/sd-v1-5"

Training Loss Not Decreasing
Symptoms:

Loss stays high (>0.5) or increases

Generated images are noisy/random

Solutions:

Increase learning rate (try 2x or 3x)

Check dataset quality (blurry images? bad captions?)

Increase number of epochs

Reduce batch_size

Try different training method

Slow Training on Good GPU
Symptoms:

Training takes hours on RTX 4060

GPU utilization is low

Solutions:

Enable xFormers

Increase batch_size (if VRAM allows)

Reduce image_size to 384

Check if another process is using GPU

Update GPU drivers