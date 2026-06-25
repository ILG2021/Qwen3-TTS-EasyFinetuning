# Qwen3-TTS Easy Finetuning

<p align="center">
  <img src="https://img.shields.io/github/stars/mozi1924/Qwen3-TTS-EasyFinetuning?style=for-the-badge&color=ffd700" alt="GitHub Stars">
  <img src="https://img.shields.io/github/license/mozi1924/Qwen3-TTS-EasyFinetuning?style=for-the-badge&color=blue" alt="License">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
</p>

<p align="center">
  <b>English</b> | <a href="./README_zh.md">简体中文</a>
</p>

An easy-to-use workspace for fine-tuning the **Qwen3-TTS** model. This repository streamlines the entire process—from raw audio ingestion to creating high-stability, expressive custom voice models.

---

### 📚 Tutorial
For a comprehensive step-by-step guide with illustrations, please refer to my article:

👉 [English Article](https://mozi1924.com/article/qwen3-tts-finetuning-en/) | [中文文章](https://mozi1924.com/article/qwen3-tts-finetuning-zh/)

### 🎙️ Why Fine-tuning instead of Zero-shot?

While zero-shot voice cloning is convenient for quick tests, **Supervised Fine-Tuning (SFT)** offers significant advantages for production-grade results:

*   **Timbre Stability**: Fine-tuned models capture the intricate nuances of the target speaker more accurately, ensuring consistent output across diverse text contexts.
*   **Expressive Control**: SFT supports natural language tone and rhythm guidance (e.g., "Speak sadly", "Faster pace"), enabling more emotive and human-like synthesis.
*   **Accent-free Cross-lingual Synthesis**: Effectively prevents "original language accents" during cross-lingual inference (e.g., a Chinese-sounding voice used for English speech will sound like a native English speaker).

---

## ✨ Key Features

*   **Integrated Pipeline**: Automated audio splitting, ASR transcription, dataset cleaning, and tokenization in a single workflow.
*   **Modern WebUI**: A premium Gradio interface for seamless data preparation, training monitoring, and interactive inference.
*   **Robust CLI**: Unified command-line interface for professional automation and remote server management.
*   **Optimized Presets**: Hardcoded, expert-tuned training parameters for different model variants (0.6B / 1.7B).
*   **Docker Ready**: Out-of-the-box environment support via pre-configured Docker images.

---

## 💻 Environment & Requirements

### Host Development Environment
Information about the environment used for developing and testing this project:
- **OS**: Ubuntu 24.04.4 LTS (Kernel 6.17.0-14-generic)
- **CPU**: 2 x Intel(R) Xeon(R) Platinum 8259CL (KVM, 32 cores)
- **Memory**: 32 GB
- **GPU**: 2 x NVIDIA GeForce RTX 3080 (10 GB VRAM each)
- **Driver & CUDA**: NVIDIA Driver 590.48.01 / CUDA 13.1
- **Python**: 3.11.14

### Recommended Training Environment
To ensure stable training and avoid Out-of-Memory (OOM) errors, we recommend:
- **GPU**: NVIDIA GPU with >= 16 GB VRAM (24 GB recommended for 1.7B model)
- **Memory**: >= 32 GB RAM
- **Storage**: SSD with at least 50 GB free space
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **Software**: CUDA 12.4+ (v12.8+ recommended), Python 3.10+

> **⚠️ Special Note for Windows Users (GPU Training):**
> Due to architectural limitations, **do not use Rancher Desktop** if you require GPU support on Windows, as it lacks native Nvidia GPU capabilities. Instead, choose one of the following:
> 1. **Run on a native Linux GPU host** (Highly Recommended: best performance and stability).
> 2. **Use pure WSL2 (Ubuntu)** (Recommended: install native Docker Engine or Python in WSL2 for seamless GPU access).
> 3. **Use Docker Desktop** (Supports GPU, but comes with significant performance overhead and is not guaranteed to be perfectly stable/usable on all Windows configurations).

---

## 🚀 Getting Started

### 1. Installation

**Using Docker (Recommended)**
```bash
# Pull the pre-built image from GHCR (Default)
docker compose up -d

# Force a local build
docker compose up -d --build
```

**Using Python Virtual Environment**
```bash
# Running this directly in a Windows environment is not actively maintained and is not planned for further maintenance. Please use Docker, which is the fastest, most stable, and most efficient recommended method.

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Flash Attention matching your CUDA/Torch version
pip install flash-attn==2.8.3 --no-build-isolation
```

### 2. Using the WebUI (Easiest)
Launch the Gradio WebUI to manage the entire lifecycle through your browser:
```bash
python src/webui.py
```
*   **Data Prep**: Upload raw audio -> Split -> ASR -> Tokenize.
*   **Training**: Select dataset -> Configure settings -> Launch Tensorboard -> Start Training.
*   **Inference**: Load your trained checkpoint and test your custom voice!

### 3. Using the CLI (Professional)
The `src/cli.py` serves as a unified entry point for all operations:

**Step A: Prepare Data**
Place your raw `.wav` files in a directory (e.g., `raw-dataset/my_speaker/`).
```bash
python src/cli.py prepare --input_dir raw-dataset/my_speaker --speaker_name my_speaker
```

**Step B: Start Training**
```bash
python src/cli.py train --experiment_name exp1 --speaker_name my_speaker --epochs 3
```

For `CustomVoice` fine-tuning, generate the speaker embedding once after ASR and before training:
```bash
python src/cli.py embed --speaker_name my_speaker --init_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
python src/cli.py train --experiment_name exp1 --speaker_name my_speaker --init_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --epochs 3
```

**Step C: Run Inference**
```bash
python src/cli.py infer --checkpoint output/exp1/checkpoint-epoch-2 --speaker my_speaker --text "Hello world! This is my custom voice."
```

---

## 📂 Project Structure

*   `src/webui.py`: Main Gradio interface.
*   `src/cli.py`: Unified command-line interface.
*   `src/sft_12hz.py`: Core fine-tuning logic.
*   `src/step1_audio_split.py`: Audio preprocessing & segmentation.
*   `src/step2_asr_clean.py`: Automatic transcription & labeling.
*   `src/prepare_data.py`: Pre-tokenizing audio into discrete codes (Step 3).

---

## ⚠️ Disclaimer

By using this tool to fine-tune models, you agree that you will not use it for any illegal purposes or to infringe upon the rights of others. The author is not responsible for any direct or indirect consequences arising from the use of this tool, including but not limited to hardware damage or legal disputes.

---

## 🤝 Acknowledgments

*   Built upon [Qwen3-TTS](https://github.com/qwenLM/Qwen3-tts) and [Qwen3-ASR](https://github.com/qwenLM/Qwen3-asr).
*   Training presets inspired by community research (e.g., [rekuenkdr](https://github.com/rekuenkdr)).

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)](https://www.star-history.com/#mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)
