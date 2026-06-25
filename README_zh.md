# Qwen3-TTS Easy Finetuning

<p align="center">
  <a href="./README.md">English</a> | <b>简体中文</b>
</p>

这是一个专为 **Qwen3-TTS** 模型设计的易用微调工作站。它简化了从原始音频数据到构建高质量、高稳定性、极具表现力的自定义语音模型的全流程。

---

### 📚 教程文章

您可以阅读我的详细图文教程以获取完整指南：

👉 [中文教程](https://mozi1924.com/article/qwen3-tts-finetuning-zh/) | [English Tutorial](https://mozi1924.com/article/qwen3-tts-finetuning-en/)

### 🎙️ 为什么选择微调而非零样本 (Zero-shot)？

虽然零样本语音克隆非常便捷，但针对生产级的应用，微调 (SFT) 具有显著优势：

- **更高的音色稳定性**: 微调模型能更精确地捕捉目标人物的音质特征，确保不同文本下的输出高度一致。
- **卓越的口吻控制**: 支持通过自然语言进行情绪和节奏转换引导（如“悲伤地说话”、“语速加快”），让合成语音更具表现力。
- **无母语口音干扰**: 彻底解决跨语言合成时的“原本语种口音”问题（例如：用纯正中文音色合成英文时，听起来会像母语级英语使用者，而非带有中式口音）。

---

## ✨ 核心特性

- **一站式流水线**: 集成音频切分、ASR 转录、多轮清洗及 Tokenization，一键完成数据准备。
- **现代化 WebUI**: 基于 Gradio 设计的高级界面，涵盖数据准备、训练监控及推理测试。
- **强大 CLI 工具**: 提供完整的命令行接口，便于自动化脚本集成。
- **针对性预设**: 针对 0.6B 和 1.7B 不同规模的模型，内置了经过优化的训练参数。
- **完善的 Docker 支持**: 预配置环境镜像，实现即插即用。

---

## 💻 环境与配置要求

### 开发环境 (当前主机)
用于本项目开发和测试的硬件环境：
- **操作系统**: Ubuntu 24.04.4 LTS (内核 6.17.0-14-generic)
- **处理器 (CPU)**: 2 x Intel(R) Xeon(R) Platinum 8259CL (KVM 虚拟机, 32 核)
- **内存 (RAM)**: 32 GB
- **显卡 (GPU)**: 2 x NVIDIA GeForce RTX 3080 (10 GB 显存)
- **驱动与 CUDA**: NVIDIA Driver 590.48.01 / CUDA 13.1
- **Python 版本**: 3.11.14

### 推荐训练环境
为了确保 **Qwen3-TTS** 模型变体（0.6B / 1.7B）的稳定训练，并避免内存溢出 (OOM)，建议配置：
- **GPU**: NVIDIA 显卡，显存 >= 16 GB (1.7B 模型建议 24 GB)
- **内存**: >= 32 GB RAM
- **存储**: SSD 固态硬盘，剩余空间至少 50 GB
- **系统**: Linux (推荐 Ubuntu 20.04+)
- **软件**: CUDA 12.4+ (推荐 v12.8+), Python 3.10+

> **⚠️ Windows 用户特别注意（关于 GPU 训练）:**
> 由于底层架构限制，在 Windows 平台下强依赖 GPU 时，请**避免使用 Rancher Desktop**（无法原生支持 Nvidia GPU）。请选择以下三种方案之一：
> 1. **在真实的 Linux GPU 主机上运行**（最推荐，性能最好，最稳定）。
> 2. **使用原生的 WSL2 (Ubuntu) 环境**（推荐，通过 WSL2 直接运行原生 Docker Engine 或配置原生 Python 环境，可无缝调用 GPU）。
> 3. **使用 Docker Desktop**（支持 GPU，但需注意 Docker Desktop 本身的性能开销极大，且不保证在所有 Windows 环境下绝对可用/稳定）。

---

## 🚀 快速上手

### 1. 安装环境

**使用 Docker (推荐)**
```bash
# 如果需要强制本地构建
docker compose up -d --build
```

**使用 Python 虚拟环境**
```
- **数据准备面板**: 上传音频 -> 切分 -> ASR 识别 -> 转换为编码。
- **模型训练面板**: 选择数据集 -> 配置参数 -> 开启 Tensorboard -> 开始训练。
- **语音推理面板**: 加载训练好的 Checkpoint，立即生成您的专属语音！

### 3. 使用 CLI (进阶/专业)
`src/cli.py` 提供了一个统一的操作入口：

**步骤 A: 准备数据**
将原始 `.wav` 文件放入文件夹（如 `raw-dataset/my_speaker/`）。
```bash
python src/cli.py prepare --input_dir raw-dataset/my_speaker --speaker_name my_speaker
```

**步骤 B: 开始训练**
```bash
python src/cli.py train --experiment_name exp1 --speaker_name my_speaker --epochs 3
```

如果是 `CustomVoice` 微调，请在 ASR 之后、训练之前先生成一次 speaker embedding：
```bash
python src/cli.py embed --speaker_name my_speaker --init_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
python src/cli.py train --experiment_name exp1 --speaker_name my_speaker --init_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --epochs 3
```

**步骤 C: 执行推理**
```bash
python src/cli.py infer --checkpoint output/exp1/checkpoint-epoch-2 --speaker my_speaker --text "你好，世界！这是我微调的自定义音色。"
```

---

## 📂 项目结构

- `src/webui.py`: 主 Gradio 交互界面。
- `src/cli.py`: 统一命令行入口。
- `src/sft_12hz.py`: 核心微调逻辑（监督微调）。
- `src/step1_audio_split.py`: 音频预处理与分段。
- `src/step2_asr_clean.py`: 自动化语音转文字及数据清洗。
- `src/prepare_data.py`: 将音频预处理为离散编码 (Step 3)。

---

## ⚠️ 免责声明

使用本工具微调模型即表示您同意，不会将其用于任何违法、不道德或侵犯他人权益的目的。作者对使用本工具所产生的任何直接或间接后果（包括但不限于硬件损坏或法律纠纷）概不负责。

---

## 🤝 致谢

- 基于 [Qwen3-TTS](https://github.com/qwenLM/Qwen3-tts) 和 [Qwen3-ASR](https://github.com/qwenLM/Qwen3-asr)。
- 训练预设逻辑参考了社区贡献者（如 [rekuenkdr](https://github.com/rekuenkdr)）。

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)](https://www.star-history.com/#mozi1924/Qwen3-TTS-EasyFinetuning&type=date&legend=top-left)
