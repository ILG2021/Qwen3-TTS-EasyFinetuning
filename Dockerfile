FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV USE_HF=1

RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    sox \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*
EXPOSE 7860 6006

ENV PYTHONPATH="/workspace/src"

CMD ["python", "src/webui.py"]
