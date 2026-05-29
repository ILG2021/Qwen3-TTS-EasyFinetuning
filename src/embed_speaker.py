#!/usr/bin/env python3
"""
Pre-compute speaker embeddings from reference audio(s).

This script extracts speaker embeddings using the Base model's speaker_encoder
and saves them as safetensors. The training loop can then load these pre-computed
embeddings directly, avoiding redundant GPU computation in every training step.

Usage:
  # Per-speaker ref from JSONL (default)
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-0.6B-Base

  # Average all audio per speaker
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base --mode avg_all

  # Use HuggingFace instead of the project's default model source
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-0.6B-Base --model_source HuggingFace

  # Custom ref audio for a single speaker
  python src/embed_speaker.py --speaker my_speaker --ref /path/to/ref.wav

  # Multi-ref averaging
  python src/embed_speaker.py --speaker my_speaker --ref a.wav,b.wav,c.wav

Output: final-dataset/{speaker}/speaker_emb.safetensors (1024-dim tensor)
"""

import argparse
import gc
import json
import os
import sys

import torch
import librosa
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from utils import get_model_path, resolve_embed_base_model, resolve_path


def get_runtime_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_speaker_encoder(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda:0"):
    """Load Base model just for its speaker_encoder."""
    base = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="flash_attention_2" if device.startswith("cuda") else "eager",
    )
    se = base.model.speaker_encoder.to(device)
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return se


def extract_embedding(se, audio_path, device):
    """Extract 1024-dim speaker embedding from audio file."""
    audio, sr = librosa.load(audio_path, sr=24000, mono=True)
    mel = mel_spectrogram(
        torch.from_numpy(audio).unsqueeze(0).to(device),
        n_fft=1024, num_mels=128, sampling_rate=24000,
        hop_size=256, win_size=1024, fmin=0, fmax=12000,
    ).transpose(1, 2)
    mel = mel.to(torch.bfloat16 if device.startswith("cuda") else torch.float32)
    with torch.no_grad():
        emb = se(mel).detach()  # [1, 1024]
    return emb[0].cpu()


def parse_speaker_names(speaker_arg):
    if not speaker_arg:
        return None
    return [item.strip() for item in str(speaker_arg).split(",") if item.strip()]


def resolve_existing_audio_path(path_value, fallback_dirs=None):
    """
    Resolve audio paths from JSONL or CLI input into an existing filesystem path.

    Dependencies: resolve_path anchors project-relative JSONL entries at the repo
    root, while fallback_dirs keeps backwards compatibility with short filenames
    that are meant to live under a speaker's audio_24k directory.
    """
    if not path_value:
        return ""

    raw_path = os.path.expanduser(str(path_value).strip())
    if not raw_path:
        return ""

    candidates = []
    if os.path.isabs(raw_path):
        candidates.append(raw_path)
    else:
        for base_dir in fallback_dirs or []:
            candidates.append(os.path.join(base_dir, raw_path))
        candidates.append(resolve_path(raw_path))
        candidates.append(os.path.abspath(raw_path))

    seen = set()
    for candidate in candidates:
        normalized = os.path.abspath(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if os.path.exists(normalized):
            return normalized
    return ""


def load_jsonl_entries(jsonl_path):
    """Load a speaker JSONL file, ignoring blank lines for easier manual cleanup."""
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_embedding_job(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_source="ModelScope",
    mode="ref",
    speaker=None,
    ref=None,
    output=None,
):
    device = get_runtime_device()
    use_hf = model_source == "HuggingFace"
    base_model = resolve_embed_base_model(model_name)
    resolved_base_model = get_model_path(base_model, use_hf=use_hf)
    print(f"Loading speaker_encoder from {base_model}")
    print(f"Resolved model path: {resolved_base_model}")
    se = load_speaker_encoder(resolved_base_model, device=device)

    dataset_path = resolve_path("final-dataset")
    if not os.path.isdir(dataset_path):
        print(f"{dataset_path} not found. Run the data pipeline first.")
        sys.exit(1)

    explicit_speakers = parse_speaker_names(speaker)
    speakers = explicit_speakers if explicit_speakers else [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]

    for spk in speakers:
        spk_dir = os.path.join(dataset_path, spk)
        jsonl = os.path.join(spk_dir, "tts_train.jsonl")
        audio_dir = os.path.join(spk_dir, "audio_24k")
        out_path = resolve_path(output) if output else os.path.join(spk_dir, "speaker_emb.safetensors")

        embeddings = []

        if ref:
            for ref_file in ref.split(","):
                ref_path = resolve_existing_audio_path(ref_file, fallback_dirs=[audio_dir])
                if ref_path:
                    print(f"  {spk}: {os.path.basename(ref_path)}")
                    embeddings.append(extract_embedding(se, ref_path, device))
                else:
                    print(f"  {spk}: ref audio not found: {ref_file.strip()}")
        elif mode == "avg_all":
            if os.path.exists(jsonl):
                entries = load_jsonl_entries(jsonl)
                for entry in entries:
                    audio_path = resolve_existing_audio_path(entry.get("audio"), fallback_dirs=[audio_dir, spk_dir])
                    if audio_path:
                        embeddings.append(extract_embedding(se, audio_path, device))
                print(f"  {spk}: averaged {len(embeddings)}/{len(entries)} samples")
            else:
                print(f"  {spk}: no JSONL, skipping")
                continue
        else:
            if os.path.exists(jsonl):
                entries = load_jsonl_entries(jsonl)
                if entries:
                    ref_audio = entries[0].get("ref_audio")
                    ref_path = resolve_existing_audio_path(ref_audio, fallback_dirs=[audio_dir, spk_dir])
                    if ref_path:
                        print(f"  {spk}: {os.path.basename(ref_path)}")
                        embeddings.append(extract_embedding(se, ref_path, device))
                    else:
                        print(f"  {spk}: ref_audio not found in JSONL, skipping")
                        continue
                else:
                    print(f"  {spk}: JSONL is empty, skipping")
                    continue
            else:
                print(f"  {spk}: no JSONL, skipping")
                continue

        if embeddings:
            avg_emb = torch.stack(embeddings).mean(dim=0).squeeze(0)
            from safetensors.torch import save_file
            save_file({"emb": avg_emb}, out_path)
            print(f"  -> saved {out_path} (norm={avg_emb.norm():.2f})")
        else:
            print(f"  {spk}: no embeddings extracted")

    del se
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "--init_model", dest="base_model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="Training model ID; CustomVoice models will automatically reuse the matching Base speaker encoder")
    parser.add_argument("--model_source", default="ModelScope", choices=["HuggingFace", "ModelScope"],
                        help="Model download source, consistent with the rest of the project")
    parser.add_argument("--mode", default="ref", choices=["ref", "avg_all"],
                        help="ref=use ref_audio from JSONL, avg_all=average all samples per speaker")
    parser.add_argument("--speaker", default=None, help="Process single speaker")
    parser.add_argument("--ref", default=None, help="Custom ref audio(s), comma-separated")
    parser.add_argument("--output", default=None, help="Custom output path")
    args = parser.parse_args()
    run_embedding_job(
        model_name=args.base_model,
        model_source=args.model_source,
        mode=args.mode,
        speaker=args.speaker,
        ref=args.ref,
        output=args.output,
    )


if __name__ == "__main__":
    main()
