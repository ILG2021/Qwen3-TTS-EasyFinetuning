#!/usr/bin/env python3
"""
Reusable speaker embedding utilities for Qwen3-TTS fine-tuning.

This module extracts speaker embeddings with the Base model's speaker_encoder
and saves them as safetensors. CLI and WebUI flows call the same module API so
embedding behavior stays consistent across entry points.

Preferred CLI usage:
  python src/cli.py embed --speaker_name my_speaker --init_model Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice

Legacy direct module usage:
  # Per-speaker ref from JSONL (default)
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-0.6B-Base

  # Average all audio per speaker
  python src/embed_speaker.py --base_model Qwen/Qwen3-TTS-12Hz-1.7B-Base --mode avg_all

  # Custom ref audio for a single speaker
  python src/embed_speaker.py --speaker my_speaker --ref /path/to/ref.wav

  # Multi-ref averaging
  python src/embed_speaker.py --speaker my_speaker --ref a.wav,b.wav,c.wav

Output: final-dataset/{speaker}/speaker_emb.safetensors (1024-dim tensor)
"""

import gc
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
import librosa
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from utils import get_model_path, resolve_embed_base_model, resolve_path


ProgressCallback = Optional[Callable[[float, str], None]]
LogCallback = Optional[Callable[[str], None]]


@dataclass
class SpeakerEmbeddingResult:
    """Result metadata for one speaker embedding generation attempt."""

    speaker: str
    output_path: str
    status: str
    message: str
    sample_count: int = 0
    norm: Optional[float] = None


def get_runtime_device():
    """Choose a runtime device based on local CUDA availability."""
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def load_speaker_encoder(model_id="Qwen/Qwen3-TTS-12Hz-0.6B-Base", device="cuda:0"):
    """
    Load the Base model speaker_encoder.

    Dependencies: Qwen3TTSModel supplies the speaker_encoder and torch controls
    dtype/attention choices for CUDA vs CPU execution.
    """
    base = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=device,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        attn_implementation="sdpa",
    )
    se = base.model.speaker_encoder.to(device)
    del base
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return se


def extract_embedding(se, audio_path, device):
    """
    Extract a 1024-dim speaker embedding from one audio file.

    Dependencies: librosa normalizes input audio to 24 kHz mono; Qwen3-TTS'
    mel_spectrogram matches the speaker_encoder's expected mel features.
    """
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
    """Parse comma-separated speaker names into a clean list."""
    if not speaker_arg:
        return []
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


def discover_speaker_names(dataset_path):
    """Return speaker directory names from a prepared final-dataset folder."""
    if not os.path.isdir(dataset_path):
        return []
    return [
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ]


def _emit_log(callback: LogCallback, message: str):
    """Route module logs to CLI print, WebUI buffers, or no-op callers."""
    if callback:
        callback(message)


def _emit_progress(callback: ProgressCallback, value: float, desc: str):
    """Report normalized progress without coupling the module to Gradio."""
    if callback:
        callback(max(0.0, min(1.0, value)), desc)


def _resolve_reference_audio_paths(spk, spk_dir, mode, ref):
    """
    Resolve reference audio paths for a speaker.

    Dependencies: speaker JSONL files created by the data pipeline provide
    `audio` and `ref_audio` fields; resolve_existing_audio_path handles absolute,
    project-relative, and speaker-directory-relative paths.
    """
    jsonl = os.path.join(spk_dir, "tts_train.jsonl")
    audio_dir = os.path.join(spk_dir, "audio_24k")

    if ref:
        return [
            resolved
            for resolved in (
                resolve_existing_audio_path(item, fallback_dirs=[audio_dir])
                for item in ref.split(",")
            )
            if resolved
        ], None

    if not os.path.exists(jsonl):
        return [], "no JSONL, skipping"

    entries = load_jsonl_entries(jsonl)
    if not entries:
        return [], "JSONL is empty, skipping"

    if mode == "avg_all":
        paths = [
            resolved
            for resolved in (
                resolve_existing_audio_path(entry.get("audio"), fallback_dirs=[audio_dir, spk_dir])
                for entry in entries
            )
            if resolved
        ]
        return paths, f"averaged {len(paths)}/{len(entries)} samples"

    ref_audio = entries[0].get("ref_audio")
    ref_path = resolve_existing_audio_path(ref_audio, fallback_dirs=[audio_dir, spk_dir])
    if not ref_path:
        return [], "ref_audio not found in JSONL, skipping"
    return [ref_path], None


def _average_embeddings(se, audio_paths: Iterable[str], device: str):
    """
    Extract and average embeddings for one speaker.

    Dependencies: extract_embedding performs the model-specific feature
    extraction; torch.stack/mean keeps multi-reference averaging deterministic.
    """
    embeddings = [extract_embedding(se, path, device) for path in audio_paths]
    if not embeddings:
        return None
    return torch.stack(embeddings).mean(dim=0).squeeze(0)


def save_embedding(embedding, output_path):
    """
    Save a speaker embedding tensor to safetensors format.

    Dependencies: safetensors keeps checkpoint loading fast and avoids pickle.
    """
    from safetensors.torch import save_file

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_file({"emb": embedding}, output_path)


def generate_speaker_embeddings(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_source="HuggingFace",
    mode="ref",
    speakers=None,
    ref=None,
    output=None,
    device=None,
    resolved_model_path=None,
    progress_callback: ProgressCallback = None,
    log_callback: LogCallback = None,
):
    """
    Generate speaker embedding files and return per-speaker result metadata.

    Dependencies: utils resolves model IDs and project paths; the caller may pass
    resolved_model_path to reuse WebUI download progress handling.
    """
    if mode not in {"ref", "avg_all"}:
        raise ValueError(f"Unsupported embedding mode: {mode}")

    dataset_path = resolve_path("final-dataset")
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"{dataset_path} not found. Run the data pipeline first.")

    speaker_names = speakers or discover_speaker_names(dataset_path)
    if isinstance(speaker_names, str):
        speaker_names = parse_speaker_names(speaker_names)
    speaker_names = [str(spk).strip() for spk in speaker_names if str(spk).strip()]
    if not speaker_names:
        raise ValueError("No speakers specified and no speaker directories were found.")

    device = device or get_runtime_device()
    use_hf = model_source == "HuggingFace"
    base_model = resolve_embed_base_model(model_name)
    resolved_base_model = resolved_model_path or get_model_path(base_model, use_hf=use_hf)
    _emit_log(log_callback, f"Loading speaker_encoder from {base_model}")
    _emit_log(log_callback, f"Resolved model path: {resolved_base_model}")
    _emit_progress(progress_callback, 0.05, f"Loading speaker_encoder from {base_model}...")
    se = load_speaker_encoder(resolved_base_model, device=device)

    results: List[SpeakerEmbeddingResult] = []
    try:
        for index, spk in enumerate(speaker_names):
            spk_dir = os.path.join(dataset_path, spk)
            out_path = resolve_path(output) if output else os.path.join(spk_dir, "speaker_emb.safetensors")
            _emit_progress(
                progress_callback,
                0.10 + 0.80 * (index / max(len(speaker_names), 1)),
                f"Embedding {spk}...",
            )

            audio_paths, note = _resolve_reference_audio_paths(spk, spk_dir, mode, ref)
            if ref and not audio_paths:
                message = "ref audio not found"
                _emit_log(log_callback, f"  {spk}: {message}")
                results.append(SpeakerEmbeddingResult(spk, out_path, "skipped", message))
                continue
            if note:
                _emit_log(log_callback, f"  {spk}: {note}")
            if not audio_paths:
                message = note or "no embeddings extracted"
                results.append(SpeakerEmbeddingResult(spk, out_path, "skipped", message))
                continue

            for audio_path in audio_paths:
                _emit_log(log_callback, f"  {spk}: {os.path.basename(audio_path)}")

            avg_emb = _average_embeddings(se, audio_paths, device)
            if avg_emb is None:
                message = "no embeddings extracted"
                _emit_log(log_callback, f"  {spk}: {message}")
                results.append(SpeakerEmbeddingResult(spk, out_path, "skipped", message))
                continue

            save_embedding(avg_emb, out_path)
            norm = float(avg_emb.norm().item())
            message = f"saved {out_path} (norm={norm:.2f})"
            _emit_log(log_callback, f"  -> {message}")
            results.append(
                SpeakerEmbeddingResult(
                    speaker=spk,
                    output_path=out_path,
                    status="saved",
                    message=message,
                    sample_count=len(audio_paths),
                    norm=norm,
                )
            )
    finally:
        del se
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _emit_progress(progress_callback, 1.0, "Done")

    return results


def run_embedding_job(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    model_source="HuggingFace",
    mode="ref",
    speaker=None,
    ref=None,
    output=None,
):
    """CLI-compatible wrapper around generate_speaker_embeddings."""
    results = generate_speaker_embeddings(
        model_name=model_name,
        model_source=model_source,
        mode=mode,
        speakers=parse_speaker_names(speaker),
        ref=ref,
        output=output,
        log_callback=print,
    )
    print("\nDone.")
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", "--init_model", dest="base_model", default="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        help="Training model ID; CustomVoice models will automatically reuse the matching Base speaker encoder")
    parser.add_argument("--model_source", default="HuggingFace", choices=["HuggingFace"],
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
