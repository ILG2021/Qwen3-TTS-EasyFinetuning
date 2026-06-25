import os
import re

from model_repository import (
    ModelDownloadTracker,
    ensure_model_path,
    format_bytes,
    get_model_local_dir,
    get_project_root,
    is_model_dir_ready,
    normalize_model_source,
    resolve_existing_model_path,
    resolve_path,
)

DEFAULT_TTS_TRAIN_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SUPPORTED_TTS_TRAIN_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
]


def configure_visible_cuda_device(gpu):
    '''Limit CUDA visibility for single-device CLI jobs and return the remapped runtime device.'''
    gpu = str(gpu or "cpu").strip()
    if gpu == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    if gpu == "cuda":
        return "cuda:0"
    if gpu.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu.split(":", 1)[1]
        return "cuda:0"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    return "cuda:0"


def normalize_resume_checkpoint_arg(value):
    '''Normalize CLI resume values into latest, None, or an absolute checkpoint path.'''
    if value is None:
        return "latest"
    value = str(value).strip()
    if not value:
        return "latest"
    if value.lower() in {"none", "false", "no", "off", "0"}:
        return None
    if value.lower() == "latest":
        return "latest"
    return resolve_path(value)


def is_model_downloaded(model_id):
    local_dir = get_model_local_dir(model_id)
    return is_model_dir_ready(local_dir)



def get_model_path(model_id, use_hf=True, progress_tracker=None):
    """
    Resolve a model/checkpoint into a local from_pretrained path.

    Dependencies: model_repository handles HuggingFace downloads and
    canonical local materialization. This compatibility wrapper preserves the
    legacy `use_hf` argument used by CLI and WebUI modules.
    """
    return ensure_model_path(model_id, use_hf=use_hf, progress_tracker=progress_tracker)


def resolve_embed_base_model(model_id):
    model_id = model_id.strip() if isinstance(model_id, str) else ""
    if not model_id:
        return DEFAULT_TTS_TRAIN_MODEL
    if model_id.endswith("-Base"):
        return model_id
    if "0.6B" in model_id:
        return "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
    if "1.7B" in model_id:
        return "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    return DEFAULT_TTS_TRAIN_MODEL


def is_custom_voice_model(model_id):
    return isinstance(model_id, str) and model_id.endswith("-CustomVoice")


def missing_speaker_embeddings(speaker_names):
    missing = []
    if not speaker_names:
        return missing
    for speaker_name in speaker_names:
        speaker_name = str(speaker_name).strip()
        if not speaker_name:
            continue
        emb_path = resolve_path(os.path.join("final-dataset", speaker_name, "speaker_emb.safetensors"))
        if not os.path.exists(emb_path):
            missing.append(speaker_name)
    return missing


def speaker_key(value):
    return re.sub(r'[^a-z0-9]+', '', str(value).lower())

def resolve_speaker_choice(speaker, supported_speakers):
    if not speaker or not supported_speakers:
        return speaker
    if speaker in supported_speakers:
        return speaker
    lower_map = {str(s).lower(): s for s in supported_speakers}
    lowered = str(speaker).lower()
    if lowered in lower_map:
        return lower_map[lowered]
    normalized = speaker_key(speaker)
    normalized_map = {}
    for s in supported_speakers:
        normalized_map.setdefault(speaker_key(s), s)
    return normalized_map.get(normalized, speaker)
