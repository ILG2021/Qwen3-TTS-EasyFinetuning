# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import json
import os
import sys
import gc
import torch
import soundfile as sf
from pydub import AudioSegment

from qwen_tts import Qwen3TTSTokenizer
from utils import resolve_path

BATCH_INFER_NUM = 8
TARGET_SAMPLE_RATE = 24000

def log_progress(progress, desc):
    print(json.dumps({"type": "progress", "progress": progress, "desc": desc}), flush=True)

def log_done(msg):
    print(json.dumps({"type": "done", "msg": msg}), flush=True)

def log_error(msg):
    print(json.dumps({"type": "error", "msg": msg}), flush=True)


def _audio_needs_resample(path):
    if os.path.splitext(path)[1].lower() != ".wav":
        return True
    try:
        info = sf.info(path)
        return info.samplerate != TARGET_SAMPLE_RATE or info.channels != 1
    except Exception:
        return True


def _prepared_audio_path(source_path, cache_dir):
    digest = hashlib.sha1(os.path.abspath(source_path).encode("utf-8")).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return os.path.join(cache_dir, f"{stem}_{digest}.wav")


def ensure_24k_audio(path, cache_dir, memo):
    """
    Return a 24 kHz mono wav for tokenizer/speaker metadata.

    Already-compatible wav files are used in place. Other files are materialized
    into a deterministic cache next to the imported dataset JSONL.
    """
    resolved = resolve_path(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"Audio file not found: {resolved}")
    if resolved in memo:
        return memo[resolved]
    if not _audio_needs_resample(resolved):
        memo[resolved] = resolved
        return resolved

    os.makedirs(cache_dir, exist_ok=True)
    prepared = _prepared_audio_path(resolved, cache_dir)
    if not os.path.exists(prepared):
        audio = AudioSegment.from_file(resolved)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(1)
        audio.export(prepared, format="wav")
    memo[resolved] = prepared
    return prepared

def run_prepare(device, tokenizer_model_path, input_jsonl, output_jsonl):
    try:
        yield {"type": "progress", "progress": 0.05, "desc": f"Loading Tokenizer: {tokenizer_model_path}..."}
        tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_model_path,
            device_map=device,
        )

        total_lines = open(input_jsonl, encoding="utf-8").readlines()
        total_lines = [json.loads(line.strip()) for line in total_lines]
        total_count = len(total_lines)

        final_lines = []
        batch_lines = []
        batch_audios = []
        resample_cache = os.path.join(os.path.dirname(os.path.abspath(input_jsonl)), "audio_24k")
        prepared_audio_memo = {}
        
        yield {"type": "progress", "progress": 0.1, "desc": f"Starting tokenization of {total_count} files..."}
        
        for idx, line in enumerate(total_lines):
            # Convert to tokenizer-ready 24 kHz wav paths for robust storage.
            line['audio'] = ensure_24k_audio(line['audio'], resample_cache, prepared_audio_memo)
            if line.get('ref_audio'):
                line['ref_audio'] = ensure_24k_audio(line['ref_audio'], resample_cache, prepared_audio_memo)
                
            batch_lines.append(line)
            batch_audios.append(line['audio'])

            if len(batch_lines) >= BATCH_INFER_NUM:
                with torch.inference_mode():
                    enc_res = tokenizer_12hz.encode(batch_audios)
                for code, item in zip(enc_res.audio_codes, batch_lines):
                    item['audio_codes'] = code.cpu().tolist()
                    final_lines.append(item)
                batch_lines.clear()
                batch_audios.clear()
                
                yield {"type": "progress", "progress": 0.1 + 0.8 * (idx / max(total_count, 1)), "desc": f"Tokenizing: {idx}/{total_count}"}

        if len(batch_audios) > 0:
            with torch.inference_mode():
                enc_res = tokenizer_12hz.encode(batch_audios)
            for code, item in zip(enc_res.audio_codes, batch_lines):
                item['audio_codes'] = code.cpu().tolist()
                final_lines.append(item)
            batch_lines.clear()
            batch_audios.clear()

        yield {"type": "progress", "progress": 0.95, "desc": "Saving JSONL output..."}
        final_lines = [json.dumps(line, ensure_ascii=False) for line in final_lines]

        with open(output_jsonl, 'w', encoding="utf-8") as f:
            for line in final_lines:
                f.writelines(line + '\n')
                
        yield {"type": "done", "msg": f"Successfully tokenized {len(final_lines)} entries."}
    except Exception as e:
        yield {"type": "error", "msg": f"Error during tokenization: {str(e)}"}
    finally:
        if 'tokenizer_12hz' in locals():
            del tokenizer_12hz
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


