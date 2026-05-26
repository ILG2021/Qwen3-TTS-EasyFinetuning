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
import json
import sys
import gc
import os
import time
import soundfile as sf

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

from qwen_tts import Qwen3TTSTokenizer
from utils import resolve_path

DEFAULT_BATCH_INFER_NUM = 4
TARGET_SR = 24000

def log_progress(progress, desc):
    print(json.dumps({"type": "progress", "progress": progress, "desc": desc}), flush=True)

def log_done(msg):
    print(json.dumps({"type": "done", "msg": msg}), flush=True)

def log_error(msg):
    print(json.dumps({"type": "error", "msg": msg}), flush=True)


def get_audio_info(path):
    path = resolve_path(path)
    info = sf.info(path)
    return path, info.samplerate, info.channels

def _cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_cuda_oom(exc):
    cuda_oom_error = getattr(torch.cuda, "OutOfMemoryError", None)
    if cuda_oom_error is not None and isinstance(exc, cuda_oom_error):
        return True
    message = str(exc).lower()
    return "cuda out of memory" in message or "out of memory" in message


def _count_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _entry_key(item):
    return resolve_path(item["audio"])


def _has_audio_codes(item):
    return bool(item.get("audio_codes"))


def _copy_completed_entries(path, f_out, completed_keys):
    copied = 0
    if not os.path.exists(path):
        return copied

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("audio") and _has_audio_codes(item):
                key = _entry_key(item)
                if key in completed_keys:
                    continue
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                completed_keys.add(key)
                copied += 1
    return copied


def _copy_completed_entries_if_available(path, f_out, completed_keys):
    try:
        return _copy_completed_entries(path, f_out, completed_keys), None
    except OSError as e:
        return 0, e


def _safe_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def run_prepare(device, tokenizer_model_path, input_jsonl, output_jsonl, batch_size=DEFAULT_BATCH_INFER_NUM):
    batch_size = max(1, int(batch_size or DEFAULT_BATCH_INFER_NUM))
    current_batch_size = batch_size
    tmp_output_jsonl = output_jsonl + ".tmp"
    run_tmp_jsonl = f"{output_jsonl}.tmp.{os.getpid()}.{int(time.time())}"

    try:
        yield {"type": "progress", "progress": 0.05, "desc": f"Loading Tokenizer: {tokenizer_model_path}..."}
        tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
            tokenizer_model_path,
            device_map=device,
        )

        batch_lines = []
        batch_audios = []
        written_count = 0
        skipped_count = 0
        
        non_24k_count = 0
        non_mono_count = 0

        total_count = _count_jsonl(input_jsonl)
        completed_keys = set()

        yield {"type": "progress", "progress": 0.1, "desc": f"Starting tokenization of {total_count} files with batch size {current_batch_size}..."}

        with open(run_tmp_jsonl, 'w', encoding="utf-8") as f_out:
            copied, copy_error = _copy_completed_entries_if_available(output_jsonl, f_out, completed_keys)
            skipped_count += copied
            if copy_error:
                yield {
                    "type": "progress",
                    "progress": 0.1,
                    "desc": f"Could not read existing output for resume: {copy_error}",
                }

            copied, copy_error = _copy_completed_entries_if_available(tmp_output_jsonl, f_out, completed_keys)
            skipped_count += copied
            if copy_error:
                yield {
                    "type": "progress",
                    "progress": 0.1,
                    "desc": f"Previous temp file is locked; skipping it for this run: {copy_error}",
                }
            f_out.flush()

            if skipped_count:
                yield {
                    "type": "progress",
                    "progress": 0.1,
                    "desc": f"Found {skipped_count} already-tokenized entries; resuming remaining files...",
                }

            with open(input_jsonl, "r", encoding="utf-8") as f_in:
                input_iter = enumerate(f_in)
                for idx, raw_line in input_iter:
                    if not raw_line.strip():
                        continue
                    line = json.loads(raw_line)

                    audio_key = _entry_key(line)
                    if audio_key in completed_keys:
                        continue

                    # Convert to absolute paths for tokenization and robust storage. The tokenizer
                    # resamples audio in memory, so we do not duplicate large datasets on disk.
                    line['audio'], sr, channels = get_audio_info(line['audio'])
                    if sr != TARGET_SR:
                        non_24k_count += 1
                    if channels != 1:
                        non_mono_count += 1
                    if line.get('ref_audio'):
                        line['ref_audio'] = resolve_path(line['ref_audio'])
                        
                    batch_lines.append(line)
                    batch_audios.append(line['audio'])

                    while len(batch_lines) >= current_batch_size:
                        chunk_size = min(current_batch_size, len(batch_lines))
                        try:
                            enc_res = tokenizer_12hz.encode(batch_audios[:chunk_size])
                        except Exception as e:
                            if _is_cuda_oom(e) and chunk_size > 1:
                                current_batch_size = max(1, chunk_size // 2)
                                _cleanup_cuda()
                                yield {
                                    "type": "progress",
                                    "progress": 0.1 + 0.8 * ((written_count + skipped_count) / max(total_count, 1)),
                                    "desc": f"CUDA OOM; reducing tokenizer batch size to {current_batch_size} and retrying...",
                                }
                                continue
                            raise

                        for code, item in zip(enc_res.audio_codes, batch_lines[:chunk_size]):
                            item['audio_codes'] = code.cpu().tolist()
                            f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                            written_count += 1
                            completed_keys.add(_entry_key(item))

                        f_out.flush()
                        del enc_res
                        del batch_lines[:chunk_size]
                        del batch_audios[:chunk_size]
                        _cleanup_cuda()
                        
                        yield {
                            "type": "progress",
                            "progress": 0.1 + 0.8 * ((written_count + skipped_count) / max(total_count, 1)),
                            "desc": f"Tokenizing: {written_count + skipped_count}/{total_count} (new {written_count}, skipped {skipped_count}, batch {current_batch_size})",
                        }

            while batch_lines:
                chunk_size = min(current_batch_size, len(batch_lines))
                try:
                    enc_res = tokenizer_12hz.encode(batch_audios[:chunk_size])
                except Exception as e:
                    if _is_cuda_oom(e) and chunk_size > 1:
                        current_batch_size = max(1, chunk_size // 2)
                        _cleanup_cuda()
                        yield {
                            "type": "progress",
                            "progress": 0.1 + 0.8 * ((written_count + skipped_count) / max(total_count, 1)),
                            "desc": f"CUDA OOM; reducing tokenizer batch size to {current_batch_size} and retrying...",
                        }
                        continue
                    raise

                for code, item in zip(enc_res.audio_codes, batch_lines[:chunk_size]):
                    item['audio_codes'] = code.cpu().tolist()
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                    written_count += 1
                    completed_keys.add(_entry_key(item))

                f_out.flush()
                del enc_res
                del batch_lines[:chunk_size]
                del batch_audios[:chunk_size]
                _cleanup_cuda()

        yield {"type": "progress", "progress": 0.95, "desc": "Saving JSONL output..."}
        os.replace(run_tmp_jsonl, output_jsonl)
        _safe_remove(tmp_output_jsonl)
                
        audio_msg = ""
        if non_24k_count or non_mono_count:
            audio_msg = (
                f" Tokenizer handled {non_24k_count} non-24k file(s)"
                f" and {non_mono_count} non-mono file(s) in memory."
            )
        yield {
            "type": "done",
            "msg": (
                f"Successfully tokenized {written_count} new entries and skipped {skipped_count} existing entries"
                f" with final batch size {current_batch_size}.{audio_msg}"
            ),
        }
    except Exception as e:
        yield {"type": "error", "msg": f"Error during tokenization: {str(e)}"}
    finally:
        _safe_remove(run_tmp_jsonl)
        if 'tokenizer_12hz' in locals():
            del tokenizer_12hz
        _cleanup_cuda()
