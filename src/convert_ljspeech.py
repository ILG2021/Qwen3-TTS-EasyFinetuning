#!/usr/bin/env python3
"""
Convert LJSpeech-style metadata into this project's tts_train.jsonl format.

Supported input rows:
  filename.wav|text
  filename|text
  filename|normalized text|text
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def _resolve_audio(audio_root, item_id):
    item_id = item_id.strip()
    candidates = []
    raw = Path(item_id)
    if raw.suffix:
        candidates.append(audio_root / raw)
    else:
        candidates.append(audio_root / f"{item_id}.wav")
        candidates.append(audio_root / item_id)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def convert_ljspeech(metadata, audio_root, output_jsonl, speaker_name, ref_audio=None, text_column=1, copy_audio=False):
    metadata = Path(metadata)
    audio_root = Path(audio_root)
    output_jsonl = Path(output_jsonl)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    speaker_dir = output_jsonl.parent
    audio_out_dir = speaker_dir / "audio"
    if copy_audio:
        audio_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with metadata.open("r", encoding="utf-8-sig") as f:
        for line_number, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            parts = raw_line.split("|")
            if len(parts) <= text_column:
                raise ValueError(f"Line {line_number} has no text column {text_column}: {raw_line}")

            source_audio = _resolve_audio(audio_root, parts[0])
            if not source_audio.exists():
                raise FileNotFoundError(f"Audio file not found for line {line_number}: {source_audio}")
            audio_path = source_audio
            if copy_audio:
                audio_path = audio_out_dir / source_audio.name
                if not audio_path.exists():
                    shutil.copy2(source_audio, audio_path)

            rows.append(
                {
                    "audio": os.path.normpath(str(audio_path)),
                    "text": parts[text_column].strip(),
                    "ref_audio": os.path.normpath(str(ref_audio or "")),
                    "speaker_id": speaker_name,
                }
            )

    if ref_audio is None and rows:
        rows[0]["ref_audio"] = rows[0]["audio"]
        for row in rows[1:]:
            row["ref_audio"] = rows[0]["audio"]

    with output_jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows), output_jsonl


def main():
    parser = argparse.ArgumentParser(description="Convert LJSpeech metadata to tts_train.jsonl")
    parser.add_argument("--metadata", required=True, help="Path to metadata.csv")
    parser.add_argument("--audio_root", required=True, help="Directory containing wav files")
    parser.add_argument("--speaker_name", required=True, help="Speaker/dataset name")
    parser.add_argument("--output_jsonl", default=None, help="Output JSONL path")
    parser.add_argument("--ref_audio", default=None, help="Reference audio path. Defaults to first item.")
    parser.add_argument("--text_column", type=int, default=1, help="0-based text column index after splitting by |")
    parser.add_argument("--copy_audio", action="store_true", help="Copy audio files next to the JSONL")
    args = parser.parse_args()

    output_jsonl = args.output_jsonl or os.path.join("final-dataset", args.speaker_name, "tts_train.jsonl")
    count, path = convert_ljspeech(
        metadata=args.metadata,
        audio_root=args.audio_root,
        output_jsonl=output_jsonl,
        speaker_name=args.speaker_name,
        ref_audio=args.ref_audio,
        text_column=args.text_column,
        copy_audio=args.copy_audio,
    )
    print(f"Converted {count} rows to {path}")


if __name__ == "__main__":
    main()
