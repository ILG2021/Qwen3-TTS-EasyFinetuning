import json
import os
import threading
import time

import gradio as gr

from utils import (
    DEFAULT_TTS_TRAIN_MODEL,
    ModelDownloadTracker,
    format_bytes,
    is_custom_voice_model,
    missing_speaker_embeddings,
    resolve_path,
)


def checkpoint_sort_key(output_path, exp_name, checkpoint_name):
    checkpoint_dir = os.path.join(output_path, exp_name, checkpoint_name)
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    global_step = -1
    epoch = -1
    if os.path.exists(trainer_state_path):
        try:
            with open(trainer_state_path, "r", encoding="utf-8") as f:
                trainer_state = json.load(f)
            global_step = int(trainer_state.get("global_step", -1))
            epoch = int(trainer_state.get("epoch", -1))
        except Exception:
            pass
    return (global_step, epoch, checkpoint_name)


def get_checkpoints(experiment_name=None, include_specials=True):
    output_path = resolve_path("output")
    ckpts = ["latest", "none"] if include_specials else []

    if not os.path.exists(output_path):
        return ckpts

    exps = [experiment_name] if experiment_name else os.listdir(output_path)
    found_ckpts = []
    for exp in exps:
        exp_dir = os.path.join(output_path, exp)
        if not os.path.isdir(exp_dir):
            continue
        for item in os.listdir(exp_dir):
            if item.startswith("checkpoint-step-") or item.startswith("checkpoint-epoch-"):
                found_ckpts.append((exp, item))

    found_ckpts.sort(key=lambda x: checkpoint_sort_key(output_path, x[0], x[1]), reverse=True)
    return ckpts + [os.path.join(exp, item) for exp, item in found_ckpts]


def normalize_speaker_name(speaker_name):
    if isinstance(speaker_name, list):
        return ",".join(s.strip() for s in speaker_name if s and s.strip())
    return speaker_name.strip() if speaker_name else ""


def parse_speaker_names(speaker_name):
    """
    Normalize Gradio speaker selections into an ordered list.

    Dependencies: Gradio returns multiselect dropdown values as lists, while
    restored experiment configs may provide comma-separated strings.
    """
    if isinstance(speaker_name, list):
        return [s.strip() for s in speaker_name if s and str(s).strip()]
    if isinstance(speaker_name, str):
        return [s.strip() for s in speaker_name.split(",") if s.strip()]
    return []


def normalize_resume_checkpoint(resume_from_checkpoint):
    if resume_from_checkpoint == "none":
        return None
    if resume_from_checkpoint and resume_from_checkpoint != "latest" and not os.path.isabs(resume_from_checkpoint):
        return resolve_path(os.path.join("output", resume_from_checkpoint))
    return resume_from_checkpoint


def save_training_config(output_dir, config_data):
    config_path = os.path.join(output_dir, "training_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)


def build_training_kwargs(
    experiment_name,
    speaker_name_str,
    resolved_init_model,
    output_dir,
    train_jsonl,
    batch_size,
    lr,
    epochs,
    grad_acc,
    final_resume,
    use_experimental_speedup,
    save_strategy,
    save_steps,
    keep_last_n_checkpoints,
    use_accelerator,
):
    return {
        "experiment_name": experiment_name,
        "init_model_path": resolved_init_model,
        "output_model_path": output_dir,
        "train_jsonl": train_jsonl,
        "speaker_name": speaker_name_str,
        "batch_size": batch_size,
        "lr": float(lr) if isinstance(lr, str) else lr,
        "num_epochs": epochs,
        "gradient_accumulation_steps": grad_acc,
        "resume_from_checkpoint": final_resume,
        "use_experimental_speedup": use_experimental_speedup,
        "save_strategy": save_strategy,
        "save_steps": save_steps,
        "keep_last_n_checkpoints": keep_last_n_checkpoints,
        "use_accelerator": use_accelerator,
    }


class TrainingLogBuffer:
    """
    Keep WebUI training logs focused on actionable state instead of every tick.

    Dependencies: handle_training_message feeds structured progress dictionaries
    from sft_12hz.run_train; raw string messages are preserved as recent events.
    """

    def __init__(self, limit=12):
        self.limit = limit
        self.events = []
        self.status = "Starting..."
        self.progress_line = ""
        self.last_event = ""

    def record_event(self, message):
        message = str(message).strip()
        if not message:
            return self.render()
        self.status = message
        if message != self.last_event:
            timestamp = time.strftime("%H:%M:%S")
            self.events.append(f"[{timestamp}] {message}")
            self.events = self.events[-self.limit:]
            self.last_event = message
        return self.render()

    def record_progress(self, message):
        message = str(message).strip()
        if message:
            self.progress_line = message
        return self.render()

    def render(self):
        lines = [f"Status: {self.status}"]
        if self.progress_line:
            lines.append(f"Current: {self.progress_line}")
        if self.events:
            lines.extend(["", "Recent events:"])
            lines.extend(self.events[-self.limit:])
        return "\n".join(lines)


def append_log(log_history, message, limit=30):
    if hasattr(log_history, "record_event"):
        return log_history.record_event(message)
    log_history.append(message)
    return "\n".join(log_history[-limit:])


def append_progress_log(log_history, message):
    if hasattr(log_history, "record_progress"):
        return log_history.record_progress(message)
    return append_log(log_history, message)


def format_training_progress(item, total_epochs):
    epoch = item.get("epoch", 0)
    step = item.get("step", 0)
    loss = item.get("loss", 0.0)
    steps_in_epoch = item.get("steps_in_epoch")
    global_step = item.get("global_step")
    epoch_progress = item.get("epoch_progress")

    if isinstance(epoch_progress, (int, float)):
        current_progress = min(0.999, (epoch + float(epoch_progress)) / max(total_epochs, 1))
    else:
        current_progress = epoch / max(total_epochs, 1)

    step_prefix = f"Global Step {global_step} | " if isinstance(global_step, int) and global_step >= 0 else ""
    if isinstance(step, int):
        if isinstance(steps_in_epoch, int) and steps_in_epoch > 0:
            desc_str = f"Epoch {epoch + 1}/{total_epochs} | {step_prefix}Step {step + 1}/{steps_in_epoch} | Loss: {loss:.4f}"
        else:
            desc_str = f"Epoch {epoch + 1}/{total_epochs} | {step_prefix}Step {step + 1} | Loss: {loss:.4f}"
    else:
        desc_str = f"Epoch {epoch + 1}/{total_epochs} | {step_prefix}{step}"

    return max(0.0, min(current_progress, 0.999)), desc_str


def handle_training_message(item, progress, total_epochs, last_status, log_history):
    msg_type = item.get("type", "")
    if msg_type == "progress":
        progress(item.get("progress", 0), desc=item.get("desc", ""))
        last_status = f"Running: {item.get('desc', '')}"
        return last_status, append_log(log_history, last_status), False
    if msg_type in {"train_progress", "epoch_start"}:
        current_progress, desc_str = format_training_progress(item, total_epochs)
        progress(current_progress, desc=desc_str)
        return gr.update(), append_progress_log(log_history, desc_str), False
    if msg_type == "done":
        progress(1.0, desc="Done")
        last_status = f"Success: {item.get('msg', 'Completed')}"
        return last_status, append_log(log_history, last_status), True
    if msg_type == "error":
        progress(0, desc="Error")
        last_status = f"Error: {item.get('msg', 'Unknown Error')}"
        return last_status, append_log(log_history, last_status), True
    if hasattr(log_history, "render"):
        return last_status, log_history.render(), False
    return last_status, "\n".join(log_history[-30:]), False


def get_training_prerequisites(experiment_name, speaker_name, init_model):
    """
    Inspect the current WebUI selections and return workflow gate state.

    Dependencies: tokenization writes logs/{experiment}/tts_train_with_codes.jsonl;
    embedding writes final-dataset/{speaker}/speaker_emb.safetensors.
    """
    experiment = experiment_name.strip() if isinstance(experiment_name, str) else ""
    speakers = parse_speaker_names(speaker_name)
    missing = []

    if not experiment:
        missing.append("Create or select an experiment.")
    if not speakers:
        missing.append("Select at least one target speaker.")

    missing_jsonl = []
    for speaker in speakers:
        jsonl = resolve_path(os.path.join("final-dataset", speaker, "tts_train.jsonl"))
        if not os.path.exists(jsonl):
            missing_jsonl.append(speaker)
    if missing_jsonl:
        missing.append("Run Data Preparation Step 1 and Step 2 for: " + ", ".join(missing_jsonl))

    tokenized_path = resolve_path(os.path.join("logs", experiment, "tts_train_with_codes.jsonl")) if experiment else ""
    tokenized_ready = bool(tokenized_path and os.path.exists(tokenized_path))
    if experiment and not tokenized_ready:
        missing.append("Run Tokenize Data for this experiment.")

    missing_embeddings = missing_speaker_embeddings(speakers)
    embeddings_ready = not missing_embeddings
    if missing_embeddings:
        model_note = " CustomVoice training is blocked until this is done." if is_custom_voice_model(init_model) else ""
        missing.append("Run Embed Speakers for: " + ", ".join(missing_embeddings) + "." + model_note)

    return {
        "experiment": experiment,
        "speakers": speakers,
        "jsonl_ready": not missing_jsonl,
        "tokenized_ready": tokenized_ready,
        "embeddings_ready": embeddings_ready,
        "missing_embeddings": missing_embeddings,
        "ready": bool(experiment and speakers and tokenized_ready and embeddings_ready and not missing_jsonl),
        "missing": missing,
    }


def render_training_gate(experiment_name, speaker_name, init_model):
    """
    Build a compact readiness panel and button states for the guided workflow.

    Dependencies: Gradio Markdown displays the HTML summary; Button updates are
    returned to make the workflow intentionally hard to run out of order.
    """
    state = get_training_prerequisites(experiment_name, speaker_name, init_model)
    status = "Ready to train" if state["ready"] else "Blocked"
    tone = "#16a34a" if state["ready"] else "#f59e0b"
    speaker_text = ", ".join(state["speakers"]) if state["speakers"] else "none selected"
    token_text = "ready" if state["tokenized_ready"] else "missing"
    embed_text = "ready" if state["embeddings_ready"] else "missing"
    requirements = state["missing"] or ["All prerequisites are satisfied."]
    requirement_html = "".join(f"<li>{item}</li>" for item in requirements)
    html = f"""
<div class="workflow-gate">
  <div class="gate-title"><span style="color:{tone}">●</span> Training gate: {status}</div>
  <div class="gate-grid">
    <span>Experiment: <b>{state["experiment"] or "not selected"}</b></span>
    <span>Speakers: <b>{speaker_text}</b></span>
    <span>Tokenized data: <b>{token_text}</b></span>
    <span>Speaker embeddings: <b>{embed_text}</b></span>
  </div>
  <ul>{requirement_html}</ul>
</div>
"""
    can_tokenize = bool(state["experiment"] and state["speakers"] and state["jsonl_ready"])
    can_embed = bool(state["speakers"] and state["jsonl_ready"])
    return html, gr.update(interactive=can_tokenize), gr.update(interactive=can_embed), gr.update(interactive=state["ready"])


def stream_worker_updates(stream, progress, success_prefix="Success"):
    last_status = "Starting..."
    for item in stream:
        if isinstance(item, dict):
            msg_type = item.get("type", "")
            if msg_type == "progress":
                progress(item.get("progress", 0), desc=item.get("desc", ""))
                last_status = f"Running: {item.get('desc', '')}"
            elif msg_type == "done":
                progress(1.0, desc="Done")
                yield f"{success_prefix}: {item.get('msg', 'Completed')}"
                return
            elif msg_type == "error":
                progress(0, desc="Error")
                yield f"Error: {item.get('msg', 'Unknown Error')}"
                return
        elif isinstance(item, str):
            last_status = item
        yield last_status


def get_deeplink_state(request: gr.Request):
    query = getattr(request, "query_params", {}) or {}
    return {
        "exp": query.get("exp", ""),
        "ckpt": query.get("ckpt", ""),
        "tab": query.get("tab", ""),
    }


def load_experiment_config(experiment_name):
    config_path = os.path.join("output", experiment_name, "training_config.json")
    checkpoint_choices = gr.update(choices=get_checkpoints(experiment_name=experiment_name, include_specials=True))
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            preset = "Latest Config"
            return (
                preset,
                data.get("init_model", "Qwen/Qwen3-TTS-12Hz-0.6B-Base"),
                data.get("batch_size", 2),
                data.get("lr", 1e-7),
                data.get("epochs", 2),
                data.get("grad_acc", 4),
                data.get("speaker_name", "").split(",") if data.get("speaker_name") else [],
                data.get("use_experimental_speedup", False),
                data.get("resume_from_checkpoint", "latest"),
                data.get("save_strategy", "both"),
                data.get("save_steps", 200),
                data.get("keep_last_n_checkpoints", 3),
                data.get("use_accelerator", False),
                f"Loaded configuration for experiment '{experiment_name}'",
                checkpoint_choices,
            )
        except Exception as e:
            return (
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), f"Failed to load config: {e}", checkpoint_choices
            )

    return (
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
        gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), "New experiment / No config found.", checkpoint_choices
    )


def on_new_experiment(name, get_experiments_fn):
    if not name or not name.strip():
        return [gr.update()] * 14 + ["Error: Experiment name cannot be empty.", gr.update()]

    name = name.strip()
    output_dir = resolve_path(os.path.join("output", name))

    if os.path.exists(output_dir):
        res = list(load_experiment_config(name))
        res[-2] = f"Experiment '{name}' already exists. Switched to it and loaded configuration."
        return [gr.update(choices=get_experiments_fn(), value=name), gr.update(value="")] + res

    try:
        os.makedirs(output_dir, exist_ok=True)
        return [
            gr.update(choices=get_experiments_fn(), value=name),
            gr.update(value=""),
            "0.6B Base",
            DEFAULT_TTS_TRAIN_MODEL,
            2,
            "1e-7",
            2,
            4,
            [],
            False,
            "latest",
            "both",
            200,
            3,
            False,
            f"Successfully created new experiment: {name}",
            gr.update(choices=get_checkpoints(experiment_name=name, include_specials=True), value="latest"),
        ]
    except Exception as e:
        return [gr.update()] * 15 + [f"Error creating experiment folder: {e}", gr.update()]


def run_with_polling(fn, progress, progress_start=0.02, progress_end=0.95, desc_prefix="Downloading"):
    """
    Run a blocking job in a worker thread and keep Gradio progress responsive.

    Dependencies: a callable may expose `progress_tracker` from model_repository
    for model downloads. Older callers can still expose `target_dir`, which is
    scanned as a fallback to show size and speed.
    """
    result = {"value": None, "error": None}

    def worker():
        try:
            result["value"] = fn()
        except Exception as exc:
            result["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    last_time = time.time()
    last_size = 0
    target_dir = None
    tracker = getattr(fn, "progress_tracker", None)
    try:
        maybe_path = getattr(fn, "target_dir", None)
        if maybe_path:
            target_dir = maybe_path
    except Exception:
        target_dir = None

    while thread.is_alive():
        current_time = time.time()
        if isinstance(tracker, ModelDownloadTracker):
            snapshot = tracker.snapshot()
            if snapshot.fraction is not None:
                progress_value = progress_start + (progress_end - progress_start) * min(snapshot.fraction, 0.995)
            else:
                # Unknown totals are common with hub downloads; advance slowly while
                # still reporting real downloaded bytes and instantaneous speed.
                elapsed_factor = min(snapshot.elapsed_seconds / 300.0, 0.90)
                progress_value = progress_start + (progress_end - progress_start) * elapsed_factor
            total_text = f" / {format_bytes(snapshot.total_bytes)}" if snapshot.total_bytes else ""
            speed_text = f"{format_bytes(snapshot.speed_bps)}/s"
            progress(
                progress_value,
                desc=(
                    f"{desc_prefix} | {snapshot.stage} | "
                    f"{format_bytes(snapshot.downloaded_bytes)}{total_text} | {speed_text}"
                ),
            )
        elif target_dir and os.path.exists(target_dir):
            total_size = 0
            for root, _, files in os.walk(target_dir):
                for name in files:
                    try:
                        total_size += os.path.getsize(os.path.join(root, name))
                    except OSError:
                        pass
            delta_t = max(current_time - last_time, 1e-6)
            speed = max(total_size - last_size, 0) / delta_t
            downloaded_mb = total_size / (1024 ** 2)
            speed_mb = speed / (1024 ** 2)
            progress(progress_start, desc=f"{desc_prefix}... {downloaded_mb:.1f} MB | {speed_mb:.2f} MB/s")
            last_time = current_time
            last_size = total_size
        else:
            progress(progress_start, desc=f"{desc_prefix}...")
        time.sleep(0.3)

    thread.join()
    if result["error"] is not None:
        raise result["error"]
    progress(progress_end, desc=f"{desc_prefix} complete")
    return result["value"]
