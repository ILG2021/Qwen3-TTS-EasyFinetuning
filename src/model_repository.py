import os
import shutil
import threading
import time
from dataclasses import dataclass


PAYLOAD_MARKERS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
    "tokenizer.json",
    "tokenizer.model",
)


def get_project_root():
    """Detect the project root directory shared by all model storage helpers."""
    if os.path.exists("/.dockerenv") or os.environ.get("IS_DOCKER"):
        return "/workspace"
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    """Normalize a project-relative path into an absolute filesystem path."""
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(get_project_root(), path))


def get_model_local_dir(model_id):
    """Return the canonical project-owned directory for a remote model id."""
    return os.path.join(get_project_root(), "models", model_id)


def normalize_model_source(source=None, use_hf=None):
    """Normalize legacy and UI source values into a stable source name."""
    source = "HuggingFace" if use_hf is not False else source
    source = str(source or "HuggingFace").strip().lower()
    if source in {"hf", "huggingface", "hugging_face"}:
        return "HuggingFace"
    raise ValueError(f"Unsupported model source: {source}")


def is_model_dir_ready(path):
    """
    Check whether a directory has the minimum payload needed by from_pretrained.

    Dependencies: local filesystem only. Hub SDK cache metadata is deliberately
    ignored so model reads never depend on HuggingFace internals.
    """
    if not path or not os.path.isdir(path):
        return False

    if any(os.path.exists(os.path.join(path, marker)) for marker in PAYLOAD_MARKERS):
        return True

    try:
        entries = os.listdir(path)
    except OSError:
        return False

    return any(entry.endswith((".safetensors", ".bin", ".model")) for entry in entries)


def format_bytes(num_bytes):
    """Format byte counts for logs and WebUI progress descriptions."""
    if num_bytes is None:
        return "--"
    value = float(max(num_bytes, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def _directory_size(path):
    total_size = 0
    if not path or not os.path.exists(path):
        return total_size
    for root, dirs, files in os.walk(path):
        dirs[:] = [name for name in dirs if name not in {".git", "__pycache__"}]
        for name in files:
            try:
                total_size += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total_size


@dataclass
class DownloadSnapshot:
    """Immutable progress sample consumed by the WebUI polling layer."""

    stage: str
    downloaded_bytes: int
    total_bytes: int | None
    speed_bps: float
    elapsed_seconds: float

    @property
    def fraction(self):
        if not self.total_bytes:
            return None
        return max(0.0, min(self.downloaded_bytes / max(self.total_bytes, 1), 1.0))


class ModelDownloadTracker:
    """
    Thread-safe progress tracker for hub downloads.

    The downloader only records stage, expected size, and active directories.
    The UI thread samples filesystem size from those directories, which keeps
    progress reporting independent from each hub SDK's private callback API.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stage = "Preparing"
        self._paths = []
        self._total_bytes = None
        self._start_time = time.time()
        self._last_time = self._start_time
        self._last_bytes = 0

    def set_stage(self, stage):
        with self._lock:
            self._stage = stage

    def set_total_bytes(self, total_bytes):
        with self._lock:
            self._total_bytes = total_bytes if total_bytes and total_bytes > 0 else None

    def set_active_paths(self, paths):
        clean_paths = [path for path in paths if path]
        with self._lock:
            self._paths = clean_paths

    def snapshot(self):
        with self._lock:
            stage = self._stage
            paths = list(self._paths)
            total_bytes = self._total_bytes
            start_time = self._start_time
            last_time = self._last_time
            last_bytes = self._last_bytes

        downloaded_bytes = sum(_directory_size(path) for path in paths)
        current_time = time.time()
        delta_t = max(current_time - last_time, 1e-6)
        speed_bps = max(downloaded_bytes - last_bytes, 0) / delta_t

        with self._lock:
            self._last_time = current_time
            self._last_bytes = downloaded_bytes

        return DownloadSnapshot(
            stage=stage,
            downloaded_bytes=downloaded_bytes,
            total_bytes=total_bytes,
            speed_bps=speed_bps,
            elapsed_seconds=current_time - start_time,
        )


def _estimate_huggingface_size(model_id):
    """Best-effort remote size lookup for accurate HuggingFace progress."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model_id, files_metadata=True)
        sizes = [sibling.size for sibling in getattr(info, "siblings", []) if getattr(sibling, "size", None)]
        return sum(sizes) or None
    except Exception:
        return None


def _hardlink_or_copy(src, dst):
    """Materialize a downloaded file without duplicating bytes when possible."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        try:
            if os.path.getsize(src) == os.path.getsize(dst):
                return
        except OSError:
            pass
    try:
        if os.path.exists(dst):
            os.remove(dst)
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _materialize_snapshot(source_dir, target_dir):
    """
    Copy or hardlink SDK download output into the canonical project directory.

    Dependencies: only local filesystem primitives. Hidden SDK bookkeeping
    directories are skipped because runtime model loading should use payload
    files, not hub cache state.
    """
    if not source_dir or not os.path.isdir(source_dir):
        return
    os.makedirs(target_dir, exist_ok=True)
    for root, dirs, files in os.walk(source_dir):
        dirs[:] = [name for name in dirs if name not in {".cache", ".git", "__pycache__"}]
        rel_root = os.path.relpath(root, source_dir)
        dst_root = target_dir if rel_root == "." else os.path.join(target_dir, rel_root)
        os.makedirs(dst_root, exist_ok=True)
        for name in files:
            if name.endswith((".lock", ".tmp", ".incomplete")):
                continue
            _hardlink_or_copy(os.path.join(root, name), os.path.join(dst_root, name))


def resolve_existing_model_path(model_ref):
    """Resolve direct paths and local checkpoints without contacting any hub."""
    resolved_input = resolve_path(model_ref)
    if os.path.exists(resolved_input):
        return resolved_input

    output_candidate = resolve_path(os.path.join("output", model_ref))
    if os.path.exists(output_candidate):
        print(f"Found local checkpoint at {output_candidate}")
        return output_candidate

    local_dir = get_model_local_dir(model_ref)
    if is_model_dir_ready(local_dir):
        print(f"Found local model at {local_dir}, skipping download!")
        return local_dir
    return None


def _download_huggingface(model_id, target_dir, tracker):
    source_dir = os.path.join(get_project_root(), "models", ".downloads", "huggingface", model_id)
    os.makedirs(os.path.dirname(source_dir), exist_ok=True)
    tracker.set_stage("Querying HuggingFace model files")
    tracker.set_total_bytes(_estimate_huggingface_size(model_id))
    tracker.set_active_paths([source_dir])

    from huggingface_hub import snapshot_download

    kwargs = {"repo_id": model_id, "local_dir": source_dir}
    try:
        kwargs["local_dir_use_symlinks"] = False
        downloaded_path = snapshot_download(**kwargs)
    except TypeError:
        kwargs.pop("local_dir_use_symlinks", None)
        downloaded_path = snapshot_download(**kwargs)

    tracker.set_stage("Preparing local model directory")
    tracker.set_active_paths([downloaded_path, target_dir])
    _materialize_snapshot(downloaded_path, target_dir)
    return target_dir


def ensure_model_path(model_ref, source="HuggingFace", use_hf=None, progress_tracker=None, download=True):
    """
    Return a local filesystem path that can be passed to from_pretrained.

    HuggingFace is used only as a download transport. Once a snapshot is
    available, payload files are materialized under ./models/<namespace>/<model>,
    and every caller reads from that directory.
    """
    existing_path = resolve_existing_model_path(model_ref)
    if existing_path:
        if progress_tracker:
            progress_tracker.set_stage("Model already available locally")
            progress_tracker.set_active_paths([existing_path])
        return existing_path

    if not download:
        return model_ref

    source = normalize_model_source(source, use_hf=use_hf)
    target_dir = get_model_local_dir(model_ref)
    os.makedirs(os.path.dirname(target_dir), exist_ok=True)
    tracker = progress_tracker or ModelDownloadTracker()
    tracker.set_stage(f"Downloading from {source}")
    tracker.set_active_paths([target_dir])

    print(f"Downloading model {model_ref} from {source} into {target_dir}...")
    try:
        resolved_path = _download_huggingface(model_ref, target_dir, tracker)
    except Exception as exc:
        raise RuntimeError(f"Failed to download {model_ref} from {source}: {exc}") from exc

    if is_model_dir_ready(resolved_path):
        tracker.set_stage("Model ready")
        tracker.set_active_paths([resolved_path])
        return resolved_path

    raise RuntimeError(f"Downloaded model is incomplete: {resolved_path}")
