import os
import platform
import re
import shutil
import socket
import subprocess
import sys


def get_project_root():
    '''Detect the project root directory.
    In Docker, it's usually /workspace.
    Otherwise, it's the parent directory of this src file.
    '''
    if os.path.exists('/.dockerenv') or os.environ.get('IS_DOCKER'):
        return '/workspace'
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(path):
    '''Normalize a path to be absolute within the project root if it is relative.'''
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(get_project_root(), path))


def get_model_local_dir(model_id):
    root = get_project_root()
    return os.path.join(root, 'models', model_id)



def is_model_dir_ready(path):
    if not path or not os.path.isdir(path):
        return False
    marker_files = (
        'config.json',
        'tokenizer_config.json',
        'preprocessor_config.json',
        'model.safetensors.index.json',
        'pytorch_model.bin.index.json',
    )
    if any(os.path.exists(os.path.join(path, marker)) for marker in marker_files):
        return True
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    return any(entry.endswith(('.safetensors', '.bin', '.json', '.model')) for entry in entries)



def _cleanup_empty_dir(path):
    if not os.path.isdir(path) or os.path.islink(path):
        return
    try:
        if not os.listdir(path):
            os.rmdir(path)
    except OSError:
        pass



def _ensure_shared_model_dir(model_id, downloaded_path):
    local_dir = get_model_local_dir(model_id)
    resolved_downloaded_path = os.path.realpath(downloaded_path) if downloaded_path else downloaded_path
    resolved_local_dir = os.path.realpath(local_dir)

    if is_model_dir_ready(local_dir):
        return local_dir

    if not downloaded_path or not is_model_dir_ready(downloaded_path):
        return downloaded_path

    if resolved_downloaded_path == resolved_local_dir:
        return local_dir

    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    _cleanup_empty_dir(local_dir)

    if not os.path.exists(local_dir):
        try:
            os.symlink(downloaded_path, local_dir, target_is_directory=True)
            print(f'Linked shared model directory {local_dir} -> {downloaded_path}')
            return local_dir
        except OSError:
            pass

    if not is_model_dir_ready(local_dir):
        shutil.copytree(downloaded_path, local_dir, dirs_exist_ok=True)
        print(f'Copied model into shared directory {local_dir}')

    return local_dir if is_model_dir_ready(local_dir) else downloaded_path



def is_model_downloaded(model_id):
    local_dir = get_model_local_dir(model_id)
    return is_model_dir_ready(local_dir)



def get_model_path(model_id, use_hf=True):
    resolved_input = resolve_path(model_id)
    if os.path.exists(resolved_input):
        return resolved_input
    output_candidate = resolve_path(os.path.join('output', model_id))
    if os.path.exists(output_candidate):
        print(f'Found local checkpoint at {output_candidate}')
        return output_candidate
    local_dir = get_model_local_dir(model_id)
    if is_model_downloaded(model_id):
        print(f'Found local model at {local_dir}, skipping download!')
        return local_dir
    print(f'Downloading model {model_id} into {local_dir}...')
    os.makedirs(os.path.dirname(local_dir), exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
        downloaded_path = snapshot_download(repo_id=model_id, local_dir=local_dir)
        return _ensure_shared_model_dir(model_id, downloaded_path)
    except Exception as e:
        print(f'Warning: Download failed, falling back to id: {e}')
        return model_id


def is_windows():
    return platform.system().lower() == "windows"


def is_flash_attention_available():
    if is_windows():
        return False
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def get_attn_implementation(device=None):
    device_text = str(device or "").lower()
    if device_text == "cpu":
        return None
    return "flash_attention_2" if is_flash_attention_available() else None


def is_port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, int(port))) == 0


def start_tensorboard(logdir="logs", port=6006):
    if is_port_open(port):
        return None
    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        "--logdir",
        logdir,
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_tensorboard_process(process=None, port=6006):
    if process is not None and process.poll() is None:
        process.terminate()
        return True
    if not is_port_open(port):
        return False
    if is_windows():
        subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["taskkill", "/F", "/IM", "python.exe", "/FI", "WINDOWTITLE eq tensorboard*"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["pkill", "-f", f"tensorboard.*--port {port}"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


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
