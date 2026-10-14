"""Local paths and executable/DLL lookup for the venv-based pipeline."""
import os
from pathlib import Path
import sys

# Plain HTTP downloads avoid large Xet transfer buffers on this 12 GB machine.
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("STANDUP_DATA_ROOT", ROOT / "data" / "standup")).resolve()
_DLL_HANDLES = []


def configure_runtime():
    scripts = Path(sys.executable).parent
    os.environ["PATH"] = str(scripts) + os.pathsep + os.environ.get("PATH", "")
    if os.name == "nt":
        _DLL_HANDLES.append(os.add_dll_directory(str(scripts)))


configure_runtime()


def local_model(repository):
    """Cache complete inference files without Windows symlink privileges."""
    if Path(repository).exists():
        return str(Path(repository).resolve())
    from huggingface_hub import model_info, snapshot_download
    destination = ROOT / 'data' / 'models' / repository.replace('/', '--')
    # Query once per load to choose safetensors over duplicate PyTorch weights.
    filenames = {item.rfilename for item in model_info(repository).siblings}
    weights = 'model.safetensors' if 'model.safetensors' in filenames else 'pytorch_model.bin'
    if 'model.bin' in filenames:
        weights = 'model.bin'  # CTranslate2 faster-whisper models
    return snapshot_download(repository, local_dir=destination, max_workers=2,
                             allow_patterns=['*.json', '*.txt', '*.model', weights])
