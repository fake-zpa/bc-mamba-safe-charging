"""Device utilities for hardware detection and configuration."""
import os
import platform
import psutil
import torch


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device.

    Args:
        prefer_gpu: Whether to prefer GPU over CPU.

    Returns:
        torch.device for computation.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Collect comprehensive device and system information.

    Returns:
        Dictionary with hardware info.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cpu_count": os.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "ram_available_gb": round(psutil.virtual_memory().available / 1e9, 1),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu_name": props.name,
            "gpu_memory_gb": round(props.total_memory / 1e9, 1),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        })
    return info


def print_device_info():
    """Print device information to stdout."""
    info = get_device_info()
    print("=" * 60)
    print("DEVICE INFORMATION")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)


def get_amp_dtype() -> torch.dtype:
    """Get the best AMP dtype for current hardware.

    Returns:
        torch.bfloat16 if supported, else torch.float16.
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
    return torch.float16
