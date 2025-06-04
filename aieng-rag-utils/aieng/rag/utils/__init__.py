import platform
import torch

def get_device_name() -> str:
    """
    Get the name of the device to be used for computation.
    This function checks the system platform and available hardware
    to determine whether to use 'cuda', 'mps', or 'cpu'.
    Returns:
        str: The name of the device ('cuda', 'mps', or 'cpu').
    """

    device = "cpu"

    if platform.system() == "Linux" and torch.cuda.is_available():
        device = "cuda"
    elif platform.system() == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

    return device
