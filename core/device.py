import torch


def get_device(preference: str = "auto") -> torch.device:
    """
    Select compute device for PyTorch operations.

    Args:
        preference: One of "auto", "cpu", "cuda", "mps"

    Returns:
        torch.device for the selected backend

    Raises:
        ValueError: If requested device is not available
    """
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available")
        return torch.device("cuda")

    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("MPS requested but not available")
        return torch.device("mps")

    raise ValueError(f"Unknown device: {preference}. Use 'auto', 'cpu', 'cuda', or 'mps'")
