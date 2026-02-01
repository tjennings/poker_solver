import pytest
import torch
from core.device import get_device


class TestDeviceSelection:
    def test_auto_returns_device(self):
        """Auto should return a valid torch device."""
        device = get_device("auto")
        assert isinstance(device, torch.device)

    def test_cpu_returns_cpu(self):
        """Explicit cpu should return cpu device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_invalid_device_raises(self):
        """Invalid device string should raise."""
        with pytest.raises(ValueError):
            get_device("invalid_device_xyz")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_when_available(self):
        """Should return CUDA device when available and requested."""
        device = get_device("cuda")
        assert device.type == "cuda"

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
    def test_mps_when_available(self):
        """Should return MPS device when available and requested."""
        device = get_device("mps")
        assert device.type == "mps"
