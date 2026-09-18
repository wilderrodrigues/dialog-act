from enum import StrEnum

import random
import numpy as np
import torch


class DeviceChoice(StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"


def select_device(choice: DeviceChoice = DeviceChoice.AUTO) -> torch.device:
    available = {
        DeviceChoice.CPU: True,
        DeviceChoice.CUDA: torch.cuda.is_available(),
        DeviceChoice.MPS: torch.backends.mps.is_available(),
    }
    if choice == DeviceChoice.AUTO:
        for candidate in (DeviceChoice.CUDA, DeviceChoice.MPS, DeviceChoice.CPU):
            if available[candidate]:
                return torch.device(candidate.value)
    if not available[choice]:
        raise ValueError(f"Requested device '{choice.value}' is unavailable in this PyTorch runtime.")
    return torch.device(choice.value)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)