import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_torch(device: torch.device) -> None:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def conv_memory_format(device: torch.device) -> torch.memory_format:
    if device.type in {"cuda", "mps"}:
        return torch.channels_last
    return torch.contiguous_format


def prepare_conv_module(module: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    module = module.to(device)
    if device.type in {"cuda", "mps"}:
        module = module.to(memory_format=torch.channels_last)
    return module


def move_batch_tensor(
    tensor: torch.Tensor, device: torch.device, non_blocking: bool = False
) -> torch.Tensor:
    tensor = tensor.to(device, non_blocking=non_blocking)
    if tensor.ndim == 4 and device.type in {"cuda", "mps"}:
        tensor = tensor.contiguous(memory_format=torch.channels_last)
    return tensor
