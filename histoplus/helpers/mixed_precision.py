"""Optimized inference."""

from typing import List, Optional, Union

import torch


class InferenceModule(torch.nn.Module):
    """Wrap around a module to run inference with mixed precision and multi-GPU support.

    This class combines both mixed precision and data parallel functionality for
    efficient inference.

    Parameters
    ----------
    module : torch.nn.Module
        The module (typically a model) to run in inference mode.
    device_ids : Optional[List[int]] = None
        List of GPU IDs to use for DataParallel. If None and device is 'cuda',
        all available GPUs will be used.
    mixed_precision : bool = True
        Whether to use mixed precision during inference.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        device_ids: Optional[List[int]] = None,
        mixed_precision: bool = True,
    ):
        super().__init__()

        self._internal_module = module
        self.mixed_precision = mixed_precision

        if not torch.cuda.is_available():
            self.device = "cpu"
            self.module = module
            self.module.to(self.device)
        else:
            self.device = f"cuda:{0 if device_ids is None else device_ids[0]}"
            self.module = torch.nn.DataParallel(module, device_ids=device_ids)
            self.module.to(self.device)

        self.module.eval()
        self.module.requires_grad_(False)

    def __getattr__(self, value):
        try:
            return super().__getattr__(value)
        except AttributeError:
            return getattr(self._internal_module, value)

    def forward(self, *args, **kwargs):
        """Forward pass with optional mixed precision and data parallel support."""
        if self.mixed_precision and self.device.startswith("cuda"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = self.module(*args, **kwargs)
        else:
            output = self.module(*args, **kwargs)

        return self._cast_output_to_float32(output)

    @staticmethod
    def _cast_output_to_float32(output: Union[dict, list, torch.Tensor]):
        if isinstance(output, dict):
            return {
                k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v
                for k, v in output.items()
            }
        elif isinstance(output, torch.Tensor):
            return output.to(torch.float32)
        elif isinstance(output, list) and all(isinstance(out, dict) for out in output):
            return [
                {
                    k: v.to(torch.float32) if isinstance(v, torch.Tensor) else v
                    for k, v in out.items()
                }
                for out in output
            ]
        else:
            return output


def prepare_module(
    module: torch.nn.Module,
    gpu: Union[None, int, list[int]] = None,
    mixed_precision: bool = False,
) -> torch.nn.Module:
    """Prepare a module for inference with mixed precision and multi-GPU support.

    Parameters
    ----------
    module: torch.nn.Module
        The module to prepare.

    gpu: Union[None, int, list[int]] = None
        GPUs to use.
        If None, will use all available GPUs.
        If -1, extraction will run on CPU.

    mixed_precision: bool = False
        Whether to use mixed precision (improved throughput on modern GPU cards).

    Returns
    -------
    torch.nn.Module
        The prepared inference module ready for efficient processing.
    """
    if gpu == -1:
        module.to("cpu")
        module.eval()
        module.requires_grad_(False)
        return module

    if gpu is None:
        gpu_ids = None
    elif isinstance(gpu, int):
        gpu_ids = [gpu]
    elif isinstance(gpu, list):
        gpu_ids = gpu
    else:
        raise TypeError("GPU identifiers must be None, int or list[int].")

    return InferenceModule(module, gpu_ids, mixed_precision)
