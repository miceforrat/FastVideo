# SPDX-License-Identifier: Apache-2.0
"""Preallocated CUDA Green Context stream pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

try:
    from fastvideo.green_context import _greenctx
except ImportError as error:
    raise ImportError(
        "FastVideo Green Context extension is not built. Run "
        "`cd fastvideo/green_context && python setup.py build_ext --inplace`."
    ) from error


@dataclass(frozen=True)
class GreenContextPair:
    """One SM partition and its two PyTorch stream wrappers."""

    requested_dit_sms: int
    actual_dit_sms: int
    actual_vae_sms: int
    total_sms: int
    dit_stream: torch.cuda.ExternalStream
    vae_stream: torch.cuda.ExternalStream


class GreenContextPairPool:
    """Own and expose a fixed set of Green Context stream pairs.

    Native Green Context objects are retained for the complete pool lifetime;
    otherwise destroying one would detach its external CUDA streams.
    """

    def __init__(
        self,
        dit_sm_counts: Iterable[int],
        device: int | torch.device | None = None,
        ignore_sm_coscheduling: bool = False,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Green Contexts require CUDA.")
        if device is None:
            self.device = torch.cuda.current_device()
        elif isinstance(device, int):
            self.device = device
        else:
            parsed_device = torch.device(device)
            self.device = (
                torch.cuda.current_device()
                if parsed_device.index is None
                else parsed_device.index
            )
        requested_counts = tuple(sorted(set(dit_sm_counts)))
        if not requested_counts:
            raise ValueError("dit_sm_counts must contain at least one split.")

        self.full_dit_stream = torch.cuda.Stream(device=self.device)
        self.full_vae_stream = torch.cuda.Stream(device=self.device)
        self._native_contexts: dict[int, object] = {}
        self._pairs: dict[int, GreenContextPair] = {}

        for requested_dit_sms in requested_counts:
            native_context = _greenctx.GreenContext(
                requested_dit_sms,
                self.device,
                ignore_sm_coscheduling,
            )
            pair = GreenContextPair(
                requested_dit_sms=requested_dit_sms,
                actual_dit_sms=native_context.dit_sm_count(),
                actual_vae_sms=native_context.vae_sm_count(),
                total_sms=native_context.total_sm_count(),
                dit_stream=torch.cuda.ExternalStream(
                    native_context.dit_stream(), device=self.device),
                vae_stream=torch.cuda.ExternalStream(
                    native_context.vae_stream(), device=self.device),
            )
            self._native_contexts[requested_dit_sms] = native_context
            self._pairs[requested_dit_sms] = pair

    def __contains__(self, requested_dit_sms: int) -> bool:
        return requested_dit_sms in self._pairs

    def __getitem__(self, requested_dit_sms: int) -> GreenContextPair:
        try:
            return self._pairs[requested_dit_sms]
        except KeyError as error:
            raise KeyError(
                f"No preallocated Green Context pair for "
                f"dit_sms={requested_dit_sms}."
            ) from error

    @property
    def requested_dit_sm_counts(self) -> tuple[int, ...]:
        return tuple(self._pairs)

    def synchronize(self) -> None:
        """Wait for every stream owned by the pool."""
        self.full_dit_stream.synchronize()
        self.full_vae_stream.synchronize()
        for pair in self._pairs.values():
            pair.dit_stream.synchronize()
            pair.vae_stream.synchronize()
