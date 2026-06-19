from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


FP4_CODEBOOK = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_quant_cfg(cfg: Any) -> Dict[str, Any]:
    weight_bits = int(_cfg_get(cfg, "weight_bits", 4))
    weight_format = _cfg_get(cfg, "weight_format", "fp4" if weight_bits == 4 else "int8")
    return {
        "runtime": _cfg_get(cfg, "runtime", "fake"),
        "weight_bits": weight_bits,
        "activation_bits": int(_cfg_get(cfg, "activation_bits", 8)),
        "weight_format": str(weight_format),
        "group_size": int(_cfg_get(cfg, "group_size", 128)),
        "pack_format": _cfg_get(cfg, "pack_format", "uint8" if weight_bits == 4 else "int8"),
    }


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


def _pad_last_dim(x: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    pad = (-x.shape[-1]) % group_size
    if pad:
        x = F.pad(x, (0, pad))
    return x, pad


def _group_view_weight(weight: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    padded, pad = _pad_last_dim(weight, group_size)
    return padded.reshape(weight.shape[0], -1, group_size), pad


def fake_quant_act_symmetric(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (2 ** (bits - 1)) - 1
    x_float = x.to(torch.float32)
    scale = x_float.detach().abs().amax().clamp(min=torch.finfo(torch.float32).tiny) / qmax
    q = torch.clamp(_round_ste(x_float / scale), -qmax, qmax)
    return (q * scale).to(x.dtype)


def quantize_weight_int8(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, int]:
    grouped, pad = _group_view_weight(weight.detach().to(torch.float32), group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q = torch.clamp(torch.round(grouped / scale), -127, 127).to(torch.int8)
    return q.reshape(weight.shape[0], -1), scale.to(torch.float32), pad


def fake_quant_weight_int8(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    grouped, _ = _group_view_weight(weight.to(torch.float32), group_size)
    scale = grouped.detach().abs().amax(dim=-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny) / 127.0
    q = torch.clamp(_round_ste(grouped / scale), -127, 127)
    dequant = (q * scale).reshape(weight.shape[0], -1)[:, : weight.shape[1]]
    return dequant.to(weight.dtype)


def dequantize_weight_int8(
    qweight: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int = 128,
) -> torch.Tensor:
    grouped = qweight.to(torch.float32).reshape(out_features, -1, group_size)
    weight = grouped * scale.to(torch.float32)
    return weight.reshape(out_features, -1)[:, :in_features]


def _fp4_delta(grouped: torch.Tensor) -> torch.Tensor:
    w_max = grouped.detach().abs().amax(dim=-1, keepdim=True)
    safe = w_max.clamp(min=torch.finfo(torch.float32).tiny)
    delta = 3.0 + math.log2(1.5) - torch.log2(safe)
    return torch.where(w_max > 0, delta, torch.zeros_like(delta)).to(torch.float32)


def quantize_weight_fp4_values(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, int]:
    grouped, pad = _group_view_weight(weight.to(torch.float32), group_size)
    delta = _fp4_delta(grouped)
    w_max = 1.5 * torch.pow(2.0, 3.0 - delta)
    clipped = torch.clamp(grouped, -w_max, w_max)
    log_scales = torch.clamp(
        torch.floor(torch.log2(torch.abs(clipped) + 1e-5) + delta).detach(),
        min=1.0,
    )
    scales = torch.pow(2.0, log_scales - 1.0 - delta)
    mantissa = _round_ste(grouped / scales)
    dequant = mantissa * scales
    return dequant.reshape(weight.shape[0], -1)[:, : weight.shape[1]], delta, pad


def quantize_weight_fp4_codes(weight: torch.Tensor, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor, int]:
    grouped, pad = _group_view_weight(weight.detach().to(torch.float32), group_size)
    delta = _fp4_delta(grouped)
    w_max = 1.5 * torch.pow(2.0, 3.0 - delta)
    clipped = torch.clamp(grouped, -w_max, w_max)
    log_scales = torch.clamp(
        torch.floor(torch.log2(torch.abs(clipped) + 1e-5) + delta).detach(),
        min=1.0,
    )
    scales = torch.pow(2.0, log_scales - 1.0 - delta)
    dequant = torch.round(grouped / scales) * scales
    normalized = dequant / torch.pow(2.0, -delta) / 2.0
    codebook = FP4_CODEBOOK.to(device=weight.device, dtype=normalized.dtype)
    distances = torch.abs(normalized.unsqueeze(-1) - codebook)
    codes = distances.argmin(dim=-1).to(torch.uint8)
    return codes.reshape(weight.shape[0], -1), delta.to(torch.float32), pad


def dequantize_weight_fp4_codes(
    codes: torch.Tensor,
    delta: torch.Tensor,
    out_features: int,
    in_features: int,
    group_size: int = 128,
) -> torch.Tensor:
    codebook = FP4_CODEBOOK.to(device=codes.device, dtype=torch.float32)
    grouped_codes = codes.to(torch.long).reshape(out_features, -1, group_size)
    normalized = codebook[grouped_codes]
    weight = normalized * torch.pow(2.0, -delta.to(torch.float32)) * 2.0
    return weight.reshape(out_features, -1)[:, :in_features]


def pack_int8x2_to_uint16(qweight: torch.Tensor) -> torch.Tensor:
    flat = qweight.reshape(qweight.shape[0], -1).to(torch.int16)
    pad = (-flat.shape[-1]) % 2
    if pad:
        flat = F.pad(flat, (0, pad))
    unsigned = (flat.to(torch.int16) + 128).to(torch.int32)
    packed = unsigned[:, 0::2] | (unsigned[:, 1::2] << 8)
    return packed.to(torch.int32)


def unpack_int8x2_from_uint16(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    packed = packed.to(torch.int32)
    lo = (packed & 0xFF) - 128
    hi = ((packed >> 8) & 0xFF) - 128
    out = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    return out[:, :n_values].to(torch.int8)


def pack_fp4_codes(codes: torch.Tensor) -> torch.Tensor:
    flat = codes.reshape(codes.shape[0], -1).to(torch.uint8)
    pad = (-flat.shape[-1]) % 2
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat[:, 0::2] | (flat[:, 1::2] << 4)


def unpack_fp4_codes(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    packed = packed.to(torch.uint8)
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    return out[:, :n_values].to(torch.uint8)


class TrivialQuantLinear(nn.Module):
    def __init__(self, in_feature: int, out_feature: int, bias: bool = False, quant: Any = None):
        super().__init__()
        self.in_features = int(in_feature)
        self.out_features = int(out_feature)
        self.quant = _as_quant_cfg(quant)
        self.runtime = self.quant["runtime"]
        self.group_size = self.quant["group_size"]
        self.weight_bits = self.quant["weight_bits"]
        self.activation_bits = self.quant["activation_bits"]
        self.weight_format = self.quant["weight_format"]
        self.pack_format = self.quant["pack_format"]
        self.padded_in_features = int(math.ceil(self.in_features / self.group_size) * self.group_size)
        self.num_groups = self.padded_in_features // self.group_size

        if self.runtime == "packed":
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features)))

        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features))
        else:
            self.register_parameter("bias", None)

        self._register_packed_buffers()

    def _register_packed_buffers(self) -> None:
        if self.weight_bits == 4 or self.weight_format == "fp4":
            packed_width = math.ceil(self.padded_in_features / 2)
            self.register_buffer("qweight_packed", torch.zeros((self.out_features, packed_width), dtype=torch.uint8))
            self.register_buffer("weight_delta", torch.zeros((self.out_features, self.num_groups, 1), dtype=torch.float32))
            self.register_buffer("weight_scale", torch.empty(0), persistent=False)
        else:
            if self.pack_format == "int8x2_uint16":
                packed_width = math.ceil(self.padded_in_features / 2)
                self.register_buffer("qweight_packed", torch.zeros((self.out_features, packed_width), dtype=torch.int32))
                self.register_buffer("qweight", torch.empty(0, dtype=torch.int8), persistent=False)
            else:
                self.register_buffer("qweight", torch.zeros((self.out_features, self.padded_in_features), dtype=torch.int8))
                self.register_buffer("qweight_packed", torch.empty(0, dtype=torch.int32), persistent=False)
            self.register_buffer("weight_scale", torch.zeros((self.out_features, self.num_groups, 1), dtype=torch.float32))
            self.register_buffer("weight_delta", torch.empty(0), persistent=False)

    def _fake_quant_weight(self) -> torch.Tensor:
        if self.weight_bits == 4 or self.weight_format == "fp4":
            qweight, _, _ = quantize_weight_fp4_values(self.weight, self.group_size)
            return qweight
        return fake_quant_weight_int8(self.weight, self.group_size)

    def _packed_weight(self) -> torch.Tensor:
        if self.weight_bits == 4 or self.weight_format == "fp4":
            n_values = self.padded_in_features
            codes = unpack_fp4_codes(self.qweight_packed, n_values)
            return dequantize_weight_fp4_codes(
                codes, self.weight_delta, self.out_features, self.in_features, self.group_size
            )
        if self.pack_format == "int8x2_uint16":
            qweight = unpack_int8x2_from_uint16(self.qweight_packed, self.padded_in_features)
        else:
            qweight = self.qweight
        return dequantize_weight_int8(
            qweight, self.weight_scale, self.out_features, self.in_features, self.group_size
        )

    def load_packed_from_float_weight(self, weight: torch.Tensor) -> None:
        weight = weight.detach().to(torch.float32)
        if self.weight_bits == 4 or self.weight_format == "fp4":
            codes, delta, _ = quantize_weight_fp4_codes(weight, self.group_size)
            self.qweight_packed.copy_(pack_fp4_codes(codes).to(self.qweight_packed.device))
            self.weight_delta.copy_(delta.to(self.weight_delta.device))
        else:
            qweight, scale, _ = quantize_weight_int8(weight, self.group_size)
            self.weight_scale.copy_(scale.to(self.weight_scale.device))
            if self.pack_format == "int8x2_uint16":
                self.qweight_packed.copy_(pack_int8x2_to_uint16(qweight).to(self.qweight_packed.device))
            else:
                self.qweight.copy_(qweight.to(self.qweight.device))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.runtime == "packed":
            weight = self._packed_weight().to(device=input.device, dtype=input.dtype)
            x = input
        else:
            weight = self._fake_quant_weight()
            x = fake_quant_act_symmetric(input, self.activation_bits)
        return F.linear(x, weight.to(dtype=x.dtype, device=x.device), self.bias)


class LinearA(TrivialQuantLinear):
    def forward(self, input: torch.Tensor, parameter: torch.Tensor, dequant_input: torch.Tensor):
        out = super().forward(input)
        delta = torch.ones((), dtype=out.dtype, device=out.device)
        return out, delta


class LinearB(TrivialQuantLinear):
    def forward(
        self,
        input: torch.Tensor,
        parameter: torch.Tensor,
        dequant_input: torch.Tensor = None,
        out_parameter: torch.Tensor = None,
    ):
        return super().forward(input)


class LinearC(TrivialQuantLinear):
    def forward(self, input: torch.Tensor, input_delta: torch.Tensor):
        return super().forward(input)
