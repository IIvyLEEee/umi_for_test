import torch
import numpy as np
import math

def round_ste(x: torch.Tensor):
    zero = torch.zeros_like(x)
    x = torch.where(x.abs() < 2, zero, x)
    return (x.round() - x).detach() + x

def int8_init_scale(x: torch.Tensor):
    delta = None
    x_min = min(x.data.min().item(), 0)
    x_max = max(x.data.max().item(), 0)
    x_absmax = max(abs(x_min), x_max)
    delta = x_absmax / 127.
    delta = torch.tensor(delta).type_as(x)
    return delta

def fp4_init_scale(x: torch.Tensor, channel_wise: bool = False):
    delta = None
    if channel_wise:
        x_clone = x.clone().detach()
        n_channels = x_clone.shape[0]
        if len(x.shape) == 4:
            x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
        elif len(x.shape) == 3:
            x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0]
        else:
            x_max = x_clone.abs().max(dim=-1)[0]
        delta = x_max.clone()
        # determine the scale and zero point channel-by-channel
        for c in range(n_channels):
            delta[c] = fp4_init_scale(x_clone[c], channel_wise=False)
        if len(x.shape) == 4:
            delta = delta.view(-1, 1, 1, 1)
        elif len(x.shape) == 3:
            delta = delta.view(-1, 1, 1)
        else:
            delta = delta.view(-1, 1)
    else:
        w_max = x.abs().max()
        delta = 2**2 - torch.log2(w_max) + math.log2(2 - 2 ** (-1)) - 1
        delta = delta.clone().detach().type_as(x)

    return delta

def fp4_quantizer(weight: torch.Tensor, weight_delta: torch.Tensor):
    w_max = (2 - 2 ** (-1)) * 2 ** (2**2 - 1 - weight_delta)
    w_min = -w_max
    x_R = torch.min(torch.max(weight, w_min), w_max)
    w_log_scales = torch.clamp(
        (torch.floor(torch.log2(torch.abs(x_R) + 1e-5) + weight_delta)).detach(),
        1.0,
    )
    x_scales = 2.0 ** (w_log_scales - 1 - weight_delta)
    x_quant_m = (weight / x_scales).round_()
    x_dequant = x_quant_m.mul_(x_scales)
    return x_dequant / 2.0 ** (-weight_delta) / 2

def int8_quantizer(input: torch.Tensor):
    input = input.to(torch.float)
    x_int = round_ste(input)
    x_quant = torch.clamp(x_int, -127, 127)
    return x_quant

class Convert_int8():
    def __init__(self):
        self.convert_tensor = torch.load("toint_tensor.ckpt")
    def convert(self, x: torch.Tensor):
        self.convert_tensor = self.convert_tensor.to(x.device)
        x = x.to(torch.float)
        x = x.clamp(-130.0, 130.0)
        x = x.to(torch.float16)
        int16_tensor = x.clone().view(torch.short)
        int16_tensor = int16_tensor.to(torch.long)
        int16_tensor = torch.where(int16_tensor >= 0, int16_tensor + 22545, int16_tensor + 32768)
        out = self.convert_tensor[int16_tensor].to(torch.float)
        
        return out

if __name__ == "__main__":
    conv = Convert_int8()
    float16_numbers = torch.arange(-32768, 32768, dtype=torch.short)
    float16_numbers = float16_numbers.view(-1, 1)
    float16_tensor = float16_numbers.clone().view(torch.float16)
    mask = (float16_tensor >= -130.0) & (float16_tensor <= 130.0)
    float16_tensor = float16_tensor[mask]
    x = float16_tensor
    diff = conv.convert(x) - int8_quantizer(x)
    # print(x)
    # print(conv.convert(x))
    # print(int8_quantizer(x))
    print(diff.max())
    print(diff.min())

