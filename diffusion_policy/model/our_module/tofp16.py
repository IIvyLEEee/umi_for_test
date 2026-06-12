import torch
import numpy as np

def extract_float_components(tensor):
    sign_bits = []
    exponent_bits = []
    mantissa_bits = []
    tensor_np = tensor.numpy()
    # 将 float64 转换为 IEEE 754 格式的 8 字节
    packed = np.frombuffer(tensor_np.tobytes(), dtype=np.uint64)
    # 提取符号位（第 63 位）
    sign_bits = (packed >> 63) & 0x1

    # 提取指数位（第 62 到 52 位，共 11 位）
    exponent = (packed >> 52) & 0x7FF
    exponent_bits = exponent.clip(1008, 1039) - 1023 + 15

    # 提取尾数位（第 51 到 0 位，共 52 位）
    mantissa = packed & 0xFFFFFFFFFFFFF
    mantissa_bits = np.array([f'{m:052b}' for m in mantissa])
    mantissa_bit = [m[0:10] for m in mantissa_bits]
    return sign_bits, exponent_bits, mantissa_bit

def create_float16_tensor(sign_bits, exponent_bits, mantissa_bits):
    sign_bits_tensor = torch.tensor(sign_bits.astype(np.int64), dtype=torch.float64)
    exponent_bits_tensor = torch.tensor(exponent_bits.astype(np.int64), dtype=torch.float64)
    mantissa_values = torch.tensor([int(mantissa, 2) for mantissa in mantissa_bits], dtype=torch.float64)
    exponent_value = exponent_bits_tensor - 15
    mantissa_final = 1 + mantissa_values / (2**10)
    value = (-1)**sign_bits_tensor * mantissa_final * (2**exponent_value)
    tensor = value.to(torch.float16)
    return tensor

def convert_(x: torch.tensor):
    x = x.to(torch.float64)
    device = x.device
    shape = x.shape
    x = x.cpu().flatten()
    sign_bits, exponent_bits, mantissa_bits = extract_float_components(x)
    result_tensor = create_float16_tensor(sign_bits, exponent_bits, mantissa_bits)
    zero_mask = (x == 0)
    result_tensor[zero_mask] = 0
    result_tensor = result_tensor.to(device=device).reshape(shape)
    return result_tensor

class Convert:
    def __init__(self):
        int_tensor = torch.arange(-300000, 300000, dtype=torch.int)
        float_tensor = int_tensor.to(torch.float64) / 2**14
        self.convert_tensor = convert_(float_tensor)

    def convert(self, x: torch.Tensor):
        x = x.to(torch.long)
        x = x + 300000
        self.convert_tensor = self.convert_tensor.to(x.device)
        out = self.convert_tensor[x]
        return out


if __name__ == "__main__":
    int_tensor = torch.arange(-100000, 100000, dtype=torch.int)
    float_tensor = int_tensor.to(torch.float64) / 2**14
    convert_tensor = convert_(float_tensor)
    a = torch.tensor([90000, -45675], dtype=torch.long)
    print(a)
    a = a + 100000
    b = convert_tensor[a]
    print(b.to(torch.float32) * 2**14)