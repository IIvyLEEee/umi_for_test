import torch
import torch.nn as nn

"""
Layer:  LayerNorm
Author: shuyuan-19
Data:   2024/10/30
"""

class LayerNormFunc_sim(torch.autograd.Function):
    """
    Custom autograd function for LayerNorm in BF16 precision.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.tensor,
        eps=1e-5,
    ):

        input = input #.to(torch.float16)
        _, _, C = input.size()
        mean = torch.mean(input, dim=-1, keepdim=True)
        var = torch.var(input, dim=-1, keepdim=True) / C * (C - 1)
        std = torch.sqrt(var + eps)
        normlized_x = (input - mean) / std
        output = normlized_x

        # for backward
        ctx.save_for_backward(input, mean, std)
        return output


    @staticmethod
    def backward(ctx, dout): 
        input, mean, std = ctx.saved_tensors
        dout = dout  

        norm = (input - mean) / std
        dnorm = dout

        d_input = (
            dnorm
            - dnorm.mean(dim=-1, keepdim=True)
            - norm * (dnorm * norm).mean(dim=-1, keepdim=True)
        )
        d_input = d_input / std
        return d_input, None


class LayerNorm(torch.nn.Module):
    add_result = None

    def __init__(
        self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True
    ):
        super(LayerNorm, self).__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.delta = torch.tensor([0.00390625], dtype=torch.float16).requires_grad_(False)
        self.weight = None
        self.bias = None

        if LayerNorm.add_result is None:
            LayerNorm.add_result = torch.load("/home/acts00/Desktop/umi_for_train/add_result.ckpt", map_location=self.delta.device)

        self.threshold = torch.tensor([130.0], dtype=torch.float16)

    def sum_tree(self, x:torch.Tensor):
        # x = x.reshape(..., 16, 2).sum(dim=-1)
        # x = x.reshape(..., 8, 2).sum(dim=-1)
        # x = x.reshape(..., 4, 2).sum(dim=-1)
        # x = x.reshape(..., 2, 2).sum(dim=-1)
        # x = x.sum(dim=-1)
        B, _, _, _ = x.shape
        x = self.array_sum(x.reshape(B, 16, 8, 16, 2))
        x = self.array_sum(x.reshape(B, 16, 8, 8, 2))
        x = self.array_sum(x.reshape(B, 16, 8, 4, 2))
        x = self.array_sum(x.reshape(B, 16, 8, 2, 2))
        x = self.array_sum(x)
        return x

    def alternative_sum(self, x1: torch.Tensor, x2: torch.Tensor):
        int16_tensor1 = x1.clone().view(torch.short)
        int16_tensor1 = int16_tensor1.to(torch.long)
        int16_tensor1 = torch.where(int16_tensor1 >= 0, int16_tensor1 + 22545, int16_tensor1 + 32768)

        int16_tensor2 = x2.clone().view(torch.short)
        int16_tensor2 = int16_tensor2.to(torch.long)
        int16_tensor2 = torch.where(int16_tensor2 >= 0, int16_tensor2 + 22545, int16_tensor2 + 32768)

        self.add_result = self.add_result.to(x1.device)
        output = LayerNorm.add_result[int16_tensor1, int16_tensor2]
        return output

    def array_sum(self, x: torch.Tensor):
        self.threshold = self.threshold.to(x.device)
        x = x.to(torch.float16)
        first_element = x[..., 0]
        second_element = x[..., 1]
        
        condition = (first_element.abs() <= self.threshold) & (second_element.abs() <= self.threshold)

        result = torch.zeros_like(first_element)
        first_element_ = torch.where(condition, first_element, result)
        second_element_ = torch.where(condition, second_element, result)
        result = torch.where(condition, self.alternative_sum(first_element_, second_element_), first_element + second_element)
        return result

    def forward(self, input: torch.Tensor):
        return LayerNormFunc_sim.apply(input, self.eps)

        input = input.to(torch.float16) 
        
        B, _, _ = input.shape
        # sum_r = torch.zeros([B, 16, 1], dtype=torch.float16).to(input.device)
        # sum = input.reshape(B, 16, 8, 32).sum(dim=-1)
        # sum = self.sum_tree(input.reshape(B, 16, 8, 32))
        # for i in range(8):
        #     sum_r = sum_r + sum[:, :, i:i+1]
        sum_r = input.sum(dim=-1, keepdim=True)
        mean = sum_r * self.delta.to(input.device)

        minus = (input - mean)

        sq = torch.square(minus)

        # var = torch.zeros([B, 16, 1], dtype=torch.float16).to(input.device)
        # sum = sq.reshape(B, 16, 8, 32).sum(dim=-1)
        # sum = self.sum_tree(sq.reshape(B, 16, 8, 32))
        # for i in range(8):
        #     var = var + sum[:, :, i:i+1]
        var = sq.sum(dim=-1, keepdim=True)
        
        std = 1 / (var + 1e-5)
        
        std = torch.sqrt(std)
        
        normlized_x = minus * std * 16

        return normlized_x.to(torch.float16)


if __name__ == "__main__":
    # create a small dummy example and check w.r.t PyTorch backward
    torch.set_printoptions(precision=15)
    B = 2
    T = 16
    C = 256
    # torch.manual_seed(41)
    # x = torch.randn(B, T, C, requires_grad=True, device="cuda:2")
    # y = x.clone().detach().requires_grad_(True) 
    
    # fake_layer = LayerNorm(C).to("cuda:2")
    # out = fake_layer(x)

    # layer = nn.LayerNorm(C).to("cuda:2")
    # outy = layer(y)

    # dout = torch.randn(B, T, C).to("cuda:2")

    # fakeloss = (out * dout).sum()
    # fakeloss.backward()
    # loss = (outy * dout).sum()
    # loss.backward()
    
    # print("out: ", out)
    # print("outy: ", outy)
    # print((out - outy).abs().max())

    
    # print("dx: ", x.grad)
    # print("dy: ", y.grad)
    # print((x.grad - y.grad).abs().max())


    fake_layer = LayerNorm(C).to("cuda:2")
    # x = torch.randn(B, 2, dtype = torch.float16, requires_grad=True, device="cuda:2")[1] / 10000
    # y1 = x.sum(dim=-1)
    # y2 = fake_layer.array_sum(x)
    # print(x)
    # print(y1)
    # print(y2)
    # print((y1 - y2).abs().max())

    x = torch.tensor([0.005161285400390625, 0.000223636627197265625], dtype = torch.float16, device="cuda:2")
    print(fake_layer.array_sum(x))
