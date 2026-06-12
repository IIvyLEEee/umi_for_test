import torch
import diffusion_policy.model.our_module.tofp16 as tofp16
import diffusion_policy.model.our_module.toint8 as toint8  
# import tofp16
# import toint8  

"""
Layer:  Linear layer for float16 input and float16 output
Author: cxz21
Data:   2025/07/13
"""
class LinearFunc(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        quant_weight: torch.Tensor,
        pre_parameter: torch.Tensor,
        input_delta: torch.Tensor,
        parameter: torch.Tensor,
        scaling_factor: torch.Tensor,
        convert_fp16=None,
        convert_int8=None,
        weight_delta=None
    ):
        # torch.save(pre_parameter, "./data/scaling_factor/input_emb.pre_parameter.ckpt")
        # torch.save(quant_weight, "./data/scaling_factor/input_emb.weight.ckpt")
        # torch.save(scaling_factor, "./data/scaling_factor/input_emb.scaling_factor.ckpt")

        quant_input = convert_int8(input.to(torch.float16) * pre_parameter).to(input.device)
        quant_output = quant_input.to(torch.float) @ quant_weight.to(torch.float).transpose(0, 1)
        quant_output = quant_output.to(torch.float) * 2
        quant_output = quant_output.round().to(torch.int32)
        
        quant_output = convert_fp16(quant_output).to(input.device)

        output = quant_output * scaling_factor

        ctx.save_for_backward(
            quant_input, quant_weight, input_delta, parameter, weight_delta
        )
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        quant_input, quant_weight, input_delta, parameter, weight_delta = ctx.saved_tensors
        dequant_input = quant_input * input_delta
        dequant_weight = quant_weight * (2.0 ** (-weight_delta + 1)) * parameter
        dequant_grad_output = grad_output.to(torch.float)

        grad_input = dequant_grad_output @ dequant_weight
        grad_weight = dequant_grad_output.transpose(-2, -1) @ dequant_input
        return grad_input, grad_weight, None, None, None, None, None, None, None, None


class Linear(torch.nn.Module):
    def __init__(self, in_feature: int, out_feature: int, bias: bool = False):
        super().__init__()
        self.init = True

        # fp16-int8 convertor
        self.convert_fp16 = tofp16.Convert()
        self.convert_int8 = toint8.Convert_int8()

        # weight init
        self.weight = torch.nn.Parameter(torch.empty((out_feature, in_feature)))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_feature))
        else:
            self.register_parameter("bias", None)

        # scaling factor init
        self.quant_weight = None
        self.weight_delta = torch.nn.Parameter(torch.empty((out_feature,1)), requires_grad=False)
        self.input_delta = torch.nn.Parameter(torch.full([], 0.0), requires_grad=False)
        self.parameter = torch.nn.Parameter(torch.full([], 0.0), requires_grad=False)

    def forward(self, input: torch.Tensor, parameter: torch.Tensor, dequant_input = None, out_parameter = None):
        if self.training is True:
            self.init = True
            self.weight_delta.data = toint8.fp4_init_scale(self.weight, channel_wise=True)
            self.quant_weight = toint8.fp4_quantizer(self.weight, self.weight_delta).detach().requires_grad_(False)
            self.input_delta.data = toint8.int8_init_scale(dequant_input) #* 2

            self.scaling_factor =  (self.input_delta * (2.0 ** (-self.weight_delta + 1)) * 2**13)
            self.scaling_factor = self.scaling_factor.to(torch.float16).transpose(0, 1)
            self.pre_parameter = (parameter / self.input_delta).to(torch.float16)
            self.parameter.data = self.pre_parameter * self.input_delta
        elif self.init is True and self.input_delta.data == 0:
            self.init = False
            self.weight_delta.data = toint8.fp4_init_scale(self.weight, channel_wise=True)
            self.quant_weight = toint8.fp4_quantizer(self.weight, self.weight_delta).detach().requires_grad_(False)
            self.input_delta.data = toint8.int8_init_scale(dequant_input) #* 2

            if out_parameter is not None:
                self.scaling_factor = self.input_delta * (2.0 ** (-self.weight_delta + 1)) * 2**13 / out_parameter
            else:
                self.scaling_factor = self.input_delta * (2.0 ** (-self.weight_delta + 1)) * 2**13
            self.scaling_factor = self.scaling_factor.to(torch.float16).transpose(0, 1)
            self.pre_parameter = (parameter / self.input_delta).to(torch.float16)
            self.parameter.data = self.pre_parameter * self.input_delta
        elif self.init is True:
            self.init = False
            self.weight_delta.data = toint8.fp4_init_scale(self.weight, channel_wise=True)
            self.quant_weight = toint8.fp4_quantizer(self.weight, self.weight_delta).detach().requires_grad_(False)
            if out_parameter is not None:
                self.scaling_factor = self.input_delta * (2.0 ** (-self.weight_delta + 1)) * 2**13 / out_parameter
            else:
                self.scaling_factor = self.input_delta * (2.0 ** (-self.weight_delta + 1)) * 2**13
            self.scaling_factor = self.scaling_factor.to(torch.float16).transpose(0, 1)
            self.pre_parameter = (parameter / self.input_delta).to(torch.float16)
            self.parameter.data = self.pre_parameter * self.input_delta
        
        out = LinearFunc.apply(
            input, 
            self.weight,
            self.quant_weight,
            self.pre_parameter, 
            self.input_delta,
            self.parameter,
            self.scaling_factor,
            self.convert_fp16.convert,
            self.convert_int8.convert,
            self.weight_delta
        )
        # quant_weight = (toint8.fp4_quantizer(self.weight, self.weight_delta) - self.weight / (2.0 ** (-self.weight_delta + 1))).detach() + self.weight / (2.0 ** (-self.weight_delta + 1))
        # x1 = input.to(torch.float16) * self.pre_parameter
        # quant_input = (self.convert_int8.convert(x1).to("cuda:2") - x1.to(torch.float)).detach() + x1.to(torch.float)

        # quant_output = quant_input.to(torch.float) @ quant_weight.to(torch.float).transpose(0, 1)
        # quant_output = quant_output.to(torch.float) * 2
        # def round_fp16(x:torch.Tensor):
        #     x = x.round().to(torch.int32)
        #     x = self.convert_fp16.convert(x).to("cuda:2")
        #     return x
        # quant_output = (round_fp16(quant_output) - quant_output / 2**14).detach() + quant_output / 2**14

        # out = quant_output * self.scaling_factor

        return out.to(torch.float16)

if __name__ == "__main__":
    input_features = 10
    output_features = 5
    batch_size = 3
    # torch.manual_seed(0)

    input = torch.randn(batch_size, input_features, requires_grad=True).to("cuda:2")
    x1 = input.clone().detach().requires_grad_(True)
    x2 = input.clone().detach().requires_grad_(True)
    custom_linear = Linear(input_features, output_features)
    torch_linear = torch.nn.Linear(input_features, output_features)

    weight = torch.randn(output_features, input_features, requires_grad=True).to("cuda:2")
    custom_linear.weight.data = weight.clone()
    torch_linear.weight.data = weight.clone()
    custom_linear.weight_delta.data = toint8.fp4_init_scale(weight.clone(), True).detach()
    custom_output = custom_linear(x1, 16, x1.detach().clone()*16)
    # custom_output,  next_delta = custom_linear(x1, 1, x1)
    x3 = x2*16
    torch_output = torch_linear(x3)

    print(custom_output)
    print(torch_output)

    custom_output.sum().backward()
    torch_output.sum().backward()

    print(custom_linear.weight.grad[0])
    print(torch_linear.weight.grad[0])

    print(x1.grad[0])
    print(x2.grad[0])