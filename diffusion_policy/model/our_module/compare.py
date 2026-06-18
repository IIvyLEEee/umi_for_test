import argparse
import math

import torch
import torch.nn as nn

from diffusion_policy.model.our_module.decoder import TransformerDecoderLayer
from diffusion_policy.model.our_module.layernorm import make_layer_norm
from diffusion_policy.model.our_module.linear import Linear
from diffusion_policy.model.our_module.multihead_attn import (
    MultiHeadAttentionFunc1,
    MultiHeadAttentionFunc2,
)
from diffusion_policy.model.our_module.relu import relu
from diffusion_policy.model.our_module.softmax import Softmax


def float_linear(module, input):
    return module(input.to(module.weight.dtype))


class NaiveMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.values = Linear(embed_size, embed_size, bias=False)
        self.keys = Linear(embed_size, embed_size, bias=False)
        self.queries = Linear(embed_size, embed_size, bias=False)
        self.fc_out = Linear(embed_size, embed_size, bias=False)
        self.softmax = Softmax()

    def forward(self, queries, keys, values):
        queries = float_linear(self.queries, queries)
        keys = float_linear(self.keys, keys)
        values = float_linear(self.values, values)
        qk = MultiHeadAttentionFunc1.apply(queries, keys, self.num_heads, None)
        qk = self.softmax(qk).to(torch.float)
        qkv = MultiHeadAttentionFunc2.apply(qk, values, self.num_heads)
        return float_linear(self.fc_out, qkv)


class NaiveTransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward):
        super().__init__()
        self.self_attn = NaiveMultiHeadAttention(d_model, nhead)
        self.multihead_attn = NaiveMultiHeadAttention(d_model, nhead)
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        self.norm1 = make_layer_norm(d_model, elementwise_affine=False)
        self.norm2 = make_layer_norm(d_model, elementwise_affine=False)
        self.norm3 = make_layer_norm(d_model, elementwise_affine=False)

    def forward_with_stages(self, tgt, memory):
        stages = {}
        stages["norm1"] = self.norm1(tgt)
        stages["self_attn"] = self.self_attn(
            stages["norm1"], stages["norm1"], stages["norm1"]
        )
        stages["residual1"] = tgt + stages["self_attn"]
        stages["norm2"] = self.norm2(stages["residual1"])
        stages["cross_attn"] = self.multihead_attn(
            stages["norm2"], memory, memory
        )
        stages["residual2"] = stages["residual1"] + stages["cross_attn"]
        stages["norm3"] = self.norm3(stages["residual2"])
        stages["ff"] = float_linear(
            self.linear2, relu(float_linear(self.linear1, stages["norm3"]))
        )
        stages["output"] = stages["residual2"] + stages["ff"]
        return stages


def quant_forward_with_stages(layer, tgt, memory):
    stages = {}
    stages["norm1"] = layer.norm1(tgt)
    stages["self_attn"] = layer._sa_block(stages["norm1"])
    stages["residual1"] = tgt + stages["self_attn"]
    stages["norm2"] = layer.norm2(stages["residual1"])
    stages["cross_attn"] = layer._mha_block(stages["norm2"], memory)
    stages["residual2"] = stages["residual1"] + stages["cross_attn"]
    stages["norm3"] = layer.norm3(stages["residual2"])
    stages["ff"] = layer._ff_block(stages["norm3"], False, 0)
    stages["output"] = stages["residual2"] + stages["ff"]
    return stages


def copy_weights(quant_layer, naive_layer):
    pairs = [
        (quant_layer.self_attn.queries, naive_layer.self_attn.queries),
        (quant_layer.self_attn.keys, naive_layer.self_attn.keys),
        (quant_layer.self_attn.values, naive_layer.self_attn.values),
        (quant_layer.self_attn.fc_out, naive_layer.self_attn.fc_out),
        (quant_layer.multihead_attn.queries, naive_layer.multihead_attn.queries),
        (quant_layer.multihead_attn.keys, naive_layer.multihead_attn.keys),
        (quant_layer.multihead_attn.values, naive_layer.multihead_attn.values),
        (quant_layer.multihead_attn.fc_out, naive_layer.multihead_attn.fc_out),
        (quant_layer.linear1, naive_layer.linear1),
        (quant_layer.linear2, naive_layer.linear2),
    ]
    for quant_linear, naive_linear in pairs:
        naive_linear.weight.data.copy_(quant_linear.weight.data)


def metrics(quant, naive):
    quant = quant.detach().float()
    naive = naive.detach().float()
    diff = quant - naive
    naive_l2 = torch.linalg.vector_norm(naive)
    cosine = torch.nn.functional.cosine_similarity(
        quant.reshape(1, -1), naive.reshape(1, -1)
    )
    return {
        "mae": diff.abs().mean().item(),
        "rmse": torch.sqrt(torch.mean(diff.square())).item(),
        "max_abs": diff.abs().max().item(),
        "relative_l2": (
            torch.linalg.vector_norm(diff) / naive_l2.clamp_min(1e-12)
        ).item(),
        "cosine": cosine.item(),
        "finite": bool(torch.isfinite(quant).all()),
    }


def run_case(d_model, nhead, seed):
    torch.manual_seed(seed)
    dim_feedforward = 4 * d_model
    quant_layer = TransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        dropout=0.0,
        norm_first=True,
    ).eval()
    naive_layer = NaiveTransformerDecoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=dim_feedforward,
    ).eval()

    for module in quant_layer.modules():
        if hasattr(module, "weight") and isinstance(module.weight, nn.Parameter):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    copy_weights(quant_layer, naive_layer)

    tgt = torch.randn(1, 16, d_model) * 0.2
    memory = torch.randn(1, 3, d_model) * 0.2

    with torch.no_grad():
        quant_layer(tgt, memory)
        quant_stages = quant_forward_with_stages(quant_layer, tgt, memory)
        naive_stages = naive_layer.forward_with_stages(tgt, memory)

    print(
        f"\nTransformerDecoderLayer d_model={d_model}, heads={nhead}, "
        f"ffn={dim_feedforward}"
    )
    print(
        f"{'stage':<12} {'mae':>11} {'rmse':>11} {'max_abs':>11} "
        f"{'rel_l2':>11} {'cosine':>11} {'finite':>8}"
    )
    for name in quant_stages:
        result = metrics(quant_stages[name], naive_stages[name])
        print(
            f"{name:<12} {result['mae']:11.4e} {result['rmse']:11.4e} "
            f"{result['max_abs']:11.4e} {result['relative_l2']:11.4e} "
            f"{result['cosine']:11.6f} {str(result['finite']):>8}"
        )

    direct_output = quant_layer(tgt, memory)
    direct_error = (direct_output.float() - quant_stages["output"].float()).abs().max()
    if not math.isclose(direct_error.item(), 0.0, abs_tol=1e-6):
        raise RuntimeError(f"Staged quant forward differs from layer forward: {direct_error}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", choices=("256", "768", "all"), default="all")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    cases = {
        "256": (256, 4),
        "768": (768, 8),
    }
    selected = cases if args.width == "all" else {args.width: cases[args.width]}
    for d_model, nhead in selected.values():
        run_case(d_model, nhead, args.seed)


if __name__ == "__main__":
    main()
