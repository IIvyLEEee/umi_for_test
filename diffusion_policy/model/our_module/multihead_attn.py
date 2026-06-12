import torch
import torch.nn as nn

# import torch.nn.functional as F
from diffusion_policy.model.our_module.linear import Linear
import diffusion_policy.model.our_module.quant_linear_a as qu
from diffusion_policy.model.our_module.softmax import Softmax
import diffusion_policy.model.our_module.quant_linear_b as fe
from torch.autograd import Function
from torch.nn.modules.activation import MultiheadAttention


class MultiHeadAttentionFunc1(Function):
    @staticmethod
    def forward(ctx, queries, keys, num_heads, atten_mask):
        _, key_len, _ = keys.shape
        N, query_len, embed_size = queries.shape

        head_dim = embed_size // num_heads
        keys = keys.reshape(N, key_len, num_heads, head_dim)
        queries = queries.reshape(N, query_len, num_heads, head_dim) / (
            head_dim ** (1 / 2)
        )
        energy = torch.einsum("nqhd,nlhd->nhql", [queries, keys])
        # atten_mask = atten_mask.unsqueeze(0).unsqueeze(0)
        if atten_mask is not None:
            energy = energy + atten_mask
        ctx.save_for_backward(queries, keys, atten_mask)
        ctx.embed_size = embed_size
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        # print(grad_output)
        queries, keys, _ = ctx.saved_tensors
        embed_size = ctx.embed_size
        N, num_heads, query_len, keys_len = grad_output.shape
        head_dim = embed_size // num_heads
        grad_attention = grad_output / (head_dim ** (1 / 2))
        grad_keys = torch.einsum("nhql,nqhd->nlhd", [grad_attention, queries])
        grad_keys = grad_keys.reshape(N, keys_len, embed_size)

        grad_queries = torch.einsum("nhql,nlhd->nqhd", [grad_attention, keys])
        grad_queries = grad_queries.reshape(N, query_len, embed_size)
        return grad_queries, grad_keys, None, None


class MultiHeadAttentionFunc2(Function):
    @staticmethod
    def forward(ctx, qk, values, num_heads):
        N, _, query_len, _ = qk.shape
        _, value_len, embed_size = values.shape

        head_dim = embed_size // num_heads
        values = values.reshape(N, value_len, num_heads, head_dim)
        out = torch.einsum("nhql,nlhd->nqhd", [qk, values]).reshape(
            N, query_len, embed_size
        )
        ctx.save_for_backward(qk, values)
        ctx.head_dim = head_dim
        ctx.num_heads = num_heads
        return out

    @staticmethod
    def backward(ctx, grad_output):
        qk, values = ctx.saved_tensors
        N, query_len, embed_size = grad_output.shape
        head_dim = ctx.head_dim
        num_heads = ctx.num_heads
        grad_output = grad_output.reshape(N, query_len, num_heads, head_dim)
        grad_qk = torch.einsum("nqhd,nlhd->nhql", [grad_output, values])
        grad_values = torch.einsum("nqhd,nhql->nlhd", [grad_output, qk])
        grad_values = grad_values.reshape(N, -1, embed_size)

        return grad_qk, grad_values, None


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0, batch_first=True):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        assert (
            self.head_dim * num_heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # self.values = qu.Linear(embed_size, embed_size, bias=False)
        # self.keys = qu.Linear(embed_size, embed_size, bias=False)
        # self.queries = qu.Linear(embed_size, embed_size, bias=False)
        # self.fc_out = fe.Linear(embed_size, embed_size, bias=False)
        self.values = Linear(embed_size, embed_size, bias=False)
        self.keys = Linear(embed_size, embed_size, bias=False)
        self.queries = Linear(embed_size, embed_size, bias=False)
        self.fc_out = Linear(embed_size, embed_size, bias=False)
        self.softmax = nn.Softmax(dim=3)
        # self.softmax = Softmax()
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        # queries, q_delta = self.queries(queries, 1, queries.clone().detach().to(torch.float)) 
        # keys, k_delta = self.keys(keys, 1, keys.clone().detach().to(torch.float))
        # values, v_delta = self.values(values, 1, values.clone().detach().to(torch.float))

        # queries = queries.to(torch.float) * q_delta
        # keys = keys.to(torch.float) * k_delta
        # values = values.to(torch.float) * v_delta
        queries = self.queries(queries)
        keys = self.keys(keys)
        values = self.values(values)

        qk = MultiHeadAttentionFunc1.apply(queries, keys, self.num_heads, attn_mask)
        # print("qk:", qk.max())
        qk_softmax = self.softmax(qk).to(torch.float)
        # print("softmax:", qk_softmax.max())
        
        qk_dropout = self.dropout(qk_softmax)
        qkv = MultiHeadAttentionFunc2.apply(qk_dropout, values, self.num_heads)
        out = self.fc_out(qkv, 1, qkv)
        # out = self.fc_out(qkv)
        return out


if __name__ == "__main__":
    embed_size = 256
    num_heads = 4
    dropout_p = 0.0
    batch_size = 56
    sequence_length = 10

    x1 = torch.randn(batch_size, sequence_length, embed_size, requires_grad=True)
    y1 = x1
    x2 = torch.randn(batch_size, sequence_length, embed_size, requires_grad=True)
    y2 = x2
    x3 = torch.randn(batch_size, sequence_length, embed_size, requires_grad=True)
    y3 = x3

    in_proj_weight = nn.Parameter(
        torch.randn(3 * embed_size, embed_size, requires_grad=True)
    )
    out_proj_weight = nn.Parameter(
        torch.randn(embed_size, embed_size, requires_grad=True)
    )

    atten_mask = torch.randn(sequence_length, sequence_length)

    true_layer = MultiheadAttention(embed_size, num_heads, dropout_p, False)
    true_layer.in_proj_weight = nn.Parameter(in_proj_weight)
    true_layer.out_proj.weight = nn.Parameter(out_proj_weight)
    out = true_layer(x1, x2, x3, atten_mask)[0]
    dout = torch.randn_like(out)
    loss = (out * dout).sum()
    loss.backward()

    attn_layer = MultiHeadAttention(
        embed_size=embed_size, num_heads=num_heads, dropout_p=dropout_p
    )
    attn_layer.queries.weight = nn.Parameter(in_proj_weight[0:embed_size, :])
    attn_layer.keys.weight = nn.Parameter(
        in_proj_weight[embed_size : 2 * embed_size, :]
    )
    attn_layer.values.weight = nn.Parameter(
        in_proj_weight[2 * embed_size : 3 * embed_size, :]
    )
    attn_layer.fc_out.weight = nn.Parameter(out_proj_weight)
    attn_layer.queries.bias = None
    attn_layer.keys.bias = None
    attn_layer.values.bias = None
    attn_layer.fc_out.bias = None
    fake_out = attn_layer(y1, y2, y3, atten_mask)

    # print("out:\n", out)
    # print("fake_out:\n", fake_out)
    print(out - fake_out)
