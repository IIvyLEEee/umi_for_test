import copy
from typing import Optional

import torch
import torch.nn as nn
from diffusion_policy.model.our_module.layernorm import make_layer_norm
import diffusion_policy.model.our_module.quant_linear_a as qu
import diffusion_policy.model.our_module.quant_linear_c as fe

from diffusion_policy.model.our_module.multihead_attn import MultiHeadAttention
# from diffusion_policy.model.our_module.linear import Linear
from diffusion_policy.model.our_module.relu import relu
from torch.nn.modules.container import ModuleList

class TransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.multihead_attn = MultiHeadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        # Implementation of Feedforward model
        self.linear1 = qu.Linear(d_model, dim_feedforward)
        # self.linear1 = Linear(d_model, 1024)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = fe.Linear(dim_feedforward, d_model)
        # self.linear2 = Linear(1024, d_model)
        

        self.norm_first = norm_first
        self.norm1 = make_layer_norm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.norm2 = make_layer_norm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.norm3 = make_layer_norm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = relu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        init=False,
        number=0,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            norm1 = self.norm1(x)
            # norm1 = norm1 * 16
            x = x + self._sa_block(norm1)

            # print("self_attn:", x.max())

            norm2 = self.norm2(x)

            # print("norm2:", norm2.max())
            # print("memory:", memory.max())
            # norm2 = norm2 * 16
            x = x + self._mha_block(norm2, memory)

            norm3 = self.norm3(x)
            # norm3 = norm3 * 16
            x = x + self._ff_block(norm3, init, number)
        else:
            x = self.norm1(x + self._sa_block(x))
            x = self.norm2(x + self._mha_block(x, memory))
            x = self.norm3(x + self._ff_block(x, init, number))

        return x

    # self-attention block
    def _sa_block(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        # print("self_attn:")
        x = self.self_attn(x, x, x)
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self, x: torch.Tensor, mem: torch.Tensor
    ) -> torch.Tensor:
        # print("multihead_attn:")
        x = self.multihead_attn(x, mem, mem)
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: torch.Tensor, init, number) -> torch.Tensor:
        # out1, out_delta1 = self.linear1(x, 16, x.clone().detach().to(torch.float) * 16)
        out1, out_delta1 = self.linear1(x, 1, x.clone().detach().to(torch.float))
        # out1 = self.linear1(x)

        out1 = self.activation(out1)

        # out1 = out1 * out_delta1
        x = self.linear2(out1, out_delta1)
        # x = self.linear2(out1)

        return self.dropout3(x)

class TransformerDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = self._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        init=False,
    ) -> torch.Tensor:
        output = tgt
        a = 0
        for mod in self.layers:
            output = mod(
                output,
                memory,
                init=init,
                number=a,
            )
            a += 1

        if self.norm is not None:
            output = self.norm(output)

        return output

    def _get_clones(self, module, N):
        return ModuleList([copy.deepcopy(module) for _ in range(N)])


if __name__ == "__main__":
    # parameter
    n_emb = 256
    n_head = 4
    n_layer = 5
    dropout_p = 0.0
    decoder_layer = TransformerDecoderLayer(
        d_model=n_emb,
        nhead=n_head,
        dim_feedforward=4 * n_emb,
        dropout=dropout_p,
        activation="relu",
        norm_first=True,
        batch_first=True,
    )
    fake_layer = TransformerDecoderLayer_(
        d_model=n_emb,
        nhead=n_head,
        dim_feedforward=4 * n_emb,
        dropout=dropout_p,
        norm_first=True,
    )
    in_proj_weight1 = nn.Parameter(torch.randn(3 * n_emb, n_emb, requires_grad=True))
    out_proj_weight1 = nn.Parameter(torch.randn(n_emb, n_emb, requires_grad=True))
    in_proj_weight2 = nn.Parameter(torch.randn(3 * n_emb, n_emb, requires_grad=True))
    out_proj_weight2 = nn.Parameter(torch.randn(n_emb, n_emb, requires_grad=True))
    linear1_weight = nn.Parameter(torch.randn(4 * n_emb, n_emb))
    linear2_weight = nn.Parameter(torch.randn(n_emb, 4 * n_emb))
    # decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)

    # input
    tgt = torch.randn(56, 10, 256, requires_grad=True)
    memory = torch.randn(56, 3, 256, requires_grad=True)
    fake_tgt = tgt
    fake_memory = memory

    T = 10
    S = 3
    sz = T
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    t, s = torch.meshgrid(torch.arange(T), torch.arange(S), indexing="ij")
    memory_mask = t >= (s - 1)
    memory_mask = (
        memory_mask.float()
        .masked_fill(memory_mask == 0, float("-inf"))
        .masked_fill(memory_mask == 1, float(0.0))
    )
    print(mask)
    print(memory_mask)
    decoder_layer.linear1.weight = linear1_weight
    fake_layer.linear1.weight = linear1_weight
    decoder_layer.linear2.weight = linear2_weight
    fake_layer.linear2.weight = linear2_weight
    decoder_layer.self_attn.in_proj_weight = nn.Parameter(in_proj_weight1)
    fake_layer.self_attn.queries.weight = nn.Parameter(in_proj_weight1[0:n_emb, :])
    fake_layer.self_attn.keys.weight = nn.Parameter(
        in_proj_weight1[n_emb : 2 * n_emb, :]
    )
    fake_layer.self_attn.values.weight = nn.Parameter(
        in_proj_weight1[2 * n_emb : 3 * n_emb, :]
    )
    decoder_layer.multihead_attn.in_proj_weight = nn.Parameter(in_proj_weight2)
    fake_layer.multihead_attn.queries.weight = nn.Parameter(in_proj_weight2[0:n_emb, :])
    fake_layer.multihead_attn.keys.weight = nn.Parameter(
        in_proj_weight2[n_emb : 2 * n_emb, :]
    )
    fake_layer.multihead_attn.values.weight = nn.Parameter(
        in_proj_weight2[2 * n_emb : 3 * n_emb, :]
    )
    decoder_layer.self_attn.out_proj.weight = nn.Parameter(out_proj_weight1)
    fake_layer.self_attn.fc_out.weight = nn.Parameter(out_proj_weight1)
    decoder_layer.multihead_attn.out_proj.weight = nn.Parameter(out_proj_weight2)
    fake_layer.multihead_attn.fc_out.weight = nn.Parameter(out_proj_weight2)
    out = decoder_layer(0, tgt, memory, mask, memory_mask)
    dout = torch.randn_like(out)
    loss = (out * dout).sum()
    loss.backward()
    fake_out = fake_layer(fake_tgt, fake_memory, mask, memory_mask)
    fake_loss = (fake_out * dout).sum()
    fake_loss.backward()

    print(out)
    print(fake_out)
    print(out - fake_out)

    print(tgt.grad)
    print(memory.grad)
    print(fake_tgt.grad)
    print(fake_memory.grad)
    print(tgt.grad - fake_tgt.grad)
    print(memory.grad - fake_memory.grad)
