from typing import Optional, Tuple, Union
import copy
import logging

import torch
import torch.nn as nn

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.module.relu import relu
from diffusion_policy.module.sinusoidal_posemb import SinusoidalPosEmb
from diffusion_policy.module.trivial_quant import LinearA, LinearB, LinearC, TrivialQuantLinear


logger = logging.getLogger(__name__)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.0, batch_first=True, quant=None):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        assert self.head_dim * num_heads == embed_size, "Embedding size needs to be divisible by heads"

        self.values = LinearA(embed_size, embed_size, bias=False, quant=quant)
        self.keys = LinearA(embed_size, embed_size, bias=False, quant=quant)
        self.queries = LinearA(embed_size, embed_size, bias=False, quant=quant)
        self.fc_out = LinearB(embed_size, embed_size, bias=False, quant=quant)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        queries, q_delta = self.queries(queries, 1, queries.detach().to(torch.float))
        keys, k_delta = self.keys(keys, 1, keys.detach().to(torch.float))
        values, v_delta = self.values(values, 1, values.detach().to(torch.float))

        queries = queries.to(torch.float) * q_delta
        keys = keys.to(torch.float) * k_delta
        values = values.to(torch.float) * v_delta

        batch_size, query_len, embed_size = queries.shape
        key_len = keys.shape[1]
        value_len = values.shape[1]
        queries = queries.reshape(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.reshape(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.reshape(batch_size, value_len, self.num_heads, self.head_dim).transpose(1, 2)

        energy = torch.matmul(queries / (self.head_dim ** 0.5), keys.transpose(-2, -1))
        if attn_mask is not None:
            energy = energy + attn_mask
        attention = self.dropout(self.softmax(energy))
        out = torch.matmul(attention, values)
        out = out.transpose(1, 2).reshape(batch_size, query_len, embed_size)
        return self.fc_out(out, 1, out)


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
        quant=None,
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=True, quant=quant)
        self.multihead_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, batch_first=True, quant=quant)
        self.linear1 = LinearA(d_model, dim_feedforward, quant=quant)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = LinearC(dim_feedforward, d_model, quant=quant)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, elementwise_affine=False)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = relu

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        init=False,
        number=0,
    ) -> torch.Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask)
            x = x + self._ff_block(self.norm3(x), init, number)
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask))
            x = self.norm3(x + self._ff_block(x, init, number))
        return x

    def _sa_block(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.dropout1(self.self_attn(x, x, x, attn_mask=attn_mask))

    def _mha_block(self, x: torch.Tensor, mem: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        return self.dropout2(self.multihead_attn(x, mem, mem, attn_mask=attn_mask))

    def _ff_block(self, x: torch.Tensor, init, number) -> torch.Tensor:
        out1, out_delta1 = self.linear1(x, 1, x.detach().to(torch.float))
        out1 = self.activation(out1)
        x = self.linear2(out1, out_delta1)
        return self.dropout3(x)


class TransformerDecoder(nn.Module):
    __constants__ = ["norm"]

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
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
        for idx, mod in enumerate(self.layers):
            output = mod(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                init=init,
                number=idx,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output

    def _get_clones(self, module, n):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerForActionDiffusionTrivialQAT(ModuleAttrMixin):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int = 800,
        p_drop_attn: float = 0.1,
        quant=None,
    ) -> None:
        super().__init__()

        self.input_emb = LinearB(input_dim, n_emb, quant=quant)
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_pos_emb = nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))

        decoder_layer = TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            norm_first=True,
            quant=quant,
        )
        self.decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=n_layer)

        self.ln_f = nn.LayerNorm(n_emb, elementwise_affine=False)
        self.head = LinearB(n_emb, output_dim, quant=quant)
        self.action_horizon = action_horizon

        self.apply(self._init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            TransformerDecoderLayer,
            TransformerDecoder,
            MultiHeadAttention,
            nn.ModuleList,
            nn.Softmax,
            nn.LayerNorm,
            nn.Mish,
            nn.Sequential,
        )
        if isinstance(module, TrivialQuantLinear):
            if module.weight is not None:
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, TransformerForActionDiffusionTrivialQAT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_pos_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))

    def get_optim_groups(self, weight_decay: float = 1e-3):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (TrivialQuantLinear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn
                if pn.endswith("bias") or pn.startswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        decay = decay & set(param_dict)
        no_decay = no_decay & set(param_dict)
        return [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
    ):
        optimizer = torch.optim.AdamW(self.get_optim_groups(weight_decay=weight_decay), lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: Optional[torch.Tensor] = None,
        init=False,
        **kwargs,
    ):
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)

        cond_emb = torch.cat([cond, time_emb], dim=1)
        cond_emb = cond_emb + self.cond_pos_emb[:, : cond_emb.shape[1], :]

        input_emb = self.input_emb(sample, 1, sample.detach().to(torch.float)).to(torch.float16)
        input_emb = input_emb + self.pos_emb[:, : input_emb.shape[1], :].to(torch.float16)

        x = self.decoder(tgt=input_emb, memory=cond_emb, init=init)
        x = self.ln_f(x)
        x = self.head(x, 1, x.detach())
        return x.to(torch.float)
