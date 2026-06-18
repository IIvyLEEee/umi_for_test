from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.our_module.sinusoidal_posemb import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.our_module.decoder import TransformerDecoder, TransformerDecoderLayer
from diffusion_policy.model.our_module.layernorm import LayerNorm, make_layer_norm
from diffusion_policy.model.our_module.layernorm_768 import LayerNorm as LayerNorm768
from diffusion_policy.model.our_module.softmax import Softmax

from diffusion_policy.model.our_module.linear import Linear
import diffusion_policy.model.our_module.quant_linear_a as qua
import diffusion_policy.model.our_module.quant_linear_b as qub
import diffusion_policy.model.our_module.quant_linear_c as quc

from diffusion_policy.model.our_module.multihead_attn import MultiHeadAttention

logger = logging.getLogger(__name__)

class TransformerForActionDiffusion(ModuleAttrMixin):
    def __init__(self,
        input_dim: int,
        output_dim: int,
        action_horizon: int,
        n_layer: int = 7,
        n_head: int = 8,
        n_emb: int = 768,
        max_cond_tokens: int=800,
        p_drop_attn: float = 0.1,
        ) -> None:
        super().__init__()
        
        # input embedding stem
        self.input_emb = qub.Linear(input_dim, n_emb)
        # self.input_emb = Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.randn((1, action_horizon, n_emb)))
        self.time_emb = SinusoidalPosEmb(n_emb)
        # learnable position embedding
        self.cond_pos_emb =  nn.Parameter(torch.randn((1, max_cond_tokens, n_emb)))
        
        # decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            norm_first=True, # important for stability
        )
        self.decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_layer
        )
        
        # decoder head
        self.ln_f = make_layer_norm(n_emb, elementwise_affine=False)
        self.head = qub.Linear(n_emb, output_dim)
        # self.head = Linear(n_emb, output_dim)
        
        self.action_horizon = action_horizon
        
        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            TransformerDecoderLayer,
            TransformerDecoder,
            MultiHeadAttention,
            nn.ModuleList,
            # ReLU,
            Softmax,
            LayerNorm,
            LayerNorm768,
            nn.LayerNorm,
            nn.Mish,
            nn.Sequential,
            nn.Softmax,
        )
        if isinstance(module, qua.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, quc.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, qub.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, TransformerForActionDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            print(type(module))
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            Linear,
            qua.Linear,
            qub.Linear,
            quc.Linear,
            torch.nn.Linear,
            torch.nn.MultiheadAttention,
            MultiHeadAttention,
            # qu_attn.MultiHeadAttention,
        )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        # assert (
        #     len(inter_params) == 0
        # ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        # assert (
        #     len(param_dict.keys() - union_params) == 0
        # ), "parameters %s were not separated into either decay/no_decay set!" % (
        #     str(param_dict.keys() - union_params),
        # )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, init=False, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,N,n_emb)
        output: (B,T,input_dim)
        """
        
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)
        
        # 2. process conditions
        cond_emb = torch.cat([cond, time_emb], dim=1)
        tc = cond_emb.shape[1]
        cond_pos_emb = self.cond_pos_emb[
            :, :tc, :
        ] 
        cond_emb = cond_emb + cond_pos_emb
        
        # 3. process input
        # input_emb = self.input_emb(sample, 1, sample.to(torch.float), 8).to(torch.float16)
        input_emb = self.input_emb(sample, 1, sample.clone().detach().to(torch.float)) #.to(torch.float16)
        # input_emb = self.input_emb(sample, init)

        t = input_emb.shape[1]
        pos_emb = self.pos_emb[:, :t, :] #.to(torch.float16) 
        input_emb = input_emb + pos_emb

        # print("input_emb:", input_emb.max())
        
        # 4. transformer
        x = self.decoder(
            tgt=input_emb,
            memory=cond_emb,
            init=init
        )        
        x = self.ln_f(x)

        # x = self.head(x, 16, x.clone().detach() * 16)
        x = self.head(x, 1, x.clone().detach())
        # x = self.head(x)
        # (B, T, n_out)
        # exit()
        return x.to(torch.float)
        
