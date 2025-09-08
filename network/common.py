# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from network.newact1 import *
from typing import Type

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()
        self.act1 = ActNet(
            input_dim=embedding_dim,  
            embed_dim=embedding_dim,  
            num_layers=6,  
            out_dim=mlp_dim,  
            num_freqs=mlp_dim,  
            output_activation=nn.GELU(),  
            op_order='AL',  
            
            freqs_init=lambda t: nn.init.normal_(t, mean=0, std=0.1),
            beta_init=lambda t: nn.init.kaiming_uniform_(t),
            
        )
        self.act2 = ActNet(
            input_dim=mlp_dim,  
            embed_dim=mlp_dim,  
            num_layers=6,  
            out_dim=embedding_dim,  
            num_freqs=mlp_dim,  
            output_activation=nn.GELU(),  
            op_order='AL',  
            
            freqs_init=lambda t: nn.init.normal_(t, mean=0, std=0.1),
            beta_init=lambda t: nn.init.kaiming_uniform_(t),
            
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.act2(self.act1(x)))


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
