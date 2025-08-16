
from typing import Tuple, List

import numpy as np
from numpy import ndarray

import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F

from mtorch.util.nnModuleFactory import factory_LinearReLU_Sequential

class BaseQnetwork(nn.Module):
    '''
    作成したQ networkが共通して持つ属性
    '''
    def __init__(self):
        super().__init__()
    
    def callback_everyUpdate(self):
        pass

class Qnetwork(BaseQnetwork):
    def __init__(self, in_chnls: int, hdn_chnls: Tuple[int, ...], out_chnls: int):
        super().__init__()
        
        self._network = factory_LinearReLU_Sequential(in_chnls, hdn_chnls, out_chnls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network.forward(x)
    
class DuelingQnetwork(BaseQnetwork):
    def __init__(self, in_chnls: int, hdn_chnls: List[int], out_chnls: int):
        super().__init__()

        in_chnls_forFactory = in_chnls
        hdn_chnls_forFactory = tuple(hdn_chnls[:-1])
        out_chnls_forFactory = hdn_chnls[-1]

        self._comm_network = factory_LinearReLU_Sequential(in_chnls_forFactory, hdn_chnls_forFactory, out_chnls, out_act="ReLU")

        self._base_layer = nn.Linear(out_chnls_forFactory, 1)
        self._advantage_layer = nn.Linear(out_chnls_forFactory, out_chnls)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            x: [N, in_chnls]
        
        Rets:
            out: [N, out_chnls]
        '''

        x = self._comm_network.forward(x)

        base = self._base_layer.forward(x)
        advantage = self._base_layer.forward(x)

        return base + advantage
