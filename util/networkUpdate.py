
from copy import copy, deepcopy

from torch import nn

def hard_update(source: nn.Module, target: nn.Module):
    '''
    ハードアップデート
    '''
    target = deepcopy(source)

def soft_update(source: nn.Module, target: nn.Module, tau: float):
    '''
    ソフトアップデート
    '''
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)
    

