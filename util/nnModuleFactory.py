from copy import copy, deepcopy
from typing import List, Type, Tuple

import torch
from torch import nn, optim
from torch.nn import Module

from mtorch.util.nnModuleConvert import conv_str2ActivationFunc

############################## ネットワーク ファクトリ ##############################

def factory_LinearReLU_ModuleList(in_chnls: int, hdn_chnls: Tuple[int, ...], out_chnls: int ,out_act: str="") -> nn.ModuleList:
    '''
        LinearとReLUが積み重なったネットワークを作成
        最終層はカスタム可能

        Args:
            in_chnls: 入力チャネル
            hdn_chnls: 隠れ層のチャネル数
            hdn_lays: 隠れ層の層数
            out_chnls: 出力層のチャネル数
            out_module: 最終層の指定
    '''
    layers = nn.ModuleList()

    chnls = tuple([in_chnls]) + hdn_chnls + tuple([out_chnls])

    for i in range(len(chnls) - 1):
        layers.append(nn.Linear(chnls[i], chnls[i + 1]))
        if i != len(chnls) - 2:
            layers.append(nn.ReLU())
    
    if out_act != "":
        layers.append(conv_str2ActivationFunc(out_act))

    return layers

def factory_LinearReLU_Sequential(in_chnls: int, hdn_chnls: Tuple[int, ...], out_chnls: int ,out_act: str="") -> nn.Sequential:
    '''
        LinearとReLUが積み重なったネットワークを作成
        最終層はカスタム可能

        Args:
            in_chnls: 入力チャネル
            hdn_chnls: 隠れ層のチャネル数
            hdn_lays: 隠れ層の層数
            out_chnls: 出力層のチャネル数
            out_module: 最終層の指定
    '''
    sequential = nn.Sequential(*factory_LinearReLU_ModuleList(in_chnls, hdn_chnls, out_chnls, out_act))
    return sequential