from copy import copy, deepcopy
from typing import List, Type, Tuple

import torch
from torch import nn, optim


def conv_str2ActivationFunc(module_str: str) -> nn.Module:
    '''
        文字列から活性化関数のnn.Moduleへ変換する
    '''
    mapping = {
        'ReLU': nn.ReLU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Softplus': nn.Softplus(),
        'Identity': nn.Identity()
    }
    if module_str in mapping:
        return mapping[module_str]
    else:
        raise ValueError(f"Unknown activation function: {module_str}")

def conv_str2LayerFunc(layer_str: str, **kwargs) -> nn.Module:
    '''
    文字列からPyTorchのnn.Module層を生成する。

    Args:
        layer_str: レイヤー名（例: 'Linear', 'BatchNorm1d', 'Dropout' など）
        kwargs: 該当レイヤーの引数（例: in_features=128, out_features=64）

    Returns:
        nn.Module: 対応するPyTorch層のインスタンス

    Raises:
        ValueError: 未定義のレイヤー名が指定された場合
    '''
    layer_map = {
        'Linear': nn.Linear,
        'BatchNorm1d': nn.BatchNorm1d,
        'Dropout': nn.Dropout,
        'LayerNorm': nn.LayerNorm,
        'Conv1d': nn.Conv1d,
        'Conv2d': nn.Conv2d,
        'MaxPool1d': nn.MaxPool1d,
        'MaxPool2d': nn.MaxPool2d,
        'AvgPool1d': nn.AvgPool1d,
        'AvgPool2d': nn.AvgPool2d,
    }

    if layer_str not in layer_map:
        raise ValueError(f"Unknown layer type: {layer_str}")

    layer_class = layer_map[layer_str]
    return layer_class(**kwargs)

import torch.nn as nn

def conv_str2LossFunc(loss_str: str, **kwargs) -> nn.Module:
    '''
    文字列からPyTorchの損失関数（nn.Module）を生成する。

    Args:
        loss_str: 損失関数名（例: 'MSELoss', 'CrossEntropyLoss', 'L1Loss' など）
        kwargs: 該当損失関数の引数（例: reduction='mean' など）

    Returns:
        nn.Module: 対応するPyTorch損失関数のインスタンス

    Raises:
        ValueError: 未定義の損失関数名が指定された場合
    '''
    loss_map = {
        'MSELoss': nn.MSELoss,
        'CrossEntropyLoss': nn.CrossEntropyLoss,
        'L1Loss': nn.L1Loss,
        'NLLLoss': nn.NLLLoss,
        'BCELoss': nn.BCELoss,
        'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
        'SmoothL1Loss': nn.SmoothL1Loss,
        'HuberLoss': nn.HuberLoss,
        'KLDivLoss': nn.KLDivLoss,
        'PoissonNLLLoss': nn.PoissonNLLLoss,
        'CTCLoss': nn.CTCLoss,
        'CosineEmbeddingLoss': nn.CosineEmbeddingLoss,
        'MarginRankingLoss': nn.MarginRankingLoss,
        'MultiMarginLoss': nn.MultiMarginLoss,
        'TripletMarginLoss': nn.TripletMarginLoss,
    }

    if loss_str not in loss_map:
        raise ValueError(f"Unknown loss function: {loss_str}")

    loss_class = loss_map[loss_str]
    return loss_class(**kwargs)


def conv_str2Optimizer(optimizer_str: str, params, **kwargs) -> optim.Optimizer:
    '''
    文字列からPyTorchのOptimizerを生成する

    Args:
        optimizer_str: オプティマイザの名前（例: 'Adam', 'SGD', 'RMSprop'）
        params: モデルのパラメータ（model.parameters()）
        kwargs: オプティマイザに渡す追加の引数（lrなど）

    Returns:
        PyTorch Optimizer インスタンス

    Raises:
        ValueError: 未定義のオプティマイザが指定された場合
    '''
    optimizer_map = {
        'SGD': optim.SGD,
        'Adam': optim.Adam,
        'AdamW': optim.AdamW,
        'RMSprop': optim.RMSprop,
        'Adagrad': optim.Adagrad,
        'Adadelta': optim.Adadelta,
    }

    if optimizer_str not in optimizer_map:
        raise ValueError(f"Unknown optimizer: {optimizer_str}")
    
    optimizer_class = optimizer_map[optimizer_str]
    return optimizer_class(params, **kwargs)