
from typing import Tuple, List

import torch
from torch import nn

import threading


class StripingTensor:
    '''
    マルチアクセス用torch.Tensor
    '''

    def __init__(self, baseTensor: torch.Tensor, stripe_num: int, dtype: torch.dtype, device: torch.device):
        self._device = device
        self._dtype = dtype
        self._tensor = baseTensor.to(dtype=dtype, device=device)

        self._stripe_num = stripe_num
        self._stripeBorder = torch.tensor(range(0, self._tensor.size(0), int(self._tensor.size(0) / self._stripe_num)), dtype=torch.int, device=self._device)

        self._mutex = [threading.Lock()] * self._stripe_num
    
    def write(self, values: torch.Tensor, indices: torch.Tensor):
        '''
        書き込み
        '''
    
    def read(self, indices: torch.Tensor):
        '''
        読み取り
        '''
    
    def classification_indices(self, indices: torch.Tensor) -> torch.Tensor:
        '''
        mutexが管理する範囲ごとにインデックスを分類
        '''
        


    def to(self, dtype:torch.dtype, device:torch.device):
        '''
        型の変更とデバイスの変更
        '''
        
        self._tensor.to(dtype=dtype, device=device)
        self._stripeBorder.to(device=device)
        self._device = device
        self._dtype = dtype

    def __len__(self) -> int:
        return len(self._tensor)
    
    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape
    
    def size(self, dim: int) -> int:
        return self._tensor.size(dim)
    
    @property
    def device(self):
        return self._tensor.device
    

    @property
    def stripe_num(self):
        return self._stripe_num
    
    @property
    def stripe_border(self):
        return self._stripeBorder
    
    @property
    def mutex(self):
        return self._mutex