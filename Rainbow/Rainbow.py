

import torch
from torch import nn

from gymnasium import Env 

from typing import Tuple
import numpy as np
from numpy import ndarray

from copy import copy, deepcopy

# from ReplayBuffer.Buffer import ReplayBuffer
from ReplayBuffer.Buffer_v2 import BaseReplayBuffer, NstepReplayBuffer
from usefulParam.Param import ScalarParam, makeConstant, makeLinear, makeMultiply
from mtorch.Network.Qnet import BaseQnetwork
from mtorch.util.nnModuleConvert import conv_str2LossFunc, conv_str2Optimizer
from mtorch.util.networkUpdate import soft_update, hard_update

class RainbowAgent:
    '''
    Rainbow Agent
    '''

    def __init__(self, 
                 gamma: ScalarParam, lr: ScalarParam, tau: ScalarParam,                       # hyper param
                 state_size: Tuple, action_size: int, action_kinds: int,      # task info
                 qnet: BaseQnetwork, 
                 lossF: str, optimizer: str, sync_interval: int,                                                  # q net learn function
                 replayBuf: BaseReplayBuffer, batch_size: int,                  # replay buffer
                 device: torch.device=torch.device('cpu')
                 ):
        self._device = device

        # Hyper Parameter
        self._gamma = gamma
        self._lr = lr
        self._tau = tau

        # task info
        self._state_size = state_size
        self._action_size = action_size
        self._action_kinds = action_kinds

        ### main q net
        self._qnet = deepcopy(qnet)
        self._target_qnet = deepcopy(qnet)

        self._q_net_lossF = conv_str2LossFunc(lossF, reduction='mean')
        self._q_net_optimizer = conv_str2Optimizer(optimizer, self._qnet.parameters(), lr=self._lr.value)

        self._sync_interval = sync_interval
        self._sync_interval_count = 0


        # replay buffer
        self._replayBuf = replayBuf
        self._batch_size = batch_size

        # log
        self._q_net_loss_history = list()

    def get_action(self, status: torch.Tensor) -> torch.Tensor:
        '''
        Actionを取得する

        Args:
            status[batch, state_size]: 状況
        
        Ret:
            action[batch, action_size]: 行動
        '''
        qval = self._qnet.forward(status)
        actions = torch.argmax(qval, dim=1)
        return actions

    def update(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray):
        '''
        エージェントを更新する．
        ・リプレイバッファへ経験の追加
        ・Qネットワークの更新
        '''

        # リプレイバッファへ経験の追加
        self.add_buffer(state, action, reward, next_state, done)

        # リプレイバッファの経験数がバッチサイズ以下なら，ネットワークを更新せず終了
        if self._replayBuf.real_size < self._batch_size:
            return 

        # ネットワークの更新
        self.learn_q_net()

        self._qnet.callback_everyUpdate()
        self._target_qnet.callback_everyUpdate()
        
    def add_buffer(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray):
        self._replayBuf.add(state, action, reward, next_state, done)
    
    def learn_q_net(self):
        '''
        Qネットワークの学習
        '''
        # バッチの取り出し
        status, actions, rewards, next_status, dones = self._replayBuf.get_sample(self._batch_size)

        # q_valを計算
        self._qnet.train()
        q_val = self._qnet.forward(status)[np.arange(len(actions)), actions.squeeze(1)].unsqueeze(1)

        # q_val_targetを計算
        with torch.no_grad():
            self._qnet.eval()
            self._target_qnet.eval()
            next_actions = self._qnet.forward(next_status).argmax(dim=1)
            n_step: int
            if isinstance(self._replayBuf, NstepReplayBuffer):
                n_step = self._replayBuf.n_step
            else:
                n_step = 1
            q_val_target = (rewards + (1 - dones) * (self._gamma.tensor_value)**n_step * self._target_qnet.forward(next_status)[np.arange(len(next_actions)), next_actions].unsqueeze(1)) / n_step

        # ネットワークを更新
        # print(f'q val shape: {q_val.shape}, target shape: {q_val_target.shape}')
        loss: torch.Tensor = self._q_net_lossF(q_val, q_val_target)
        self._q_net_optimizer.zero_grad()
        loss.backward()
        self._q_net_optimizer.step()

        self._sync_interval_count = (self._sync_interval_count + 1) % self._sync_interval
        if self._sync_interval_count == 0:
            soft_update(self._qnet, self._target_qnet, self._tau.value)

        # 記録
        self._q_net_loss_history.append(loss.item())
        
    def param_step(self):
        self._gamma.step()
        self._lr.step()
        self._tau.step()
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def q_net_loss_history(self):
        return self._q_net_loss_history


# class RainbowAgent_legacy:
#     '''
#     Rainbow Agent
#     '''

#     def __init__(self, 
#                  gamma: ScalarParam, lr: ScalarParam, tau: ScalarParam,                       # hyper param
#                  state_size: int, action_size: int, action_kinds: int,      # task info
#                  hdn_chnls: Tuple[int, ...],  # q net
#                  lossF: str, optimizer: str, sync_interval: int,                                                  # q net learn function
#                  replayBuf: BaseReplayBuffer, batch_size: int,                  # replay buffer
#                  device: torch.device=torch.device('cpu'), 
#                  noisy: bool=True, sigma_init: float=0.5, 
#                  epsilon_greedy: bool=True, epsilon: ScalarParam=makeMultiply(1.0, 0.998, 1e-4, 1.0, torch.device('cpu')), 
#                  dueling: bool=True):
#         self._device = device

#         # condition
#         self._noisy = noisy
#         self._dueling = dueling
#         self._epsilon_greedy = epsilon_greedy

#         # Hyper Parameter
#         self._gamma = gamma
#         self._lr = lr
#         self._tau = tau
#         self._epsilon = epsilon

#         # task info
#         self._state_size = state_size
#         self._action_size = action_size
#         self._action_kinds = action_kinds

#         ### main q net
#         self._q_net: BaseQnetwork
#         self._target_q_net: BaseQnetwork
#         if noisy and dueling:
#             self._q_net = DuelingNoisyNet(state_size, hdn_chnls, action_kinds, sigma_init).to(self._device)
#             self._target_q_net = DuelingNoisyNet(state_size, hdn_chnls, action_kinds, sigma_init).to(self._device)
#         elif noisy and not dueling:
#             self._q_net = NoisyNet(state_size, hdn_chnls, action_kinds, sigma_init).to(self._device)
#             self._target_q_net = NoisyNet(state_size, hdn_chnls, action_kinds, sigma_init).to(self._device)
#         elif not noisy and dueling:
#             pass
#         else:       # not noisy and not dueling
#             self._q_net = Qnetwork(state_size, hdn_chnls, action_kinds).to(self._device)
#             self._target_q_net = Qnetwork(state_size, hdn_chnls, action_kinds).to(self._device)

#         self._target_q_net.load_state_dict(self._q_net.state_dict())

#         self._q_net_lossF = conv_str2LossFunc(lossF, reduction='mean')
#         self._q_net_optimizer = conv_str2Optimizer(optimizer, self._q_net.parameters(), lr=self._lr.value)

#         self._sync_interval = sync_interval
#         self._sync_interval_count = 0


#         # replay buffer
#         self._replayBuf = replayBuf
#         self._batch_size = batch_size

#         # log
#         self._q_net_loss_history = list()

#     def get_action(self, status: torch.Tensor) -> torch.Tensor:
#         '''
#         Actionを取得する

#         Args:
#             status[batch, state_size]: 状況
        
#         Ret:
#             action[batch, action_size]: 行動
#         '''

#         if self._epsilon_greedy:
#             if np.random.random() < self._epsilon.value:
#                 return torch.tensor(np.random.choice(range(self._action_kinds)))

#         q_val = self._q_net.forward(status)
#         action = torch.argmax(q_val, dim=1)
#         return action

#     def update(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray):
#         '''
#         エージェントを更新する．
#         ・リプレイバッファへ経験の追加
#         ・Qネットワークの更新
#         '''

#         # リプレイバッファへ経験の追加
#         self.add_buffer(state, action, reward, next_state, done)

#         # リプレイバッファの経験数がバッチサイズ以下なら，ネットワークを更新せず終了
#         if self._replayBuf.real_size < self._batch_size:
#             return 

#         # ネットワークの更新
#         self.learn_q_net()

#         if isinstance(self._q_net, NoisyNetInterface) and isinstance(self._target_q_net, NoisyNetInterface):
#             self.noise_reset()
        
#     def add_buffer(self, state: ndarray, action: ndarray, reward: ndarray, next_state: ndarray, done: ndarray):
#         self._replayBuf.add(state, action, reward, next_state, done)
    
#     def learn_q_net(self):
#         '''
#         Qネットワークの学習
#         '''
#         # バッチの取り出し
#         status, actions, rewards, next_status, dones = self._replayBuf.get_sample(self._batch_size)

#         # q_valを計算
#         self._q_net.train()
#         q_val = self._q_net.forward(status)[np.arange(len(actions)), actions.squeeze(1)].unsqueeze(1)

#         # q_val_targetを計算
#         with torch.no_grad():
#             self._q_net.eval()
#             self._target_q_net.eval()
#             next_actions = self._q_net.forward(next_status).argmax(dim=1)
#             n_step: int
#             if isinstance(self._replayBuf, NstepReplayBuffer):
#                 n_step = self._replayBuf.n_step
#             else:
#                 n_step = 1
#             q_val_target = (rewards + (1 - dones) * (self._gamma.tensor_value)**n_step * self._target_q_net.forward(next_status)[np.arange(len(next_actions)), next_actions].unsqueeze(1)) / n_step

#         # ネットワークを更新
#         # print(f'q val shape: {q_val.shape}, target shape: {q_val_target.shape}')
#         loss: torch.Tensor = self._q_net_lossF(q_val, q_val_target)
#         self._q_net_optimizer.zero_grad()
#         loss.backward()
#         self._q_net_optimizer.step()

#         self._sync_interval_count = (self._sync_interval_count + 1) % self._sync_interval
#         if self._sync_interval_count == 0:
#             soft_update(self._q_net, self._target_q_net, self._tau.value)

#         # 記録
#         self._q_net_loss_history.append(loss.item())
    
#     def noise_reset(self):
#         '''
#         ノイズのリセット
#         q_netがNoisyでなければ何もしない
#         '''
#         if isinstance(self._q_net, NoisyNetInterface) and isinstance(self._target_q_net, NoisyNetInterface):
#             self._q_net.noise_reset()
#             # self._target_q_net.noise_reset()
#         else:
#             raise RuntimeError(f"q netがNoisy系でないのに，noise resetが呼ばれた")
        
#     def param_step(self):
#         self._gamma.step()
#         self._lr.step()
#         self._tau.step()
#         self._epsilon.step()
    
#     @property
#     def device(self) -> torch.device:
#         return self._device
    
#     @property
#     def q_net_loss_history(self):
#         return self._q_net_loss_history