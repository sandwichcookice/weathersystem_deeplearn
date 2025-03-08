#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import random
import numpy as np

def derive_action_dict_policy(probs, output_ids, threshold=0.5):
    """
    將模型輸出機率 (tensor, shape: (1, num_actions)) 轉換為布林決策字典：
    當機率 >= threshold 時返回 True，否則返回 False。
    """
    probs = probs.squeeze(0)
    decisions = [bool(p.item() >= threshold) for p in probs]
    return dict(zip(output_ids, decisions))

def compute_returns(rewards, gamma):
    """
    計算折扣回報，對於每個時間步 t, G_t = r_t + gamma * r_{t+1} + ...。
    REINFORCE 算法中可直接用整個 episode 的累積 reward。
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
