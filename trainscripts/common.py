#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import json
import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import traceback

# 設定錯誤日誌記錄到文件
logging.basicConfig(
    filename='error_log.txt',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s:%(message)s'
)

def generate_time_weights(batch_size, seq_len, device="cpu", low=0.5, high=1.5):
    pattern = random.choice(["U", "increasing", "decreasing"])
    if pattern == "U":
        half = seq_len // 2
        if seq_len % 2 == 0:
            left = np.linspace(low, high, half)
            right = np.linspace(high, low, half)
            weights = np.concatenate([left, right])
        else:
            left = np.linspace(low, high, half + 1)
            right = np.linspace(high, low, half + 1)[1:]
            weights = np.concatenate([left, right])
    elif pattern == "increasing":
        weights = np.linspace(low, high, seq_len)
    elif pattern == "decreasing":
        weights = np.linspace(high, low, seq_len)
    else:
        raise ValueError("Unknown pattern.")
    weights = np.tile(weights, (batch_size, 1)).reshape(batch_size, seq_len, 1)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def load_json_file(filepath):
    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                print(f"file : {filepath} is loaded")
                return json.load(f)
        except Exception as e:
            print(f"使用編碼 {enc} 讀取檔案 {filepath} 失敗：{e}")
    raise Exception(f"無法解析檔案 {filepath}，請確認其編碼格式。")

def load_json_files(directory):
    files = [os.path.join(directory, fname) for fname in os.listdir(directory) 
             if fname.endswith(".json") and not fname.startswith("._")]
    regions = []
    for f in files:
        region = load_json_file(f)
        if "station_id" not in region:
            # 如果 JSON 裡沒有 "station_id"，則以檔案名稱作為 station_id
            region["station_id"] = os.path.basename(f)
        regions.append(region)
    return regions


class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        scores = torch.bmm(Q, K.transpose(1,2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.bmm(attn_weights, V)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.buffer)

def evaluate_reward(rules, real_record, action_dict):
    """
    依據傳入的 real_record 與 action_dict (模型推導結果) 計算 reward。
    這裡假設 action_dict 中已包含所有 reward 規則中需要的變數。
    """
    variables = {}
    variables.update(real_record)
    variables.update({k: bool(v) for k, v in action_dict.items()})
    total_reward = 0.0
    for rule in rules:
        try:
            if eval(rule.get("condition", ""), {}, variables):
                total_reward += float(rule.get("reward", 0))
        except Exception as e:
            print("Reward eval error:", rule, variables, e)
            logging.error(f"Error in reward evaluation: {rule}, {variables}, {traceback.format_exc()}")
    return total_reward

class WeatherEnv(gym.Env):
    def __init__(self, global_config, data_config):
        super(WeatherEnv, self).__init__()
        self.input_ids = global_config.get("input_ids", [])
        self.output_ids = global_config.get("output_ids", [])
        self.rules = global_config.get("rules", [])
        self.predicted_records = data_config.get("predicted_records", [])
        self.real_records = data_config.get("real_records", [])
        self.data_len = len(self.predicted_records)
        if self.data_len < 24:
            raise Exception("Insufficient data: need at least 24 records.")
        self.observation_space = gym.spaces.Dict({
            "weather": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24, len(self.input_ids)), dtype=np.float32),
            "time_weight": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        })
        # action_space 保留，但實際上我們使用字典傳入
        self.action_space = gym.spaces.Discrete(11)

    def reset(self):
        start = random.randint(0, self.data_len - 24)
        self.current_index = start
        block = self.predicted_records[start:start+24]
        weather = np.array([[rec.get(k, 0.0) for k in self.input_ids] for rec in block], dtype=np.float32)
        time_weight = generate_time_weights(1, 24, device="cpu")[0, :, 0].cpu().numpy()
        return {"weather": weather, "time_weight": time_weight}

    def step(self, action):
        # action 預期為模型推理得到的字典
        block = self.real_records[self.current_index:self.current_index+24]
        total_reward = sum([evaluate_reward(self.rules, rec, action) for rec in block])
        done = True
        return self.reset(), total_reward, done, {}

class WeatherDQNModel(nn.Module):
    def __init__(self):
        super(WeatherDQNModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # 輸出維度更新為 22 (11個決策，每個決策兩個 Q 值)
        self.q_out = nn.Linear(128, 22)
    def forward(self, x, time_weight):
        lstm1_out, _ = self.lstm1(x)
        fixed_weight = time_weight.detach().unsqueeze(-1)
        attn1_out = lstm1_out * fixed_weight
        lstm2_out, _ = self.lstm2(attn1_out)
        attn_out = self.attention(lstm2_out)
        pooled = torch.mean(attn_out, dim=1)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        q_vals = self.q_out(shared)
        return q_vals
    
class WeatherActorCriticModel(nn.Module):
    def __init__(self):
        super(WeatherActorCriticModel, self).__init__()
        # 兩層 Bi-LSTM 前端特徵提取 (輸入維度固定為7)
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # Actor 分支：輸出 11 個動作（各動作輸出一個機率，使用 sigmoid 激活）
        self.actor = nn.Linear(128, 11)
        # Critic 分支：輸出單一標量，作為狀態價值估計
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x, time_weight):
        """
        x: Tensor, shape (B, 24, 7) － 輸入天氣數據
        time_weight: Tensor, shape (B, 24) － 固定時間權重（不參與反向傳播）
        """
        lstm1_out, _ = self.lstm1(x)  # (B, 24, 128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B, 24, 1)
        attn1_out = lstm1_out * fixed_weight               # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)               # (B, 24, 128)
        pooled = torch.mean(lstm2_out, dim=1)              # (B, 128)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        policy_logits = self.actor(shared)               # (B, 11)
        policy_probs = torch.sigmoid(policy_logits)      # 每個數值介於 0 與 1
        value = self.critic(shared)                      # (B, 1)
        return policy_probs, value