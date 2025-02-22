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
from torch.utils.tensorboard import SummaryWriter  # 新增
from tqdm import tqdm

#########################################
# 生成時間權重 (來自舊訓練腳本)
#########################################
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

#########################################
# 讀取 training_data.json 檔案
#########################################
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), "training_data.json")
with open(TRAINING_DATA_PATH, "r", encoding="utf-8") as f:
    training_data = json.load(f)

# 取得必要參數與超參數
INPUT_IDS = training_data.get("input_ids", [])
OUTPUT_IDS = training_data.get("output_ids", [])
RULES = training_data.get("rules", [])
training_params = training_data.get("training", {})
learning_rate = training_params.get("learning_rate", 0.001)
batch_size = training_params.get("batch_size", 32)
num_episodes = training_params.get("num_epochs", 100)  # 這裡用 num_epochs 當作總 episode 數
gamma = training_params.get("gamma", 0.99)
learning_starts = training_params.get("learning_starts", 1000)
replay_buffer_capacity = training_params.get("replay_buffer_size", 50000)
exploration_config = training_params.get("exploration_config", {"epsilon_timesteps": 10000, "final_epsilon": 0.02})

# 設定 epsilon 探索參數
initial_epsilon = 1.0
final_epsilon = exploration_config.get("final_epsilon", 0.02)
epsilon_decay_steps = exploration_config.get("epsilon_timesteps", 10000)

#########################################
# 讀取 split_weather_continuous.json 檔案 (預測與真實資料)
#########################################
SPLIT_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "split_weather_continuous.json")
with open(SPLIT_DATA_PATH, "r", encoding="utf-8") as f:
    split_data = json.load(f)
# 如果 split_data 中有 input_ids，則使用；否則沿用 training_data 的
INPUT_IDS = split_data.get("input_ids", INPUT_IDS)
PREDICTED_RECORDS = split_data.get("predicted_records", [])
REAL_RECORDS = split_data.get("real_records", [])
DATA_LEN = len(PREDICTED_RECORDS)

#########################################
# 定義 Replay Buffer
#########################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

#########################################
# 定義自注意力層
#########################################
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        Q = self.query(x)          # (B, T, d_model)
        K = self.key(x)            # (B, T, d_model)
        V = self.value(x)          # (B, T, d_model)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        attn_weights = torch.softmax(scores, dim=-1)            # (B, T, T)
        out = torch.bmm(attn_weights, V)                          # (B, T, d_model)
        return out

#########################################
# 定義模型架構 – WeatherDDQNModel
#########################################
class WeatherDDQNModel(nn.Module):
    def __init__(self):
        super(WeatherDDQNModel, self).__init__()
        # 第一層 Bi-LSTM：輸入 7 維，隱藏單元 64，雙向 (輸出維度 = 64*2 = 128)
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        # 第二層 Bi-LSTM：輸入 128 維，隱藏單元 64，雙向 (輸出 128)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        
        # 第一層注意力：使用固定時間權重與 lstm1 輸出作乘法融合
        # 第二層注意力：自注意力層
        self.attention2 = SelfAttention(d_model=128)
        
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Advantage branch
        self.adv_hidden = nn.Linear(128, 128)
        self.adv_stream = nn.Linear(128, 22)  # 22 = 2 x 11
        
        # Value branch
        self.val_hidden = nn.Linear(128, 128)
        self.val_stream = nn.Linear(128, 11)    # 每個建議一個 value

    def forward(self, x, time_weight):
        """
        x: Tensor，形狀 (B, 24, 7)
        time_weight: Tensor，形狀 (B, 24)；僅用於第一層注意力，故在使用前 detach
        """
        lstm1_out, _ = self.lstm1(x)  # (B, 24, 128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B, 24, 1)
        attn1_out = lstm1_out * fixed_weight                # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)                # (B, 24, 128)
        attn2_out = self.attention2(lstm2_out)              # (B, 24, 128)
        pooled = torch.mean(attn2_out, dim=1)               # (B, 128)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        
        adv = F.relu(self.adv_hidden(shared))
        adv = self.adv_stream(adv)                          # (B, 22)
        adv = adv.view(-1, 11, 2)                           # (B, 11, 2)
        
        val = F.relu(self.val_hidden(shared))
        val = self.val_stream(val)                          # (B, 11)
        val = val.view(-1, 11, 1)                           # (B, 11, 1)
        
        # Dueling 結構：Q = V + (A - mean(A))
        adv_mean = adv.mean(dim=2, keepdim=True)
        q_vals = val + (adv - adv_mean)                     # (B, 11, 2)
        
        # 聚合 value 輸出：取 11 個 value 的平均 (B, 11, 1) -> (B,)
        self._value_out = torch.mean(val, dim=1).squeeze(-1)
        return q_vals

#########################################
# 定義 Gym 環境 – WeatherEnv
#########################################
class WeatherEnv(gym.Env):
    def __init__(self, env_config=None):
        super(WeatherEnv, self).__init__()
        self.observation_space = gym.spaces.Dict({
            "weather": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24, 7), dtype=np.float32),
            "time_weight": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        })
        self.action_space = gym.spaces.MultiBinary(11)
        
        self.predicted_records = PREDICTED_RECORDS
        self.real_records = REAL_RECORDS
        self.rules = RULES
        
        if len(self.predicted_records) < 24 or len(self.real_records) < 24:
            raise Exception("資料數量不足，無法組成 24 筆連續資料")
        self.data_len = len(self.predicted_records)
    
    def reset(self):
        start_idx = np.random.randint(0, self.data_len - 24 + 1)
        self.current_index = start_idx
        block = self.predicted_records[start_idx : start_idx + 24]
        weather_block = []
        for rec in block:
            row = [rec.get(key, 0.0) for key in INPUT_IDS]
            weather_block.append(row)
        weather_block = np.array(weather_block, dtype=np.float32)
        # 生成時間權重
        time_weight_tensor = generate_time_weights(1, 24, device="cpu", low=0.5, high=1.5)
        time_weight = time_weight_tensor[0, :, 0].cpu().numpy()
        obs = {"weather": weather_block, "time_weight": time_weight}
        return obs

    def step(self, action):
        block = self.real_records[self.current_index : self.current_index + 24]
        total_reward = 0.0
        for rec in block:
            total_reward += evaluate_reward(self.rules, rec, action)
        done = True  # 每個 episode 為單步（24 筆資料）
        info = {"start_index": self.current_index}
        return self.reset(), total_reward, done, info

#########################################
# Reward 評估函數
#########################################
def evaluate_reward(rules, real_record, action):
    variables = {}
    variables.update(real_record)
    for i, key in enumerate(OUTPUT_IDS):
        variables[key] = bool(action[i])
    total_reward = 0.0
    for rule in rules:
        condition = rule.get("condition", "")
        try:
            if eval(condition, {}, variables):
                total_reward += float(rule.get("reward", 0))
        except Exception as e:
            print("條件評估錯誤:", condition, variables, e)
    return total_reward

#########################################
# 定義 Replay Buffer 類別（如上）
#########################################
# 已經定義在前面，此處重複即可

#########################################
# 主程序 – 手動實作 DDQN 演算法的訓練流程
#########################################
if __name__ == "__main__":
    
    # 設定裝置：自動選擇 CUDA 或 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 建立 TensorBoard 寫入器
    writer = SummaryWriter(log_dir="./logs")
    
    # 建立環境
    env = WeatherEnv()
    # 建立主網路與目標網路
    model = WeatherDDQNModel().to(device)
    target_model = WeatherDDQNModel().to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_buffer_capacity)

    total_steps = 0
    target_update_freq = 1000  # 每 1000 個步驟更新目標網路
    episode_rewards = []

    # 使用 tqdm 進度條包裹外層訓練迴圈
    for episode in tqdm(range(num_episodes), desc="Training Episodes"):
        obs = env.reset()
        done = False
        episode_reward = 0

        total_steps += 1

        epsilon = max(final_epsilon, initial_epsilon - (initial_epsilon - final_epsilon) * (total_steps / epsilon_decay_steps))
        if random.random() < epsilon:
            action = np.random.randint(0, 2, size=(11,))
        else:
            model.eval()
            with torch.no_grad():
                state_weather = torch.tensor(obs["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                state_time = torch.tensor(obs["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = model(state_weather, state_time)
                q_vals = q_vals.squeeze(0)
                action = q_vals.argmax(dim=1).cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        episode_reward += reward

        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs

        if len(replay_buffer) >= learning_starts:
            model.train()
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            states_weather = torch.tensor(np.array([s["weather"] for s in states]), dtype=torch.float32, device=device)
            states_time = torch.tensor(np.array([s["time_weight"] for s in states]), dtype=torch.float32, device=device)
            next_states_weather = torch.tensor(np.array([s["weather"] for s in next_states]), dtype=torch.float32, device=device)
            next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]), dtype=torch.float32, device=device)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.long, device=device)
            rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32, device=device)
            dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32, device=device)

            q_vals = model(states_weather, states_time)
            actions_tensor_expanded = actions_tensor.unsqueeze(2)
            chosen_q_vals = torch.gather(q_vals, 2, actions_tensor_expanded).squeeze(2)
            current_q = chosen_q_vals.mean(dim=1)

            with torch.no_grad():
                next_q_vals_main = model(next_states_weather, next_states_time)
                next_actions = next_q_vals_main.argmax(dim=2)
                next_actions_expanded = next_actions.unsqueeze(2)
                next_q_vals_target = target_model(next_states_weather, next_states_time)
                next_chosen_q_vals = torch.gather(next_q_vals_target, 2, next_actions_expanded).squeeze(2)
                next_q = next_chosen_q_vals.mean(dim=1)
                target_q = rewards_tensor + gamma * next_q * (1 - dones_tensor)

            loss = F.mse_loss(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_steps % 100 == 0:
                print(f"Step: {total_steps}, Loss: {loss.item():.4f}")
                writer.add_scalar("Loss/train", loss.item(), total_steps)  # 記錄 loss

            if total_steps % target_update_freq == 0:
                target_model.load_state_dict(model.state_dict())

        episode_rewards.append(episode_reward)
        writer.add_scalar("Reward/episode", episode_reward, episode+1)  # 記錄 reward

        if (episode + 1) % 1000 == 0:
            os.makedirs("./model/log", exist_ok=True)
            torch.save(model.state_dict(), f"./model/log/model_checkpoint_episode_{episode+1}.pth")

    writer.close()  # 結束時關閉 writer
    print("Training completed.")