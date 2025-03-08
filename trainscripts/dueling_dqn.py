#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import traceback

from common import (
    generate_time_weights,
    evaluate_reward,
    load_json_file,
    load_json_files,
    ReplayBuffer,
    WeatherEnv
)

# ---------------------
# 輔助函數：將模型輸出轉換為布林決策字典
def derive_action_dict(q_vals, output_ids):
    """
    將模型輸出 q_vals (shape (1, 22) 或 (1,11,2)) 重塑為 (11,2)，
    並取每組 argmax 轉換為布林值後建立字典。
    """
    # 若 q_vals shape 為 (1,22) 則 reshape 為 (11,2)
    if len(q_vals.shape) == 2:
        q_vals = q_vals.view(11, 2)
    else:
        q_vals = q_vals.squeeze(0)
    decisions = q_vals.argmax(dim=1)
    bool_decisions = [bool(val.item()) for val in decisions]
    return dict(zip(output_ids, bool_decisions))

# 輔助函數：將 action 字典轉換為 0/1 向量 (長度 11)
def action_dict_to_index(action_dict, output_ids):
    return [0 if not action_dict.get(k, False) else 1 for k in output_ids]

# ---------------------
# 定義 Dueling DQN 模型
class WeatherDuelingDQNModel(nn.Module):
    def __init__(self):
        super(WeatherDuelingDQNModel, self).__init__()
        # 前端特徵提取層：兩層 Bi-LSTM
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # 自注意力層（第二層注意力）
        # 此處使用 common 中定義的 SelfAttention（需在 common.py 中已定義）
        # 若沒有，也可直接複製其實作：
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # Advantage branch
        self.adv_hidden = nn.Linear(128, 128)
        self.adv_stream = nn.Linear(128, 11 * 2)  # 輸出 22 個單元，之後 reshape 成 (batch, 11, 2)
        # Value branch
        self.val_hidden = nn.Linear(128, 128)
        self.val_stream = nn.Linear(128, 11)      # 輸出 11 個值

    def forward(self, x, time_weight):
        """
        x: (B, 24, 7) 輸入天氣資料
        time_weight: (B, 24) 固定的時間權重
        """
        # 第一層 Bi-LSTM
        lstm1_out, _ = self.lstm1(x)               # (B, 24, 128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B, 24, 1)
        attn1_out = lstm1_out * fixed_weight        # (B, 24, 128)
        # 第二層 Bi-LSTM
        lstm2_out, _ = self.lstm2(attn1_out)         # (B, 24, 128)
        # 自注意力（此處直接對 lstm2_out 做平均池化即可，也可加入更複雜的自注意力）
        pooled = torch.mean(lstm2_out, dim=1)         # (B, 128)
        # 共享全連接層
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        # Advantage branch
        adv = F.relu(self.adv_hidden(shared))
        adv = self.adv_stream(adv)                    # (B, 11*2)
        adv = adv.view(-1, 11, 2)                       # (B, 11, 2)
        # Value branch
        val = F.relu(self.val_hidden(shared))
        val = self.val_stream(val)                    # (B, 11)
        val = val.view(-1, 11, 1)                       # (B, 11, 1)
        # 組合 Q 值：Q = V + (A - mean(A))
        adv_mean = adv.mean(dim=2, keepdim=True)        # (B, 11, 1)
        q_vals = val + (adv - adv_mean)                 # (B, 11, 2)
        # 最後取每個 action 的決策：這裡仍保持 2 維數據，後續 derive_action_dict 會取 argmax
        # 若需要最終 Q 向量為 (B, 11) 可進一步取 mean 或 max（取決於設計）
        return q_vals.view(-1, 11*2)

# ---------------------
# 讀取配置與資料
base_dir = os.path.dirname(__file__)
global_config = load_json_file(os.path.join(base_dir, "..", "config.json"))
training_data_list = load_json_files(os.path.join(base_dir, "..", "data", "training_data"))
check_data_list = load_json_files(os.path.join(base_dir, "..", "data", "check"))
verify_data_list = load_json_files(os.path.join(base_dir, "..", "data", "val"))

# ---------------------
# 訓練參數
lr = global_config["training"].get("learning_rate", 0.001)
batch_size = global_config["training"].get("batch_size", 64)
num_epochs = global_config["training"].get("num_epochs", 100)
gamma = global_config["training"].get("gamma", 0.99)
learning_starts = global_config["training"].get("learning_starts", 1000)
replay_capacity = global_config["training"].get("replay_buffer_size", 50000)
epsilon_timesteps = global_config["training"]["exploration_config"].get("epsilon_timesteps", 10000)
final_epsilon = global_config["training"]["exploration_config"].get("final_epsilon", 0.02)
initial_epsilon = 1.0

device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "dueling_dqn_full"))

# ---------------------
# 建立模型、目標模型與 optimizer
model = WeatherDuelingDQNModel().to(device)
target_model = WeatherDuelingDQNModel().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
buffer = ReplayBuffer(replay_capacity)
total_steps = 0
target_update_freq = 1000

# ---------------------
# 訓練資料分批加載設定，每批 10 個檔案
batch_files = 50
num_total_files = len(training_data_list)
num_batches = (num_total_files + batch_files - 1) // batch_files
global_epoch = 0

print("開始訓練所有區域 (Dueling DQN) - 分批加載：")
for batch_idx in range(num_batches):
    batch_data = training_data_list[batch_idx * batch_files : (batch_idx + 1) * batch_files]
    print(f"\n== 訓練批次 {batch_idx+1}/{num_batches} (檔案數: {len(batch_data)}) ==")
    for region in batch_data:
        region_id = region.get("station_id", "unknown")
        print(f"訓練區域: {region_id}")
        env = WeatherEnv(global_config, region)
        for ep in tqdm(range(num_epochs), desc=f"訓練 {region_id}"):
            state = env.reset()
            done = False
            ep_reward = 0
            while not done:
                total_steps += 1
                epsilon = max(final_epsilon, initial_epsilon - (initial_epsilon - final_epsilon)*(total_steps/epsilon_timesteps))
                if random.random() < epsilon:
                    random_actions = [bool(random.getrandbits(1)) for _ in range(len(env.output_ids))]
                    action_dict = dict(zip(env.output_ids, random_actions))
                else:
                    try:
                        model.eval()
                        with torch.no_grad():
                            state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                            time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                            q_vals = model(state_tensor, time_tensor)
                            action_dict = derive_action_dict(q_vals, global_config.get("output_ids", []))
                    except Exception as e:
                        logging.error(f"Error in action selection: {traceback.format_exc()}")
                        random_actions = [bool(random.getrandbits(1)) for _ in range(len(env.output_ids))]
                        action_dict = dict(zip(env.output_ids, random_actions))
                try:
                    next_state, reward, done, _ = env.step(action_dict)
                except Exception as e:
                    logging.error(f"Error in env.step for region {region_id}: {traceback.format_exc()}")
                    reward = 0
                    next_state = env.reset()
                    done = True
                buffer.push(state, action_dict, reward, next_state, done)
                state = next_state
                ep_reward += reward

                if len(buffer) >= learning_starts:
                    try:
                        model.train()
                        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                        states_weather = torch.tensor(np.array([s["weather"] for s in states]), dtype=torch.float32, device=device)
                        states_time = torch.tensor(np.array([s["time_weight"] for s in states]), dtype=torch.float32, device=device)
                        next_states_weather = torch.tensor(np.array([s["weather"] for s in next_states]), dtype=torch.float32, device=device)
                        next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]), dtype=torch.float32, device=device)
                        
                        actions_indices = [action_dict_to_index(a, global_config.get("output_ids", [])) for a in actions]
                        actions_tensor = torch.tensor(actions_indices, dtype=torch.long, device=device)
                        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
                        
                        q_vals = model(states_weather, states_time)
                        q_vals = q_vals.view(-1, len(global_config.get("output_ids", [])), 2)
                        actions_tensor_expanded = actions_tensor.unsqueeze(2)
                        chosen_q = torch.gather(q_vals, 2, actions_tensor_expanded).squeeze(2)
                        current_q = chosen_q.mean(dim=1)
                        
                        with torch.no_grad():
                            next_q_vals_main = model(next_states_weather, next_states_time)
                            next_q_vals_main = next_q_vals_main.view(-1, len(global_config.get("output_ids", [])), 2)
                            next_actions = next_q_vals_main.argmax(dim=2)
                            next_q_vals_target = target_model(next_states_weather, next_states_time)
                            next_q_vals_target = next_q_vals_target.view(-1, len(global_config.get("output_ids", [])), 2)
                            next_chosen_q = torch.gather(next_q_vals_target, 2, next_actions.unsqueeze(2)).squeeze(2)
                            next_q = next_chosen_q.mean(dim=1)
                            target_q = rewards_tensor + gamma * next_q * (1 - dones_tensor)
                        
                        loss = F.mse_loss(current_q, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        if total_steps % target_update_freq == 0:
                            target_model.load_state_dict(model.state_dict())
                        writer.add_scalar("Loss/train", loss.item(), total_steps)
                    except Exception as e:
                        logging.error(f"Error during training update: {traceback.format_exc()}")
            writer.add_scalar("Reward/episode", ep_reward, total_steps)
            global_epoch += 1
            if global_epoch % 1000 == 0:
                print(f"Saving model checkpoint... (global epoch {global_epoch})")
                checkpoint_dir = os.path.join(base_dir, "..", "model", "dueling_dqn")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"dueling_dqn_checkpoint_ep{global_epoch}.pth"))
    torch.cuda.empty_cache()
    print(f"批次 {batch_idx+1} 訓練完成。")

print("所有區域訓練完成。")

# ---------------------
# 測試部分（分批處理並加入進度條）
print("開始測試...")
test_rewards = []
test_batch_files = 10
num_test_files = len(check_data_list)
num_test_batches = (num_test_files + test_batch_files - 1) // test_batch_files
for batch_idx in tqdm(range(num_test_batches), desc="Testing Batches"):
    batch_test = check_data_list[batch_idx * test_batch_files : (batch_idx + 1) * test_batch_files]
    for data in tqdm(batch_test, desc="Testing Files", leave=False):
        try:
            env = WeatherEnv(global_config, data)
            state = env.reset()
            model.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = model(state_tensor, time_tensor)
                action_dict = derive_action_dict(q_vals, global_config.get("output_ids", []))
            _, reward, _, _ = env.step(action_dict)
            test_rewards.append(reward)
            writer.add_scalar("Test/Reward", reward, total_steps)
        except Exception as e:
            logging.error(f"Error during testing on file: {traceback.format_exc()}")
    torch.cuda.empty_cache()
avg_test_reward = np.mean(test_rewards) if test_rewards else 0.0
writer.add_scalar("Test/AverageReward", avg_test_reward, total_steps)
print(f"測試平均 Reward: {avg_test_reward:.4f}")

# ---------------------
# 驗證部分（分批處理並加入進度條）
print("開始驗證...")
file_rewards = []
file_idx = 0
verify_batch_files = 10
num_verify_files = len(verify_data_list)
num_verify_batches = (num_verify_files + verify_batch_files - 1) // verify_batch_files
for batch_idx in tqdm(range(num_verify_batches), desc="Validation Batches"):
    batch_verify = verify_data_list[batch_idx * verify_batch_files : (batch_idx + 1) * verify_batch_files]
    for data in tqdm(batch_verify, desc="Validation Files", leave=False):
        block_rewards = []
        env = WeatherEnv(global_config, data)
        num_blocks = len(data.get("predicted_records", [])) // 24
        file_name = data.get("station_id", f"file_{file_idx}")
        for i in range(num_blocks):
            state = env.reset()  # 每次 reset 取得一個 block（24 筆資料）
            try:
                model.eval()
                with torch.no_grad():
                    state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                    time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = model(state_tensor, time_tensor)
                    action_dict = derive_action_dict(q_vals, global_config.get("output_ids", []))
                _, block_reward, _, _ = env.step(action_dict)
                block_rewards.append(block_reward)
                writer.add_scalar("Validation/BlockReward", block_reward, total_steps + i)
            except Exception as e:
                logging.error(f"Error during validation on block {i} of file {file_name}: {traceback.format_exc()}")
        avg_val_reward = np.mean(block_rewards) if block_rewards else 0.0
        writer.add_scalar("Validation/AverageReward", avg_val_reward, total_steps)
        writer.add_scalar(f"Validation/{file_name}_AverageReward", avg_val_reward, total_steps)
        print(f"驗證檔案 {file_name} 平均 Reward: {avg_val_reward:.4f}")
        file_rewards.append((file_name, avg_val_reward))
        file_idx += 1
    torch.cuda.empty_cache()

writer.close()
print("Dueling DQN 完整訓練、測試與驗證完成。")
