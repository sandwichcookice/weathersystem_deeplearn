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
    WeatherEnv
)
from OnPolicycommon import (
    derive_action_dict_policy,
    compute_returns
)

# ---------------------
# 定義 Policy Gradient (REINFORCE) 模型
class WeatherPolicyGradientModel(nn.Module):
    def __init__(self):
        super(WeatherPolicyGradientModel, self).__init__()
        # 兩層 Bi-LSTM
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # 輸出層：產生 11 個動作的概率（經過 sigmoid 激活）
        self.policy = nn.Linear(128, 11)
    
    def forward(self, x, time_weight):
        """
        x: Tensor, shape (B, 24, 7) —— 天氣數據
        time_weight: Tensor, shape (B, 24) —— 固定時間權重（不參與反向傳播）
        """
        lstm1_out, _ = self.lstm1(x)                       # (B, 24, 128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B, 24, 1)
        attn1_out = lstm1_out * fixed_weight                # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)                # (B, 24, 128)
        pooled = torch.mean(lstm2_out, dim=1)               # (B, 128)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        probs = torch.sigmoid(self.policy(shared))          # (B, 11) 每個數值介於0與1
        return probs

# ---------------------
# 讀取配置及資料
base_dir = os.path.dirname(__file__)
global_config = load_json_file(os.path.join(base_dir, "..", "config.json"))
training_data_list = load_json_files(os.path.join(base_dir, "..", "data", "training_data"))
check_data_list = load_json_files(os.path.join(base_dir, "..", "data", "check"))
verify_data_list = load_json_files(os.path.join(base_dir, "..", "data", "val"))

# ---------------------
# 訓練參數
lr = global_config["training"].get("learning_rate", 0.001)
num_epochs = global_config["training"].get("num_epochs", 100)  # 每個檔案的訓練 epoch 數
gamma = global_config["training"].get("gamma", 0.99)             # 單步環境：G = reward
# REINFORCE 為 on-policy，不使用 replay buffer

device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "reinforce_lightTraining"))

# ---------------------
# 建立模型與 optimizer
model = WeatherPolicyGradientModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
total_steps = 0

# ---------------------
# 分批加載訓練資料，每批 10 個檔案
batch_files = 100
num_total_files = len(training_data_list)
num_batches = (num_total_files + batch_files - 1) // batch_files
global_epoch = 0

print("開始訓練所有區域 (REINFORCE) - 分批加載：")
for batch_idx in range(num_batches):
    batch_data = training_data_list[batch_idx * batch_files : (batch_idx + 1) * batch_files]
    print(f"\n== 訓練批次 {batch_idx+1}/{num_batches} (檔案數: {len(batch_data)}) ==")
    for region in batch_data:
        region_id = region.get("station_id", "unknown")
        print(f"訓練區域: {region_id}")
        env = WeatherEnv(global_config, region)
        for ep in tqdm(range(num_epochs), desc=f"訓練 {region_id}"):
            state = env.reset()
            log_probs = []
            rewards = []
            done = False
            model.train()
            while not done:
                total_steps += 1
                try:
                    state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                    time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                    probs = model(state_tensor, time_tensor)  # (1, 11)
                    # 以概率建立 Bernoulli 分布，抽樣動作
                    m = torch.distributions.Bernoulli(probs)
                    action_sample = m.sample()  # (1, 11)
                    log_prob = m.log_prob(action_sample).sum()  # 總和所有動作的 log probability
                    action_dict = dict(zip(global_config.get("output_ids", []),
                                             [bool(int(a.item())) for a in action_sample.squeeze(0)]))
                except Exception as e:
                    logging.error(f"Error in action selection: {traceback.format_exc()}")
                    random_actions = [bool(random.getrandbits(1)) for _ in range(len(env.output_ids))]
                    action_dict = dict(zip(env.output_ids, random_actions))
                    log_prob = torch.tensor(0.0, device=device)
                try:
                    next_state, reward, done, _ = env.step(action_dict)
                except Exception as e:
                    logging.error(f"Error in env.step for region {region_id}: {traceback.format_exc()}")
                    reward = 0
                    next_state = env.reset()
                    done = True
                log_probs.append(log_prob)
                rewards.append(reward)
                state = next_state
            # 對於單步環境，累計 reward 即為 G
            G = sum(rewards)
            loss = - sum(log_probs) * G
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("Loss/train", loss.item(), total_steps)
            writer.add_scalar("Reward/episode", G, total_steps)
            global_epoch += 1
            if global_epoch % 1000 == 0:
                checkpoint_dir = os.path.join(base_dir, "..", "model", "reinforce_lightTraining")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"reinforce_checkpoint_ep{global_epoch}.pth"))
        # 單一區域訓練結束後
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
                probs = model(state_tensor, time_tensor)
                action_dict = dict(zip(global_config.get("output_ids", []),
                                         [bool(p.item() >= 0.5) for p in probs.squeeze(0)]))
            # 注意：測試時環境內部依然使用真實天氣計算 reward
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
# 驗證部分（分批處理並加入進度條，記錄每個驗證檔案的平均 Reward）
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
                    probs = model(state_tensor, time_tensor)
                    action_dict = dict(zip(global_config.get("output_ids", []),
                                             [bool(p.item() >= 0.5) for p in probs.squeeze(0)]))
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
print("REINFORCE 完整訓練、測試與驗證完成。")
