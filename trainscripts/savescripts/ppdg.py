#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import time
import threading
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import traceback

from common import (
    generate_time_weights,
    evaluate_reward,
    load_json_file,
    load_json_files,
    #ReplayBuffer,
    WeatherEnv,
    WeatherDQNModel
)

# -------------------- 輔助函數：批次資料加載 --------------------
def get_file_paths(data_dir):
    """取得 data_dir 下所有 JSON 檔案的完整路徑，排除以 "._" 開頭的檔案。"""
    file_paths = [f for f in glob.glob(os.path.join(data_dir, "*.json"))
                  if not os.path.basename(f).startswith("._")]
    return file_paths

def load_json_files_batch(file_paths):
    """
    讀取一批 JSON 檔案，返回一個列表，每個元素為該檔案內容（字典）。
    若檔案中未包含 "station_id"，則自動補上（使用檔案 basename）。
    """
    data_batch = []
    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                print(f"file : {path} is loaded")
                data = json.load(f)
                data["file_name"] = os.path.basename(path)
                if "station_id" not in data:
                    data["station_id"] = data.get("file_name", "unknown")
                data_batch.append(data)
        except Exception as e:
            logging.error(f"Error loading file {path}: {traceback.format_exc()}")
    return data_batch

# -------------------- 輔助函數：將模型輸出轉換為行動決策 --------------------
def derive_action_continuous(actor_output, threshold=0.5):
    """
    將 actor 輸出（shape (1,11)）視為連續控制值，
    並以 threshold 將其轉換為布林決策列表（0->False, 1->True）。
    """
    decisions = (actor_output >= threshold).squeeze(0)
    bool_decisions = [bool(int(val.item())) for val in decisions]
    return bool_decisions

def action_vector_to_dict(action_vector, output_ids, threshold=0.5):
    """將連續行動向量（tensor shape (1, 11)）轉換為字典。"""
    bool_actions = (action_vector >= threshold).squeeze(0).tolist()
    bool_actions = [bool(int(a)) for a in bool_actions]
    return dict(zip(output_ids, bool_actions))

# -------------------- Replay Buffer --------------------
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
        current_batch_size = batch_size
        if len(self.buffer) < batch_size:
            current_batch_size = len(self.buffer)
            # 若真的想取 batch_size 筆可以用 replace=True，但一般情況下直接用現有數量也可
            indices = np.random.choice(len(self.buffer), current_batch_size, replace=False)
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------- DDPG 模型定義 --------------------
# Actor 模型：使用兩層 Bi-LSTM 提取特徵，再透過全連接層輸出 11 維連續行動值 (經 Sigmoid 限制在 [0,1])
class WeatherDDPGActor(nn.Module):
    def __init__(self):
        super(WeatherDDPGActor, self).__init__()
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.actor = nn.Linear(128, 11)

    def forward(self, x, time_weight):
        lstm1_out, _ = self.lstm1(x)
        fixed_weight = time_weight.detach().unsqueeze(-1)
        attn1_out = lstm1_out * fixed_weight
        lstm2_out, _ = self.lstm2(attn1_out)
        pooled = torch.mean(lstm2_out, dim=1)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        actions = torch.sigmoid(self.actor(shared))
        return actions

# Critic 模型：提取狀態特徵，再結合行動向量後輸出 Q 值
class WeatherDDPGCritic(nn.Module):
    def __init__(self):
        super(WeatherDDPGCritic, self).__init__()
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128 + 11, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, time_weight, action):
        lstm1_out, _ = self.lstm1(x)
        fixed_weight = time_weight.detach().unsqueeze(-1)
        attn1_out = lstm1_out * fixed_weight
        lstm2_out, _ = self.lstm2(attn1_out)
        pooled = torch.mean(lstm2_out, dim=1)
        state_features = F.relu(self.fc1(pooled))
        x_cat = torch.cat([state_features, action], dim=1)
        x_cat = F.relu(self.fc2(x_cat))
        q_value = self.fc3(x_cat)
        return q_value

# -------------------- 噪聲策略：Ornstein-Uhlenbeck --------------------
class OUNoise:
    """Ornstein-Uhlenbeck Process for temporally correlated exploration noise."""
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(*self.size)
        self.state = self.state + dx
        return self.state

# -------------------- DDPG Worker 函數 --------------------
def ddpg_worker(worker_id, global_actor, global_critic, target_actor, target_critic,
                actor_optimizer, critic_optimizer, global_config, training_batch,
                global_step, max_global_steps, lock, device, gamma, replay_buffer):
    # 每個 worker 建立自己的 TensorBoard writer
    worker_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "ddpg_full")
    writer = SummaryWriter(log_dir=worker_log_dir)
    print(f"DDPG Worker {worker_id} started.")
    
    local_actor = WeatherDDPGActor().to(device)
    local_actor.load_state_dict(global_actor.state_dict())
    local_critic = WeatherDDPGCritic().to(device)
    local_critic.load_state_dict(global_critic.state_dict())
    
    noise = OUNoise(size=(1, 11))
    
    seed = 1234 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pbar = tqdm(total=max_global_steps, desc=f"DDPG Worker {worker_id}", position=worker_id, leave=False)
    
    while True:
        with global_step.get_lock():
            if global_step.value >= max_global_steps:
                break
        if not training_batch:
            logging.error(f"Worker {worker_id}: Training batch is empty!")
            break
        region = random.choice(training_batch)
        region_id = region.get("station_id", "unknown")
        try:
            env = WeatherEnv(global_config, region)
        except Exception as e:
            logging.error(f"Worker {worker_id} 初始化環境失敗: {traceback.format_exc()}")
            continue
        state = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            with global_step.get_lock():
                global_step.value += 1
                current_step = global_step.value
            state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
            time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
            local_actor.eval()
            with torch.no_grad():
                action = local_actor(state_tensor, time_tensor)  # (1, 11)
            noise_sample = torch.tensor(noise.sample(), dtype=torch.float32, device=device)
            action = action + noise_sample
            action = torch.clamp(action, 0.0, 1.0)
            # 以 0.5 為閾值轉換為布林行動字典 (DDPG 用連續輸出更新 Critic，但環境 step 需要字典)
            action_dict = dict(zip(global_config.get("output_ids", []),
                                     [bool(int(a.item() >= 0.5)) for a in action.squeeze(0)]))
            try:
                next_state, reward, done, _ = env.step(action_dict)
            except Exception as e:
                logging.error(f"Worker {worker_id} env.step error for region {region_id}: {traceback.format_exc()}")
                reward = 0
                next_state = env.reset()
                done = True
            ep_reward += reward
            # 儲存 transition 到全局 replay buffer
            transition = (state, action.cpu().squeeze(0).tolist(), reward, next_state, done)
            replay_buffer.push(*transition)
            print("Buffer size: ", len(replay_buffer))
            state = next_state
            
            # 當 replay buffer 足夠大時更新模型
            if len(replay_buffer) >= global_config["training"].get("learning_starts", 1000):
                states, actions, rewards, next_states, dones = replay_buffer.sample(global_config["training"].get("batch_size", 64))
                states_tensor = torch.tensor(np.array([s["weather"] for s in states]), dtype=torch.float32, device=device)
                states_time = torch.tensor(np.array([s["time_weight"] for s in states]), dtype=torch.float32, device=device)
                next_states_tensor = torch.tensor(np.array([s["weather"] for s in next_states]), dtype=torch.float32, device=device)
                next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]), dtype=torch.float32, device=device)
                actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)  # (batch, 11)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
                
                with torch.no_grad():
                    next_actions = target_actor(next_states_tensor, next_states_time)
                    target_q = target_critic(next_states_tensor, next_states_time, next_actions)
                    y = rewards_tensor + gamma * target_q * (1 - dones_tensor)
                current_q = local_critic(states_tensor, states_time, actions_tensor)
                critic_loss = F.mse_loss(current_q, y)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                local_actor.train()
                actor_actions = local_actor(states_tensor, states_time)
                actor_loss = -local_critic(states_tensor, states_time, actor_actions).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # 軟更新目標網路
                tau = global_config["training"].get("tau", 0.005)
                for target_param, param in zip(target_actor.parameters(), global_actor.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for target_param, param in zip(target_critic.parameters(), global_critic.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                total_loss = actor_loss + critic_loss
                writer.add_scalar("Loss/train", total_loss.item(), current_step)
            pbar.set_postfix({"ep_reward": ep_reward})
        writer.add_scalar("Reward/episode", ep_reward, current_step)
        pbar.update(0)
    pbar.close()
    writer.close()
    print(f"DDPG Worker {worker_id} finished.")

# -------------------- Checkpoint Saver --------------------
def checkpoint_saver(global_actor, global_step, checkpoint_interval, model_dir, max_global_steps):
    next_checkpoint = checkpoint_interval
    while global_step.value < max_global_steps:
        time.sleep(5)
        if global_step.value >= next_checkpoint:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, f"ddpg_checkpoint_step{global_step.value}.pth")
            torch.save(global_actor.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {global_step.value}")
            next_checkpoint += checkpoint_interval

# -------------------- 主程式 --------------------
def main():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "..", "config.json")
    global_config = load_json_file(config_path)
    print(f"file : {config_path} is loaded")
    
    # 取得所有檔案路徑並分批加載資料
    train_dir = os.path.join(base_dir, "..", "data", "training_data")
    test_dir = os.path.join(base_dir, "..", "data", "check")
    val_dir = os.path.join(base_dir, "..", "data", "val")
    
    all_train_files = get_file_paths(train_dir)
    all_test_files = get_file_paths(test_dir)
    all_val_files = get_file_paths(val_dir)
    
    print(f"共掃描到 {len(all_train_files)} 個訓練檔案。")
    print(f"共掃描到 {len(all_test_files)} 個測試檔案。")
    print(f"共掃描到 {len(all_val_files)} 個驗證檔案。")
    
    batch_size_files = global_config["training"].get("batch_file_size", 50)
    train_batches = [all_train_files[i:i+batch_size_files] for i in range(0, len(all_train_files), batch_size_files)]
    training_data_list = []
    for batch_paths in train_batches:
        batch_data = load_json_files_batch(batch_paths)
        training_data_list.extend(batch_data)
        # 若需釋放記憶體，可在此批訓練完後釋放 batch_data，再加載下一批
    
    check_data_list = load_json_files_batch(all_test_files)
    verify_data_list = load_json_files_batch(all_val_files)
    
    print(f"共讀取到 {len(training_data_list)} 筆訓練資料。")
    print(f"共讀取到 {len(check_data_list)} 筆測試資料。")
    print(f"共讀取到 {len(verify_data_list)} 筆驗證資料。")
    print("資料讀取完成。")
    
    # 訓練參數
    lr = global_config["training"].get("learning_rate", 0.001)
    gamma = global_config["training"].get("gamma", 0.99)
    max_global_steps = global_config["training"].get("max_global_steps", 10830)
    checkpoint_interval = global_config["training"].get("checkpoint_interval", 1000)
    model_dir = os.path.join(base_dir, "..", "model", "ddpg")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "ddpg_full"))
    
    global_actor = WeatherDDPGActor().to(device)
    global_critic = WeatherDDPGCritic().to(device)
    global_actor.share_memory()
    global_critic.share_memory()
    
    target_actor = WeatherDDPGActor().to(device)
    target_actor.load_state_dict(global_actor.state_dict())
    target_critic = WeatherDDPGCritic().to(device)
    target_critic.load_state_dict(global_critic.state_dict())
    
    actor_optimizer = torch.optim.Adam(global_actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(global_critic.parameters(), lr=lr)
    
    global_step = mp.Value('i', 0)
    lock = mp.Lock()
    
    replay_buffer = ReplayBuffer(global_config["training"].get("replay_buffer_size", 50000))
    
    # 啟動 checkpoint saver 線程
    checkpoint_thread = threading.Thread(target=checkpoint_saver, args=(global_actor, global_step, checkpoint_interval, model_dir, max_global_steps))
    checkpoint_thread.daemon = True
    checkpoint_thread.start()
    
    num_workers = 1  # 可根據需要調整 worker 數量
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=ddpg_worker, args=(worker_id, global_actor, global_critic, target_actor, target_critic,
                                                  actor_optimizer, critic_optimizer, global_config, training_data_list,
                                                  global_step, max_global_steps, lock, device, gamma, replay_buffer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    writer.close()
    print("DDPG 完整訓練完成。")
    
    # -------------------------------
    # 測試部分
    print("開始測試（DDPG）...")
    test_rewards = []
    pbar_test = tqdm(total=len(check_data_list), desc="Testing")
    for data in check_data_list:
        try:
            env = WeatherEnv(global_config, data)
            state = env.reset()
            global_actor.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                action = global_actor(state_tensor, time_tensor)  # (1, 11)
                action_dict = dict(zip(global_config.get("output_ids", []),
                                         [bool(int(a.item() >= 0.5)) for a in action.squeeze(0)]))
            _, reward, _, _ = env.step(action_dict)
            test_rewards.append(reward)
            writer.add_scalar("Test/Reward", reward, global_step.value)
        except Exception as e:
            logging.error(f"Error during testing: {traceback.format_exc()}")
        pbar_test.update(1)
    pbar_test.close()
    avg_test_reward = np.mean(test_rewards) if test_rewards else 0.0
    writer.add_scalar("Test/AverageReward", avg_test_reward, global_step.value)
    print(f"測試平均 Reward: {avg_test_reward:.4f}")
    
    # -------------------------------
    # 驗證部分：分 block 驗證，每個 block 使用 24 筆資料
    print("開始驗證（DDPG）...")
    file_rewards = []
    file_idx = 0
    pbar_val = tqdm(total=len(verify_data_list), desc="Validation Files")
    for data in verify_data_list:
        env = WeatherEnv(global_config, data)
        records = data.get("predicted_records", [])
        num_blocks = len(records) // 24
        file_name = data.get("station_id", f"file_{file_idx}")
        block_rewards = []
        for i in tqdm(range(num_blocks), desc=f"Validating {file_name}", leave=False):
            state = env.reset()
            try:
                global_actor.eval()
                with torch.no_grad():
                    state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                    time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                    action = global_actor(state_tensor, time_tensor)
                    action_dict = dict(zip(global_config.get("output_ids", []),
                                             [bool(int(a.item() >= 0.5)) for a in action.squeeze(0)]))
                _, block_reward, _, _ = env.step(action_dict)
                block_rewards.append(block_reward)
                writer.add_scalar("Validation/BlockReward", block_reward, global_step.value + i)
            except Exception as e:
                logging.error(f"Error during validation on block {i} of file {file_name}: {traceback.format_exc()}")
        avg_val_reward = np.mean(block_rewards) if block_rewards else 0.0
        writer.add_scalar("Validation/AverageReward", avg_val_reward, global_step.value)
        writer.add_scalar(f"Validation/{file_name}_AverageReward", avg_val_reward, global_step.value)
        print(f"驗證檔案 {file_name} 平均 Reward: {avg_val_reward:.4f}")
        file_rewards.append((file_name, avg_val_reward))
        file_idx += 1
        pbar_val.update(1)
    pbar_val.close()
    
    writer.close()
    print("DDPG 完整訓練、測試與驗證完成。")
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
