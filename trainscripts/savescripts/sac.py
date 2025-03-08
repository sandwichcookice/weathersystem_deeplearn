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


# -------------------- 輔助函數：批次資料加載 -------------------------
def get_file_paths(data_dir):
    """
    取得 data_dir 目錄下所有 JSON 檔案的完整路徑，排除以 "._" 開頭的檔案。
    """
    file_paths = [f for f in glob.glob(os.path.join(data_dir, "*.json"))
                  if not os.path.basename(f).startswith("._")]
    return file_paths

def load_json_files_batch(file_paths):
    """
    讀取一批 JSON 檔案，返回列表，每個元素為該檔案內容（字典），
    並若未包含 "station_id"，則以檔案 basename 補上。
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

# -------------------- 輔助函數：將連續行動向量轉為布林決策字典 -------------------------
def action_vector_to_dict(action_vector, output_ids, threshold=0.5):
    """
    將連續行動向量（shape (1,11)）經過 threshold 轉換為布林決策字典。
    """
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
        if batch_size > len(self.buffer):
            raise ValueError("Sample size larger than buffer size.")
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# -------------------- SAC 網路定義 --------------------
# Actor：輸出每個行動的平均值與 log_std，利用 reparameterization 來採樣
class WeatherSACActor(nn.Module):
    def __init__(self):
        super(WeatherSACActor, self).__init__()
        # 兩層 Bi-LSTM 前端特徵提取
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # 輸出層：產生 mean 與 log_std，每個維度為 11
        self.mean_layer = nn.Linear(128, 11)
        self.log_std_layer = nn.Linear(128, 11)
    
    def forward(self, x, time_weight):
        """
        x: (B, 24, 7)
        time_weight: (B, 24)
        返回：mean (B,11), log_std (B,11)
        """
        lstm1_out, _ = self.lstm1(x)  # (B, 24, 128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B, 24, 1)
        attn1_out = lstm1_out * fixed_weight  # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)  # (B, 24, 128)
        pooled = torch.mean(lstm2_out, dim=1)  # (B, 128)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        mean = self.mean_layer(shared)
        log_std = self.log_std_layer(shared)
        # 限制 log_std 的範圍，避免過大或過小
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

# Critic：輸出 Q 值
class WeatherSACCritic(nn.Module):
    def __init__(self):
        super(WeatherSACCritic, self).__init__()
        # 狀態前端特徵提取
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 128)
        # 將狀態特徵與行動連結（行動維度為 11）
        self.fc2 = nn.Linear(128 + 11, 128)
        self.fc3 = nn.Linear(128, 1)
    
    def forward(self, x, time_weight, action):
        """
        x: (B, 24, 7)
        time_weight: (B, 24)
        action: (B, 11)
        """
        lstm1_out, _ = self.lstm1(x)  # (B,24,128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B,24,1)
        attn1_out = lstm1_out * fixed_weight  # (B,24,128)
        lstm2_out, _ = self.lstm2(attn1_out)  # (B,24,128)
        pooled = torch.mean(lstm2_out, dim=1)  # (B,128)
        state_features = F.relu(self.fc1(pooled))  # (B,128)
        x_cat = torch.cat([state_features, action], dim=1)  # (B,128+11)
        x_cat = F.relu(self.fc2(x_cat))  # (B,128)
        q_value = self.fc3(x_cat)  # (B,1)
        return q_value

# -------------------- Actor 動作採樣與 log_prob 計算 --------------------
def sample_action(actor, state_tensor, time_tensor):
    """
    透過 actor 進行 forward pass，利用 reparameterization trick 進行採樣，
    返回：行動 (B,11) 與對應 log_prob (B,)
    """
    mean, log_std = actor(state_tensor, time_tensor)
    std = log_std.exp()
    # 重新參數化採樣
    normal = torch.distributions.Normal(mean, std)
    z = normal.rsample()  # reparameterization trick
    # 採用 Sigmoid 將輸出壓至 [0,1]
    action = torch.sigmoid(z)
    # 計算 log_prob，需加上 Sigmoid 的變數變換修正項
    # 參考：https://arxiv.org/abs/1812.05905 (SAC 實作)
    log_prob = normal.log_prob(z) - torch.log(action * (1 - action) + 1e-6)
    log_prob = log_prob.sum(dim=1, keepdim=True)
    return action, log_prob

# -------------------- SAC Worker 函數 --------------------
def sac_worker(worker_id, global_actor, global_critic1, global_critic2, target_actor, target_critic1, target_critic2,
               actor_optimizer, critic_optimizer, global_config, training_batch, global_step, max_global_steps, lock, device, replay_buffer, gamma, alpha, policy_update_delay):
    worker_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "sac_full")
    writer = SummaryWriter(log_dir=worker_log_dir)
    print(f"SAC Worker {worker_id} started.")
    
    local_actor = WeatherSACActor().to(device)
    local_actor.load_state_dict(global_actor.state_dict())
    local_critic1 = WeatherSACCritic().to(device)
    local_critic1.load_state_dict(global_critic1.state_dict())
    local_critic2 = WeatherSACCritic().to(device)
    local_critic2.load_state_dict(global_critic2.state_dict())
    
    seed = 1234 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pbar = tqdm(total=max_global_steps, desc=f"SAC Worker {worker_id}", position=worker_id, leave=False)
    update_counter = 0
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
                action, log_prob = sample_action(local_actor, state_tensor, time_tensor)
            # 將連續行動轉換為布林字典（以 0.5 為閾值）
            action_dict = dict(zip(global_config.get("output_ids", []),
                                     [bool(int(a.item()>=0.5)) for a in action.squeeze(0)]))
            try:
                next_state, reward, done, _ = env.step(action_dict)
            except Exception as e:
                logging.error(f"Worker {worker_id} env.step error for region {region_id}: {traceback.format_exc()}")
                reward = 0
                next_state = env.reset()
                done = True
            ep_reward += reward
            # 儲存 transition 到 replay buffer
            transition = (state, action.cpu().squeeze(0).tolist(), reward, next_state, done)
            with lock:
                replay_buffer.push(*transition)
            state = next_state
            pbar.update(1)
            
            # 當 replay buffer 足夠大時進行模型更新
            if len(replay_buffer) >= global_config["training"].get("learning_starts", 1000):
                try:
                    batch_size_train = global_config["training"].get("batch_size", 64)
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size_train)
                    
                    states_tensor = torch.tensor(np.array([s["weather"] for s in states]), dtype=torch.float32, device=device)
                    states_time = torch.tensor(np.array([s["time_weight"] for s in states]), dtype=torch.float32, device=device)
                    next_states_tensor = torch.tensor(np.array([s["weather"] for s in next_states]), dtype=torch.float32, device=device)
                    next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]), dtype=torch.float32, device=device)
                    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)  # (batch, 11)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
                    
                    # Critic更新
                    with torch.no_grad():
                        # 目標行動採用 target_actor 與 reparameterization，並計算 log_prob_target
                        next_action, next_log_prob = sample_action(target_actor, next_states_tensor, next_states_time)
                        target_q1 = target_critic1(next_states_tensor, next_states_time, next_action)
                        target_q2 = target_critic2(next_states_tensor, next_states_time, next_action)
                        target_q = torch.min(target_q1, target_q2) - alpha * next_log_prob
                        y = rewards_tensor + gamma * target_q * (1 - dones_tensor)
                    
                    current_q1 = local_critic1(states_tensor, states_time, actions_tensor)
                    current_q2 = local_critic2(states_tensor, states_time, actions_tensor)
                    critic_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
                    
                    # Actor 更新：使用 reparameterization 並最大化 Q - α * log_prob
                    update_counter += 1
                    if update_counter % global_config["training"].get("policy_update_delay", 1) == 0:
                        local_actor.train()
                        new_action, new_log_prob = sample_action(local_actor, states_tensor, states_time)
                        actor_loss = (- local_critic1(states_tensor, states_time, new_action) + alpha * new_log_prob).mean()
                        
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()
                        
                        # 軟更新目標網路
                        tau = global_config["training"].get("tau", 0.005)
                        for target_param, param in zip(target_actor.parameters(), global_actor.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                        for target_param, param in zip(target_critic1.parameters(), global_critic1.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                        for target_param, param in zip(target_critic2.parameters(), global_critic2.parameters()):
                            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                        
                        total_update_loss = actor_loss.item() + critic_loss.item()
                        writer.add_scalar("Loss/train", total_update_loss, current_step)
                except Exception as e:
                    logging.error(f"Worker {worker_id} update error: {traceback.format_exc()}")
        pbar.set_postfix({"ep_reward": ep_reward})
        writer.add_scalar("Reward/episode", ep_reward, current_step)
    pbar.close()
    writer.close()
    print(f"SAC Worker {worker_id} finished.")

# -------------------- Checkpoint Saver --------------------
def checkpoint_saver(global_actor, global_step, checkpoint_interval, model_dir, max_global_steps):
    next_checkpoint = checkpoint_interval
    while global_step.value < max_global_steps:
        time.sleep(5)
        if global_step.value >= next_checkpoint:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, f"sac_checkpoint_step{global_step.value}.pth")
            torch.save(global_actor.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {global_step.value}")
            next_checkpoint += checkpoint_interval

# -------------------- 主程式 --------------------
def main():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "..", "config.json")
    global_config = load_json_file(config_path)
    print(f"file : {config_path} is loaded")
    
    # 取得所有資料檔案路徑，並分批加載
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
    model_dir = os.path.join(base_dir, "..", "model", "sac")
    batch_size_train = global_config["training"].get("batch_size", 64)
    policy_update_delay = global_config["training"].get("policy_update_delay", 1)
    alpha = global_config["training"].get("alpha", 0.2)  # 熵係數
    noise_scale = global_config["training"].get("exploration_noise", 0.2)
    noise_clip = global_config["training"].get("noise_clip", 0.5)  # 此 SAC 版本主要用於 target policy 平滑噪聲
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "sac_full"))
    
    # 建立全局網路： Actor 與 Twin Critics，並建立對應的目標網路
    global_actor = WeatherSACActor().to(device)
    global_critic1 = WeatherSACCritic().to(device)
    global_critic2 = WeatherSACCritic().to(device)
    global_actor.share_memory()
    global_critic1.share_memory()
    global_critic2.share_memory()
    
    target_actor = WeatherSACActor().to(device)
    target_actor.load_state_dict(global_actor.state_dict())
    target_critic1 = WeatherSACCritic().to(device)
    target_critic1.load_state_dict(global_critic1.state_dict())
    target_critic2 = WeatherSACCritic().to(device)
    target_critic2.load_state_dict(global_critic2.state_dict())
    
    actor_optimizer = torch.optim.Adam(global_actor.parameters(), lr=lr)
    critic_optimizer = torch.optim.Adam(list(global_critic1.parameters()) + list(global_critic2.parameters()), lr=lr)
    
    global_step = mp.Value('i', 0)
    lock = mp.Lock()
    
    # 建立全局 Replay Buffer
    replay_buffer = ReplayBuffer(global_config["training"].get("replay_buffer_size", 50000))
    
    # 啟動 checkpoint saver 線程
    checkpoint_thread = threading.Thread(target=checkpoint_saver, args=(global_actor, global_step, checkpoint_interval, model_dir, max_global_steps))
    checkpoint_thread.daemon = True
    checkpoint_thread.start()
    
    num_workers = 1  # 可根據需求調整 worker 數量
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=sac_worker, args=(worker_id, global_actor, global_critic1, global_critic2,
                                                  target_actor, target_critic1, target_critic2,
                                                  actor_optimizer, critic_optimizer, global_config, training_data_list,
                                                  global_step, max_global_steps, lock, device, replay_buffer, gamma, alpha, policy_update_delay))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    writer.close()
    print("SAC 完整訓練完成。")
    
    # -------------------------------
    # 測試部分：每個測試檔案單次 forward pass
    print("開始測試（SAC）...")
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
                mean, log_std = global_actor(state_tensor, time_tensor)
                action, _ = sample_action(global_actor, state_tensor, time_tensor)
                action_dict = dict(zip(global_config.get("output_ids", []),
                                         [bool(int(a.item()>=0.5)) for a in action.squeeze(0)]))
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
    # 驗證部分：對每個驗證檔案，將資料分成若干個 block，每 block 用 env.reset() 取得一組 24 筆資料後計算 reward
    print("開始驗證（SAC）...")
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
                    mean, log_std = global_actor(state_tensor, time_tensor)
                    action, _ = sample_action(global_actor, state_tensor, time_tensor)
                    action_dict = dict(zip(global_config.get("output_ids", []),
                                             [bool(int(a.item()>=0.5)) for a in action.squeeze(0)]))
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
    print("SAC 完整訓練、測試與驗證完成。")
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
