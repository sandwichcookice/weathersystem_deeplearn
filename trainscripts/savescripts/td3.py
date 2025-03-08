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
    讀取一批 JSON 檔案，返回一個列表，每個元素為該檔案內容（字典）。
    同時若檔案中未包含 "station_id"，則以檔案 basename 作為 station_id。
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

# -------------------- 輔助函數：將模型輸出轉換為布林決策字典 -------------------------
def derive_action_dict(q_vals, output_ids):
    """
    將模型輸出 q_vals (shape (1, 22)) 重塑為 (11, 2)，取每組的 argmax，
    將 0 轉為 False，1 轉為 True，返回對應的字典。
    """
    q_vals = q_vals.view(11, 2)
    decisions = q_vals.argmax(dim=1)
    bool_decisions = [bool(int(val.item())) for val in decisions]
    return dict(zip(output_ids, bool_decisions))

def action_dict_to_index(action_dict, output_ids):
    """將 action 字典轉換為 0/1 向量 (長度 11)。"""
    return [0 if not action_dict.get(k, False) else 1 for k in output_ids]

# -------------------- Replay Buffer 定義 -------------------------
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

# -------------------- TD3 網路定義 -------------------------
class WeatherTD3Actor(nn.Module):
    def __init__(self):
        super(WeatherTD3Actor, self).__init__()
        # 兩層 Bi-LSTM 作為前端特徵提取（輸入維度固定為 7）
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # Actor 分支：輸出 11 個連續行動值 (經 Sigmoid 激活，介於 0 與 1)
        self.actor = nn.Linear(128, 11)

    def forward(self, x, time_weight):
        # x: (B, 24, 7), time_weight: (B, 24)
        lstm1_out, _ = self.lstm1(x)  # (B, 24, 128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B, 24, 1)
        attn1_out = lstm1_out * fixed_weight  # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)  # (B, 24, 128)
        pooled = torch.mean(lstm2_out, dim=1)  # (B, 128)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        actions = torch.sigmoid(self.actor(shared))  # (B, 11)
        return actions

class WeatherTD3Critic(nn.Module):
    def __init__(self):
        super(WeatherTD3Critic, self).__init__()
        # 前端狀態特徵提取（與 Actor 相同的部分）
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(128, 128)
        # 融合狀態與行動（行動維度 11）
        self.fc2 = nn.Linear(128 + 11, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, time_weight, action):
        # x: (B, 24, 7), time_weight: (B, 24), action: (B, 11)
        lstm1_out, _ = self.lstm1(x)  # (B, 24, 128)
        fixed_weight = time_weight.detach().unsqueeze(-1)  # (B, 24, 1)
        attn1_out = lstm1_out * fixed_weight  # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)  # (B, 24, 128)
        pooled = torch.mean(lstm2_out, dim=1)  # (B, 128)
        state_features = F.relu(self.fc1(pooled))  # (B, 128)
        x_cat = torch.cat([state_features, action], dim=1)  # (B, 128+11)
        x_cat = F.relu(self.fc2(x_cat))  # (B, 128)
        q_value = self.fc3(x_cat)  # (B, 1)
        return q_value

# -------------------- 噪聲策略：目標政策平滑 (target policy noise) --------------------
def add_noise(action, noise_scale, noise_clip):
    """
    為目標政策加上平滑噪聲，並 clip 噪聲。
    action: Tensor, shape (B, 11)
    """
    noise = torch.randn_like(action) * noise_scale
    noise = noise.clamp(-noise_clip, noise_clip)
    return (action + noise).clamp(0.0, 1.0)

# -------------------- Worker 函數：TD3 --------------------
def td3_worker(worker_id, global_actor, global_critic1, global_critic2, target_actor, target_critic1, target_critic2,
                actor_optimizer, critic_optimizer, global_config, training_batch, global_step, max_global_steps, lock, device, replay_buffer, gamma, policy_delay, noise_scale, noise_clip):
    # 每個 worker 建立獨立的 TensorBoard writer
    worker_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "td3_full")
    writer = SummaryWriter(log_dir=worker_log_dir)
    print(f"TD3 Worker {worker_id} started.")
    
    local_actor = WeatherTD3Actor().to(device)
    local_actor.load_state_dict(global_actor.state_dict())
    local_critic1 = WeatherTD3Critic().to(device)
    local_critic1.load_state_dict(global_critic1.state_dict())
    local_critic2 = WeatherTD3Critic().to(device)
    local_critic2.load_state_dict(global_critic2.state_dict())
    
    seed = 1234 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pbar = tqdm(total=max_global_steps, desc=f"TD3 Worker {worker_id}", position=worker_id, leave=False)
    
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
            # 將 state 轉 tensor
            state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
            time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
            # Actor forward pass (探索採用直接加上噪聲)
            local_actor.eval()
            with torch.no_grad():
                action = local_actor(state_tensor, time_tensor)  # (1, 11)
            # 加上探索噪聲（可選用 OU Noise 或簡單高斯噪聲）
            exploration_noise = torch.randn_like(action) * noise_scale
            action_expl = (action + exploration_noise).clamp(0.0, 1.0)
            # 將連續行動轉換成字典（以 0.5 為閾值）
            action_dict = dict(zip(global_config.get("output_ids", []),
                                     [bool(int(a.item()>=0.5)) for a in action_expl.squeeze(0)]))
            try:
                next_state, reward, done, _ = env.step(action_dict)
            except Exception as e:
                logging.error(f"Worker {worker_id} env.step error for region {region_id}: {traceback.format_exc()}")
                reward = 0
                next_state = env.reset()
                done = True
            ep_reward += reward
            # 儲存 transition 到全域 replay buffer
            transition = (state, action_expl.cpu().squeeze(0).tolist(), reward, next_state, done)
            with lock:
                replay_buffer.push(*transition)
            state = next_state
            
            # 當 replay buffer 足夠大時進行更新
            if len(replay_buffer) >= global_config["training"].get("learning_starts", 1000):
                try:
                    # Sample minibatch
                    batch_size_train = global_config["training"].get("batch_size", 64)
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size_train)
                    
                    states_tensor = torch.tensor(np.array([s["weather"] for s in states]), dtype=torch.float32, device=device)
                    states_time = torch.tensor(np.array([s["time_weight"] for s in states]), dtype=torch.float32, device=device)
                    next_states_tensor = torch.tensor(np.array([s["weather"] for s in next_states]), dtype=torch.float32, device=device)
                    next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]), dtype=torch.float32, device=device)
                    actions_tensor = torch.tensor(actions, dtype=torch.float32, device=device)  # (batch, 11)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1)
                    
                    # 目標行動：使用 target_actor 加噪聲，再 clip
                    with torch.no_grad():
                        target_actions = target_actor(next_states_tensor, next_states_time)
                        target_actions = add_noise(target_actions, noise_scale, noise_clip)
                        target_q1 = target_critic1(next_states_tensor, next_states_time, target_actions)
                        target_q2 = target_critic2(next_states_tensor, next_states_time, target_actions)
                        target_q = torch.min(target_q1, target_q2)
                        y = rewards_tensor + gamma * target_q * (1 - dones_tensor)
                    
                    # 更新 Critic 1 與 Critic 2
                    current_q1 = local_critic1(states_tensor, states_time, actions_tensor)
                    current_q2 = local_critic2(states_tensor, states_time, actions_tensor)
                    critic_loss1 = F.mse_loss(current_q1, y)
                    critic_loss2 = F.mse_loss(current_q2, y)
                    critic_loss = critic_loss1 + critic_loss2
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    critic_optimizer.step()
                    
                    # 延遲 Actor 更新：每 policy_delay 次更新一次 Actor 與目標網路
                    update_counter += 1
                    if update_counter % policy_delay == 0:
                        local_actor.train()
                        actor_actions = local_actor(states_tensor, states_time)
                        # Actor loss：最大化 Q(s, μ(s))，因此取負值
                        actor_loss = -local_critic1(states_tensor, states_time, actor_actions).mean()
                        
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
    print(f"TD3 Worker {worker_id} finished.")

# -------------------- Checkpoint Saver --------------------
def checkpoint_saver(global_actor, global_step, checkpoint_interval, model_dir, max_global_steps):
    next_checkpoint = checkpoint_interval
    while global_step.value < max_global_steps:
        time.sleep(5)
        if global_step.value >= next_checkpoint:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, f"td3_checkpoint_step{global_step.value}.pth")
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
        # 可根據需求在每批訓練後釋放該批資料
    
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
    model_dir = os.path.join(base_dir, "..", "model", "td3")
    batch_size_train = global_config["training"].get("batch_size", 64)
    policy_delay = global_config["training"].get("policy_delay", 2)
    noise_scale = global_config["training"].get("exploration_noise", 0.2)
    noise_clip = global_config["training"].get("noise_clip", 0.5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "td3_full"))
    
    # 建立全局網路： Actor 與 Twin Critics，並建立對應的目標網路
    global_actor = WeatherTD3Actor().to(device)
    global_critic1 = WeatherTD3Critic().to(device)
    global_critic2 = WeatherTD3Critic().to(device)
    global_actor.share_memory()
    global_critic1.share_memory()
    global_critic2.share_memory()
    
    target_actor = WeatherTD3Actor().to(device)
    target_actor.load_state_dict(global_actor.state_dict())
    target_critic1 = WeatherTD3Critic().to(device)
    target_critic1.load_state_dict(global_critic1.state_dict())
    target_critic2 = WeatherTD3Critic().to(device)
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
        p = mp.Process(target=td3_worker, args=(worker_id, global_actor, global_critic1, global_critic2,
                                                 target_actor, target_critic1, target_critic2,
                                                 actor_optimizer, critic_optimizer, global_config, training_data_list,
                                                 global_step, max_global_steps, lock, device, replay_buffer, gamma,
                                                 policy_delay, noise_scale, noise_clip))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    writer.close()
    print("TD3 完整訓練完成。")
    
    # -------------------------------
    # 測試部分：每個測試檔案單次 forward pass
    print("開始測試（TD3）...")
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
                action = global_actor(state_tensor, time_tensor)
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
    # 驗證部分：對每個驗證檔案，將資料分成若干個 block，每 block 用 env.reset() 取得一組 24 筆資料後計算 reward
    print("開始驗證（TD3）...")
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
    print("TD3 完整訓練、測試與驗證完成。")
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
