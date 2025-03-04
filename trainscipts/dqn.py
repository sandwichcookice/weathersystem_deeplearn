#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
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
    WeatherEnv,
    WeatherDQNModel
)

# 輔助函數：將模型輸出轉換為布林決策字典
def derive_action_dict(q_vals, output_ids):
    """
    將模型輸出 q_vals (shape (1, 22)) 重塑為 (11,2)，並對每組取 argmax，
    將 0 轉換為 False，1 轉換為 True，返回對應字典。
    """
    q_vals = q_vals.view(11, 2)
    decisions = q_vals.argmax(dim=1)
    bool_decisions = [bool(val.item()) for val in decisions]
    return dict(zip(output_ids, bool_decisions))

# 輔助函數：將 action 字典轉換為 0/1 向量 (長度 11)
def action_dict_to_index(action_dict, output_ids):
    return [0 if not action_dict.get(k, False) else 1 for k in output_ids]

# 讀取配置及資料
base_dir = os.path.dirname(__file__)
global_config = load_json_file(os.path.join(base_dir, "..", "config.json"))
training_data_list = load_json_files(os.path.join(base_dir, "..", "data", "training_data"))
check_data_list = load_json_files(os.path.join(base_dir, "..", "data", "check"))
verify_data_list = load_json_files(os.path.join(base_dir, "..", "data", "val"))

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

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "dqn_full"))

# 建立模型與目標模型
model = WeatherDQNModel().to(device)
target_model = WeatherDQNModel().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
buffer = ReplayBuffer(replay_capacity)
total_steps = 0
target_update_freq = 1000

print("開始訓練所有區域...")
for region in training_data_list:
    env = WeatherEnv(global_config, region)
    for ep in tqdm(range(num_epochs), desc="Training Region"):
        state = env.reset()
        done = False
        ep_reward = 0
        while not done:
            total_steps += 1
            epsilon = max(final_epsilon, initial_epsilon - (initial_epsilon - final_epsilon) * (total_steps / epsilon_timesteps))
            if random.random() < epsilon:
                # 隨機產生 11 個布林值
                random_actions = [bool(random.getrandbits(1)) for _ in range(len(env.output_ids))]
                action_dict = dict(zip(env.output_ids, random_actions))
            else:
                try:
                    model.eval()
                    with torch.no_grad():
                        state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                        time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                        q_vals = model(state_tensor, time_tensor)  # shape (1,22)
                        action_dict = derive_action_dict(q_vals, global_config.get("output_ids", []))
                except Exception as e:
                    logging.error(f"Error in action selection: {traceback.format_exc()}")
                    random_actions = [bool(random.getrandbits(1)) for _ in range(len(env.output_ids))]
                    action_dict = dict(zip(env.output_ids, random_actions))
            try:
                next_state, reward, done, _ = env.step(action_dict)
            except Exception as e:
                logging.error(f"Error in env.step for region {region}: {traceback.format_exc()}")
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
                    
                    # 將 buffer 中的 actions (字典) 轉換成 indices (shape: (batch, 11))
                    actions_indices = [action_dict_to_index(a, global_config.get("output_ids", [])) for a in actions]
                    actions_tensor = torch.tensor(actions_indices, dtype=torch.long, device=device)
                    rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                    dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
                    
                    q_vals = model(states_weather, states_time)  # shape (batch,22)
                    # 重塑成 (batch, 11, 2)
                    q_vals = q_vals.view(-1, len(global_config.get("output_ids", [])), 2)
                    actions_tensor_expanded = actions_tensor.unsqueeze(2)  # (batch,11,1)
                    chosen_q = torch.gather(q_vals, 2, actions_tensor_expanded).squeeze(2)  # (batch, 11)
                    current_q = chosen_q.mean(dim=1)
                    
                    with torch.no_grad():
                        next_q_vals = target_model(next_states_weather, next_states_time)
                        next_q_vals = next_q_vals.view(-1, len(global_config.get("output_ids", [])), 2)
                        max_next_q = next_q_vals.max(dim=2)[0]
                        next_q = max_next_q.mean(dim=1)
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
        if (ep + 1) % 1000 == 0:
            os.makedirs("./model/log", exist_ok=True)
            torch.save(model.state_dict(), f"./model/log/dqn_checkpoint_ep{ep+1}.pth")

print("所有區域訓練完成。")

print("開始測試...")
test_rewards = []
for data in check_data_list:
    try:
        # 以測試資料建立環境
        env = WeatherEnv(global_config, data)
        state = env.reset()
        model.eval()
        with torch.no_grad():
            # 將當前狀態轉為 tensor
            state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
            time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
            # 模型推理得到 Q 值
            q_vals = model(state_tensor, time_tensor)  # shape (1, 22)
            # 轉換為布林決策字典（依據 global_config 中定義的 output_ids）
            action_dict = derive_action_dict(q_vals, global_config.get("output_ids", []))
        # 執行一次環境 step，以測試模型產生的動作對 reward 的影響
        _, reward, _, _ = env.step(action_dict)
        test_rewards.append(reward)
        writer.add_scalar("Test/Reward", reward, total_steps)
    except Exception as e:
        logging.error(f"Error during testing on file: {traceback.format_exc()}")

avg_test_reward = np.mean(test_rewards) if test_rewards else 0.0
writer.add_scalar("Test/AverageReward", avg_test_reward, total_steps)
print(f"測試平均 Reward: {avg_test_reward:.4f}")

print("開始驗證...")
# 用於儲存每個驗證檔案的平均 reward，方便後續整體統計
file_rewards = []
file_idx = 0
for data in verify_data_list:
    rewards_list = []
    env = WeatherEnv(global_config, data)
    records = data.get("predicted_records", [])
    num_blocks = len(records) // 24
    # 為了方便紀錄，這裡取檔案名稱或索引作為標識
    file_name = data.get("station_id", f"file_{file_idx}")
    for i in range(num_blocks):
        block = records[i*24:(i+1)*24]
        weather = np.array([[rec.get(k, 0.0) for k in global_config.get("input_ids", [])] for rec in block], dtype=np.float32)
        time_weight = generate_time_weights(1, 24, device=device)[0, :, 0].cpu().numpy()
        state = {"weather": weather, "time_weight": time_weight}
        try:
            model.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                q_vals = model(state_tensor, time_tensor)
                action_dict = derive_action_dict(q_vals, global_config.get("output_ids", []))
            # 針對此 block，每筆記錄計算 reward，再求平均作為 block reward
            block_rewards = [evaluate_reward(global_config.get("rules", []), rec, action_dict) for rec in block]
            avg_block_reward = np.mean(block_rewards)
            rewards_list.append(avg_block_reward)
            # 寫入每個 block 的 reward，方便檢視細節
            writer.add_scalar("Validation/BlockReward", avg_block_reward, total_steps + i)
        except Exception as e:
            logging.error(f"Error during validation on block {i} of file {file_name}: {traceback.format_exc()}")
    if rewards_list:
        avg_val_reward = np.mean(rewards_list)
    else:
        avg_val_reward = 0.0
    writer.add_scalar("Validation/AverageReward", avg_val_reward, total_steps)
    writer.add_scalar(f"Validation/{file_name}_AverageReward", avg_val_reward, total_steps)
    print(f"驗證檔案 {file_name} 平均 Reward: {avg_val_reward:.4f}")
    file_rewards.append((file_name, avg_val_reward))
    file_idx += 1

writer.close()
print("DQN 完整訓練、測試與驗證完成。")