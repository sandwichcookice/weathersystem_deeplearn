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

# ---------------------
# 輔助函數：將模型輸出轉換為布林決策字典
def derive_action_dict(q_vals, output_ids):
    """
    將模型輸出 q_vals (shape (1,22)) 重塑為 (11,2)，對每組取 argmax，
    將 0 轉換為 False，1 轉換為 True，返回對應字典。
    """
    q_vals = q_vals.view(11, 2)
    decisions = q_vals.argmax(dim=1)
    bool_decisions = [bool(val.item()) for val in decisions]
    return dict(zip(output_ids, bool_decisions))

# 輔助函數：將 action 字典轉換為 0/1 向量 (長度 11)
def action_dict_to_index(action_dict, output_ids):
    return [0 if not action_dict.get(k, False) else 1 for k in output_ids]

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
batch_size = global_config["training"].get("batch_size", 64)
num_epochs = global_config["training"].get("num_epochs", 100)   # 每個檔案訓練的 epoch 數
gamma = global_config["training"].get("gamma", 0.99)
learning_starts = global_config["training"].get("learning_starts", 1000)
replay_capacity = global_config["training"].get("replay_buffer_size", 50000)
epsilon_timesteps = global_config["training"]["exploration_config"].get("epsilon_timesteps", 10000)
final_epsilon = global_config["training"]["exploration_config"].get("final_epsilon", 0.02)
initial_epsilon = 1.0

device = torch.device("mps" if torch.backends.mps.is_available() else 
                        "cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "ddqn_full"))

# ---------------------
# 建立模型與目標模型
model = WeatherDQNModel().to(device)
target_model = WeatherDQNModel().to(device)
target_model.load_state_dict(model.state_dict())
target_model.eval()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
buffer = ReplayBuffer(replay_capacity)
total_steps = 0
target_update_freq = 1000

# ---------------------
# 訓練資料分批加載設定
batch_files = 10
num_total_files = len(training_data_list)
num_batches = (num_total_files + batch_files - 1) // batch_files
global_epoch = 0  # 累計訓練 epoch 數

print("開始訓練所有區域 (DDQN) - 分批加載：")
for batch_idx in range(num_batches):
    batch_data = training_data_list[batch_idx * batch_files : (batch_idx + 1) * batch_files]
    print(f"\n== 開始訓練批次 {batch_idx + 1}/{num_batches} (檔案數: {len(batch_data)}) ==")
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
                epsilon = max(final_epsilon, initial_epsilon - (initial_epsilon - final_epsilon) * (total_steps / epsilon_timesteps))
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
                        
                        q_vals = model(states_weather, states_time)  # shape (batch,22)
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
                checkpoint_dir = os.path.join(base_dir, "..", "model", "ddqn")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"ddqn_checkpoint_ep{global_epoch}.pth"))
    # 完成一批檔案後清空 GPU 記憶體
    torch.cuda.empty_cache()
    print(f"批次 {batch_idx + 1} 訓練完成。")

print("所有區域訓練完成。")

# ---------------------
# 測試部分（分批處理，加入進度條）
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
# 驗證部分（分批處理，加入進度條）
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
print("DDQN 完整訓練、測試與驗證完成。")
