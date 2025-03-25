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
    WeatherEnv,
    WeatherD3QNModel
)

# ----------------------------------------------------
# 1. 輔助函數：將模型輸出轉換為布林決策字典
# ----------------------------------------------------
def derive_action_dict(q_vals, output_ids):
    """
    q_vals: shape (1, num_decisions*2) 或 (1, num_decisions, 2)
    取每個決策 (2維) 的 argmax -> 布林決策
    """
    if len(q_vals.shape) == 3:
        # (1, num_decisions, 2)
        q_vals = q_vals.squeeze(0)
    else:
        # (1, num_decisions*2) -> reshape
        num_actions = q_vals.shape[1]
        num_decisions = num_actions // 2
        q_vals = q_vals.view(num_decisions, 2)
    decisions = q_vals.argmax(dim=1)
    bool_decisions = [bool(val.item()) for val in decisions]
    return dict(zip(output_ids, bool_decisions))

def action_dict_to_index(action_dict, output_ids):
    return [0 if not action_dict.get(k, False) else 1 for k in output_ids]

# ----------------------------------------------------
# 3. 主程式：訓練 / 測試 / 驗證
# ----------------------------------------------------
def main():
    # 讀取配置與資料
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

    target_update_freq = 1000

    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "d3qn_full"))

    # 讀取輸出決策清單
    output_ids = global_config.get("output_ids", [])
    num_decisions = len(output_ids)

    # 從第一筆資料來取得 weather_dim
    env_sample = WeatherEnv(global_config, training_data_list[0])
    init_state = env_sample.reset()

    # 建立 D3QN Online/Target Model
    online_model = WeatherD3QNModel().to(device)
    target_model = WeatherD3QNModel().to(device)
    target_model.load_state_dict(online_model.state_dict())
    target_model.eval()

    optimizer = torch.optim.Adam(online_model.parameters(), lr=lr)
    buffer = ReplayBuffer(replay_capacity)
    total_steps = 0

    # 分批載入訓練資料
    batch_files = 10
    num_total_files = len(training_data_list)
    num_batches = (num_total_files + batch_files - 1) // batch_files
    global_epoch = 0

    print("開始訓練所有區域 (D3QN) - 分批加載：")
    for batch_idx in range(num_batches):
        batch_data = training_data_list[batch_idx * batch_files : (batch_idx + 1) * batch_files]
        print(f"\n== 開始訓練批次 {batch_idx+1}/{num_batches} (檔案數: {len(batch_data)}) ==")

        for region in batch_data:
            region_id = region.get("station_id", "unknown")
            print(f"訓練區域: {region_id}")
            env = WeatherEnv(global_config, region)

            for ep in tqdm(range(num_epochs), desc=f"D3QN訓練 {region_id}"):
                state = env.reset()
                done = False
                ep_reward = 0

                while not done:
                    total_steps += 1
                    # Epsilon 動態衰減
                    epsilon = max(final_epsilon, initial_epsilon - (initial_epsilon - final_epsilon)*(total_steps / epsilon_timesteps))

                    # 動作選擇
                    if random.random() < epsilon:
                        # random
                        random_actions = [bool(random.getrandbits(1)) for _ in range(num_decisions)]
                        action_dict = dict(zip(output_ids, random_actions))
                    else:
                        # greedy
                        online_model.eval()
                        with torch.no_grad():
                            w_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                            t_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                            q_vals = online_model(w_tensor, t_tensor)  # (1, num_decisions*2)
                            action_dict = derive_action_dict(q_vals, output_ids)

                    # 與環境互動
                    try:
                        next_state, reward, done, _ = env.step(action_dict)
                    except Exception as e:
                        logging.error(f"Error in env.step for region {region_id}: {traceback.format_exc()}")
                        reward = 0
                        next_state = env.reset()
                        done = True
                    ep_reward += reward

                    # push 到 Buffer
                    buffer.push(state, action_dict, reward, next_state, done)

                    # 狀態更新
                    state = next_state

                    # 若 buffer 足夠 -> 開始訓練
                    if len(buffer) >= learning_starts and len(buffer) >= batch_size:
                        online_model.train()
                        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                        # 處理成 Tensor
                        states_weather = torch.tensor(np.array([s["weather"] for s in states]), dtype=torch.float32, device=device)
                        states_time = torch.tensor(np.array([s["time_weight"] for s in states]), dtype=torch.float32, device=device)
                        next_states_weather = torch.tensor(np.array([s["weather"] for s in next_states]), dtype=torch.float32, device=device)
                        next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]), dtype=torch.float32, device=device)

                        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

                        # 動作 (batch_size, num_decisions)
                        actions_indices = [action_dict_to_index(a, output_ids) for a in actions]
                        actions_tensor = torch.tensor(actions_indices, dtype=torch.long, device=device)

                        # ---- 計算當前 Q(s,a) ----
                        curr_q_all = online_model(states_weather, states_time)  # (B, num_decisions*2)
                        # reshape -> (B, num_decisions, 2)
                        curr_q_all = curr_q_all.view(-1, num_decisions, 2)
                        # gather
                        # actions_tensor shape=(B,num_decisions) => expand (B,num_decisions,1)
                        actions_expanded = actions_tensor.unsqueeze(-1)  # (B, num_decisions,1)
                        chosen_q = torch.gather(curr_q_all, 2, actions_expanded).squeeze(-1)  # (B, num_decisions)

                        # 這裡可取 mean / sum 等做合併；原DDQN腳本中取 mean:
                        curr_q = chosen_q.mean(dim=1)  # (B,)

                        with torch.no_grad():
                            # ---- Double DQN ----
                            # 1) 用 online_model 選動作
                            next_q_online = online_model(next_states_weather, next_states_time)  # (B, num_decisions*2)
                            next_q_online = next_q_online.view(-1, num_decisions, 2)
                            best_actions = next_q_online.argmax(dim=2)  # (B,num_decisions)

                            # 2) 用 target_model 估計 Q
                            next_q_target = target_model(next_states_weather, next_states_time)  # (B, num_decisions*2)
                            next_q_target = next_q_target.view(-1, num_decisions, 2)
                            best_actions_expanded = best_actions.unsqueeze(-1)  # (B,num_decisions,1)
                            next_chosen_q = torch.gather(next_q_target, 2, best_actions_expanded).squeeze(-1)  # (B, num_decisions)
                            # 取 mean
                            next_q = next_chosen_q.mean(dim=1)  # (B,)

                            target_q = rewards_tensor + gamma * (1 - dones_tensor) * next_q

                        loss = F.mse_loss(curr_q, target_q)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if total_steps % target_update_freq == 0:
                            target_model.load_state_dict(online_model.state_dict())

                        writer.add_scalar("Loss/train", loss.item(), total_steps)

                # 一個episode結束
                writer.add_scalar("Reward/episode", ep_reward, total_steps)
                global_epoch += 1
                if global_epoch % 1000 == 0:
                    save_dir = os.path.join(base_dir, "..", "model", "d3qn")
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(online_model.state_dict(), os.path.join(save_dir, f"d3qn_checkpoint_{global_epoch}.pth"))

            torch.cuda.empty_cache()
            print(f"區域 {region_id} 訓練完成。")

        print(f"批次 {batch_idx+1} 訓練完成。")

    print("所有區域訓練完成。")

    # -------------------------------------------------------
    # 測試階段
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
                online_model.eval()
                with torch.no_grad():
                    w_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                    t_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                    q_vals = online_model(w_tensor, t_tensor)
                    action_dict = derive_action_dict(q_vals, output_ids)
                _, reward, _, _ = env.step(action_dict)
                test_rewards.append(reward)
                writer.add_scalar("Test/Reward", reward, total_steps)
            except Exception as e:
                logging.error(f"Error during testing: {traceback.format_exc()}")
        torch.cuda.empty_cache()
    avg_test_reward = np.mean(test_rewards) if test_rewards else 0.0
    writer.add_scalar("Test/AverageReward", avg_test_reward, total_steps)
    print(f"測試平均 Reward: {avg_test_reward:.4f}")

    # -------------------------------------------------------
    # 驗證階段
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
                state = env.reset()
                try:
                    online_model.eval()
                    with torch.no_grad():
                        w_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                        t_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                        q_vals = online_model(w_tensor, t_tensor)
                        action_dict = derive_action_dict(q_vals, output_ids)
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
    print("D3QN 完整訓練、測試與驗證完成。")


if __name__ == "__main__":
    main()
