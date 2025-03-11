#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import traceback

# 從 common.py 匯入所需輔助函數與類別
from common import (
    generate_time_weights,
    load_json_file,
    load_json_files,
    ReplayBuffer,
    WeatherEnv
)

# ------------------------------------------------------------------------------
# 分布投影函數：將目標網路預測的分布進行投影，使其落在固定支撐 [Vmin, Vmax] 上
def projection_distribution(next_dist, rewards, dones, gamma, support, Vmin, Vmax):
    batch_size = rewards.size(0)
    num_atoms = support.size(0)
    delta_z = (Vmax - Vmin) / (num_atoms - 1)
    
    Tz = rewards.unsqueeze(1) + gamma * (1 - dones.unsqueeze(1)) * support.unsqueeze(0)
    Tz = Tz.clamp(Vmin, Vmax)
    b = (Tz - Vmin) / delta_z
    l = b.floor().long()
    u = b.ceil().long()
    
    lower_weight = (u.float() - b)
    upper_weight = (b - l.float())
    lower_weight = lower_weight.masked_fill(u == l, 1.0)
    upper_weight = upper_weight.masked_fill(u == l, 0.0)
    
    proj_dist = torch.zeros_like(next_dist)
    for i in range(num_atoms):
        p = next_dist[:, i]
        l_index = l[:, i].unsqueeze(1)
        u_index = u[:, i].unsqueeze(1)
        proj_dist.scatter_add_(1, l_index, (p * lower_weight[:, i]).unsqueeze(1))
        proj_dist.scatter_add_(1, u_index, (p * upper_weight[:, i]).unsqueeze(1))
    return proj_dist

# ------------------------------------------------------------------------------
# 修正後的 WeatherC51Model
class WeatherC51Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_decisions, num_atoms, Vmin, Vmax):
        """
        模型採用 Bi-LSTM 進行時序特徵抽取，並以全連接層產出分布式 Q 值。
        輸入資料為時序資料，其中每個 timestep 結合天氣特徵與時間權重。
        """
        super(WeatherC51Model, self).__init__()
        # 使用雙向 LSTM 處理時序資料
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        # 透過全連接層進行特徵融合
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # 最後輸出層：產生 num_decisions * 2 * num_atoms 個節點
        self.fc_out = nn.Linear(hidden_dim, num_decisions * 2 * num_atoms)
        self.num_decisions = num_decisions  # 每筆資料共 num_decisions 個決策，每個決策兩個動作
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        # 固定支撐向量
        self.register_buffer("support", torch.linspace(Vmin, Vmax, num_atoms))
        
    def forward(self, weather, time_weight):
        """
        weather: Tensor，形狀 (B, 24, weather_dim)
        time_weight: Tensor，形狀 (B, 24)
        先將 time_weight 調整為 (B, 24, 1) 後與 weather 於 feature 維度串接，
        再透過 LSTM 與全連接層產出分布式 Q 值，重塑輸出形狀為 (B, num_decisions, 2, num_atoms)。
        """
        # 調整 time_weight 維度
        time_weight = time_weight.unsqueeze(-1)  # (B, 24, 1)
        # 串接天氣數據與時間權重，形成每個 timestep 的完整輸入特徵
        x = torch.cat([weather, time_weight], dim=2)  # (B, 24, weather_dim+1)
        lstm_out, _ = self.lstm(x)  # (B, 24, hidden_dim*2)
        # 時序平均池化
        pooled = torch.mean(lstm_out, dim=1)  # (B, hidden_dim*2)
        x = F.relu(self.fc1(pooled))
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)  # (B, num_decisions * 2 * num_atoms)
        x = x.view(-1, self.num_decisions, 2, self.num_atoms)
        # 於原子維度上使用 softmax，確保分布和為 1
        dist = F.softmax(x, dim=3)
        return dist

# ------------------------------------------------------------------------------
# 輔助函數：由 C51 模型輸出分布計算預期 Q 值並產生布林決策字典
def derive_action_dict_c51(dist, output_ids, support):
    """
    dist: Tensor，形狀 (1, num_decisions, 2, num_atoms)
    計算各決策預期 Q 值，並依 argmax 得出 0/1 決策，轉為字典格式。
    """
    expected_q = torch.sum(dist * support, dim=3)  # (1, num_decisions, 2)
    decisions = expected_q.argmax(dim=2).squeeze(0)
    bool_decisions = [bool(int(a)) for a in decisions]
    return dict(zip(output_ids, bool_decisions))

# 將 action 字典轉換為 0/1 向量
def action_dict_to_index(action_dict, output_ids):
    return [0 if not action_dict.get(k, False) else 1 for k in output_ids]

# ------------------------------------------------------------------------------
# 主程式：訓練、測試與驗證流程
def main():
    # 讀取全域配置與資料
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

    # C51 參數
    num_atoms = global_config["training"].get("num_atoms", 51)
    Vmin = global_config["training"].get("Vmin", -10)
    Vmax = global_config["training"].get("Vmax", 10)
    
    # 設定運算裝置與 TensorBoard 寫入
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "c51_full"))
    
    # 從 global_config 取得決策項目，並以第一筆資料初始化環境以獲得輸入維度
    output_ids = global_config.get("output_ids", [])
    num_decisions = len(output_ids)
    env_sample = WeatherEnv(global_config, training_data_list[0])
    init_state = env_sample.reset()
    # init_state["weather"] shape: (24, len(input_ids))
    weather_dim = init_state["weather"].shape[1]
    # 加上時間權重後，單 timestep 輸入維度為 weather_dim + 1
    input_dim = weather_dim + 1

    # 建立 C51 模型與目標網路
    model = WeatherC51Model(input_dim=input_dim, hidden_dim=128,
                             num_decisions=num_decisions, num_atoms=num_atoms,
                             Vmin=Vmin, Vmax=Vmax).to(device)
    target_model = WeatherC51Model(input_dim=input_dim, hidden_dim=128,
                                    num_decisions=num_decisions, num_atoms=num_atoms,
                                    Vmin=Vmin, Vmax=Vmax).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = ReplayBuffer(replay_capacity)
    total_steps = 0
    target_update_freq = 1000

    # 分批載入訓練資料設定
    batch_files = 10
    num_total_files = len(training_data_list)
    num_batches = (num_total_files + batch_files - 1) // batch_files
    global_epoch = 0

    print("開始訓練所有區域 (C51) - 分批加載：")
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
                    # 根據 ε-greedy 選擇動作
                    if random.random() < epsilon:
                        random_actions = [bool(random.getrandbits(1)) for _ in range(num_decisions)]
                        action_dict = dict(zip(output_ids, random_actions))
                    else:
                        try:
                            model.eval()
                            with torch.no_grad():
                                # 將 state 中資料轉換為正確形狀：weather (1, 24, weather_dim)，time_weight (1, 24)
                                weather_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                                time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                                dist = model(weather_tensor, time_tensor)
                                action_dict = derive_action_dict_c51(dist, output_ids, model.support)
                        except Exception as e:
                            logging.error(f"Error in action selection: {traceback.format_exc()}")
                            random_actions = [bool(random.getrandbits(1)) for _ in range(num_decisions)]
                            action_dict = dict(zip(output_ids, random_actions))
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

                    # 當累積足夠經驗後進行模型更新
                    if len(buffer) >= learning_starts:
                        try:
                            model.train()
                            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                            
                            # 轉換狀態資料，注意保持正確形狀：
                            # weather: (B, 24, weather_dim)，time_weight: (B, 24)
                            states_weather = torch.tensor(np.array([s["weather"] for s in states]),
                                                          dtype=torch.float32, device=device)
                            states_time = torch.tensor(np.array([s["time_weight"] for s in states]),
                                                       dtype=torch.float32, device=device)
                            next_states_weather = torch.tensor(np.array([s["weather"] for s in next_states]),
                                                               dtype=torch.float32, device=device)
                            next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]),
                                                            dtype=torch.float32, device=device)
                            
                            # 轉換動作：將 action 字典轉換為 0/1 向量，形狀 (B, num_decisions)
                            actions_indices = [action_dict_to_index(a, output_ids) for a in actions]
                            actions_tensor = torch.tensor(actions_indices, dtype=torch.long, device=device)
                            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)
                            
                            # 模型預測分布，形狀 (B, num_decisions, 2, num_atoms)
                            curr_dist = model(states_weather, states_time)
                            # 擷取採用的動作分布：使用 gather 操作
                            # 將 actions_tensor 擴展至 (B, num_decisions, 1, num_atoms)
                            actions_tensor_expanded = actions_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, num_atoms)
                            pred_dist = curr_dist.gather(2, actions_tensor_expanded).squeeze(2)  # (B, num_decisions, num_atoms)
                            
                            # 使用目標網路計算下一狀態分布
                            with torch.no_grad():
                                next_dist_full = target_model(next_states_weather, next_states_time)  # (B, num_decisions, 2, num_atoms)
                                next_q = torch.sum(next_dist_full * target_model.support, dim=3)  # (B, num_decisions, 2)
                                next_actions = next_q.argmax(dim=2)  # (B, num_decisions)
                                next_actions_expanded = next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, num_atoms)
                                next_dist = next_dist_full.gather(2, next_actions_expanded).squeeze(2)  # (B, num_decisions, num_atoms)
                                
                                # 將 rewards 與 dones 廣播至各決策，視為獨立樣本
                                rewards_expanded = rewards_tensor.unsqueeze(1).expand(-1, num_decisions)
                                dones_expanded = dones_tensor.unsqueeze(1).expand(-1, num_decisions)
                                # 調整形狀 (B*num_decisions, num_atoms)
                                next_dist_reshaped = next_dist.view(-1, num_atoms)
                                rewards_reshaped = rewards_expanded.contiguous().view(-1)
                                dones_reshaped = dones_expanded.contiguous().view(-1)
                                
                                target_dist = projection_distribution(next_dist_reshaped, rewards_reshaped, dones_reshaped,
                                                                      gamma, target_model.support, Vmin, Vmax)
                                # 調整回 (B, num_decisions, num_atoms)
                                target_dist = target_dist.view(batch_size, num_decisions, num_atoms)
                            
                            # 計算交叉熵損失：針對每個決策所有原子分布
                            loss = - (target_dist * torch.log(pred_dist + 1e-8)).sum(dim=2).mean()
                            
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
                    checkpoint_dir = os.path.join(base_dir, "..", "model", "c51")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"c51_checkpoint_ep{global_epoch}.pth"))
            torch.cuda.empty_cache()
            print(f"區域 {region_id} 訓練完成。")
        print(f"批次 {batch_idx + 1} 訓練完成。")
    
    print("所有區域訓練完成。")
    
    # ------------------------------------------------------------------------------
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
                model.eval()
                with torch.no_grad():
                    weather_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                    time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                    dist = model(weather_tensor, time_tensor)
                    action_dict = derive_action_dict_c51(dist, output_ids, model.support)
                _, reward, _, _ = env.step(action_dict)
                test_rewards.append(reward)
                writer.add_scalar("Test/Reward", reward, total_steps)
            except Exception as e:
                logging.error(f"Error during testing on file: {traceback.format_exc()}")
        torch.cuda.empty_cache()
    avg_test_reward = np.mean(test_rewards) if test_rewards else 0.0
    writer.add_scalar("Test/AverageReward", avg_test_reward, total_steps)
    print(f"測試平均 Reward: {avg_test_reward:.4f}")
    
    # ------------------------------------------------------------------------------
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
                state = env.reset()  # 每次 reset 取得一個 block（24 筆資料）
                try:
                    model.eval()
                    with torch.no_grad():
                        weather_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                        time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                        dist = model(weather_tensor, time_tensor)
                        action_dict = derive_action_dict_c51(dist, output_ids, model.support)
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
    print("C51 完整訓練、測試與驗證完成。")

if __name__ == "__main__":
    main()
