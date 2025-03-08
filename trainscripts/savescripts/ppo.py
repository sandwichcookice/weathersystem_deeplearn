#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import traceback

# 從 common 載入必要的共用函數與類別
from common import (
    generate_time_weights,
    evaluate_reward,
    load_json_file,
    load_json_files,
    WeatherEnv,
    WeatherActorCriticModel  # 此模型輸出：(policy_probs, value)
)

# --------------------
# 輔助函數：將模型的 policy 輸出轉換成動作決策字典
def derive_action_dict_policy(policy_probs, output_ids, threshold=0.5):
    """
    policy_probs: Tensor, shape (1, 11)，每個數值表示採取 True 的機率
    以 threshold 為閥值轉換為布林值
    """
    bool_actions = [bool(float(p)) >= threshold if False else bool(float(p) >= threshold) for p in policy_probs.squeeze(0)]
    # 此處直接判斷 p>= threshold 即為 True
    bool_actions = [bool(p >= threshold) for p in policy_probs.squeeze(0)]
    return dict(zip(output_ids, bool_actions))

# 輔助函數：計算 discounted returns
def compute_returns(rewards, gamma):
    """
    給定一個 rewards 列表，計算折扣回報 G_t = r_t + gamma*r_{t+1} + ... 
    返回一個 numpy 陣列，與 rewards 大小相同
    """
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns

# --------------------
# PPO 超參數
clip_epsilon = 0.2          # clip 範圍
ppo_epochs = 4              # 每次 rollout 重複更新次數
value_coef = 0.5            # critic loss 的權重
entropy_coef = 0.01         # entropy bonus 的權重

# --------------------
# 主程式：訓練、測試與驗證 PPO
def main():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, "..", "config.json")
    global_config = load_json_file(config_path)
    print(f"Config loaded from {config_path}")

    # 資料目錄（採用之前分批加載方法，此處示範一次性加載所有檔案）
    training_data_list = load_json_files(os.path.join(base_dir, "..", "data", "training_data"))
    check_data_list = load_json_files(os.path.join(base_dir, "..", "data", "check"))
    verify_data_list = load_json_files(os.path.join(base_dir, "..", "data", "val"))
    
    print(f"Loaded {len(training_data_list)} training files, {len(check_data_list)} test files, {len(verify_data_list)} validation files.")
    
    # 訓練參數
    lr = global_config["training"].get("learning_rate", 0.0003)
    gamma = global_config["training"].get("gamma", 0.99)
    num_epochs = global_config["training"].get("num_epochs", 100)  # 每個檔案的訓練輪數
    max_timesteps_per_ep = global_config["training"].get("max_timesteps_per_ep", 24)  # 假設每次 env.reset() 回傳一個 block 24 筆資料
    batch_size = global_config["training"].get("ppo_batch_size", 32)  # PPO mini-batch size (若使用 mini-batch更新)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "ppo_full"))
    
    model = WeatherActorCriticModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    global_step = 0

    print("開始訓練 PPO...")
    # 逐一遍歷每個訓練檔案（或區域）
    for region in training_data_list:
        region_id = region.get("station_id", region.get("file_name", "unknown"))
        print(f"Training on region: {region_id}")
        env = WeatherEnv(global_config, region)
        global_epoch = 0
        # 每個檔案訓練 num_epochs 輪
        for ep in tqdm(range(num_epochs), desc=f"Region {region_id} Epoch"):
            global_epoch += 1
            state = env.reset()
            done = False
            episode_rewards = []
            # 用於儲存 rollout 軌跡
            log_probs = []
            values = []
            rewards = []
            states = []
            time_weights = []
            while not done:
                states.append(state["weather"])
                time_weights.append(state["time_weight"])
                state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                model.eval()
                with torch.no_grad():
                    policy_probs, value = model(state_tensor, time_tensor)
                # 以 Bernoulli 進行抽樣，每個動作獨立（on-policy）
                m = torch.distributions.Bernoulli(policy_probs)
                action_sample = m.sample()
                log_prob = m.log_prob(action_sample).sum()
                action_dict = dict(zip(global_config.get("output_ids", []),
                                         [bool(int(a.item())) for a in action_sample.squeeze(0)]))
                log_probs.append(log_prob)
                values.append(value)
                try:
                    next_state, reward, done, _ = env.step(action_dict)
                except Exception as e:
                    logging.error(f"Error in env.step for region {region_id}: {traceback.format_exc()}")
                    reward = 0
                    next_state = env.reset()
                    done = True
                rewards.append(reward)
                episode_rewards.append(reward)
                state = next_state
                global_step += 1
            # 當一個 rollout 完成後，計算 returns 與 advantage
            returns = compute_returns(rewards, gamma)
            returns_tensor = torch.tensor(returns, dtype=torch.float32, device=device)
            values_tensor = torch.cat(values)
            advantages = returns_tensor - values_tensor.squeeze(1)
            # 將 rollout 中的數據轉為 tensor（批次維度為 rollout 長度）
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32, device=device)
            time_tensor_all = torch.tensor(np.array(time_weights), dtype=torch.float32, device=device)
            log_probs_tensor = torch.stack(log_probs)
            # 進行 PPO 更新：這裡簡單重複多次整個 rollout 更新
            ppo_loss = 0.0
            for _ in range(ppo_epochs):
                model.train()
                # 前向傳播
                new_policy, new_value = model(states_tensor, time_tensor_all)
                m_new = torch.distributions.Bernoulli(new_policy)
                new_log_probs = m_new.log_prob(action_sample.squeeze(0))
                new_log_probs = new_log_probs.sum(dim=1)
                # 計算 ratio
                ratio = torch.exp(new_log_probs - log_probs_tensor)
                # PPO 剪裁損失
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(new_value.squeeze(1), returns_tensor)
                entropy_loss = - m_new.entropy().mean()
                loss = actor_loss + value_coef * critic_loss + entropy_coef * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ppo_loss += loss.item()
            avg_loss = ppo_loss / ppo_epochs
            writer.add_scalar("Loss/train", avg_loss, global_step)
            writer.add_scalar("Reward/episode", np.sum(episode_rewards), global_step)
        # 儲存每個檔案訓練完成後的 checkpoint
        if global_epoch % 1000 == 0:
                print(f"Saving model checkpoint... (global epoch {global_epoch})")
                checkpoint_dir = os.path.join(base_dir, "..", "model", "ppo")
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"ppo_checkpoint_ep{global_epoch}.pth"))
        print(f"Checkpoint saved for region {region_id} at global step {global_step}")

    print("所有區域訓練完成。")
    
    # -------------------------------
    # 測試部分：以測試資料建立環境，單次 forward pass
    print("開始測試（PPO）...")
    test_rewards = []
    pbar_test = tqdm(total=len(check_data_list), desc="Testing")
    for data in check_data_list:
        try:
            env = WeatherEnv(global_config, data)
            state = env.reset()
            model.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                policy_probs, _ = model(state_tensor, time_tensor)
                action_dict = derive_action_dict_policy(policy_probs, global_config.get("output_ids", []))
            _, reward, _, _ = env.step(action_dict)
            test_rewards.append(reward)
            writer.add_scalar("Test/Reward", reward, global_step)
        except Exception as e:
            logging.error(f"Error during testing: {traceback.format_exc()}")
        pbar_test.update(1)
    pbar_test.close()
    avg_test_reward = np.mean(test_rewards) if test_rewards else 0.0
    writer.add_scalar("Test/AverageReward", avg_test_reward, global_step)
    print(f"測試平均 Reward: {avg_test_reward:.4f}")
    
    # -------------------------------
    # 驗證部分：對每個驗證檔案，將資料分成若干個 block，每 block 使用 env.reset() 得到 24 筆資料，計算 reward
    print("開始驗證（PPO）...")
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
                model.eval()
                with torch.no_grad():
                    state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                    time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                    policy_probs, _ = model(state_tensor, time_tensor)
                    action_dict = derive_action_dict_policy(policy_probs, global_config.get("output_ids", []))
                _, block_reward, _, _ = env.step(action_dict)
                block_rewards.append(block_reward)
                writer.add_scalar("Validation/BlockReward", block_reward, global_step + i)
            except Exception as e:
                logging.error(f"Error during validation on block {i} of file {file_name}: {traceback.format_exc()}")
        avg_val_reward = np.mean(block_rewards) if block_rewards else 0.0
        writer.add_scalar("Validation/AverageReward", avg_val_reward, global_step)
        writer.add_scalar(f"Validation/{file_name}_AverageReward", avg_val_reward, global_step)
        print(f"驗證檔案 {file_name} 平均 Reward: {avg_val_reward:.4f}")
        file_rewards.append((file_name, avg_val_reward))
        file_idx += 1
        pbar_val.update(1)
    pbar_val.close()
    
    writer.close()
    print("PPO 完整訓練、測試與驗證完成。")
    
if __name__ == '__main__':
    # 若使用多進程時，請確保使用 spawn 模式
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()