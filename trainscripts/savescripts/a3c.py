#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import time
import threading
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import traceback
import glob
import json

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

# -------------------- 從 common 載入其他必要內容 -------------------------
from common import (
    generate_time_weights,
    evaluate_reward,
    load_json_file,
    WeatherEnv,
    WeatherActorCriticModel  # 模型定義已更新，存放在 common.py 中
)

# -------------------- Worker 函數 -------------------------
def worker(worker_id, global_model, optimizer, global_config, training_batch, global_step, max_global_steps, lock, device):
    # 為每個 worker 建立獨立的 TensorBoard writer
    worker_log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", "a3c_full")
    worker_writer = SummaryWriter(log_dir=worker_log_dir)
    print(f"Worker {worker_id} started.")
    
    local_model = WeatherActorCriticModel().to(device)
    local_model.load_state_dict(global_model.state_dict())
    seed = 1234 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    pbar = tqdm(total=max_global_steps, desc=f"Worker {worker_id}", position=worker_id, leave=False)
    
    while True:
        with global_step.get_lock():
            if global_step.value >= max_global_steps:
                break
        if not training_batch:
            logging.error("Worker {}: Training batch is empty!".format(worker_id))
            break
        region = random.choice(training_batch)
        region_id = region.get("station_id", "unknown")
        try:
            env = WeatherEnv(global_config, region)
        except Exception as e:
            logging.error(f"Worker {worker_id} 初始化環境失敗: {traceback.format_exc()}")
            continue
        state = env.reset()
        log_probs = []
        values = []
        rewards = []
        done = False
        
        local_model.train()
        try:
            state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
            time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
            policy_probs, value = local_model(state_tensor, time_tensor)
            # 使用 Bernoulli 分布抽樣動作（因為每個動作輸出一個機率）
            m = torch.distributions.Bernoulli(policy_probs)
            action_sample = m.sample()  # shape (1, 11)
            log_prob = m.log_prob(action_sample).sum()  # sum over actions
            action_dict = dict(zip(global_config.get("output_ids", []),
                                     [bool(int(a.item())) for a in action_sample.squeeze(0)]))
        except Exception as e:
            logging.error(f"Worker {worker_id} action selection error: {traceback.format_exc()}")
            random_actions = [bool(random.getrandbits(1)) for _ in range(len(global_config.get("output_ids", [])))]
            action_dict = dict(zip(global_config.get("output_ids", []), random_actions))
            log_prob = torch.tensor(0.0, device=device)
            value = torch.tensor([[0.0]], device=device)
        try:
            next_state, reward, done, _ = env.step(action_dict)
        except Exception as e:
            logging.error(f"Worker {worker_id} env.step error for region {region_id}: {traceback.format_exc()}")
            reward = 0
            next_state = env.reset()
            done = True
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        with global_step.get_lock():
            global_step.value += 1
            current_step = global_step.value
        pbar.update(1)
        
        # 單步環境，G = reward
        G = sum(rewards)
        baseline = values[0]
        advantage = G - baseline.item()
        actor_loss = - torch.stack(log_probs).mean() * advantage
        critic_loss = F.mse_loss(baseline, torch.tensor([[G]], dtype=torch.float32, device=device))
        total_loss = actor_loss + critic_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        with lock:
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())
        pbar.set_postfix({"loss": total_loss.item(), "reward": G})
        # 將訓練時的 loss 與 reward 紀錄到 TensorBoard
        worker_writer.add_scalar("Loss/train", total_loss.item(), current_step)
        worker_writer.add_scalar("Reward/episode", G, current_step)
    pbar.close()
    worker_writer.close()
    print(f"Worker {worker_id} finished.")

# -------------------- Checkpoint Saver 函數 -------------------------
def checkpoint_saver(global_model, global_step, checkpoint_interval, model_dir, max_global_steps):
    next_checkpoint = checkpoint_interval
    while global_step.value < max_global_steps:
        time.sleep(5)
        if global_step.value >= next_checkpoint:
            os.makedirs(model_dir, exist_ok=True)
            checkpoint_path = os.path.join(model_dir, f"a3c_checkpoint_step{global_step.value}.pth")
            torch.save(global_model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at step {global_step.value}")
            next_checkpoint += checkpoint_interval

# -------------------- 主程式 -------------------------
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
    
    # 每批加載一定數量檔案，避免一次性占用過多記憶體
    batch_size_files = global_config["training"].get("batch_file_size", 50)
    train_batches = [all_train_files[i:i+batch_size_files] for i in range(0, len(all_train_files), batch_size_files)]
    
    # 累積每批加載的資料（您可以根據需求在每批訓練完成後釋放已使用的資料）
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
    model_dir = os.path.join(base_dir, "..", "model", "a3c")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "a3c_full"))
    
    global_model = WeatherActorCriticModel().to(device)
    global_model.share_memory()
    optimizer = torch.optim.Adam(global_model.parameters(), lr=lr)
    global_step = mp.Value('i', 0)
    lock = mp.Lock()
    
    # 啟動 checkpoint saver 線程
    checkpoint_thread = threading.Thread(target=checkpoint_saver, args=(global_model, global_step, checkpoint_interval, model_dir, max_global_steps))
    checkpoint_thread.daemon = True
    checkpoint_thread.start()
    
    num_workers = 1
    processes = []
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(worker_id, global_model, optimizer, global_config,
                                            training_data_list, global_step, max_global_steps, lock, device))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    
    writer.close()
    print("A3C (Actor-Critic) 完整訓練完成。")
    
    # -------------------------------
    # 測試部分：每個測試檔案單次 forward pass
    print("開始測試（A3C）...")
    test_rewards = []
    pbar_test = tqdm(total=len(check_data_list), desc="Testing")
    for data in check_data_list:
        try:
            env = WeatherEnv(global_config, data)
            state = env.reset()
            global_model.eval()
            with torch.no_grad():
                state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                policy_probs, _ = global_model(state_tensor, time_tensor)
                # 以 0.5 為閾值產生動作決策
                action_dict = dict(zip(global_config.get("output_ids", []),
                                         [bool(p.item() >= 0.5) for p in policy_probs.squeeze(0)]))
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
    # 驗證部分：將每個驗證檔案分成若干個 block，每 block 用 env.reset() 取得一組 24 筆資料後計算 reward
    print("開始驗證（A3C）...")
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
                global_model.eval()
                with torch.no_grad():
                    state_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                    time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                    policy_probs, _ = global_model(state_tensor, time_tensor)
                    action_dict = dict(zip(global_config.get("output_ids", []),
                                             [bool(p.item() >= 0.5) for p in policy_probs.squeeze(0)]))
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
    print("A3C 完整訓練、測試與驗證完成。")
    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
