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
import gym

# 從 common.py 匯入輔助函數與環境定義
from common import (
    generate_time_weights,
    load_json_file,
    load_json_files,
    evaluate_reward,
    WeatherEnv
)

# =============================================================================
# 修改後的 NoisyLinear：每次 forward 時重置噪聲
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.8):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input):
        if self.training:
            self.reset_noise()  # 每次 forward 時更新噪聲
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(input, weight, bias)


# =============================================================================
# 優先重放緩衝區（Prioritized Replay Buffer）
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.priorities.append(max_priority)
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
            self.priorities[self.pos] = max_priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.6):
        if len(self.buffer) == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(self.priorities[:len(self.buffer)])
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        return tuple(zip(*samples)), indices, weights

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

# =============================================================================
# Rainbow DQN 模型：結合 Dueling、分布式 Q 與 Noisy Nets
# 在 WeatherRainbowDQNModel 中新增 reset_noise 方法
class WeatherRainbowDQNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_decisions, num_atoms, Vmin, Vmax):
        super(WeatherRainbowDQNModel, self).__init__()
        self.num_decisions = num_decisions
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.register_buffer("support", torch.linspace(Vmin, Vmax, num_atoms))
        
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True, bidirectional=True)
        self.fc_shared = nn.Sequential(
            NoisyLinear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, num_atoms)
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, 2 * num_decisions * num_atoms)
        )
    
    def forward(self, weather, time_weight):
        time_weight = time_weight.unsqueeze(-1)  # (B, 24, 1)
        x = torch.cat([weather, time_weight], dim=2)  # (B, 24, weather_dim+1)
        lstm_out, _ = self.lstm(x)  # (B, 24, hidden_dim*2)
        pooled = torch.mean(lstm_out, dim=1)  # (B, hidden_dim*2)
        features = self.fc_shared(pooled)  # (B, hidden_dim)
        
        value = self.value_stream(features)  # (B, num_atoms)
        advantage = self.advantage_stream(features)  # (B, 2*num_decisions*num_atoms)
        advantage = advantage.view(-1, self.num_decisions, 2, self.num_atoms)  # (B, num_decisions, 2, num_atoms)
        
        value = value.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, num_atoms)
        advantage_mean = advantage.mean(dim=2, keepdim=True)  # (B, num_decisions, 1, num_atoms)
        q_atoms = value + (advantage - advantage_mean)  # (B, num_decisions, 2, num_atoms)
        dist = F.softmax(q_atoms, dim=3)
        return dist

    def reset_noise(self):
        # 遞歸呼叫所有 NoisyLinear 層的 reset_noise()
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    
class WeatherEnv(gym.Env):
    def __init__(self, global_config, data_config):
        super().__init__()
        self.input_ids = global_config["input_ids"]
        self.output_ids = global_config["output_ids"]
        self.rules = global_config["rules"]
        self.predicted_records = data_config["predicted_records"]
        self.real_records = data_config["real_records"]
        self.data_len = len(self.predicted_records)

    def reset(self):
        # 單次Episode：只隨機抽一段24hr資料
        self.idx = random.randint(0, self.data_len - 24)
        block = self.predicted_records[self.idx : self.idx + 24]
        weather = np.array([[rec.get(k, 0.0) for k in self.input_ids] for rec in block], dtype=np.float32)
        time_weight = generate_time_weights(1, 24, device="cpu")[0, :, 0].cpu().numpy()
        return {"weather": weather, "time_weight": time_weight}

    def step(self, action_dict):
        # 單步環境，直接結束
        # reward 由24hr資料( real_records ) & action_dict 計算
        block = self.real_records[self.idx : self.idx + 24]
        reward = sum(evaluate_reward(self.rules, rec, action_dict) for rec in block)
        done = True
        next_state = {}  # 或者直接None
        return next_state, reward, done, {}



# =============================================================================
# 輔助函數：由模型輸出分布計算預期 Q 值並衍生布林決策字典
def derive_action_dict_rainbow(dist, output_ids, support):
    """
    dist: (1, num_decisions, 2, num_atoms)
    計算各決策預期 Q 值後取 argmax，返回對應的決策字典
    """
    expected_q = torch.sum(dist * support, dim=3)  # (1, num_decisions, 2)
    decisions = expected_q.argmax(dim=2).squeeze(0)  # (num_decisions,)
    bool_decisions = [bool(int(a)) for a in decisions]
    return dict(zip(output_ids, bool_decisions))

def action_dict_to_index(action_dict, output_ids):
    return [0 if not action_dict.get(k, False) else 1 for k in output_ids]

# =============================================================================
# 投影分布函數：分布式 Q 學習必備，將目標分布投影到固定支撐上
def projection_distribution(next_dist, rewards, dones, gamma, support, Vmin, Vmax):
    batch_size, num_atoms = next_dist.size()
    delta_z = (Vmax - Vmin) / (num_atoms - 1)
    projected_dist = torch.zeros_like(next_dist)

    # Tz 計算目標位置
    rewards = rewards.unsqueeze(-1).expand(-1, num_atoms)
    dones = dones.unsqueeze(-1).expand(-1, num_atoms)
    support = support.unsqueeze(0).expand(batch_size, -1)

    Tz = rewards + gamma * support * (1 - dones)
    Tz = Tz.clamp(Vmin, Vmax)

    # 計算對應的上下限索引
    b_j = (Tz - Vmin) / delta_z
    l = b_j.floor().long()
    u = b_j.ceil().long()

    # 修正邊界條件 (確保索引合法)
    l = l.clamp(0, num_atoms - 1)
    u = u.clamp(0, num_atoms - 1)

    # 投影分布
    offset = (torch.arange(batch_size) * num_atoms).unsqueeze(1).expand(batch_size, num_atoms).to(next_dist.device)

    next_dist_flat = next_dist.view(-1)

    projected_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist_flat * (u.float() - b_j).view(-1)))
    projected_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist_flat * (b_j - l.float()).view(-1)))

    return projected_dist


# =============================================================================
# 主程式：訓練、測試與驗證流程
def main():
    # 讀取全域配置與資料
    base_dir = os.path.dirname(__file__)
    global_config = load_json_file(os.path.join(base_dir, "..", "config.json"))
    training_data_list = load_json_files(os.path.join(base_dir, "..", "data", "training_data"))
    check_data_list = load_json_files(os.path.join(base_dir, "..", "data", "check"))
    verify_data_list = load_json_files(os.path.join(base_dir, "..", "data", "val"))
    
    # 訓練參數
    lr = global_config["training"].get("learning_rate", 0.0005)
    batch_size = global_config["training"].get("batch_size", 64)
    num_epochs = global_config["training"].get("num_epochs", 100)
    gamma = global_config["training"].get("gamma", 0.99)
    learning_starts = global_config["training"].get("learning_starts", 1000)
    replay_capacity = global_config["training"].get("replay_buffer_size", 50000)
    # 優先重放的 beta 參數初始值
    beta_start = 0.4
    beta_frames = 50000
    
    # C51 參數
    num_atoms = global_config["training"].get("num_atoms", 51)
    Vmin = global_config["training"].get("Vmin", -10)
    Vmax = global_config["training"].get("Vmax", 10)
    
    # 設定裝置與 TensorBoard
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                          "cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=os.path.join(base_dir, "..", "logs", "rainbow_full"))
    
    # 從 global_config 取得決策項目與輸入維度資訊
    output_ids = global_config.get("output_ids", [])
    num_decisions = len(output_ids)
    env_sample = WeatherEnv(global_config, training_data_list[0])
    init_state = env_sample.reset()
    weather_dim = init_state["weather"].shape[1]
    # 輸入維度 = weather_dim + 1 (加上時間權重)
    input_dim = weather_dim + 1
    
    # 建立 Rainbow 模型與目標網路
    model = WeatherRainbowDQNModel(input_dim=input_dim, hidden_dim=128,
                                   num_decisions=num_decisions, num_atoms=num_atoms,
                                   Vmin=Vmin, Vmax=Vmax).to(device)
    target_model = WeatherRainbowDQNModel(input_dim=input_dim, hidden_dim=128,
                                          num_decisions=num_decisions, num_atoms=num_atoms,
                                          Vmin=Vmin, Vmax=Vmax).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    buffer = PrioritizedReplayBuffer(replay_capacity)
    total_steps = 0
    target_update_freq = 5000

    # 分批載入訓練資料設定
    batch_files = 10
    num_total_files = len(training_data_list)
    num_batches = (num_total_files + batch_files - 1) // batch_files
    global_epoch = 0

    model.train()

    print("開始訓練所有區域 (Rainbow DQN) - 分批加載：")
    for batch_idx in range(num_batches):
        batch_data = training_data_list[batch_idx * batch_files : (batch_idx + 1) * batch_files]
        print(f"\n== 開始訓練批次 {batch_idx+1}/{num_batches} (檔案數: {len(batch_data)}) ==")
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
                    # 探索策略：使用 Noisy Nets 整合探索，故直接採用模型輸出
                    # 在訓練迴圈中進行動作選擇時
                    try:
                        with torch.no_grad():
                            model.reset_noise()  # 重置所有 Noisy 層的噪聲
                            weather_tensor = torch.tensor(state["weather"], dtype=torch.float32, device=device).unsqueeze(0)
                            time_tensor = torch.tensor(state["time_weight"], dtype=torch.float32, device=device).unsqueeze(0)
                            dist = model(weather_tensor, time_tensor)
                            action_dict = derive_action_dict_rainbow(dist, output_ids, model.support)
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
                            beta = min(1.0, beta_start + total_steps * (1.0 - beta_start) / beta_frames)
                            (states, actions, rewards, next_states, dones), indices, weights = buffer.sample(batch_size, beta)

                            states_weather = torch.tensor(np.array([s["weather"] for s in states]), dtype=torch.float32, device=device)
                            states_time = torch.tensor(np.array([s["time_weight"] for s in states]), dtype=torch.float32, device=device)
                            next_states_weather = torch.tensor(np.array([s["weather"] for s in next_states]), dtype=torch.float32, device=device)
                            next_states_time = torch.tensor(np.array([s["time_weight"] for s in next_states]), dtype=torch.float32, device=device)

                            actions_indices = [action_dict_to_index(a, output_ids) for a in actions]
                            actions_tensor = torch.tensor(actions_indices, dtype=torch.long, device=device)
                            rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
                            dones_tensor = torch.tensor(dones, dtype=torch.float32, device=device)

                            curr_dist = model(states_weather, states_time)
                            actions_tensor_expanded = actions_tensor.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, num_atoms)
                            pred_dist = curr_dist.gather(2, actions_tensor_expanded).squeeze(2)

                            with torch.no_grad():
                                next_dist_full = target_model(next_states_weather, next_states_time)
                                next_q = torch.sum(next_dist_full * target_model.support, dim=3)
                                next_actions = next_q.argmax(dim=2)
                                next_actions_expanded = next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, num_atoms)
                                next_dist = next_dist_full.gather(2, next_actions_expanded).squeeze(2)

                                rewards_expanded = rewards_tensor.unsqueeze(1).expand(-1, num_decisions)
                                dones_expanded = dones_tensor.unsqueeze(1).expand(-1, num_decisions)

                                next_dist_reshaped = next_dist.view(-1, num_atoms)
                                rewards_reshaped = rewards_expanded.contiguous().view(-1)
                                dones_reshaped = dones_expanded.contiguous().view(-1)

                                target_dist = projection_distribution(
                                    next_dist_reshaped, rewards_reshaped, dones_reshaped,
                                    gamma, target_model.support, Vmin, Vmax
                                ).view(batch_size, num_decisions, num_atoms)

                            log_pred = torch.log(pred_dist + 1e-8)
                            sample_loss = -(target_dist * log_pred).sum(dim=2).mean(dim=1)

                            weighted_loss = sample_loss * weights.to(device)
                            loss = weighted_loss.mean()

                            if torch.isnan(loss) or torch.isinf(loss):
                                logging.error("Loss 出現數值異常，跳過本次更新。")
                                optimizer.zero_grad()
                                continue

                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            td_errors = torch.abs(sample_loss.detach()).cpu().numpy()
                            buffer.update_priorities(indices, td_errors + 1e-6)

                            if total_steps % target_update_freq == 0:
                                target_model.load_state_dict(model.state_dict())

                            writer.add_scalar("Loss/train", loss.item(), global_epoch)

                        except Exception as e:
                            logging.error(f"Error during training update: {traceback.format_exc()}")

                writer.add_scalar("Reward/episode", ep_reward, global_epoch)
                global_epoch += 1
                if global_epoch % 1000 == 0:
                    print(f"Saving model checkpoint... (global epoch {global_epoch})")
                    checkpoint_dir = os.path.join(base_dir, "..", "model", "rainbow")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"rainbow_checkpoint_ep{global_epoch}.pth"))
            torch.cuda.empty_cache()
            print(f"區域 {region_id} 訓練完成。")
        print(f"批次 {batch_idx+1} 訓練完成。")
    
    print("所有區域訓練完成。")
    
    # =============================================================================
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
                    action_dict = derive_action_dict_rainbow(dist, output_ids, model.support)
                _, reward, _, _ = env.step(action_dict)
                test_rewards.append(reward)
                writer.add_scalar("Test/Reward", reward, total_steps)
            except Exception as e:
                logging.error(f"Error during testing on file: {traceback.format_exc()}")
        torch.cuda.empty_cache()
    avg_test_reward = np.mean(test_rewards) if test_rewards else 0.0
    writer.add_scalar("Test/AverageReward", avg_test_reward, total_steps)
    print(f"測試平均 Reward: {avg_test_reward:.4f}")
    
    # =============================================================================
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
                        action_dict = derive_action_dict_rainbow(dist, output_ids, model.support)
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
    print("Rainbow DQN 完整訓練、測試與驗證完成。")

if __name__ == "__main__":
    main()
