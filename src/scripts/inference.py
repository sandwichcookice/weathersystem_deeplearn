#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import torch
import numpy as np
import torch.nn as nn
import random
import torch.nn.functional as F

# 輔助函數：將模型輸出轉換為布林決策字典
def derive_action_dict(q_vals, output_ids):
    """
    將模型輸出 q_vals (shape (1, 22) 或 (1,11,2)) 重塑為 (11,2)，
    並取每組 argmax 轉換為布林值後建立字典。
    """
    if len(q_vals.shape) == 2:
        q_vals = q_vals.view(11, 2)
    else:
        q_vals = q_vals.squeeze(0)
    decisions = q_vals.argmax(dim=1)
    bool_decisions = [bool(val.item()) for val in decisions]
    return dict(zip(output_ids, bool_decisions))

# 定義 Dueling DQN 模型
class WeatherDuelingDQNModel(nn.Module):
    def __init__(self):
        super(WeatherDuelingDQNModel, self).__init__()
        # 前端特徵提取層：兩層 Bi-LSTM
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # 自注意力層（簡單實作）
        self.attention = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # Advantage branch
        self.adv_hidden = nn.Linear(128, 128)
        self.adv_stream = nn.Linear(128, 11 * 2)  # 輸出 22 個單元，之後 reshape 成 (batch, 11, 2)
        # Value branch
        self.val_hidden = nn.Linear(128, 128)
        self.val_stream = nn.Linear(128, 11)      # 輸出 11 個值

    def forward(self, x, time_weight):
        """
        x: (B, 24, 7) 輸入氣象資料
        time_weight: 預期為 (B, 24) 或 (B, 24, 1)
        """
        lstm1_out, _ = self.lstm1(x)  # (B, 24, 128)
        # 若 time_weight 已為 3D (B, 24, 1)，則直接使用；若為 2D則 unsqueeze
        if time_weight.dim() == 2:
            fixed_weight = time_weight.detach().unsqueeze(-1)
        elif time_weight.dim() == 3:
            fixed_weight = time_weight.detach()
        else:
            raise ValueError("Unexpected time_weight dimensions: {}".format(time_weight.dim()))
        attn1_out = lstm1_out * fixed_weight  # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)   # (B, 24, 128)
        # 採用平均池化
        pooled = torch.mean(lstm2_out, dim=1)   # (B, 128)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        adv = F.relu(self.adv_hidden(shared))
        adv = self.adv_stream(adv)              # (B, 11*2)
        adv = adv.view(-1, 11, 2)                # (B, 11, 2)
        val = F.relu(self.val_hidden(shared))
        val = self.val_stream(val)              # (B, 11)
        val = val.view(-1, 11, 1)                # (B, 11, 1)
        adv_mean = adv.mean(dim=2, keepdim=True)  # (B, 11, 1)
        q_vals = val + (adv - adv_mean)           # (B, 11, 2)
        return q_vals.view(-1, 11*2)

def generate_time_weights(batch_size, seq_len, device="cpu", low=0.5, high=1.5):
    pattern = random.choice(["U", "increasing", "decreasing"])
    if pattern == "U":
        half = seq_len // 2
        if seq_len % 2 == 0:
            left = np.linspace(low, high, half)
            right = np.linspace(high, low, half)
            weights = np.concatenate([left, right])
        else:
            left = np.linspace(low, high, half + 1)
            right = np.linspace(high, low, half + 1)[1:]
            weights = np.concatenate([left, right])
    elif pattern == "increasing":
        weights = np.linspace(low, high, seq_len)
    elif pattern == "decreasing":
        weights = np.linspace(high, low, seq_len)
    else:
        raise ValueError("Unknown pattern.")
    weights = np.tile(weights, (batch_size, 1)).reshape(batch_size, seq_len, 1)
    return torch.tensor(weights, dtype=torch.float32, device=device)

def main():
    parser = argparse.ArgumentParser(description="Dueling DQN 推理腳本 (適配 testinput.json)")
    parser.add_argument("--config", type=str, default="./config.json", help="配置文件路徑")
    parser.add_argument("--model_path", type=str, default="model/time1/dueling_dqn/leatest.pth", help="模型檢查點路徑")
    parser.add_argument("--input_file", type=str, default="data/testinput.json", help="推理輸入文件路徑")
    args = parser.parse_args()

    # 載入配置文件
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print("無法載入配置文件，使用預設配置。", e)
        config = {}

    # 基本參數設定
    seq_len = config.get("sequence_length", 24)
    input_ids = config.get("input_ids", [
        "Temperature", "DewPoint", "ApparentTemperature",
        "RelativeHumidity", "WindSpeed", "ProbabilityOfPrecipitation",
        "ComfortIndex"
    ])
    output_ids = config.get("output_ids", [
        "wearing_warm_clothes", "carry_umbrella", "damp_cold", "damp_hot",
        "dry_cold", "dry_hot", "wind_cold", "wind_hot",
        "outdoor_recommended", "increase_water_intake", "ground_slippery_warning"
    ])
    input_dim = len(input_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入推理輸入資料，適配 testinput.json 的結構
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        if isinstance(sample_data, dict) and "predicted_records" in sample_data:
            file_input_ids = sample_data.get("input_ids", input_ids)
            records = sample_data["predicted_records"]
            # 每筆資料的氣象數值皆包在 "ElementValue" 內
            sample = [
                [float(record["ElementValue"].get(col, 0)) for col in file_input_ids]
                for record in records
            ]
        elif isinstance(sample_data, list):
            sample = sample_data
        else:
            raise ValueError("推理輸入資料格式不正確！")
    except Exception as e:
        print("無法載入推理輸入文件，將使用全零數據。", e)
        sample = np.zeros((seq_len, input_dim), dtype=np.float32)

    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)
    time_weights = generate_time_weights(batch_size=1, seq_len=seq_len, device=device)

    # 載入模型檢查點時，設定 weights_only=True 以提高安全性
    model = WeatherDuelingDQNModel().to(device)
    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print("載入模型檢查點失敗：", e)
        return

    model.eval()
    with torch.no_grad():
        q_values = model(sample_tensor, time_weights)

    action_dict = derive_action_dict(q_values, output_ids)

    print("推理得到的 Q 值:")
    print(q_values.cpu().numpy())
    print("模型輸出的決策 (True/False):")
    print(action_dict)

if __name__ == "__main__":
    main()
