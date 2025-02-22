#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os

# -------------------------------
# 定義 SelfAttention（與訓練時一致）
# -------------------------------
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key   = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
    
    def forward(self, x):
        # x: (B, T, d_model)
        Q = self.query(x)                      # (B, T, d_model)
        K = self.key(x)                        # (B, T, d_model)
        V = self.value(x)                      # (B, T, d_model)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # (B, T, T)
        attn_weights = torch.softmax(scores, dim=-1)            # (B, T, T)
        out = torch.bmm(attn_weights, V)                          # (B, T, d_model)
        return out

# -------------------------------
# 定義模型架構 – WeatherDDQNModel（與訓練時一致）
# -------------------------------
class WeatherDDQNModel(nn.Module):
    def __init__(self):
        super(WeatherDDQNModel, self).__init__()
        # 第一層 Bi-LSTM：輸入 7 維，隱藏單元 64，雙向 → 輸出維度 = 64*2 = 128
        self.lstm1 = nn.LSTM(input_size=7, hidden_size=64, batch_first=True, bidirectional=True)
        # 第二層 Bi-LSTM：輸入 128 維，隱藏單元 64，雙向 → 輸出 128
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        # 第二層注意力層：自注意力，d_model=128
        self.attention2 = SelfAttention(d_model=128)
        # 共享全連接層
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        # Advantage branch
        self.adv_hidden = nn.Linear(128, 128)
        self.adv_stream = nn.Linear(128, 22)  # 22 = 2 x 11 (每個決策2個 Q 值)
        # Value branch
        self.val_hidden = nn.Linear(128, 128)
        self.val_stream = nn.Linear(128, 11)    # 每個決策1個 value

    def forward(self, x, time_weight):
        """
        x: Tensor，形狀 (B, 24, 7)
        time_weight: Tensor，形狀 (B, 24) 或 (B,24,1)；用於第一層注意力
        """
        lstm1_out, _ = self.lstm1(x)  # (B, 24, 128)
        if time_weight.dim() == 2:
            fixed_weight = time_weight.detach().unsqueeze(-1)
        else:
            fixed_weight = time_weight.detach()
        attn1_out = lstm1_out * fixed_weight                # (B, 24, 128)
        lstm2_out, _ = self.lstm2(attn1_out)                # (B, 24, 128)
        attn2_out = self.attention2(lstm2_out)              # (B, 24, 128)
        pooled = torch.mean(attn2_out, dim=1)               # (B, 128)
        shared = F.relu(self.fc1(pooled))
        shared = F.relu(self.fc2(shared))
        adv = F.relu(self.adv_hidden(shared))
        adv = self.adv_stream(adv)                          # (B, 22)
        adv = adv.view(-1, 11, 2)                           # (B, 11, 2)
        val = F.relu(self.val_hidden(shared))
        val = self.val_stream(val)                          # (B, 11)
        val = val.view(-1, 11, 1)                           # (B, 11, 1)
        adv_mean = adv.mean(dim=2, keepdim=True)
        q_vals = val + (adv - adv_mean)                     # (B, 11, 2)
        self._value_out = torch.mean(val, dim=1).squeeze(-1)  # (B,)
        return q_vals

# -------------------------------
# 生成時間權重（推理時使用，隨機選擇模式）
# -------------------------------
def generate_time_weights(batch_size, seq_len, device="cpu", low=0.5, high=1.5):
    import random
    pattern = random.choice(["U", "increasing", "decreasing"])
    print(f"Selected time weight pattern: {pattern}")
    if pattern == "U":
        half = seq_len // 2
        if seq_len % 2 == 0:
            left = np.linspace(low, high, half)
            right = np.linspace(high, low, half)
            weights = np.concatenate([left, right])
        else:
            left = np.linspace(low, high, half+1)
            right = np.linspace(high, low, half+1)[1:]
            weights = np.concatenate([left, right])
    elif pattern == "increasing":
        weights = np.linspace(low, high, seq_len)
    elif pattern == "decreasing":
        weights = np.linspace(high, low, seq_len)
    else:
        raise ValueError("Unknown pattern.")
    weights = np.tile(weights, (batch_size, 1)).reshape(batch_size, seq_len, 1)
    return torch.tensor(weights, dtype=torch.float32, device=device)

# -------------------------------
# 定義 reward 評估函數（計算扣分分數，只計算負獎勵）
# -------------------------------
def evaluate_penalty(rules, real_record, action, output_ids):
    """
    參數：
      rules: 獎勵規則列表，每個 rule 為 dict，包含 "condition" 與 "reward"
      real_record: dict，包含實際天氣資訊
      action: 模型輸出的決策列表（長度 11），布林值
      output_ids: 用於設定變數名稱，與 training_data.json 中 output_ids 一致
    回傳：
      總扣分分數（對 reward 為負的規則，累加絕對值）
    """
    variables = {}
    variables.update(real_record)
    for i, key in enumerate(output_ids):
        variables[key] = bool(action[i])
    total_penalty = 0.0
    for rule in rules:
        condition = rule.get("condition", "")
        try:
            if eval(condition, {}, variables):
                reward_val = float(rule.get("reward", 0))
                if reward_val < 0:
                    total_penalty += abs(reward_val)
        except Exception as e:
            print("條件評估錯誤:", condition, variables, e)
    return total_penalty

# -------------------------------
# 定義函數：將陣列型態的 real_file 轉換為字典
# -------------------------------
def convert_input_to_real_record(data):
    """
    假設 data 為 list of lists，形狀 (24, 7)，按照順序：
      Temperature, DewPoint, ApparentTemperature, RelativeHumidity, WindSpeed, ProbabilityOfPrecipitation, ComfortIndex
    轉換方法：
      actual_temp: 平均 Temperature
      dew_point: 平均 DewPoint
      apparent_temp: 平均 ApparentTemperature
      relative_humidity: 平均 RelativeHumidity
      wind_speed: 平均 WindSpeed
      prob_precip: 平均 ProbabilityOfPrecipitation
      comfort_index: 平均 ComfortIndex
    並根據 prob_precip 決定 actual_rain 與 carry_umbrella
    """
    data = np.array(data, dtype=np.float32)
    avg_vals = data.mean(axis=0)
    actual_temp = float(avg_vals[0])
    dew_point = float(avg_vals[1])
    apparent_temp = float(avg_vals[2])
    relative_humidity = float(avg_vals[3])
    wind_speed = float(avg_vals[4])
    prob_precip = float(avg_vals[5])
    comfort_index = float(avg_vals[6])
    real_record = {
        "actual_rain": True if prob_precip >= 0.5 else False,
        "carry_umbrella": True if prob_precip >= 0.5 else False,
        "apparent_temp": apparent_temp,
        "actual_temp": actual_temp,
        "relative_humidity": relative_humidity,
        "dew_point": dew_point,
        "wind_speed": wind_speed,
        "comfort_index": comfort_index
    }
    return real_record

# -------------------------------
# 主程式：推理與驗證
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference & Validation script for WeatherDDQNModel (outputs T/F decisions and penalty score)")
    parser.add_argument("--config", type=str, default="model/config.json", help="Path to config file")
    parser.add_argument("--model_path", type=str, default="model/log/leatest.pth", help="Path to model checkpoint")
    parser.add_argument("--input_file", type=str, default="testinput.json", help="Path to test input JSON")
    parser.add_argument("--real_file", type=str, default="testinput.json", help="Path to real record JSON for validation (should be array format)")
    args = parser.parse_args()

    # 載入配置（若存在）
    try:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception as e:
        print("無法載入配置文件，使用預設值。")
        config = {}

    seq_len = config.get("sequence_length", 24)
    # 取得 input_ids 與 output_ids
    input_ids = config.get("input_ids", ["Temperature", "DewPoint", "ApparentTemperature", "RelativeHumidity", "WindSpeed", "ProbabilityOfPrecipitation", "ComfortIndex"])
    output_ids = config.get("output_ids", ["wearing_warm_clothes", "carry_umbrella", "damp_cold", "damp_hot", "dry_cold", "dry_hot", "wind_cold", "wind_hot", "outdoor_recommended", "increase_water_intake", "ground_slippery_warning"])
    input_dim = len(input_ids)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立模型並載入權重
    model = WeatherDDQNModel().to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 載入推理資料（期望 JSON 文件內容為 [seq_len, input_dim]）
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        sample = np.array(sample_data, dtype=np.float32)
    except Exception as e:
        print("無法載入推理輸入文件，使用預設全 0 輸入。")
        sample = np.zeros((seq_len, input_dim), dtype=np.float32)

    sample_tensor = torch.tensor(sample, dtype=torch.float32, device=device).unsqueeze(0)  # (1, seq_len, input_dim)

    # 生成時間權重（shape 為 (1, seq_len, 1)）
    time_weights = generate_time_weights(batch_size=1, seq_len=seq_len, device=device)

    with torch.no_grad():
        q_values = model(sample_tensor, time_weights)  # (1, 11, 2)

    # 對每個決策取 argmax 得到 0 或 1，轉換為布林值（0→False, 1→True）
    actions_tensor = torch.argmax(q_values, dim=2)  # (1, 11)
    actions_array = actions_tensor.squeeze(0).cpu().numpy()
    actions_bool = [True if a == 1 else False for a in actions_array]

    print("推理得到的 Q 值:")
    print(q_values.cpu().numpy())
    print("模型輸出的 T/F 決策:")
    print(actions_bool)

    # 處理真實資料（real_file）：由於實際上與 input_file 同樣都是陣列，
    # 則轉換為字典（取平均值等）以便與規則進行評估
    try:
        with open(args.real_file, "r", encoding="utf-8") as f:
            real_data = json.load(f)
        if isinstance(real_data, list):
            real_record = convert_input_to_real_record(real_data)
            print("已將 real_file（列表格式）轉換為字典。")
        elif isinstance(real_data, dict):
            real_record = real_data
        else:
            print("real_file 格式不正確，使用預設 dummy 資料。")
            real_record = {
                "actual_rain": True,
                "carry_umbrella": False,
                "apparent_temp": 12,
                "actual_temp": 12,
                "relative_humidity": 80,
                "dew_point": 5,
                "wind_speed": 8,
                "comfort_index": 10
            }
    except Exception as e:
        print("無法載入真實資料文件，使用預設 dummy 資料。")
        real_record = {
            "actual_rain": True,
            "carry_umbrella": False,
            "apparent_temp": 12,
            "actual_temp": 12,
            "relative_humidity": 80,
            "dew_point": 5,
            "wind_speed": 8,
            "comfort_index": 10
        }

    penalty_score = evaluate_penalty(config.get("rules", []), real_record, actions_bool, output_ids)
    print("驗證結果 - 總扣分分數:")
    print(penalty_score)

if __name__ == "__main__":
    main()