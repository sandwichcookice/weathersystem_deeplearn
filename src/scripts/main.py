#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 輔助函數：將模型輸出轉換為布林決策字典
def derive_action_dict(q_vals, output_ids):
    """
    將模型輸出 q_vals (shape (1, 22) 或 (1, 11, 2)) 重塑為 (11, 2)，
    並對每組取 argmax，轉換為布林值後建立決策字典。
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
        x: (B, seq_len, 7) 輸入氣象資料
        time_weight: 預期為 (B, seq_len) 或 (B, seq_len, 1)
        """
        lstm1_out, _ = self.lstm1(x)  # (B, seq_len, 128)
        # 處理 time_weight 維度
        if time_weight.dim() == 2:
            fixed_weight = time_weight.detach().unsqueeze(-1)
        elif time_weight.dim() == 3:
            fixed_weight = time_weight.detach()
        else:
            raise ValueError("Unexpected time_weight dimensions: {}".format(time_weight.dim()))
        attn1_out = lstm1_out * fixed_weight  # (B, seq_len, 128)
        lstm2_out, _ = self.lstm2(attn1_out)   # (B, seq_len, 128)
        pooled = torch.mean(lstm2_out, dim=1)    # (B, 128)
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

def main():
    parser = argparse.ArgumentParser(description="打包後之 Dueling DQN 推理執行檔")
    parser.add_argument("--model_path", type=str, required=True, help="模型檢查點路徑")
    parser.add_argument("--input_data", type=str, required=True, help="輸入資料的 JSON 文件路徑")
    parser.add_argument("--time_weight", type=str, required=True, help="time weight 的 JSON 文件路徑")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入輸入資料，處理兩種格式：
    # 若為 dict 且包含 predicted_records，則從中提取固定七個欄位的資料，資料缺失則補 0；
    # 否則預期為二維陣列
    try:
        with open(args.input_data, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        if isinstance(input_data, dict) and "predicted_records" in input_data:
            fixed_input_ids = ["Temperature", "DewPoint", "ApparentTemperature", "RelativeHumidity", "WindSpeed", "ProbabilityOfPrecipitation", "ComfortIndex"]
            records = input_data["predicted_records"]
            sample = []
            for record in records:
                element = record.get("ElementValue", {})
                row = []
                for key in fixed_input_ids:
                    try:
                        value = float(element.get(key, 0))
                    except:
                        value = 0.0
                    row.append(value)
                sample.append(row)
            input_array = np.array(sample, dtype=np.float32)
        else:
            input_array = np.array(input_data, dtype=np.float32)
        if input_array.ndim == 2:
            sample_tensor = torch.tensor(input_array, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            raise ValueError("輸入資料必須為二維陣列 (seq_len, input_dim)")
    except Exception as e:
        print("無法載入輸入資料：", e)
        return

    # 載入 time weight 資料 (預期格式為一維或二維陣列)
    try:
        with open(args.time_weight, "r", encoding="utf-8") as f:
            time_weight_data = json.load(f)
        time_weight_array = np.array(time_weight_data, dtype=np.float32)
        if time_weight_array.ndim == 1:
            time_weight_tensor = torch.tensor(time_weight_array, dtype=torch.float32, device=device)\
                                    .unsqueeze(0).unsqueeze(-1)
        elif time_weight_array.ndim == 2:
            time_weight_tensor = torch.tensor(time_weight_array, dtype=torch.float32, device=device)\
                                    .unsqueeze(-1)
        else:
            raise ValueError("time weight 資料必須為一維或二維陣列")
    except Exception as e:
        print("無法載入 time weight 資料：", e)
        return

    # 使用預設的 output_ids
    output_ids = [
        "wearing_warm_clothes", "carry_umbrella", "damp_cold", "damp_hot",
        "dry_cold", "dry_hot", "wind_cold", "wind_hot",
        "outdoor_recommended", "increase_water_intake", "ground_slippery_warning"
    ]

    # 載入模型檢查點 (僅載入權重)
    model = WeatherDuelingDQNModel().to(device)
    try:
        state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    except Exception as e:
        print("載入模型檢查點失敗：", e)
        return

    model.eval()
    with torch.no_grad():
        q_values = model(sample_tensor, time_weight_tensor)

    action_dict = derive_action_dict(q_values, output_ids)
    # 僅輸出決策字典 (以 JSON 格式輸出)
    print(json.dumps(action_dict, ensure_ascii=False))

if __name__ == "__main__":
    main()