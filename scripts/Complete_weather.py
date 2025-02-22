import json
import math
import numpy as np
import pandas as pd
import torch
from datetime import datetime

#############################################
# 1. 讀取原始資料（假設資料為 JSON 陣列，每筆為一個字典）
#############################################
def load_raw_data(filepath="data/Data.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # raw 直接是一個列表
    return raw

#############################################
# 2. 缺失值處理：對於 -99，採用線性插值補全（包括連續缺失情況）
#############################################
def clean_data(df, fields):
    for field in fields:
        df[field] = df[field].replace(-99, np.nan)
        df[field] = df[field].interpolate(method="linear")
        df[field] = df[field].fillna(method="ffill").fillna(method="bfill")
    return df

#############################################
# 3. 衍生特徵計算
#############################################
def compute_dew_point(T, RH):
    RH = max(RH, 1)
    numerator = 243.04 * (math.log(RH/100.0) + (17.625 * T)/(243.04 + T))
    denominator = 17.625 - math.log(RH/100.0) - (17.625 * T)/(243.04 + T)
    return numerator / denominator

def compute_apparent_temperature(T, V, RH):
    e = (RH/100.0) * 6.105 * math.exp((17.27 * T)/(237.7 + T))
    return 1.04 * T + 0.2 * e - 0.65 * V - 2.7

def compute_prob_precipitation(precip):
    if precip < 0.1:
        return 10.0
    else:
        return 10.0 + 70.0 * min(1.0, precip/0.5)

def compute_comfort_index(T, Td):
    diff_exp = math.exp((17.269 * Td)/(Td+237.3) - (17.269 * T)/(T+237.3))
    return T - 0.55 * (1 - diff_exp) * (T - 14)

#############################################
# 4. 將原始資料轉換為我們需要的特徵格式（保留字典形式）
#############################################
def transform_data(df):
    transformed = []
    for _, row in df.iterrows():
        T = row["air_temperature"]
        RH = row["relative_humidity"]
        precip = row.get("precipitation", 0)
        V = row.get("wind_speed", 0)
        Td = compute_dew_point(T, RH)
        T_app = compute_apparent_temperature(T, V, RH)
        P_rain = compute_prob_precipitation(precip)
        CI = compute_comfort_index(T, Td)
        transformed.append({
            "Temperature": T,
            "DewPoint": Td,
            "ApparentTemperature": T_app,
            "RelativeHumidity": RH,
            "WindSpeed": V,
            "ProbabilityOfPrecipitation": P_rain,
            "ComfortIndex": CI,
        })
    return transformed

#############################################
# 5. 將連續資料依時間順序分組成固定長度的序列（例如24小時一組）
#############################################
def group_into_sequences(records, seq_len=24):
    sequences = []
    num_records = len(records)
    # 固定分組：每 seq_len 筆作為一組，忽略不足 seq_len 筆的最後一組
    for i in range(0, num_records - seq_len + 1, seq_len):
        seq = records[i:i+seq_len]
        sequences.append(seq)  # 每個 seq 為一個 list，每筆記錄仍為字典
    return sequences

#############################################
# 6. 生成時間權重（隨機選擇 "U", "increasing", "decreasing" 模式）
#############################################
def generate_time_weights(batch_size, seq_len, device="cpu", low=0.5, high=1.5):
    import random
    if seq_len < 2:
        raise ValueError("Sequence length must be at least 2.")
    pattern = random.choice(["U", "increasing", "decreasing"])
    print(f"隨機選擇的時間權重模式: {pattern}")
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

#############################################
# 7. 儲存處理後的結果（保留字典形式的序列與 input_ids 映射）
#############################################
def tensor_converter(o):
    if isinstance(o, (np.ndarray, torch.Tensor)):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def save_sequences(data, filename="data/processed_weather.json"):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=tensor_converter)
    print(f"處理後資料已儲存至 {filename}")

#############################################
# 主程式
#############################################
def main():
    # 讀取原始資料
    raw_records = load_raw_data("data/Data.json")
    # 清洗欄位：air_temperature, relative_humidity, precipitation, wind_speed
    fields_to_clean = ["air_temperature", "relative_humidity", "precipitation", "wind_speed"]
    df = pd.DataFrame(raw_records)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df.sort_values("date_time", inplace=True)
    df = clean_data(df, fields_to_clean)
    print("清洗後資料筆數:", len(df))
    
    # 轉換成衍生特徵（保留字典形式，每筆記錄含鍵值）
    transformed_records = transform_data(df)
    
    print("生成序列數:", len(transformed_records))
    
    # 組合最終輸出，保留 input_ids 映射
    output_data = {
        "input_ids": [
            "Temperature",
            "DewPoint",
            "ApparentTemperature",
            "RelativeHumidity",
            "WindSpeed",
            "ProbabilityOfPrecipitation",
            "ComfortIndex"
        ],
        "sequences": transformed_records  # 每個序列是一個 list，每筆記錄為字典
    }
    
    # # 生成時間權重，形狀為 [num_sequences, 24, 1]
    # batch_size = len(transformed_records)
    # time_weights = generate_time_weights(batch_size, device="cpu")
    # output_data["time_weights"] = time_weights
    
    # 儲存結果
    save_sequences(output_data, filename="data/processed_weather.json")

if __name__ == "__main__":
    main()