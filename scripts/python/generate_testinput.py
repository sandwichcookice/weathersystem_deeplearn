import json
import random
import os

# 來源與目標檔案路徑（請依實際路徑調整）
src_path = r"d:/AI-Module-training/WeatherSystem_Deeplearn/weathersystem_deeplearn/data/split_weather_continuous.json"
dst_path = r"d:/AI-Module-training/WeatherSystem_Deeplearn/weathersystem_deeplearn/data/testinput.json"

# 讀取 JSON 檔案
with open(src_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 取得預測資料與欄位定義
predicted_records = data.get("predicted_records", [])
input_ids = data.get("input_ids", [])

# 檢查是否有足夠的資料
if len(predicted_records) < 24:
    raise Exception("預測資料筆數不足 24 筆，無法抽取連續資料！")

# 隨機選擇起始位置：0 到 (len - 24)
start_idx = random.randint(0, len(predicted_records) - 24)
selected_records = predicted_records[start_idx:start_idx + 24]

# 構建推理用的輸入資料（可依需求更改格式）
inference_input = {
    "input_ids": input_ids,
    "predicted_records": selected_records
}

# 儲存至 testinput.json
with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(inference_input, f, indent=4)

print(f"已從 index {start_idx} 至 {start_idx+23} 抽取資料，並存成：{dst_path}")