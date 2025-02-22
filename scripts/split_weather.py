import json
import random

def add_random_noise(value, min_pct=0.02, max_pct=0.05):
    """
    為一個數值添加隨機誤差，誤差百分比在 min_pct 與 max_pct 之間，
    隨機決定正負方向。
    """
    pct = random.uniform(min_pct, max_pct)
    if random.random() < 0.5:
        pct = -pct
    return value * (1 + pct)

def split_predicted_real(json_in="data/processed_weather.json", json_out="data/split_weather_continuous.json"):
    """
    從原始處理後的資料中分離出預測資料與真實資料：
      - 預測資料：針對 input_ids 裡的數值，每筆記錄添加隨機誤差（±2% ~ ±5%）。
      - 真實資料：根據與預測不同的鍵來構造（例如 actual_temp、dew_point、apparent_temp、
        relative_humidity、wind_speed、precipitation、actual_rain），同樣對數值添加隨機誤差。
      
    輸出結構為一個鍵值表，包含：
      "input_ids": [ ... ],
      "predicted_records": [ {record1}, {record2}, ... ],
      "real_records": [ {record1}, {record2}, ... ]
    每筆記錄都保留原有順序與日期時間等其他欄位。
    """
    with open(json_in, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 讀取預測用的 input_ids（例如：Temperature, DewPoint, ...）
    input_ids = data.get("input_ids", [])
    # 讀取所有連續記錄，假設存放在 "sequences" 或 "records" 中
    records = data.get("sequences", data.get("records", []))
    
    predicted_records = []
    real_records = []
    
    # 對每筆記錄產生預測與真實兩套資料
    for rec in records:
        pred_rec = {}
        # 對於預測資料：針對 input_ids 中的每個鍵，取原始數值並加入隨機誤差
        for key in input_ids:
            try:
                orig = float(rec.get(key, 0))
            except:
                orig = 0.0
            pred_rec[key] = add_random_noise(orig)
        # 如果有 date_time 等其他欄位，保留原始值
        for key in rec:
            if key not in input_ids:
                pred_rec[key] = rec[key]
        predicted_records.append(pred_rec)
        
        # 生成真實資料：使用另一組鍵，與預測不同，並參考規則中使用的真實欄位
        real_rec = {}
        # 將原始的 Temperature 對應為 actual_temp（加入獨立誤差）
        try:
            temp = float(rec.get("Temperature", 0))
        except:
            temp = 0.0
        real_rec["actual_temp"] = add_random_noise(temp)
        
        # DewPoint 對應為 dew_point
        try:
            dew = float(rec.get("DewPoint", 0))
        except:
            dew = 0.0
        real_rec["dew_point"] = add_random_noise(dew)
        
        # ApparentTemperature 對應為 apparent_temp
        try:
            app_temp = float(rec.get("ApparentTemperature", 0))
        except:
            app_temp = 0.0
        real_rec["apparent_temp"] = add_random_noise(app_temp)
        
        # RelativeHumidity 對應為 relative_humidity
        try:
            rh = float(rec.get("RelativeHumidity", 0))
        except:
            rh = 0.0
        real_rec["relative_humidity"] = add_random_noise(rh)
        
        # WindSpeed 對應為 wind_speed
        try:
            ws = float(rec.get("WindSpeed", 0))
        except:
            ws = 0.0
        real_rec["wind_speed"] = add_random_noise(ws)
        
        # 將 ProbabilityOfPrecipitation 作為 proxy 來表示 precipitation
        try:
            precip = float(rec.get("ProbabilityOfPrecipitation", 0))
        except:
            precip = 0.0
        real_rec["precipitation"] = add_random_noise(precip)
        
        # 生成 actual_rain：假設當 precipitation（這裡 proxy） > 50 時，視為下雨
        real_rec["actual_rain"] = True if precip > 50 else False
        
        # 保留其他欄位，如 date_time
        for key in rec:
            if key not in input_ids:
                # 如果已經處理過，則不重複（例如 date_time 已加入）
                if key not in real_rec:
                    real_rec[key] = rec[key]
        
        real_records.append(real_rec)
    
    out_data = {
        "input_ids": input_ids,
        "predicted_records": predicted_records,
        "real_records": real_records
    }
    
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=4)
    
    print(f"已儲存分離後的預測與真實資料至 {json_out}")

if __name__ == "__main__":
    split_predicted_real()