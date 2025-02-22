import json
import math
import numpy as np
import pandas as pd
import random

#############################################
# 1. 讀取原始資料（假設資料為 JSON 陣列，每筆為一個字典）
#############################################
def load_raw_data(filepath="data/Data.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
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
# 5. 為數值添加隨機誤差，用於生成預測與真實資料
#############################################
def add_random_noise(value, min_pct=0.02, max_pct=0.05):
    pct = random.uniform(min_pct, max_pct)
    if random.random() < 0.5:
        pct = -pct
    return value * (1 + pct)

#############################################
# 6. 根據原始轉換資料生成預測與真實資料
#############################################
def split_predicted_real(transformed_records, input_ids):
    predicted_records = []
    real_records = []
    
    for rec in transformed_records:
        pred_rec = {}
        # 預測資料：針對 input_ids 中的每個鍵加入誤差
        for key in input_ids:
            try:
                orig = float(rec.get(key, 0))
            except:
                orig = 0.0
            pred_rec[key] = add_random_noise(orig)
        # 保留其他欄位，例如日期時間
        for key in rec:
            if key not in input_ids:
                pred_rec[key] = rec[key]
        predicted_records.append(pred_rec)
        
        # 真實資料：依照對應規則轉換
        real_rec = {}
        try:
            temp = float(rec.get("Temperature", 0))
        except:
            temp = 0.0
        real_rec["actual_temp"] = add_random_noise(temp)
        
        try:
            dew = float(rec.get("DewPoint", 0))
        except:
            dew = 0.0
        real_rec["dew_point"] = add_random_noise(dew)
        
        try:
            app_temp = float(rec.get("ApparentTemperature", 0))
        except:
            app_temp = 0.0
        real_rec["apparent_temp"] = add_random_noise(app_temp)
        
        try:
            rh = float(rec.get("RelativeHumidity", 0))
        except:
            rh = 0.0
        real_rec["relative_humidity"] = add_random_noise(rh)
        
        try:
            ws = float(rec.get("WindSpeed", 0))
        except:
            ws = 0.0
        real_rec["wind_speed"] = add_random_noise(ws)
        
        try:
            precip = float(rec.get("ProbabilityOfPrecipitation", 0))
        except:
            precip = 0.0
        real_rec["precipitation"] = add_random_noise(precip)
        
        try:
            ci = float(rec.get("ComfortIndex", 0))
        except:
            ci = 0.0
        real_rec["comfort_index"] = add_random_noise(ci)
        
        real_rec["actual_rain"] = True if precip > 50 else False
        
        # 保留其他欄位
        for key in rec:
            if key not in input_ids and key not in real_rec:
                real_rec[key] = rec[key]
                
        real_records.append(real_rec)
    
    return predicted_records, real_records

#############################################
# 7. 儲存結果
#############################################
def save_combined_data(input_ids, predicted_records, real_records, filename="data/split_weather_continuous.json"):
    output_data = {
        "input_ids": input_ids,
        "predicted_records": predicted_records,
        "real_records": real_records
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4)
    print(f"已儲存分離後的預測與真實資料至 {filename}")

#############################################
# 主程式
#############################################
def main():
    # 讀取原始資料
    raw_records = load_raw_data("data/Data.json")
    fields_to_clean = ["air_temperature", "relative_humidity", "precipitation", "wind_speed"]
    df = pd.DataFrame(raw_records)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df.sort_values("date_time", inplace=True)
    df = clean_data(df, fields_to_clean)
    print("清洗後資料筆數:", len(df))
    
    # 轉換衍生特徵，得到每個記錄（保留日期時間資訊）
    transformed_records = transform_data(df)
    print("生成記錄數:", len(transformed_records))
    
    # 定義 input_ids，作為要處理的主要欄位
    input_ids = [
        "Temperature",
        "DewPoint",
        "ApparentTemperature",
        "RelativeHumidity",
        "WindSpeed",
        "ProbabilityOfPrecipitation",
        "ComfortIndex"
    ]
    
    # 依照轉換後的資料，生成預測與真實資料
    predicted_records, real_records = split_predicted_real(transformed_records, input_ids)
    
    # 儲存結果至同一份文件
    save_combined_data(input_ids, predicted_records, real_records, filename="data/split_weather_continuous.json")

if __name__ == "__main__":
    main()