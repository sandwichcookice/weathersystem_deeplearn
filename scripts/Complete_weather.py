import json
import math
import numpy as np
import pandas as pd
from datetime import datetime

# 讀取原始 JSON 資料（請確保檔案路徑正確）
def load_raw_data(filepath="data/Data.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # 假設資料存放在 raw["records"]["Locations"][0]["Location"][...]
    # 根據您提供的結構進行調整，這裡以第一個 Location 為例：
    location_data = raw["records"]["Locations"][0]["Location"]
    # 將每筆資料展開，每筆資料可能包含多個 WeatherElement
    # 這裡我們假設從中提取下列欄位：
    # "air_temperature", "relative_humidity", "precipitation", "wind_speed", "air_pressure", "date_time"
    records = []
    for loc in location_data:
        # loc 為一個字典，內含 "WeatherElement" 列表
        # 我們遍歷 WeatherElement，根據 ElementName 選取需要的值
        # 這裡以簡化方式假設 Temperature 來自 ElementName=="溫度"的第一個 ElementValue，RH來自 "RelativeHumidity"等
        # 請根據實際情況調整：
        for we in loc.get("WeatherElement", []):
            ename = we.get("ElementName", "")
            for t in we.get("Time", []):
                dt = t.get("DataTime") or t.get("StartTime")
                # 取第一個 ElementValue
                val_dict = t.get("ElementValue", [{}])[0]
                record = {"date_time": dt}
                if ename == "溫度":
                    record["Temperature"] = float(val_dict.get("Temperature", -99))
                elif ename == "相對濕度":
                    record["RelativeHumidity"] = float(val_dict.get("RelativeHumidity", -99))
                elif ename == "降水量":  # 假設降水量
                    record["Precipitation"] = float(val_dict.get("Precipitation", -99))
                elif ename == "風速":
                    record["WindSpeed"] = float(val_dict.get("WindSpeed", -99))
                elif ename == "氣壓":
                    record["AirPressure"] = float(val_dict.get("AirPressure", -99))
                # 可依需要添加更多欄位
                records.append(record)
    return records

# 缺失值處理：對於 -99，使用前後有效值的線性插值
def clean_data(records, field_names):
    # 轉換成 DataFrame（依照 date_time 排序）
    df = pd.DataFrame(records)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df.sort_values("date_time", inplace=True)
    # 對指定欄位，將 -99 視為缺失值
    for field in field_names:
        df[field] = df[field].replace(-99, np.nan)
        # 使用線性插值填補缺失值
        df[field] = df[field].interpolate(method="linear")
        # 如首尾仍有缺失，用前/後值填充
        df[field] = df[field].fillna(method="ffill").fillna(method="bfill")
    return df

# 計算露點溫度（Magnus公式）
def compute_dew_point(T, RH):
    # T: 氣溫 (°C), RH: 相對濕度 (%)
    # 防止 RH==0
    RH = max(RH, 1)
    numerator = 243.04 * (np.log(RH/100.0) + (17.625 * T)/(243.04+T))
    denominator = 17.625 - np.log(RH/100.0) - (17.625 * T)/(243.04+T)
    return numerator / denominator

# 計算體感溫度
def compute_apparent_temperature(T, V, RH):
    # T: 氣溫 (°C), V: 風速 (m/s), RH: 相對濕度 (%)
    e = (RH/100.0) * 6.105 * math.exp((17.27 * T)/(237.7+T))
    return 1.04 * T + 0.2 * e - 0.65 * V - 2.7

# 降雨機率（根據降水量，簡單映射）
def compute_prob_precipitation(precip):
    # precip: 降水量 (mm)
    if precip < 0.1:
        return 10.0
    else:
        return 10.0 + 70.0 * min(1.0, precip/0.5)

# 舒適度指數（THI）公式
def compute_comfort_index(T, Td):
    # T: 氣溫, Td: 露點溫度
    return T - 0.55 * (1 - math.exp((17.269 * Td)/(Td+237.3) - (17.269 * T)/(T+237.3))) * (T - 14)

# 將原始資料轉換為模型輸入格式
def transform_data(df):
    # 假設我們需要的欄位：Temperature, RelativeHumidity, Precipitation, WindSpeed
    # 由此計算：DewPoint, ApparentTemperature, ComfortIndex, ProbabilityOfPrecipitation
    transformed = []
    for idx, row in df.iterrows():
        T = row["Temperature"]
        RH = row["RelativeHumidity"]
        precip = row.get("Precipitation", 0)  # 若缺失，假設為0
        V = row.get("WindSpeed", 0)
        # 計算露點
        Td = compute_dew_point(T, RH)
        # 計算體感溫度
        T_app = compute_apparent_temperature(T, V, RH)
        # 降雨機率
        P_rain = compute_prob_precipitation(precip)
        # 舒適度指數
        CI = compute_comfort_index(T, Td)
        record = {
            "Temperature": T,
            "DewPoint": Td,
            "ApparentTemperature": T_app,
            "RelativeHumidity": RH,
            "WindSpeed": V,
            "ProbabilityOfPrecipitation": P_rain,
            "ComfortIndex": CI,
            "date_time": row["date_time"]
        }
        transformed.append(record)
    return transformed

# 將連續資料依照時間順序分組成固定長度的序列（例如 24 小時一組）
def group_into_sequences(records, seq_len=24):
    # records 已經依 date_time 排序
    sequences = []
    num_records = len(records)
    # 依次取連續的 seq_len 筆資料（可以採用滑動窗口或固定分組）
    # 這裡採用固定分組：從頭開始每 24 筆作為一組，忽略不足 24 筆的最後一組
    for i in range(0, num_records - seq_len + 1, seq_len):
        seq = records[i:i+seq_len]
        sequences.append(seq)
    return sequences

def main():
    # 載入原始資料
    raw_records = load_raw_data("data/Data.json")  # 根據實際檔案位置
    # 欄位需要清洗：Temperature, RelativeHumidity, Precipitation, WindSpeed
    fields_to_clean = ["Temperature", "RelativeHumidity", "Precipitation", "WindSpeed"]
    df = clean_data(raw_records, fields_to_clean)
    print("清洗後資料筆數:", len(df))
    
    # 轉換成我們需要的特徵
    transformed_records = transform_data(df)
    
    # 根據 date_time 排序（若尚未排序）
    transformed_records.sort(key=lambda x: x["date_time"])
    
    # 分組成連續序列，每組 24 筆
    # sequences = group_into_sequences(transformed_records, seq_len=24)
    # print("生成序列數:", len(sequences))
    
    # 最後，可將序列資料儲存為 JSON 或 CSV 用於後續訓練
    output_filepath = "data/processed_weather.json"
    with open(output_filepath, "w", encoding="utf-8") as f:
        json.dump(transformed_records, f, indent=4, default=str)
    print(f"處理後資料已儲存至 {output_filepath}")

if __name__ == "__main__":
    main()