#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import re
import json
from datetime import datetime

# === 1) 欄位對應表 ===
#   - 收錄不同 CSV 格式可能出現的原始欄位 (左) 與「最終欄位名稱」(右)
#   - Precipitation 與 H_24R 均視為「日累積雨量」並對應到 "precipitation"
KNOWN_FIELDS = {
    "station_id":       "station_id",
    "obsTime":          "date_time",
    "WDIR":             "wind_direction",
    "WDSD":             "wind_speed",
    "TEMP":             "air_temperature",
    "HUMD":             "relative_humidity",
    "PRES":             "air_pressure",
    "H_24R":            "precipitation",   # 日累積雨量

    "StationId":        "station_id",
    "WindDirection":    "wind_direction",
    "WindSpeed":        "wind_speed",
    "AirTemperature":   "air_temperature",
    "RelativeHumidity": "relative_humidity",
    "AirPressure":      "air_pressure",
    "Precipitation":    "precipitation",   # 日累積雨量
    "Weather":          "weather",
    "DateTime":         "date_time",
    "phenomenonTime":   "date_time"
}

def unify_row_data(header, row):
    """
    傳入 CSV 的表頭 (header) 與資料 (row)，
    回傳統一後的 dict 格式。
    資料型態需求：
      - station_id: str
      - weather: str (可有可無)
      - precipitation: float
      - wind_direction: int
      - wind_speed: float
      - relative_humidity: int
      - air_temperature: float
      - air_pressure: float
      - date_time: "yyyy-MM-dd HH:mm:ss"
    """

    # 先將當前 row 與 header 組成字典
    row_dict = dict(zip(header, row))

    # 預設結構
    unified = {
        "id": None,               # 稍後於程式中動態指定
        "station_id": None,       # str
        "weather": "N/A",         # str, 若找不到欄位時預設
        "precipitation": 0.0,     # float
        "wind_direction": None,   # int
        "wind_speed": None,       # float
        "relative_humidity": None,# int
        "air_temperature": None,  # float
        "air_pressure": None,     # float
        "date_time": None         # "yyyy-MM-dd HH:mm:ss"
    }

    # === 2) 欄位對應 ===
    for original_col, mapped_col in KNOWN_FIELDS.items():
        if original_col in row_dict:
            value = row_dict[original_col]
            unified[mapped_col] = value

    # === 3) 型態轉換 / 後處理 ===

    # 3.1 wind_direction -> int
    if unified["wind_direction"] is not None:
        try:
            unified["wind_direction"] = int(float(unified["wind_direction"]))
        except ValueError:
            unified["wind_direction"] = None

    # 3.2 wind_speed, precipitation, air_temperature, air_pressure -> float
    float_cols = ["wind_speed", "precipitation", "air_temperature", "air_pressure"]
    for col in float_cols:
        val = unified[col]
        if val is not None:
            try:
                unified[col] = float(val)
            except ValueError:
                unified[col] = None

    # 3.3 relative_humidity -> int
    if unified["relative_humidity"] is not None:
        try:
            unified["relative_humidity"] = int(float(unified["relative_humidity"]))
        except ValueError:
            unified["relative_humidity"] = None

    # 3.4 date_time -> "yyyy-MM-dd HH:mm:ss"
    if unified["date_time"] is not None:
        dt_str = unified["date_time"]
        # 嘗試常見格式；若全部失敗則保留原字串或設 None
        possible_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S"
        ]
        for fmt in possible_formats:
            try:
                dt_obj = datetime.strptime(dt_str, fmt)
                unified["date_time"] = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                break
            except ValueError:
                pass
        # 若都無法解析，可視需要:
        # unified["date_time"] = None

    return unified

def main():
    # === 4) 基本路徑設定 ===
    input_folder = "./data/Data_history"   # TODO: 放置原始CSV的資料夾
    output_folder = "./data/Data_history_clean"    # TODO: 儲存分檔後結果的資料夾
    summary_file = "./data/Data_history_clean/summary.json"      # TODO: 總結檔案(包含各測站資料量與異常標註)

    # 若 output_folder 不存在，先行建立
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # === 5) 檔名篩選 (正則) ===
    csv_pattern = re.compile(r'^(weather_)?auto_\d{8}\.csv$')

    # === 6) 用於收集各測站資料與 Meta 資訊 ===
    stations_data = {}  # { station_id: [ {record1}, {record2}, ... ] }
    stations_meta = {}  # { station_id: { "count": 0, "id_min": ..., "id_max": ... } }

    record_id = 1  # 全域 ID 遞增

    # === 7) 遞迴掃描資料夾 & 處理CSV ===
    for root, dirs, files in os.walk(input_folder):
        for fname in files:
            # 篩選檔名
            if csv_pattern.match(fname):
                full_path = os.path.join(root, fname)
                if not os.path.isfile(full_path):
                    continue

                print(f"正在處理檔案: {full_path} ...")  # 執行過程輸出 (progress)

                # 讀取 CSV
                with open(full_path, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)
                    except StopIteration:
                        print(f"檔案無內容，跳過: {fname}")
                        continue

                    line_count = 0
                    for row in reader:
                        if len(row) != len(header):
                            # 欄位數不符，可能損毀或雜訊
                            continue
                        
                        unified_dict = unify_row_data(header, row)
                        unified_dict["id"] = record_id
                        record_id += 1
                        line_count += 1

                        sid = unified_dict["station_id"]
                        if not sid:
                            # station_id 為空時無法分檔
                            continue

                        # 加入 stations_data
                        if sid not in stations_data:
                            stations_data[sid] = []
                            stations_meta[sid] = {
                                "count": 0,
                                "id_min": float('inf'),
                                "id_max": -1
                            }
                        
                        stations_data[sid].append(unified_dict)

                        # 更新測站 Meta
                        stations_meta[sid]["count"] += 1
                        if record_id < stations_meta[sid]["id_min"]:
                            stations_meta[sid]["id_min"] = record_id
                        if record_id > stations_meta[sid]["id_max"]:
                            stations_meta[sid]["id_max"] = record_id
                    
                    print(f"檔案 {fname} 讀取完畢，共處理 {line_count} 筆資料。")

    # === 8) 分檔輸出 ===
    print("開始依測站分檔輸出...")
    for sid, records in stations_data.items():
        output_path = os.path.join(output_folder, f"{sid}.json")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(records, out_f, ensure_ascii=False, indent=4)

    print(f"分檔輸出完成，共處理 {record_id - 1} 筆紀錄。")

    # === 9) 統計 & 判斷「異常測站」 ===
    if len(stations_meta) == 0:
        print("無測站資料，無法產生 summary。")
        return

    # (1) 計算每個測站的筆數
    total_counts = sum(meta["count"] for meta in stations_meta.values())
    total_stations = len(stations_meta)
    avg_count = total_counts / total_stations  # 平均每個測站筆數

    # (2) 判斷閾值 (範例：小於平均值 80% 即標註為「異常」)
    threshold_ratio = 0.8
    threshold_value = avg_count * threshold_ratio

    summary_dict = {
        "total_station_count": total_stations,
        "average_records_per_station": avg_count,
        "threshold_ratio": threshold_ratio,
        "stations": {}
    }

    for sid, meta in stations_meta.items():
        count = meta["count"]
        # 由於我們對 record_id 進行 +1 時才存，故計算 id_min/id_max 時，可能要調整
        # 這裡示範記錄「該測站的最初及最終記錄 ID」：透過 stations_data[sid] 拿 min, max
        # 先簡易示範用:
        station_ids = [r["id"] for r in stations_data[sid]]
        id_min = min(station_ids)
        id_max = max(station_ids)

        # 判斷是否異常
        status = "abnormal" if count < threshold_value else "normal"

        summary_dict["stations"][sid] = {
            "count": count,
            "id_min": id_min,
            "id_max": id_max,
            "status": status
        }

    # (3) 輸出 summary.json
    with open(summary_file, 'w', encoding='utf-8') as sf:
        json.dump(summary_dict, sf, ensure_ascii=False, indent=4)

    print(f"已產生測站資料總結檔案：{summary_file}")
    print("=== 程式執行結束 ===")

if __name__ == "__main__":
    main()
