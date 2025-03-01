#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os

def main():
    # 假設 summary.json 放在這裡
    summary_file_path = "./data/Data_history_clean/summary.json"

    # 檢查檔案是否存在
    if not os.path.isfile(summary_file_path):
        print(f"無法找到檔案: {summary_file_path}")
        return

    # 讀取 summary.json
    with open(summary_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 確認結構是否包含 "stations"
    if "stations" not in data:
        print("summary.json 結構異常，無 'stations' 欄位。")
        return

    stations_info = data["stations"]
    
    # 篩選 status = "abnormal" 的測站
    abnormal_stations = []
    for station_id, meta in stations_info.items():
        if meta.get("status") == "abnormal":
            abnormal_stations.append({
                "station_id": station_id,
                "count": meta.get("count", 0),
                "id_min": meta.get("id_min", 0),
                "id_max": meta.get("id_max", 0)
            })
    
    # 若無異常測站，提示訊息
    if not abnormal_stations:
        print("目前無異常測站。")
        return
    
    # 將結果印出，可依需求調整格式
    print("以下為偵測到之異常測站：")

    num = 0

    for st in abnormal_stations:
        num+=1
        print(f"- 測站: {st['station_id']}, 筆數: {st['count']}, ID範圍: {st['id_min']} ~ {st['id_max']}")

    print(f"總計 {num} 個異常測站。")

if __name__ == "__main__":
    main()
