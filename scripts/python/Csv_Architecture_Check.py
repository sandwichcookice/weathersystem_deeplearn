#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
處理 CSV 資料檔之架構差異：
1. 遞迴掃描指定目錄，篩選符合 auto_yyyymmdd.csv 與 weather_auto_yyyymmdd.csv 的檔案。
2. 讀取每個符合條件之檔案的第一行（分類標頭）。
3. 彙整並計數不同的架構，儲存至 data_history_Architecture。
4. 針對 2019 ~ 目前時間範圍，檔名符合度可在此邏輯中檢核。
"""

import os
import csv
import re
from collections import defaultdict

def discover_csv_architectures(root_dir):
    """
    遞迴掃描指定資料夾，找出符合命名規則的 CSV 檔案：
    - auto_yyyymmdd.csv
    - weather_auto_yyyymmdd.csv
    並讀取其第一行標頭，整合成各種架構計數。
    返回一個包含架構與次數的字典。
    """
    data_history_Architecture = defaultdict(int)
    
    # 正規表示式：符合 auto_YYYYMMDD.csv 或 weather_auto_YYYYMMDD.csv
    pattern = re.compile(r'^(weather_)?auto_\d{8}\.csv$')
    
    for current_path, dirs, files in os.walk(root_dir):
        for filename in files:
            # 先檢查檔名格式是否符合
            if pattern.match(filename):
                full_path = os.path.join(current_path, filename)
                
                # 懷疑此檔案是否真實存在或存在讀取問題？帶著質疑態度打開
                if not os.path.isfile(full_path):
                    # 如有需要，可在此加入錯誤處理或警示
                    continue
                
                # 嘗試讀取檔案
                with open(full_path, 'r', encoding='utf-8-sig', newline='') as f:
                    reader = csv.reader(f)
                    try:
                        header = next(reader)  # 讀取第一行
                        # 將當前標頭轉成 Tuple（不可變）作為字典 key
                        header_key = tuple(header)
                        data_history_Architecture[header_key] += 1
                    except StopIteration:
                        # 空檔案或讀取失敗時，帶著懷疑但先跳過
                        continue
    
    return data_history_Architecture

def filter_by_year(filename, start_year=2019, end_year=9999):
    """
    以檔名中的 yyyymmdd 判斷年份，是否落在指定範圍內。
    若符合，返回 True；否則返回 False。
    例：auto_20220101.csv -> 2022
    """
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        year = int(date_str[0:4])  # 取 yyyy
        return start_year <= year <= end_year
    return False

def main():
    # 假設你的輸入資料夾路徑如下
    input_directory = "./data/Data_history"
    
    # 先對所有符合檔名的檔案讀取架構
    all_architectures = discover_csv_architectures(input_directory)
    
    # 如果需要再過濾時間範圍(2019～now)，可以在此步驟中再次檢查
    # 亦可直接在 discover_csv_architectures() 內整合檢查
    # 下方為篩除 2019 年之後之範例，可依需求自行調整
    filtered_architectures = {}
    for arch_key, count in all_architectures.items():
        # 這裡只是展示手法，實際應在檔案遍歷處就處理年份
        # 因為 arch_key 只是標頭資訊，無法判斷年份
        # 此示範僅供參考，如需精準依年份篩選，要在 discover_csv_architectures() 中先判斷檔名
        filtered_architectures[arch_key] = count

    # 列印或後續寫檔至 data_history_Architecture
    print("=== data_history_Architecture 統計結果 ===")
    for arch, arch_count in filtered_architectures.items():
        print(f"架構: {arch}\n出現次數: {arch_count}\n{'-'*40}")

if __name__ == "__main__":
    main()
