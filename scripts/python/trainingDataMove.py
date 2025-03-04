import os
import shutil
import random

def collect_json_files(directory):
    """
    遞迴遍歷資料夾，收集最底層的 JSON 檔案，並過濾掉 macOS 生成的 `._*` 檔案
    """
    json_files = []
    
    for root, _, files in os.walk(directory):
        subdirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        if not subdirs:  # 沒有子資料夾，視為最底層
            for file in files:
                if file.lower().endswith(".json") and not file.startswith("._"):
                    json_files.append(os.path.join(root, file))
    
    return json_files

def split_and_move_files(json_files, base_dir):
    """
    按照 70%:20%:10% 的比例移動 JSON 檔案到 `train`, `check`, `val`
    """
    random.shuffle(json_files)  # 隨機打亂順序
    total_count = len(json_files)
    
    train_count = int(total_count * 0.7)
    check_count = int(total_count * 0.2)
    val_count = total_count - train_count - check_count  # 剩餘的給 val
    
    # 設定目標目錄
    train_dir = os.path.join(base_dir, "train")
    check_dir = os.path.join(base_dir, "check")
    val_dir = os.path.join(base_dir, "val")
    
    # 創建資料夾
    for folder in [train_dir, check_dir, val_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # 移動檔案
    for i, file in enumerate(json_files):
        if not os.path.exists(file):
            print(f"⚠️ 檔案不存在，跳過: {file}")
            continue  # 避免因為找不到檔案而中斷

        if i < train_count:
            target_dir = train_dir
        elif i < train_count + check_count:
            target_dir = check_dir
        else:
            target_dir = val_dir
        
        try:
            shutil.move(file, os.path.join(target_dir, os.path.basename(file)))
            print(f"📂 移動: {file} → {target_dir}")
        except Exception as e:
            print(f"❌ 移動失敗: {file}, 錯誤: {e}")

if __name__ == "__main__":
    root_dir = input("請輸入要掃描的資料夾路徑: ").strip()
    
    if not os.path.exists(root_dir):
        print("❌ 指定的目錄不存在！")
    else:
        print(f"🔍 開始遍歷目錄: {root_dir}")
        json_files = collect_json_files(root_dir)
        
        print(f"📊 找到 {len(json_files)} 個 JSON 檔案")
        if len(json_files) > 0:
            split_and_move_files(json_files, root_dir)
            print("✅ JSON 檔案已成功分類到 `train/`, `check/`, `val/`！")