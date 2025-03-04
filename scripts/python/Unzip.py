import os
import zipfile

def unzip_recursive(directory):
    """
    遞迴解壓縮目錄中的所有 ZIP 檔案
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.zip'):
                zip_path = os.path.join(root, file)
                extract_to = os.path.join(root, file.replace('.zip', ''))

                if not os.path.exists(extract_to):
                    os.makedirs(extract_to)

                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_to)
                    print(f"✅ 解壓縮: {zip_path} -> {extract_to}")
                except zipfile.BadZipFile:
                    print(f"❌ 無效的 ZIP 檔案: {zip_path}")
                    continue
                
                # 解壓縮完成後，檢查新資料夾內是否還有 ZIP 檔案
                unzip_recursive(extract_to)

if __name__ == "__main__":
    root_dir = input("請輸入要掃描的資料夾路徑: ").strip()
    
    if not os.path.exists(root_dir):
        print("❌ 指定的目錄不存在！")
    else:
        print(f"🔍 開始遞迴解壓縮 ZIP 檔案: {root_dir}")
        unzip_recursive(root_dir)
        print("🎉 所有 ZIP 檔案已解壓縮完成！")