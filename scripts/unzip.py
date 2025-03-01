import os
import zipfile

def extract_all_zips(target_folder):
    """
    遞迴處理指定資料夾內所有ZIP檔案，僅針對有效的ZIP檔案進行解壓縮。
    """
    processed = True

    while processed:
        processed = False
        for root, dirs, files in os.walk(target_folder):
            for file in files:
                if file.lower().endswith('.zip'):
                    zip_path = os.path.join(root, file)
                    
                    # 先確認檔案是否為有效的ZIP檔
                    if not zipfile.is_zipfile(zip_path):
                        print(f"跳過非ZIP檔案：{zip_path}")
                        continue
                    
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(root)
                        os.remove(zip_path)
                        print(f"已成功解壓縮：{zip_path}")
                        processed = True
                    except Exception as e:
                        print(f"解壓失敗 {zip_path}：{e}")

if __name__ == '__main__':
    # 指定目標資料夾路徑，請根據實際環境做調整
    target_folder = r"./data/Data_histroy"
    extract_all_zips(target_folder)