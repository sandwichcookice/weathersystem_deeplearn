import torch
import json

# -------------------------------
# **讀取 `training_data.json`，獲取動態的 `true_weather` 欄位**
# -------------------------------
def load_true_weather_ids(json_file="training_data.json"):
    with open(json_file, "r") as f:
        config = json.load(f)
    return config.get("true_weather_ids", ["actual_rain", "actual_temp"])  # 預設值

# -------------------------------
# **GPU 加速數據生成（由 `train.py` 傳入 `device`）**
# -------------------------------
def generate_weather(input_ids, input_specs, true_weather_ids, device, batch_size=64):
    """
    生成「預測天氣」與「真實天氣」數據，並使用 GPU 加速
    - `input_ids`: 輸入變數名稱（如 "rain_prob", "temperature"）
    - `input_specs`: 輸入數據範圍（如 {"rain_prob": {"min": 0, "max": 100}, "temperature": {"min": 0.0, "max": 35.0}}）
    - `true_weather_ids`: 真實天氣的欄位名稱（從 JSON 設定檔讀取）
    - `device`: 由 `train.py` 傳入的運算設備（"cuda", "mps", 或 "cpu"）
    - `batch_size`: 要生成多少筆數據（適合訓練時一次生成多筆）
    """

    # **1️⃣ 生成「預測天氣」數據**
    min_vals = torch.tensor([input_specs[key]["min"] for key in input_ids], device=device)
    max_vals = torch.tensor([input_specs[key]["max"] for key in input_ids], device=device)

    # **在 GPU/CPU 上生成隨機數據**
    predicted_weather_tensor = min_vals + (max_vals - min_vals) * torch.rand((batch_size, len(input_ids)), device=device)

    # **轉換為 Python 字典格式**
    predicted_weather_list = [
        {key: round(predicted_weather_tensor[i, j].cpu().item(), 2) for j, key in enumerate(input_ids)}
        for i in range(batch_size)
    ]

    # **2️⃣ 生成「真實天氣」數據**
    true_weather_list = []
    for i in range(batch_size):
        rain_prob = predicted_weather_list[i]["rain_prob"] / 100

        # **是否下雨**（基於 `rain_prob`，但有 ±10% 誤差）
        true_rain_prob = max(0, min(1, rain_prob + torch.rand(1, device=device).item() * 0.2 - 0.1))
        actual_rain = torch.rand(1, device=device).item() < true_rain_prob

        # **實際氣溫**（基於 `temperature`，但有 ±3°C 誤差）
        actual_temp = round(predicted_weather_list[i]["temperature"] + (torch.rand(1, device=device).item() * 6 - 3), 2)

        # **根據 JSON 設定，動態生成 `true_weather`**
        true_weather = {}
        for field in true_weather_ids:
            if field == "actual_rain":
                true_weather[field] = bool(actual_rain)
            elif field == "actual_temp":
                true_weather[field] = actual_temp
            elif field == "humidity":
                true_weather[field] = round(torch.rand(1, device=device).item() * 60 + 30, 2)  # 假設濕度範圍 30~90%
            elif field == "wind_speed":
                true_weather[field] = round(torch.rand(1, device=device).item() * 20, 2)  # 假設風速範圍 0~20 m/s
            else:
                true_weather[field] = None  # 如果未定義，則設為 None

        true_weather_list.append(true_weather)

    return predicted_weather_list, true_weather_list

# -------------------------------
# **新增：時間權重生成函式**
# -------------------------------
def generate_time_weights(batch_size, seq_len, device="cpu", low=0.5, high=1.5):
    """
    生成固定的時間權重，形狀為 [batch_size, seq_len, 1]。
    此版本會隨機從三種模式中選擇一種：
      - "U"         : U型，兩頭低，中間高
      - "increasing": 遞增，從 low 遞增到 high
      - "decreasing": 遞減，從 high 遞減到 low
      
    參數:
      - batch_size: 每次生成的資料筆數
      - seq_len: 序列長度（例如一天內的時間點數）
      - low: 起始或最低權重值（預設 0.5）
      - high: 終止或最高權重值（預設 1.5）
      - device: 設定返回 tensor 的運算設備
      
    回傳:
      - 一個 torch.tensor，形狀為 [batch_size, seq_len, 1]
    """
    import numpy as np
    import random

    # 隨機選擇一種模式
    pattern = random.choice(["U", "increasing", "decreasing"])
    print(f"隨機選擇的時間權重模式: {pattern}")

    if pattern == "U":
        # 產生 U 型：兩頭低，中間高
        half = seq_len // 2
        if seq_len % 2 == 0:
            left = np.linspace(low, high, half)
            right = np.linspace(high, low, half)
            weights = np.concatenate([left, right])
        else:
            left = np.linspace(low, high, half + 1)
            right = np.linspace(high, low, half + 1)[1:]
            weights = np.concatenate([left, right])
    elif pattern == "increasing":
        weights = np.linspace(low, high, seq_len)
    elif pattern == "decreasing":
        weights = np.linspace(high, low, seq_len)
    else:
        raise ValueError("Unknown pattern. This should not happen.")

    # 擴展到 [batch_size, seq_len, 1]
    weights = np.tile(weights, (batch_size, 1)).reshape(batch_size, seq_len, 1)
    return torch.tensor(weights, dtype=torch.float32, device=device)


# -------------------------------
# **新增：輸出結果函式**
# -------------------------------
def tensor_converter(o):
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

def save_weather_data(predicted_weather, true_weather, filename="generated_weather.json", print_output=True):
    """
    儲存生成的天氣數據到 JSON，並可選擇是否打印結果
    修改處：當遇到 tensor 時，自動轉換為 list。
    """
    weather_data = {
        "predicted_weather": predicted_weather,
        "true_weather": true_weather
    }
    with open(filename, "w") as f:
        json.dump(weather_data, f, indent=4, default=tensor_converter)
    if print_output:
        print("✅ 已儲存天氣數據至", filename)
        print("🔹 預測天氣：", predicted_weather[:3])  # 只顯示前 3 筆
        print("🔹 真實天氣：", true_weather[:3])  # 只顯示前 3 筆


# -------------------------------
# **測試數據生成與輸出（修改後）**
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # 定義基本參數
    input_ids = ["rain_prob", "temperature"]
    input_specs = {
        "rain_prob": {"min": 0, "max": 100},
        "temperature": {"min": 0.0, "max": 35.0}
    }
    true_weather_ids = load_true_weather_ids()

    # 測試時批次量設為較少，例如 3 筆
    batch_size = 1
    # 序列長度設為 8 (一天內 8 個時間點)
    seq_len = 24

    # 產生時間序列資料：每個時間點使用 generate_weather 生成一組數據
    predicted_weather_seq = []  # 長度為 seq_len，每個元素為長度為 batch_size 的 list of dict
    true_weather_seq = []       # 與 predicted_weather_seq 結構相同
    time_weights_seq = []       # 每個時間點的時間權重
    for t in range(seq_len):
        pred, true = generate_weather(input_ids, input_specs, true_weather_ids, device, batch_size=batch_size)
        # 生成時間權重，形狀為 [batch_size, seq_len, 1]
        time_weights = generate_time_weights(batch_size, seq_len, device=device)
        predicted_weather_seq.append(pred)
        true_weather_seq.append(true)
        time_weights_seq.append(time_weights)



    # 印出測試結果
    print("生成的時間序列預測天氣與對應的時間權重：")
    for i in range(batch_size):
        print(f"\n--- Sample {i+1} ---")
        for t in range(seq_len):
            # 印出每個時間點的預測資料
            print(f"Time {t+1:02d}: {predicted_weather_seq[t][i]}")
        # 印出該樣本的時間權重 (轉換為一維陣列顯示)
        print("Time Weights:", time_weights[i].squeeze(-1).cpu().numpy())

    # 也可以儲存生成的數據到 JSON（如果需要）
    # 組合資料：將每個樣本的序列資料組成 list
    combined_predicted = []
    combined_true = []
    for i in range(batch_size):
        sample_pred = [predicted_weather_seq[t][i] for t in range(seq_len)]
        sample_true = [true_weather_seq[t][i] for t in range(seq_len)]
        combined_predicted.append(sample_pred)
        combined_predicted.append(time_weights_seq[i])
        combined_true.append(sample_true)
        
    # 儲存結果（此函式保持不變）
    save_weather_data(combined_predicted, combined_true, filename="generated_weather_sequence.json")