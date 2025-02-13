import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch.nn.functional as F

# -------------------------------
# 從 generate_weather.py 引入資料生成函式
# -------------------------------
from .scripts.generate_weather import load_true_weather_ids, generate_weather

# -------------------------------
# 注意力層：採用乘法方式將先驗時間權重整合進來
# -------------------------------
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)  # 可訓練參數
        
    def forward(self, lstm_outputs, time_weights):
        # lstm_outputs: [batch_size, seq_len, hidden_dim]
        # time_weights: [batch_size, seq_len, 1]
        scores = self.attn(lstm_outputs)         # [batch_size, seq_len, 1]
        scores = scores * time_weights           # 將先驗時間權重乘上去
        scores = scores.squeeze(-1)              # [batch_size, seq_len]
        attn_weights = F.softmax(scores, dim=1)    # 得到權重分佈 [batch_size, seq_len]
        attn_weights = attn_weights.unsqueeze(-1) # [batch_size, seq_len, 1]
        context = torch.sum(lstm_outputs * attn_weights, dim=1)  # [batch_size, hidden_dim]
        return context, attn_weights

# -------------------------------
# LSTM 與注意力整合的 Dueling DQN 模型定義
# -------------------------------
class LSTM_Attention_DDQN(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim, lstm_layers, output_dim, seq_len):
        """
        input_dim: 每個時間步的特徵數（例如 rain_prob, temperature）
        lstm_hidden_dim: LSTM 隱藏層維度
        lstm_layers: LSTM 層數
        output_dim: 輸出決策數量（根據 output_ids 的長度）
        seq_len: 時間序列長度（例如一天 8 個時間點）
        """
        super(LSTM_Attention_DDQN, self).__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim
        
        # LSTM 層：輸入 shape 為 [batch, seq_len, input_dim]
        self.lstm = nn.LSTM(input_dim, lstm_hidden_dim, lstm_layers, batch_first=True)
        # 注意力層：輸入 LSTM 的所有時間步輸出與時間權重
        self.attention = AttentionLayer(lstm_hidden_dim)
        
        # 全連接層
        self.fc1 = nn.Linear(lstm_hidden_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        
        # Advantage 分支
        self.adv_hidden = nn.Linear(128, 128)
        self.advantage_stream = nn.Linear(128, output_dim * 2)  # 每個決策有 2 個 Q 值
        
        # Value 分支
        self.val_hidden = nn.Linear(128, 128)
        self.value_stream = nn.Linear(128, output_dim)
        
    def forward(self, x, time_weights):
        """
        x: [batch_size, seq_len, input_dim] 的時間序列數據
        time_weights: [batch_size, seq_len, 1] 的先驗時間權重
        """
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, lstm_hidden_dim]
        context, attn_weights = self.attention(lstm_out, time_weights)  # context: [batch_size, lstm_hidden_dim]
        
        out = F.relu(self.fc1(context))
        out = F.relu(self.fc2(out))
        
        adv = F.relu(self.adv_hidden(out))
        adv = self.advantage_stream(adv)
        adv = adv.view(-1, self.output_dim, 2)
        
        val = F.relu(self.val_hidden(out))
        val = self.value_stream(val)
        val = val.unsqueeze(2)
        
        q_values = val + (adv - adv.mean(dim=2, keepdim=True))
        return q_values, attn_weights

# -------------------------------
# 資料集生成函式
# -------------------------------
def generate_training_data(num_samples, seq_len, input_ids, input_specs, true_weather_ids, device):
    """
    利用 generate_weather.py 中的 generate_weather 函式來生成時間序列資料。
    為了建立時間序列，每個樣本我們依序生成 seq_len 個時間點的預測數據，
    並轉換為 tensor，形狀為 [num_samples, seq_len, input_dim]。
    """
    X_seq = []
    # 對每個時間步，呼叫 generate_weather 生成單個時間點的資料
    for t in range(seq_len):
        # 注意：每次產生的 batch 都是 num_samples 筆，產生的 predicted_weather_list 為 list of dict
        predicted_weather_list, _ = generate_weather(input_ids, input_specs, true_weather_ids, device, batch_size=num_samples)
        # 將 list of dict 轉換為 [num_samples, input_dim] 的數據矩陣
        X_t = []
        for sample in predicted_weather_list:
            # 依照 input_ids 的順序提取數值
            row = [sample[key] for key in input_ids]
            X_t.append(row)
        X_t = torch.tensor(X_t, dtype=torch.float32, device=device)  # [num_samples, input_dim]
        X_seq.append(X_t.unsqueeze(1))  # [num_samples, 1, input_dim]
    # 沿著時間軸串接得到 [num_samples, seq_len, input_dim]
    X_seq = torch.cat(X_seq, dim=1)
    return X_seq

# -------------------------------
# 建立固定的先驗時間權重函式
# -------------------------------
def get_time_weights(num_samples, seq_len, device):
    """
    回傳固定的時間權重，shape 為 [num_samples, seq_len, 1]。
    例如假設早上的權重較高，設定一個 [8] 陣列
    """
    # 這裡定義一個長度為 seq_len 的先驗權重陣列
    time_weight_array = np.array([1.0, 1.2, 1.5, 1.5, 1.2, 1.0, 0.8, 0.5], dtype=np.float32)
    if len(time_weight_array) != seq_len:
        # 若 seq_len 與陣列長度不同，可重複或截斷
        time_weight_array = np.resize(time_weight_array, seq_len)
    # 擴展為 [num_samples, seq_len, 1]
    T = np.tile(time_weight_array, (num_samples, 1)).reshape(num_samples, seq_len, 1)
    return torch.tensor(T, dtype=torch.float32, device=device)

# -------------------------------
# 訓練函式
# -------------------------------
def train(model, optimizer, criterion, train_loader, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_states, batch_targets, batch_time_weights in train_loader:
            batch_states = batch_states.to(device)
            batch_targets = batch_targets.to(device)
            batch_time_weights = batch_time_weights.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(batch_states, batch_time_weights)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# -------------------------------
# 主程式
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="訓練 LSTM_Attention_DDQN 模型")
    parser.add_argument("--config_json", type=str, required=True,
                        help="包含模型結構與訓練參數的 JSON 配置文件")
    parser.add_argument("--save_model", type=str, default="lstm_attention_ddqn_model.pth",
                        help="保存訓練後模型的檔案名稱")
    args = parser.parse_args()

    # 讀取 config.json
    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    input_ids = config["input_ids"]
    output_ids = config["output_ids"]
    input_dim = len(input_ids)
    output_dim = len(output_ids)
    
    # 取得時間序列相關參數
    lstm_config = config["model_details"]["lstm_layer"]
    seq_len = lstm_config.get("sequence_length", 8)
    lstm_hidden_dim = lstm_config.get("lstm_hidden_dim", 64)
    lstm_layers = lstm_config.get("lstm_layers", 1)
    
    training_params = config.get("training", {})
    learning_rate = training_params.get("learning_rate", 0.001)
    batch_size = training_params.get("batch_size", 32)
    num_epochs = training_params.get("num_epochs", 100)
    optimizer_type = training_params.get("optimizer", "Adam")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 從 generate_weather.py 載入 true_weather_ids
    true_weather_ids = load_true_weather_ids()
    
    # 建立訓練資料：
    num_samples = 1000
    # 利用外部資料生成腳本產生時間序列資料： shape [num_samples, seq_len, input_dim]
    X_seq = generate_training_data(num_samples, seq_len, input_ids, config["input_specs"], true_weather_ids, device)
    # 目標 Q 值：示範用隨機生成，形狀 [num_samples, output_dim, 2]
    y = np.random.rand(num_samples, output_dim, 2).astype(np.float32)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    # 取得先驗時間權重，形狀 [num_samples, seq_len, 1]
    T = get_time_weights(num_samples, seq_len, device)
    
    # 建立資料集與 DataLoader
    train_dataset = TensorDataset(X_seq, y, T)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 建立模型
    model = LSTM_Attention_DDQN(input_dim, lstm_hidden_dim, lstm_layers, output_dim, seq_len)
    model.to(device)
    
    # 選擇 optimizer
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"不支援的 optimizer 類型：{optimizer_type}")
    
    # 定義損失函數
    criterion = nn.MSELoss()
    
    print("開始訓練...")
    train(model, optimizer, criterion, train_loader, num_epochs, device)
    
    torch.save(model.state_dict(), args.save_model)
    print(f"模型已保存至 {args.save_model}")

if __name__ == "__main__":
    main()