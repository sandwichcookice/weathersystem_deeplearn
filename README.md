# 個人化即時天氣建議系統

本專案透過深度強化學習（Deep Reinforcement Learning）技術，結合氣象預報資料，建立一個可提供即時個人化天氣建議的系統，協助使用者根據未來24小時的天氣預報做出即時且適合個人需求的決策。

---

## 專案特色

- 使用 **Bi-LSTM + Attention** 作為前端特徵提取層。
- 採用 **Dueling DQN** 作為最終的強化學習決策模型。
- 支援 GPU 加速訓練，並提供友善的訓練腳本與驗證工具。
- 系統提供多項出行建議，如攜帶雨具、保暖衣物穿著建議、戶外活動適宜性評估等。

---

## 系統架構

```
氣象預報資料
     │
     ▼
前端特徵提取層 (Bi-LSTM + Attention)
     │
     ▼
強化學習模型決策 (Dueling DQN)
     │
     ▼
個人化即時天氣建議輸出
```

---

## 主要模型比較

本專案對多個基於 Q-learning 的強化學習模型進行了詳盡比較，包括：

- DQN
- Double DQN
- Dueling DQN (最終採用)
- Categorical DQN (C51)
- D3QN

> 最終選擇 **Dueling DQN** 模型，該模型在各項評估指標（學習效率、決策品質、穩定性等）皆有最佳表現。

---

## 目錄結構說明

- `/src`：存放原始程式碼與訓練腳本。
  - `/Scripts`：通用功能腳本
  - `/trainScripts`：模型訓練腳本
- `/dist`：部署打包檔案，包括模型、設定檔及資料
- `/analyze`：模型比較分析結果

---

## 安裝與使用

### 環境需求

- Python >= 3.8
- PyTorch >= 1.8
- CUDA 支援（可選，推薦使用）

### 安裝步驟

1. 克隆此專案至本地端
```bash
git clone https://github.com/sandwichcookice/weathersystem_deeplearn.git
```

2. 安裝套件

**MacOS:**
```bash
pip install -r requirements_mac.txt
```

**Windows (尚未完善，待後續版本支援):**
```bash
pip install -r requirements_win.txt
```

### 模型訓練

- 等待整理完後，後續版本將會更新

### 模型推理

使用提供的推理腳本進行推理：
```bash
python src/Scripts/inference.py --config (your_config_path) --model_path (your_model_path) --input_file (your_input_data)
```

---

## 版本更新紀錄

詳細版本更新紀錄，請參考 [UPDATELOG](./UpdateLog.md)。

---

## 未來開發方向

- 進一步提升模型泛化能力。
- 擴充更多樣化的個人化參數。
- 建立即時且輕量化的邊緣運算部署方案。

---

