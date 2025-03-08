# Changelog

## [v.0.2.0]
### (Update)
- 訓練方式從強化學習Q-learning 改成 DDQN + LSTM + 注意力

## [v.0.3.0]
### (Update)
- 更改模型架構 更改部分訓練檔案 並且有嘗試訓練模型 可以成功訊練 不過在真實參數的部分有數據錯誤 仍然繼續訓練 要改

## [v.0.4.0]
### (Update)
- 修改資料清洗腳本 確定清洗的Real Data 和 預測Data 都是正確可以被訓練使用的
- 訓練腳本引入可選用gpu訓練
- 訓練腳本引入進度條 方便觀測訓練進度
- 更新獎逞規則 在部分模糊界線添加正確的獎逞
- 修改推理資料的生成 現在推理資料將會從訓練資料中隨機抽取24個連續時間段推理
### (suggess)
- 修改一下rule的部分 目前是推測rule導致模型出現推理偏差 導致偏差部分 是在空缺部分導致的訓練異常

## [v.0.5.0]
### (Update)
- 修改rule 更新獎逞力度 和部分特殊條件判斷 例如濕熱和乾熱不可能同時出現
- 訓練腳本更新進入mac可以用的訓練方式 以便mac加快訓練速度
### (suggess)
- 增加批次驗證腳本 引入驗證資料 以確定模型是真的可用

## [v.0.5.1]
### (Update)
- 在.gitignore中增加一個vscode工作區遺忘

## [v.0.5.2]
### (Update)
- 資料來源更新，目前資料為過去六年的天氣狀況，壓縮在Data_history.zip , 預計使用其他雲端軟體同步
- unzip.py的新增 功能如腳本字面上意思

## [v.0.6.0]
### (Purpose)
- 清洗過去六年的資料相關
### (Update)
- 資料清洗和整理腳本的全面換新，目前最終清洗是採用C++來處理，編譯後的執行檔被放在bin
- Json檔案打包壓縮腳本，由C++編寫，編譯後的執行檔被放在bin
- C++腳本所需要的標頭檔案都存放在scripts/Chead中 目前有兩個，一個是JSON解析標頭 一個是miniz(zip)函式庫
- UnZip現在由C++來編寫，編譯後的執行檔被放在bin
- scripts 資料夾因為新增了C++的原因 架構重新編排，scripts的第一層子級是對應的語言名稱資料夾，他們的子級才會是各種功能腳本
- 將這個版本之前所用到的資料都轉移存放在Old_Data裡面，還留存的原因是架構保存，為防止之後有需要
- data資料夾將被永久遺忘 因為資料量的增加，導致無法上傳至github，所以由其他載體進行轉移
### (Delete)
- 原先使用py的清洗腳本(Complete_split_weather.py)被刪除，因為新的腳本是使用C++來編寫

## [v.0.7.0]
### (Update)
- Unzip.py 在MacOs上面用的
- trainingDataMove.py 用來移動訓練資料 到訓練/驗證/測試 集中
- trainscripts 這邊專門存放15種方式的訓練腳本
### (important)
- 在這個版本中 我們初步的完成了DQN的實作 並完善了他的腳本 在接下來的版本中會一一更新剩下的十四種
1. ~~DQN~~
1. Double DQN
2. Dueling DQN
3. Policy Gradient (REINFORCE)
4. Actor-Critic (例如 A3C)
5. Proximal Policy Optimization (PPO)
6. Deep Deterministic Policy Gradient (DDPG)
7. Twin Delayed DDPG (TD3)
8. Soft Actor-Critic (SAC)
9. Behavioral Cloning (BC)
10. DAgger 算法
11. Generative Adversarial Imitation Learning (GAIL)
12. Deep Q-learning from Demonstrations (DQfD)
13. Rainbow DQN
14. World Models

## [v.0.8.0]
### (Update)
- v.0.6.0 的 Change 被修正為 Update , Updata 被修正為 Update 此處為打字錯誤
- trainscripts 的名稱從 trainscipts 更正為 trainscripts 此處為打字錯誤
- Double DQN 的訓練腳本
### (important)
- 在這個版本中 我們修該了原本的目標 所以原本的強化訓練腳本變成了專注於離散動作範圍的強化訓練腳本--DQN以及其衍生版
- 其餘未被採納，但是已被寫完的腳本將會被放在trainscripts/savescripts中
- 已經被訓練完的模型，但是未被採納的，將會被存放在./savemodel中
1. ~~DQN~~
1. ~~Double DQN~~
2. ~~Dueling DQN~~
4. Categorical DQN (C51)
5. Rainbow DQN