# Reward and punishment recovery

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameter settings
# ----------------------------
window_size = 100  # Moving average window size

def moving_average(series, window):
    """Compute the moving average for a given window size."""
    return series.rolling(window=window).mean()

# ----------------------------
# Read and preprocess Reward data for Training and Validation phases
# ----------------------------
models = ['dqn', 'ddqn', 'duelingdqn', 'c51', 'd3qn']
train_reward_data = {}   # Store training reward data for each model
val_reward_data = {}     # Store validation reward data for each model

for model in models:
    train_file = f"./data/{model}_reward.csv"
    val_file   = f"./data/{model}_reward_val.csv"
    
    # Read CSV files; each file must contain "Step" and "Value" columns.
    train_df = pd.read_csv(train_file)
    val_df   = pd.read_csv(val_file)
    
    if not set(['Step', 'Value']).issubset(train_df.columns):
        raise ValueError(f"{model} training reward file must contain 'Step' and 'Value' columns")
    if not set(['Step', 'Value']).issubset(val_df.columns):
        raise ValueError(f"{model} validation reward file must contain 'Step' and 'Value' columns")
    
    # Sort by Step and compute moving average
    train_df = train_df.sort_values(by='Step').reset_index(drop=True)
    val_df   = val_df.sort_values(by='Step').reset_index(drop=True)
    
    train_df['MA'] = moving_average(train_df['Value'], window_size)
    val_df['MA']   = moving_average(val_df['Value'], window_size)
    
    train_reward_data[model] = train_df
    val_reward_data[model]   = val_df

# ----------------------------
# Compute summary metrics: Average and Standard Deviation of Reward (using MA) for each model
# ----------------------------
train_avg = {}
val_avg = {}
train_std = {}
val_std = {}

for model in models:
    train_df = train_reward_data[model]
    val_df   = val_reward_data[model]
    
    # 使用移動平均後的數據，過濾 NaN 值
    train_valid = train_df['MA'].dropna()
    val_valid   = val_df['MA'].dropna()
    
    train_avg[model] = train_valid.mean()
    train_std[model] = train_valid.std()
    val_avg[model]   = val_valid.mean()
    val_std[model]   = val_valid.std()
    
    print(f"{model}: Training Avg Reward = {train_avg[model]:.2f}, Validation Avg Reward = {val_avg[model]:.2f}")

# ----------------------------
# Visualization: Plot Training Reward Moving Average Curves
# ----------------------------
plt.figure(figsize=(10,6))
for model in models:
    df = train_reward_data[model]
    plt.plot(df['Step'], df['MA'], label=model, linewidth=2)
plt.xlabel("Step")
plt.ylabel("Training Reward (Moving Average)")
plt.title("Training Reward MA Curves")
plt.legend()
plt.tight_layout()
plt.savefig("training_reward_ma_curves.png")
plt.show()

# ----------------------------
# Visualization: Plot Validation Reward Moving Average Curves
# ----------------------------
plt.figure(figsize=(10,6))
for model in models:
    df = val_reward_data[model]
    plt.plot(df['Step'], df['MA'], label=model, linewidth=2)
plt.xlabel("Step")
plt.ylabel("Validation Reward (Moving Average)")
plt.title("Validation Reward MA Curves")
plt.legend()
plt.tight_layout()
plt.savefig("validation_reward_ma_curves.png")
plt.show()

# ----------------------------
# Visualization: Bar Chart for Average Training Reward Comparison (with error bars)
# ----------------------------
plt.figure(figsize=(8,6))
train_bars = plt.bar(models, [train_avg[m] for m in models], yerr=[train_std[m] for m in models],
                     capsize=5, color='lightblue')
plt.xlabel("Model")
plt.ylabel("Average Training Reward")
plt.title("Training Reward Comparison")
for bar, value in zip(train_bars, [train_avg[m] for m in models]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.1f}",
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig("training_reward_comparison.png")
plt.show()

# ----------------------------
# Visualization: Bar Chart for Average Validation Reward Comparison (with error bars)
# ----------------------------
plt.figure(figsize=(8,6))
val_bars = plt.bar(models, [val_avg[m] for m in models], yerr=[val_std[m] for m in models],
                   capsize=5, color='lightgreen')
plt.xlabel("Model")
plt.ylabel("Average Validation Reward")
plt.title("Validation Reward Comparison")
for bar, value in zip(val_bars, [val_avg[m] for m in models]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.1f}",
             ha='center', va='bottom')
plt.tight_layout()
plt.savefig("validation_reward_comparison.png")
plt.show()
