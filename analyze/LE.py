#Learning efficiency

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameter settings
# ----------------------------
window_size = 100            # Moving average window size: 100 steps
reward_threshold = 300       # Reward threshold for sample efficiency analysis

def moving_average(series, window):
    """Compute the moving average for a given window size."""
    return series.rolling(window=window).mean()

# ----------------------------
# Read and preprocess Reward data for each model
# ----------------------------
models = ['dqn', 'ddqn', 'duelingdqn', 'c51', 'd3qn']
reward_data = {}

for model in models:
    reward_file = f"./data/{model}_reward.csv"
    df = pd.read_csv(reward_file)
    # Check required columns
    if not set(['Step', 'Value']).issubset(df.columns):
        raise ValueError(f"{model}'s Reward file must contain 'Step' and 'Value' columns")
    df = df.sort_values(by='Step').reset_index(drop=True)
    df['MA'] = moving_average(df['Value'], window_size)
    reward_data[model] = df

# ----------------------------
# Reward Threshold and AUC Analysis
# ----------------------------
threshold_steps = {}    # Store first step reaching reward threshold
auc_values = {}         # Store area under the reward MA curve

for model in models:
    df = reward_data[model]
    # Determine first step where moving average reward exceeds the threshold
    above_threshold = df[df['MA'] >= reward_threshold]
    if not above_threshold.empty:
        threshold_step = above_threshold.iloc[0]['Step']
    else:
        threshold_step = df['Step'].max()  # 若未達門檻，則取最大步數
    threshold_steps[model] = threshold_step

    # Calculate AUC for the moving average reward curve (ignore NaN values)
    valid = df['MA'].notna()
    auc = np.trapz(df.loc[valid, 'MA'], x=df.loc[valid, 'Step'])
    auc_values[model] = auc

# Print the computed metrics for each model
for model in models:
    print(f"{model}: reaches reward threshold at Step {threshold_steps[model]} and has Reward AUC = {auc_values[model]:.2f}")

# ----------------------------
# Visualization: Reward Moving Average Curves
# ----------------------------
plt.figure(figsize=(10,6))
for model in models:
    df = reward_data[model]
    plt.plot(df['Step'], df['MA'], label=model, linewidth=2)
plt.xlabel("Step")
plt.ylabel("Reward (Moving Average)")
plt.title("Reward Moving Average Curves")
plt.legend()
plt.tight_layout()
plt.savefig("reward_ma_curves.png")
plt.show()

# ----------------------------
# Visualization: Bar Chart for Reward Threshold Comparison
# ----------------------------
plt.figure(figsize=(8,6))
bars = plt.bar(models, [threshold_steps[m] for m in models], color='lightblue')
plt.xlabel("Model")
plt.ylabel("Step to Reach Reward Threshold")
plt.title("Learning Efficiency Analysis: Reward Threshold Comparison")
for bar, value in zip(bars, [threshold_steps[m] for m in models]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{int(value)}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("reward_threshold_comparison.png")
plt.show()

# ----------------------------
# Visualization: Bar Chart for Reward AUC Comparison
# ----------------------------
plt.figure(figsize=(8,6))
bars = plt.bar(models, [auc_values[m] for m in models], color='lightgreen')
plt.xlabel("Model")
plt.ylabel("Reward AUC")
plt.title("Learning Efficiency Analysis: Reward AUC Comparison")
for bar, value in zip(bars, [auc_values[m] for m in models]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.1f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("reward_auc_comparison.png")
plt.show()
