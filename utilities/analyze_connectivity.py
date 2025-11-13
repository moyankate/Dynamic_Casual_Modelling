#### new! added by Kate ####

import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd  # For Excel export
from utilities.plot_utils import PlotGraph
import math
import numpy as np

#### Load saved results ####
with open('sdcm_results_subject5.pkl', 'rb') as f:
    data = pickle.load(f)

# --- New diagnostic prints ---
print("\n==== Pickle File Contents Preview ====")
print(f"Top-level keys: {list(data.keys())}")

# If you want to peek at shapes and types safely
for k, v in data.items():
    if hasattr(v, 'shape'):
        print(f"{k}: array with shape {v.shape}")
    elif isinstance(v, dict):
        print(f"{k}: dict with keys {list(v.keys())}")
    else:
        print(f"{k}: type {type(v)}")
print("======================================\n")
# --- End diagnostics ---

Ep = data['Ep']
A = Ep['A']
B = Ep['B']
C = Ep['C']
region_names = data['region_names']
U = data['U'] 

#### Print A matrix to terminal ####
print("\nEffective Connectivity Matrix (A):")
df_A = pd.DataFrame(A, index=region_names, columns=region_names)
print(df_A)

#### Save A matrix to Excel ####
df_A.to_excel("A_matrix.xlsx")
print("\nA matrix saved to A_matrix.xlsx")

#### Plot A matrix heatmap ####
plt.figure(figsize=(5, 4))
sns.heatmap(A, annot=True, cmap='coolwarm', center=0,
            xticklabels=region_names, yticklabels=region_names)
plt.title('Effective Connectivity (A matrix)')
plt.xlabel('From Region')
plt.ylabel('To Region')
plt.tight_layout()
plt.show()

#### Connectivity plot ####
plotter = PlotGraph(region_names, A)
plotter.display()  

# Plot actual vs predicted BOLD for each ROI
y = data['predicted_BOLD']
actual_y = data['actual_BOLD']
region_names = data['region_names']
subject_idx = 0

n_rois = y.shape[1]
n_per_fig = 3
n_figs = math.ceil(n_rois / n_per_fig)

for f in range(n_figs):
    start = f * n_per_fig
    end = min((f + 1) * n_per_fig, n_rois)
    
    plt.figure(figsize=(12, 3 * (end - start)))
    for i, roi_idx in enumerate(range(start, end)):
        plt.subplot(end - start, 1, i + 1)
        plt.plot(actual_y[:, roi_idx], label='Actual BOLD', linestyle='--', alpha=0.8)
        plt.plot(y[:, roi_idx], label='Predicted BOLD', alpha=0.9)
        plt.title(f'ROI {roi_idx + 1}: {region_names[roi_idx]}')
        plt.ylabel('Signal')
        plt.legend()
        plt.tight_layout()
    
    plt.xlabel('Time (scans)')
    plt.suptitle(f'Predicted vs Actual BOLD Signal (Subject {subject_idx}) — ROIs {start+1}–{end}', fontsize=14, y=1.02)
    plt.subplots_adjust(top=0.93)  # Adjust space for suptitle
    plt.show()

#### plot U, the stimulus time course ####
plt.figure(figsize=(12, 4))
for i in range(U['u'].shape[1]):
    time = np.arange(U['u'].shape[0]) * U['dt']
    plt.plot(time, U['u'][:, i], label=f'Stimulus: {U["name"][i]}')
plt.xlabel('Time (s)')
plt.ylabel('Input Value')
plt.title('Stimulus Inputs Over Time')
plt.legend()
plt.tight_layout()
plt.show()

