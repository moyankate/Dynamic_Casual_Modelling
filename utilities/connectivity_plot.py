#### new! added by kate #### 
import pickle
from utilities.plot_utils import PlotGraph  # if you've moved it to a separate file
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

# Load results
with open('pdcm_results_subject0.pkl', 'rb') as f:
    data = pickle.load(f)

Ep = data['Ep']
A = Ep['A']
region_names = data['region_names']

# Display matrix
print("\nEffective Connectivity Matrix (A):")
df_A = pd.DataFrame(A, index=region_names, columns=region_names)
print(df_A)

# Graph connectivity
plotter = PlotGraph(region_names, A)
plotter.display()

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
