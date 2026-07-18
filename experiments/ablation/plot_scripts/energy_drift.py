import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 18,
    "axes.titlesize": 36,
    "axes.labelsize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})
# Load ablation data
with open('ablation_full_data.pkl', 'rb') as f:
    data = pickle.load(f)

states = ['A', 'B', 'C', 'D']
groups = ['Vanilla', 'E+P', '+FF_no_Lp', '+FF+E+P']
colors = {'Vanilla': 'gray', 'E+P': 'purple', '+FF_no_Lp': 'blue', '+FF+E+P': 'red'}

fig, axes = plt.subplots(1, 4, figsize=(20, 4), dpi=300)

for col, state in enumerate(states):
    if state not in data:
        continue
        
    state_data = data[state]
    
    for group in groups:
        g_data = next((item for item in state_data if item.get('group') == group), None)
        if g_data is None:
            continue
            
        X_pred = g_data.get('X_pred')
        t_eval = g_data.get('t_eval')
        
        if X_pred is None or t_eval is None or np.isnan(X_pred).all():
            continue
            
        if 'V_pred' in g_data:
            V_pred = g_data['V_pred']
        else:
            V_pred = np.gradient(X_pred, t_eval, axis=0)
            
        delta_E = np.abs(v2 - v02) / v02
            
        delta_E = delta_E + 1e-16
        
        display_label = '+FF+E' if group == '+FF_no_Lp' else group
        
        axes[col].plot(t_eval, delta_E, label=display_label, color=colors[group], linewidth=1.5, alpha=0.8)

    axes[col].set_yscale('log')
    axes[col].set_xlabel('Time (s)', fontsize=12)
    if col == 0:
        axes[col].set_ylabel('Relative Energy Drift', fontsize=12)
    axes[col].grid(True, which="both", ls="--", alpha=0.4)
    axes[col].legend(loc='best', fontsize='small')
    axes[col].set_title(f'Case {state}', fontsize=24, fontweight='bold')

plt.tight_layout()

save_dir = 'output/figures'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'Figure_10.png'), dpi=600, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, 'Figure_10.eps'), format='eps', bbox_inches='tight')
print("Saved: Figure_10.png / Figure_10.eps")