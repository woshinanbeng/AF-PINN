import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

with open('ablation_full_data.pkl', 'rb') as f:
    data = pickle.load(f)

states = ['A', 'B', 'C', 'D']
target_groups = ['+FF_no_Lp', '+FF+E+P']

colors = {'+FF_no_Lp': '#1f77b4', '+FF+E+P': '#d62728'}
linestyles = {'+FF_no_Lp': ':', '+FF+E+P': '--'}

fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=300)

for col, state in enumerate(states):
    ax = axes[col]
    if state not in data:
        continue
    state_data = data[state]
    
    mid_x, mid_y = None, None
    
    for group in target_groups:
        g_data = next((item for item in state_data if item.get('group') == group), None)
        if not g_data: 
            continue
        
        X_pred = g_data.get('X_pred')
        X_base = g_data.get('X_base')
        
        if group == '+FF+E+P' and X_base is not None: 
            ax.plot(X_base[:, 0], X_base[:, 1], '-', color='gray', alpha=0.5, linewidth=3.5, label='Boris (Truth)')
            mid_idx = len(X_base) // 2
            mid_x, mid_y = X_base[mid_idx, 0], X_base[mid_idx, 1]
        
        if X_pred is not None and not np.isnan(X_pred).all():
            display_label = '+FF+E' if group == '+FF_no_Lp' else group
            
            ax.plot(X_pred[:, 0], X_pred[:, 1], linestyle=linestyles[group], 
                    color=colors[group], linewidth=2.0 if group == '+FF+E+P' else 1.8, label=display_label)

    axins = inset_axes(ax, width="30%", height="30%", loc='lower left', borderpad=1.5)
    
    for group in target_groups:
        g_data = next((item for item in state_data if item.get('group') == group), None)
        if not g_data: 
            continue
            
        X_pred = g_data.get('X_pred')
        X_base = g_data.get('X_base')
        
        if X_base is not None:
            axins.plot(X_base[:, 0], X_base[:, 1], '-', color='gray', alpha=0.5, linewidth=4)
            
        if X_pred is not None and not np.isnan(X_pred).all():
            axins.plot(X_pred[:, 0], X_pred[:, 1], linestyle=linestyles[group], 
                       color=colors[group], linewidth=2.5 if group == '+FF+E+P' else 2.0)

    axins.set_aspect('equal')
    axins.set_xticks([])
    axins.set_yticks([])
    
    if mid_x is not None and mid_y is not None:
        span = 0.0125
        axins.set_xlim(mid_x - span, mid_x + span)
        axins.set_ylim(mid_y - span, mid_y + span)
    
    mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.4", linestyle='--')

    ax.set_title(f'Case {state}', fontweight='bold')
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel('X (m)')
    if col == 0: 
        ax.set_ylabel('Y (m)')
    
    ax.legend(loc='upper right', framealpha=0.9)

plt.tight_layout()
save_dir = 'output/figures'
os.makedirs(save_dir, exist_ok=True)
plt.savefig(os.path.join(save_dir, 'Figure_6.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(save_dir, 'Figure_6.eps'), format='eps', dpi=300, bbox_inches='tight')
print("Saved: Figure_6.png / Figure_6.eps")