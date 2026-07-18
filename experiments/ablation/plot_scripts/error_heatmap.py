import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

os.makedirs('advanced_plots', exist_ok=True)
save_dir = r'c:\Users\shizhaojialele\Desktop\PINN\图全新'
os.makedirs(save_dir, exist_ok=True)

data_path = 'ablation_full_data.pkl'
print(f"Loading data from {data_path}...")
with open(data_path, 'rb') as f:
    all_results = pickle.load(f)

states = ['A', 'B', 'C', 'D']
groups_to_compare = ['Vanilla', 'E+P', '+FF_no_Lp', '+FF+E+P']

display_names = {
    'Vanilla': 'Vanilla',
    'E+P': 'E+P',
    '+FF_no_Lp': '+FF+E',
    '+FF+E+P': '+FF+E+P',
}

vmin_log = -8.0
vmax_log = 1.0

fig, axes = plt.subplots(4, 4, figsize=(24, 20))
fig.subplots_adjust(wspace=0.40, hspace=0.45, top=0.92, bottom=0.04, left=0.04, right=0.96)

for row, state in enumerate(states):
    if state not in all_results:
        continue
    state_data = all_results[state]
    base_data = state_data[0]
    t_eval = base_data['t_eval']
    t_max = t_eval[-1]

    for col, gname in enumerate(groups_to_compare):
        ax = axes[row, col]
        gdata = next((item for item in state_data if item['group'] == gname), None)

        if gdata is None or np.isnan(gdata['X_pred']).all() or np.isnan(gdata['final_error']):
            dummy_matrix = np.full((3, len(t_eval)), vmax_log)
            im = ax.imshow(dummy_matrix, aspect='auto', cmap='jet',
                           extent=[0, t_max, 0, 3], origin='lower',
                           vmin=vmin_log, vmax=vmax_log)
            ax.text(t_max / 2, 1.5, 'NaN\n(Gradient Explosion)',
                    color='white', fontsize=18, fontweight='bold',
                    ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.5'))
        else:
            X_pred = gdata['X_pred']
            X_base = gdata['X_base']
            abs_err = np.abs(X_pred - X_base) + 1e-12
            log_err = np.log10(abs_err).T
            im = ax.imshow(log_err, aspect='auto', cmap='jet',
                           extent=[0, t_max, 0, 3], origin='lower',
                           interpolation='nearest',
                           vmin=vmin_log, vmax=vmax_log)

        title_text = display_names.get(gname, gname)
        ax.set_title(title_text, fontsize=22.5, fontweight='bold', pad=10)
        ax.set_xlabel('Time $t$', fontsize=21)
        if col == 0:
            ax.set_ylabel('Spatial Component', fontsize=21)
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['$x$', '$y$', '$z$'], fontsize=21)
        ax.axhline(y=1, color='white', linewidth=1.5)
        ax.axhline(y=2, color='white', linewidth=1.5)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('$\log_{10}|\\text{Error}|$ (m)', fontsize=21)
        cbar.ax.tick_params(labelsize=21)

fig.suptitle('Spatiotemporal Error Evolution - Ablation Study',
             fontsize=28, fontweight='bold', y=0.98)

plt.savefig(os.path.join(save_dir, 'Figure_4.png'), format='png', dpi=200, bbox_inches='tight', pad_inches=0.15)
plt.savefig(os.path.join(save_dir, 'Figure_4.eps'), format='eps', dpi=200, bbox_inches='tight', pad_inches=0.15)
plt.close()
print(f"Saved: {os.path.join(save_dir, 'Figure_4.png')}")
print(f"Saved: {os.path.join(save_dir, 'Figure_4.eps')}")
print("Done!")
