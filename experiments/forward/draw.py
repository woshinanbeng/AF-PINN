import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

os.makedirs('advanced_plots', exist_ok=True)
save_dir = 'output/figures'
os.makedirs(save_dir, exist_ok=True)

data_path = 'forward_2500_full_data.pkl'
print(f"Loading data from {data_path}...")
with open(data_path, 'rb') as f:
    all_results = pickle.load(f)

if isinstance(all_results, dict):
    available_states = list(all_results.keys())
    print(f"Available states: {available_states}")
else:
    available_states = []

states_to_plot = available_states[:4]

def get_best_prediction(state_data):
    if isinstance(state_data, list):
        for item in state_data:
            if item.get('group') == '+FF+E+P':
                return item
        return state_data[-1]
    return state_data

def plot_combined_2x4():
    if not states_to_plot:
        return

    print("Generating combined 2x4 figure...")
    fig = plt.figure(figsize=(24, 12))

    colors = {'x': '#1f77b4', 'y': '#d62728', 'z': '#2ca02c'}

    for i, state in enumerate(states_to_plot):
        data = get_best_prediction(all_results[state])
        X_base, X_pred = data.get('X_base'), data.get('X_pred')
        t_eval = data.get('t_eval')
        case_label = state[0]

        ax1 = fig.add_subplot(2, 4, i + 1, projection='3d')

        if X_base is not None:
            ax1.plot(X_base[:, 0], X_base[:, 1], X_base[:, 2],
                     color='#FFA500', lw=2.0, alpha=0.9, label='Ground Truth')
            ax1.plot(X_pred[:, 0], X_pred[:, 1], X_pred[:, 2],
                     color='#1f77b4', lw=1.2, linestyle='--', label='F-PINN')

        ax1.set_title(f'Case {case_label}', fontsize=18, fontweight='bold', pad=10)
        ax1.set_xlabel('X (m)', fontsize=12, labelpad=5)
        ax1.set_ylabel('Y (m)', fontsize=12, labelpad=5)
        ax1.set_zlabel('Z (m)', fontsize=12, labelpad=3)
        ax1.set_box_aspect((1, 1, 0.8))
        ax1.view_init(elev=20, azim=-60)
        ax1.tick_params(axis='both', labelsize=9)
        ax1.legend(fontsize=12, loc='upper left')

        ax2 = fig.add_subplot(2, 4, 4 + i + 1)

        if t_eval is not None and X_base is not None:
            err_x, err_y, err_z = [np.abs(X_pred[:, j] - X_base[:, j]) for j in range(3)]
            ax2.plot(t_eval, err_x, color=colors['x'], lw=1.5, linestyle='--', alpha=0.85, label=r'$|\Delta x|$')
            ax2.plot(t_eval, err_y, color=colors['y'], lw=1.5, linestyle='--', alpha=0.85, label=r'$|\Delta y|$')
            ax2.plot(t_eval, err_z, color=colors['z'], lw=1.5, linestyle='--', alpha=0.85, label=r'$|\Delta z|$')

        ax2.set_yscale('log')
        ax2.set_title(f'Case {case_label}', fontsize=18, fontweight='bold')
        ax2.set_xlabel('Time (s)', fontsize=16)
        ax2.set_ylabel('Absolute Error (m)', fontsize=16)
        ax2.grid(True, which='both', ls='--', alpha=0.3)
        ax2.legend(fontsize=14, loc='lower right')
        ax2.tick_params(axis='both', labelsize=14)

    fig.subplots_adjust(
        left=0.04, right=0.96,
        top=0.88, bottom=0.06,
        hspace=0.15, wspace=0.25
    )

    fig.suptitle('2500-step Prediction: 3D Trajectories and Positional Errors',
                 fontsize=24, fontweight='bold', y=0.97)

    plt.savefig(os.path.join(save_dir, 'Figure_3.png'), format='png', dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.savefig(os.path.join(save_dir, 'Figure_3.eps'), format='eps', dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close()
    print(f"Saved: {os.path.join(save_dir, 'Figure_3.png')}")
    print(f"Saved: {os.path.join(save_dir, 'Figure_3.eps')}")

if __name__ == "__main__":
    plot_combined_2x4()
    print("Done!")
