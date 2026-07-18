"""
w-ablation study: contribution of individual Fourier frequency inputs.
Compares 4 input modes: t only, t+omega_c, t+omega_m, t+omega_c+omega_m (all).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pinn_main import (
    device, Logger,
    state_A, state_B, state_C, state_D,
    Fourier_PINN, train_pinn,
)

ABLATION_MODES = ['t', 't_c', 't_m', 'all']
MODE_COLORS = {'t': '#d62728', 't_c': '#2ca02c', 't_m': '#ff7f0e', 'all': '#1f77b4'}


# ==========================================
# Plotting
# ==========================================
def plot_3d_trajectories(state_data, state_name, output_dir):
    """1x4 3D trajectory comparison across modes."""
    fig = plt.figure(figsize=(24, 6))
    fig.suptitle(f'3D Trajectory Comparison: State {state_name}', fontsize=18, fontweight='bold')

    for i, mode in enumerate(ABLATION_MODES):
        ax = fig.add_subplot(1, 4, i + 1, projection='3d')
        X_pred = state_data[mode]['X_pred']
        X_base = state_data[mode]['X_base']
        ax.plot(X_base[:, 0], X_base[:, 1], X_base[:, 2],
                color='tab:orange', label='VPA', alpha=0.7, linewidth=2)
        ax.plot(X_pred[:, 0], X_pred[:, 1], X_pred[:, 2],
                color='tab:blue', label=f'Mode: {mode}', linewidth=1.5)
        ax.set_title(f'Mode: {mode}'); ax.legend(loc='upper right')

    plt.tight_layout()
    path = os.path.join(output_dir, f'State_{state_name}_3D_Ablation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {path}")


def plot_loss_comparison(state_data, state_name, output_dir):
    """Loss evolution comparison across modes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for mode in ABLATION_MODES:
        hist = state_data[mode]['history']
        ax.plot(hist['Epochs'], hist['L'],
                label=f'Mode: {mode}', color=MODE_COLORS.get(mode, 'black'), linewidth=2)
    ax.set_yscale('log')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss')
    ax.set_title(f'Loss Evolution: State {state_name}', fontweight='bold')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'State_{state_name}_Loss_Ablation.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    os.makedirs('output/w_ablation', exist_ok=True)
    log_file = 'output/w_ablation/running_log.txt'
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    print(f"Device: {device}")

    states_to_run = [
        (state_A, "A_Normal_Passing"),
        (state_B, "B_OffAxis_Passing"),
        (state_C, "C_Banana_Orbit"),
        (state_D, "D_HFS_StrongGrad"),
    ]

    full_data = {}

    for config, base_name in states_to_run:
        print(f"\n{'='*60}\n  State: {base_name}\n{'='*60}")
        state_data = {}

        for mode in ABLATION_MODES:
            tag = f"{base_name}_mode_{mode}"

            def make_model(s=config, m=mode):
                return Fourier_PINN(s, input_mode=m)

            X_pred, X_base, hist, t_eval = train_pinn(
                config, tag,
                model_factory=make_model,
                max_epochs=100000,
                output_dir='output/w_ablation',
            )

            state_data[mode] = {'X_pred': X_pred, 'X_base': X_base,
                                'history': hist, 't_eval': t_eval}
            full_data[tag] = state_data[mode]

        plot_3d_trajectories(state_data, base_name, 'output/w_ablation')
        plot_loss_comparison(state_data, base_name, 'output/w_ablation')

    # Save all data
    with open('output/w_ablation/ablation_data.pkl', 'wb') as f:
        pickle.dump(full_data, f)
    print(f"\nDone. Results in output/w_ablation/")
