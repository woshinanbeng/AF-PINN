"""
Ablation study: contribution of Fourier Features, Energy loss, and P_phi loss.
Compares 4 configurations across 5 orbit types.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import sys
import os
import json
import pickle
import copy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pinn_main import (
    device, Logger,
    state_A, state_B, state_C, state_D,
    AblationPINN,
    train_pinn,
    adaptive_weights, compute_losses,
    classical_trajectory, t_max, Q_M, B0, q_safety, R0,
)

# ==========================================
# Ablation group definitions
# ==========================================
ABLATION_GROUPS = [
    {
        "name": "Vanilla",
        "label": r"$L_r + L_{ic}$ (Vanilla PINN)",
        "fourier": False, "energy": False, "pphi": False,
        "color": "#7f7f7f", "marker": "o",
    },
    {
        "name": "+FF+E+P",
        "label": r"$+\mathrm{FF} + L_E + L_{P_\phi}$ (Ours)",
        "fourier": True, "energy": True, "pphi": True,
        "color": "#d62728", "marker": "D",
    },
    {
        "name": "E+P",
        "label": r"$L_E + L_{P_\phi}$ (no FF)",
        "fourier": False, "energy": True, "pphi": True,
        "color": "#9467bd", "marker": "v",
    },
    {
        "name": "+FF_no_Lp",
        "label": r"$+\mathrm{FF} + L_E$ (no $L_{P_\phi}$)",
        "fourier": True, "energy": True, "pphi": False,
        "color": "#ff7f0e", "marker": "s",
    },
]

FIXED_EPOCHS = 100000
N_COLLOC = 10000


# ==========================================
# Training (uses unified train_pinn with AblationPINN factory)
# ==========================================
def train_one_ablation(state, state_name, group_cfg, output_dir):
    gname = group_cfg["name"]

    def make_model(s=state, g=group_cfg):
        return AblationPINN(s, use_fourier=g["fourier"], freeze_omega=False)

    X_pred, X_base, history, t_eval = train_pinn(
        state, f"{state_name}_{gname}",
        model_factory=make_model,
        use_energy=group_cfg["energy"],
        use_pphi=group_cfg["pphi"],
        max_epochs=FIXED_EPOCHS,
        n_colloc=N_COLLOC,
        output_dir=output_dir,
        save_weights=(gname == "+FF+E+P"),
    )

    with torch.no_grad():
        final_err = history['Error'][-1] if history['Error'] else 0.0

    return {
        'X_pred': X_pred, 'X_base': X_base, 't_eval': t_eval,
        'history': history, 'final_error': final_err,
        'group': gname,
    }


# ==========================================
# Plotting
# ==========================================
def plot_ablation_summary(all_results, state_name, output_dir):
    """2x2 panel: error vs time, convergence, zoomed x(t), bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Ablation Study: State {state_name}', fontsize=16, fontweight='bold')

    # (a) Position error vs time
    ax = axes[0, 0]
    for i, g in enumerate(ABLATION_GROUPS):
        r = all_results[i]
        err = np.sqrt(np.sum((r['X_pred'] - r['X_base'])**2, axis=1))
        ax.plot(r['t_eval'], err, color=g['color'], linewidth=1.5, label=g['label'])
    ax.set_yscale('log')
    ax.set_xlabel('Time'); ax.set_ylabel('Position Error')
    ax.set_title('(a) Position Error vs Time'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Error convergence
    ax = axes[0, 1]
    for i, g in enumerate(ABLATION_GROUPS):
        r = all_results[i]
        ax.plot(r['history']['Epochs'], r['history']['Error'],
                color=g['color'], linewidth=1.5, label=g['label'])
    ax.set_yscale('log')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Trajectory Error')
    ax.set_title('(b) Training Error Evolution'); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) x(t) zoomed
    ax = axes[1, 0]
    t_eval = all_results[0]['t_eval']
    n = len(t_eval)
    i0, i1 = n * 2 // 5, n * 3 // 5
    ax.plot(t_eval[i0:i1], all_results[0]['X_base'][i0:i1, 0],
            'k-', linewidth=2.5, alpha=0.5, label='VPA Truth')
    for i, g in enumerate(ABLATION_GROUPS):
        ax.plot(t_eval[i0:i1], all_results[i]['X_pred'][i0:i1, 0],
                color=g['color'], linewidth=1.2, ls='--', label=g['label'])
    ax.set_xlabel('Time (zoomed)'); ax.set_ylabel('x(t)')
    ax.set_title('(c) x(t) Local Comparison'); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # (d) Bar chart
    ax = axes[1, 1]
    errors = [r['final_error'] for r in all_results]
    colors = [g['color'] for g in ABLATION_GROUPS]
    labels = [g['name'] for g in ABLATION_GROUPS]
    bars = ax.bar(range(len(ABLATION_GROUPS)), errors, color=colors, edgecolor='black')
    ax.set_yscale('log')
    ax.set_xticks(range(len(ABLATION_GROUPS)))
    ax.set_xticklabels(labels, fontsize=9, rotation=15)
    ax.set_ylabel('Final Error'); ax.set_title('(d) Final Error Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, err * 2.5,
                f'{err:.1e}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(output_dir, f'Ablation_{state_name}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {path}")


def plot_component_errors(all_results, state_name, output_dir):
    """1x3 component-wise error comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(f'Component Error: State {state_name}', fontsize=14, fontweight='bold')

    for ci, cn in enumerate(['x', 'y', 'z']):
        ax = axes[ci]
        for i, g in enumerate(ABLATION_GROUPS):
            err = np.abs(all_results[i]['X_pred'][:, ci] - all_results[i]['X_base'][:, ci])
            ax.plot(all_results[i]['t_eval'], err, color=g['color'], linewidth=1.2, label=g['label'])
        ax.set_yscale('log')
        ax.set_xlabel('Time'); ax.set_ylabel(f'|{cn}_pred - {cn}_true|')
        ax.set_title(f'{cn}-component Error'); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'Ablation_Components_{state_name}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {path}")


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    os.makedirs('output/ablation', exist_ok=True)
    log_file = 'output/ablation/running_log.txt'
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    print(f"Device: {device}")

    states_to_run = [
        (state_A, "A"), (state_B, "B"), (state_C, "C"),
        (state_D, "D"),
    ]

    all_results = {}
    for config, sname in states_to_run:
        print(f"\n{'='*60}\n  State {sname}: {len(ABLATION_GROUPS)} groups x {FIXED_EPOCHS} epochs\n{'='*60}")

        group_results = []
        for g in ABLATION_GROUPS:
            result = train_one_ablation(config, sname, g, 'output/ablation')
            group_results.append(result)

        plot_ablation_summary(group_results, sname, 'output/ablation')
        plot_component_errors(group_results, sname, 'output/ablation')
        all_results[sname] = group_results

    # Summary table
    print(f'\n{"="*90}')
    header = f'{"State":>6} | ' + ' | '.join([f'{g["name"]:>14}' for g in ABLATION_GROUPS])
    print(header)
    print('-' * 90)
    for sname in all_results:
        row = [f'{r["final_error"]:.2e}' for r in all_results[sname]]
        print(f'{sname:>6} | ' + ' | '.join([f'{v:>14}' for v in row]))
    print('=' * 90)

    # Save results
    summary = {}
    for sname in all_results:
        summary[sname] = {
            g['name']: {'error': r['final_error']}
            for g, r in zip(ABLATION_GROUPS, all_results[sname])
        }
    with open('output/ablation/ablation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    with open('output/ablation/ablation_full_data.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    print(f'\nDone. Results in output/ablation/')
