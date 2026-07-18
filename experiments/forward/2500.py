"""
Forward problem: AF-PINN trajectory prediction (2500 time-steps).
Multi-seed training with Adam + L-BFGS.
"""

import torch
import numpy as np
import sys
import os
import pickle

# Add parent directory so we can import pinn_main
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pinn_main import (
    device, Logger,
    state_A, state_B, state_C, state_D,
    AF_PINN, train_pinn, plot_6_panels, plot_error_analysis,
    classical_trajectory, t_max,
)

# ==========================================
# Main: Multi-seed training
# ==========================================
if __name__ == "__main__":
    seeds = [123456] # Main text seed
    states_to_run = [
        (state_A, "A_Normal_Passing"),
        (state_B, "B_OffAxis_Passing"),
        (state_C, "C_Banana_Orbit"),
        (state_D, "D_HFS_StrongGrad"),
    ]

    output_dir = 'output/forward_2500'
    os.makedirs(output_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(output_dir, 'running_log.txt')
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout

    print(f"Device: {device}")
    print(f"Seeds: {seeds}")

    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as s:
        s.write(f"{'Seed':>8} {'State':<25} {'Final Loss':>14} {'MSE':>14}\n")
        s.write("-" * 65 + "\n")

    for seed in seeds:
        print(f"\n{'='*60}\n  SEED = {seed}\n{'='*60}")
        torch.manual_seed(seed)
        np.random.seed(seed)

        for config, name in states_to_run:
            run_tag = f"seed{seed}_{name}"
            print(f"\n>>> [{run_tag}] Starting...")

            try:
                X_pred, X_base, hist, t_eval = train_pinn(
                    config, run_tag,
                    model_factory=lambda s=config: AF_PINN(s),
                    max_epochs=100000,
                    output_dir=output_dir,
                )

                # Save data
                result = {'X_pred': X_pred, 'X_base': X_base, 'history': hist, 't_eval': t_eval}
                pkl_path = os.path.join(output_dir, f'{run_tag}_data.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(result, f)

                # Plot
                plot_6_panels(X_pred, X_base, hist, run_tag, output_dir)
                plot_error_analysis(X_pred, X_base, t_eval, run_tag, output_dir)

                mse = hist['Error'][-1]
                with open(summary_path, 'a', encoding='utf-8') as s:
                    s.write(f"{seed:>8} {name:<25} {hist['L'][-1]:>14.4e} {mse:>14.4e}\n")
                print(f">>> [{run_tag}] Done. MSE = {mse:.4e}")

            except Exception as e:
                print(f"!!! [{run_tag}] Failed: {e}")
                with open(summary_path, 'a', encoding='utf-8') as s:
                    s.write(f"{seed:>8} {name:<25} {'FAILED':>14} --\n")

    print(f"\nAll done. Results in {output_dir}/")
