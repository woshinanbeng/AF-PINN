"""
Frequency method comparison: AF-PINN vs MFF-PINN, SV-SNN, RFF, PhysicsPrior-Fixed.
Compares 5 frequency encoding methods across 4 orbit types.
"""

import torch
import numpy as np
import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.pinn_main import (
    device, Logger,
    state_A, state_B, state_C, state_D,
    MODEL_REGISTRY, train_pinn, plot_6_panels, plot_error_analysis,
)

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    os.makedirs('output/method_comparison', exist_ok=True)
    log_file = 'output/method_comparison/running_log.txt'
    sys.stdout = Logger(log_file)
    sys.stderr = sys.stdout
    print(f"Device: {device}")

    states_to_run = [
        (state_A, "A_Normal_Passing"),
        (state_B, "B_OffAxis_Passing"),
        (state_C, "C_Banana_Orbit"),
        (state_D, "D_HFS_StrongGrad"),
    ]

    method_names = ['AF-PINN', 'MFF-PINN', 'SV-SNN', 'RFF', 'PhysicsPrior-Fixed']

    summary_path = 'output/method_comparison/summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as s:
        s.write(f"{'Method':<22} {'Case':<25} {'Final Loss':>14} {'MSE':>14}\n")
        s.write("-" * 80 + "\n")

    for method_name in method_names:
        model_class = MODEL_REGISTRY[method_name]

        for config, name in states_to_run:
            run_tag = f"{method_name}_{name}"
            print(f"\n{'='*60}\n  {run_tag}\n{'='*60}")

            try:
                X_pred, X_base, hist, t_eval = train_pinn(
                    config, run_tag,
                    model_factory=lambda s=config, mc=model_class: mc(s),
                    max_epochs=100000,
                    output_dir='output/method_comparison',
                )

                result = {'X_pred': X_pred, 'X_base': X_base, 'history': hist, 't_eval': t_eval}
                pkl_path = f'output/method_comparison/{run_tag}_data.pkl'
                with open(pkl_path, 'wb') as f:
                    pickle.dump(result, f)

                plot_6_panels(X_pred, X_base, hist, run_tag, 'output/method_comparison')
                plot_error_analysis(X_pred, X_base, t_eval, run_tag, 'output/method_comparison')

                mse = hist['Error'][-1]
                with open(summary_path, 'a', encoding='utf-8') as s:
                    s.write(f"{method_name:<22} {name:<25} {hist['L'][-1]:>14.4e} {mse:>14.4e}\n")
                print(f">>> [{run_tag}] Done. MSE = {mse:.4e}")

            except Exception as e:
                print(f"!!! [{run_tag}] Failed: {e}")
                with open(summary_path, 'a', encoding='utf-8') as s:
                    s.write(f"{method_name:<22} {name:<25} {'FAILED':>14} --\n")

    print(f"\nDone. Results in output/method_comparison/")
