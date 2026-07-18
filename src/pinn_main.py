"""
AF-PINN: Adaptive-Frequency Physics-Informed Neural Networks
for Multi-Scale Charged-Particle Dynamics in Tokamak Fields.

Shared library: physics, models, training, and utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
import time
import os
import copy
import sys
import pickle
import json

# Global device and dtype
torch.set_default_dtype(torch.float64)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Physical parameters
B0 = 5.0
q_safety = 2.5
R0 = 6.2
q_over_m = 4.8e7
norm_factor = 1e6
Q_M = q_over_m / norm_factor
interval = 4.147267104135095e-10 * norm_factor
n_steps = 2500
t_max = n_steps * interval

# Initial conditions (A-D: Normal Passing, Off-Axis Passing, Banana Orbit, HFS StrongGrad)
state_A = {'X0': [7.2, 0.0, 0.0], 'V0': [1.0, 5.0, 0.0]}       # Normal Passing
state_B = {'X0': [0.0, 6.0, 2.0], 'V0': [-4.0, 1.0, 1.0]}      # Off-Axis Passing
state_C = {'X0': [6.5, 0.0, 0.0], 'V0': [0.1, 0.5, 8.0]}       # Banana Orbit
state_D = {'X0': [5.5, 0.0, 0.0], 'V0': [3.0, 3.0, 3.0]}       # HFS StrongGrad

STATE_LABELS = {
    'A': 'Normal Passing',
    'A_Normal_Passing': 'Normal Passing',
    'B': 'Off-Axis Passing',
    'B_OffAxis_Passing': 'Off-Axis Passing',
    'C': 'Banana Orbit',
    'C_Banana_Orbit': 'Banana Orbit',
    'D': 'HFS StrongGrad',
    'D_HFS_StrongGrad': 'HFS StrongGrad',
}

# Tokamak magnetic field (ITER-like analytical model)
def B_field(x, y, z):
    """Torch version of the tokamak magnetic field."""
    r2 = x**2 + y**2 + 1e-12
    r = torch.sqrt(r2)
    Bx = B0 * (-q_safety * R0 * y + z * x) / (q_safety * r2)
    By = B0 * (q_safety * R0 * x + z * y) / (q_safety * r2)
    Bz = (B0 / q_safety) * (-1 + R0 / r)
    return Bx, By, Bz

def B_field_np(x, y, z):
    """NumPy version of the tokamak magnetic field."""
    r2 = x**2 + y**2 + 1e-12
    r = np.sqrt(r2)
    return np.array([
        B0 * (-q_safety * R0 * y + z * x) / (q_safety * r2),
        B0 * (q_safety * R0 * x + z * y) / (q_safety * r2),
        (B0 / q_safety) * (-1 + R0 / r),
    ])

# Boris integrator for reference trajectories
def classical_trajectory(state, t_span, num_points):
    """Boris pusher for generating reference trajectories."""
    dt = (t_span[1] - t_span[0]) / (num_points - 1)
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    positions = np.zeros((num_points, 3))
    x = np.array(state['X0'], dtype=np.float64)
    v = np.array(state['V0'], dtype=np.float64)
    positions[0] = x
    for i in range(1, num_points):
        B = B_field_np(x[0], x[1], x[2])
        t_vec = Q_M * B * dt / 2.0
        t_mag2 = np.dot(t_vec, t_vec)
        s_vec = 2.0 * t_vec / (1.0 + t_mag2)
        v_prime = v + np.cross(v, t_vec)
        v = v + np.cross(v_prime, s_vec)
        x = x + v * dt
        positions[i] = x
    return t_eval, positions


# Dual-output Logger
class Logger(object):
    """Tee stdout to both console and file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# Physics-based frequency initialization
def compute_physics_frequencies(state):
    """Compute gyro-frequency (omega_c) and drift-scale frequency (omega_m)
    from initial position and velocity."""
    x0, y0, z0 = state['X0']
    v0_vec = np.array(state['V0'])
    B_vec = B_field_np(x0, y0, z0)
    B_mag = np.linalg.norm(B_vec)
    b_hat = B_vec / B_mag
    v_par = np.dot(v0_vec, b_hat)
    v_perp = np.sqrt(max(0.0, np.linalg.norm(v0_vec)**2 - v_par**2))

    omega_c_init = Q_M * B_mag
    omega_drift = (v_par**2 + 0.5 * v_perp**2) / (omega_c_init * R0**2)
    omega_transit = abs(v_par) / (q_safety * R0)
    omega_m_init = omega_transit if abs(v_par) > v_perp else omega_drift

    return omega_c_init, omega_m_init


# Neural network models

def _make_mlp_3x128(in_dim):
    """3x128 MLP backbone."""
    net = nn.Sequential(
        nn.Linear(in_dim, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 3),
    )
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class AF_PINN(nn.Module):
    """AF-PINN: 2 physics-initialized trainable Fourier frequencies.
    Equivalent to AblationPINN(use_fourier=True, freeze_omega=False)."""
    def __init__(self, state):
        super().__init__()
        omega_c_init, omega_m_init = compute_physics_frequencies(state)
        self.omega_c = nn.Parameter(torch.tensor([omega_c_init], dtype=torch.float64))
        self.omega_m = nn.Parameter(torch.tensor([omega_m_init], dtype=torch.float64))
        self.net = _make_mlp_3x128(5)

    def forward(self, t):
        t_feat = torch.cat([t,
            torch.sin(self.omega_m * t), torch.cos(self.omega_m * t),
            torch.sin(self.omega_c * t), torch.cos(self.omega_c * t)], dim=1)
        return self.net(t_feat)


class AblationPINN(nn.Module):
    """PINN with configurable Fourier features and optional frequency freezing.
    Used for ablation studies.

    use_fourier=True,  freeze_omega=False → AF-PINN (trainable)
    use_fourier=True,  freeze_omega=True  → Physics-prior fixed
    use_fourier=False                     → Vanilla (no Fourier features)
    """
    def __init__(self, state, use_fourier=True, freeze_omega=False):
        super().__init__()
        self.use_fourier = use_fourier
        omega_c_init, omega_m_init = compute_physics_frequencies(state)

        if self.use_fourier and not freeze_omega:
            self.omega_c = nn.Parameter(torch.tensor([omega_c_init], dtype=torch.float64))
            self.omega_m = nn.Parameter(torch.tensor([omega_m_init], dtype=torch.float64))
        elif self.use_fourier and freeze_omega:
            self.register_buffer('omega_c', torch.tensor([omega_c_init], dtype=torch.float64))
            self.register_buffer('omega_m', torch.tensor([omega_m_init], dtype=torch.float64))
        else:
            self.omega_c = omega_c_init
            self.omega_m = omega_m_init

        input_dim = 5 if use_fourier else 1
        self.net = _make_mlp_3x128(input_dim)

    def forward(self, t):
        if self.use_fourier:
            features = torch.cat([t,
                torch.sin(self.omega_m * t), torch.cos(self.omega_m * t),
                torch.sin(self.omega_c * t), torch.cos(self.omega_c * t)], dim=1)
        else:
            features = t
        return self.net(features)


class Fourier_PINN(nn.Module):
    """Fourier-PINN with selectable input modes (for w-ablation study).
    input_mode: 'all' (default), 't' (vanilla), 't_c' (ωc only), 't_m' (ωm only).
    """
    def __init__(self, state, input_mode='all'):
        super().__init__()
        self.input_mode = input_mode
        omega_c_init, omega_m_init = compute_physics_frequencies(state)
        self.omega_c = nn.Parameter(torch.tensor([omega_c_init], dtype=torch.float64))
        self.omega_m = nn.Parameter(torch.tensor([omega_m_init], dtype=torch.float64))

        dim_map = {'t': 1, 't_c': 3, 't_m': 3, 'all': 5}
        if input_mode not in dim_map:
            raise ValueError(f"Unsupported input_mode: {input_mode}")
        self.net = _make_mlp_3x128(dim_map[input_mode])

    def forward(self, t):
        if self.input_mode == 't':
            t_feat = t
        elif self.input_mode == 't_c':
            t_feat = torch.cat([t, torch.sin(self.omega_c * t), torch.cos(self.omega_c * t)], dim=1)
        elif self.input_mode == 't_m':
            t_feat = torch.cat([t, torch.sin(self.omega_m * t), torch.cos(self.omega_m * t)], dim=1)
        elif self.input_mode == 'all':
            t_feat = torch.cat([t,
                torch.sin(self.omega_m * t), torch.cos(self.omega_m * t),
                torch.sin(self.omega_c * t), torch.cos(self.omega_c * t)], dim=1)
        return self.net(t_feat)


class MFF_PINN(nn.Module):
    """MFF-PINN: 320 fixed random frequencies in 3 groups."""
    def __init__(self, state):
        super().__init__()
        np.random.seed(123456)
        freqs1 = np.random.uniform(0, 10, int(320 * 0.25))
        freqs2 = np.random.normal(250, 50, int(320 * 0.50))
        freqs3 = np.random.uniform(250, 500, int(320 * 0.25))
        all_freqs = np.concatenate([freqs1, freqs2, freqs3])
        np.random.shuffle(all_freqs)
        self.freqs = nn.Parameter(torch.tensor(all_freqs[:320], dtype=torch.float64), requires_grad=False)
        self.net = _make_mlp_3x128(1 + 2 * 320)

    def forward(self, t):
        t_freq = t @ self.freqs.unsqueeze(0)
        sin_cos = torch.stack((torch.sin(t_freq), torch.cos(t_freq)), dim=-1)
        interleaved = sin_cos.flatten(start_dim=1)
        return self.net(torch.cat([t, interleaved], dim=1))


class SV_SNN(nn.Module):
    """SV-SNN: 320 trainable frequencies (10 modes x 32)."""
    def __init__(self, state):
        super().__init__()
        np.random.seed(123456)
        freqs1 = np.random.uniform(0, 10, int(320 * 0.25))
        freqs2 = np.random.normal(250, 50, int(320 * 0.50))
        freqs3 = np.random.uniform(250, 500, int(320 * 0.25))
        all_freqs = np.concatenate([freqs1, freqs2, freqs3])
        np.random.shuffle(all_freqs)
        self.freqs = nn.Parameter(torch.tensor(all_freqs[:320], dtype=torch.float64).reshape(10, 32))
        self.net = _make_mlp_3x128(1 + 2 * 320)

    def forward(self, t):
        freq_flat = self.freqs.reshape(-1)
        t_freq = t @ freq_flat.unsqueeze(0)
        sin_cos = torch.stack((torch.sin(t_freq), torch.cos(t_freq)), dim=-1)
        interleaved = sin_cos.flatten(start_dim=1)
        return self.net(torch.cat([t, interleaved], dim=1))


class RFF_PINN(nn.Module):
    """RFF-PINN: 96 fixed random Fourier features with 3 sigma values."""
    def __init__(self, state):
        super().__init__()
        np.random.seed(123456)
        sigmas = [1.0, 10.0, 100.0]
        all_freqs = []
        for s in sigmas:
            all_freqs.append(np.random.normal(0, s, 32))
        freqs = np.concatenate(all_freqs)
        self.freqs = nn.Parameter(torch.tensor(freqs, dtype=torch.float64), requires_grad=False)
        self.net = _make_mlp_3x128(1 + 2 * 96)

    def forward(self, t):
        t_freq = t @ self.freqs.unsqueeze(0)
        sin_cos = torch.stack((torch.sin(t_freq), torch.cos(t_freq)), dim=-1)
        interleaved = sin_cos.flatten(start_dim=1)
        return self.net(torch.cat([t, interleaved], dim=1))


class PhysicsPriorFixed_PINN(nn.Module):
    """Physics-prior PINN: omega_c, omega_m fixed at initial physics estimates."""
    def __init__(self, state):
        super().__init__()
        omega_c_init, omega_m_init = compute_physics_frequencies(state)
        self.omega_c = nn.Parameter(torch.tensor([omega_c_init], dtype=torch.float64), requires_grad=False)
        self.omega_m = nn.Parameter(torch.tensor([omega_m_init], dtype=torch.float64), requires_grad=False)
        self.net = _make_mlp_3x128(5)

    def forward(self, t):
        t_feat = torch.cat([t,
            torch.sin(self.omega_m * t), torch.cos(self.omega_m * t),
            torch.sin(self.omega_c * t), torch.cos(self.omega_c * t)], dim=1)
        return self.net(t_feat)


# Model registry for method comparison experiments
MODEL_REGISTRY = {
    'AF-PINN':            AF_PINN,
    'MFF-PINN':           MFF_PINN,
    'SV-SNN':             SV_SNN,
    'RFF':                RFF_PINN,
    'PhysicsPrior-Fixed': PhysicsPriorFixed_PINN,
}


# Training

def compute_losses(model, t_colloc, X0_t, V0_t, X_base_t, use_energy=True, use_pphi=True):
    """Compute all physics-informed loss components.

    Returns:
        loss_dict: {L_r, L_ic, L_E, L_P} as tensors
        X_pred: predicted positions
        auxiliary values (v0_sq, Pphi0) for reference
    """
    X = model(t_colloc)
    x, y, z = X[:, 0:1], X[:, 1:2], X[:, 2:3]

    # Velocities
    vx = torch.autograd.grad(x, t_colloc, torch.ones_like(x), create_graph=True)[0]
    vy = torch.autograd.grad(y, t_colloc, torch.ones_like(y), create_graph=True)[0]
    vz = torch.autograd.grad(z, t_colloc, torch.ones_like(z), create_graph=True)[0]

    # Accelerations
    ax = torch.autograd.grad(vx, t_colloc, torch.ones_like(vx), create_graph=True)[0]
    ay = torch.autograd.grad(vy, t_colloc, torch.ones_like(vy), create_graph=True)[0]
    az = torch.autograd.grad(vz, t_colloc, torch.ones_like(vz), create_graph=True)[0]

    # Lorentz residual
    Bx, By, Bz = B_field(x, y, z)
    fx = ax - Q_M * (vy * Bz - vz * By)
    fy = ay - Q_M * (vz * Bx - vx * Bz)
    fz = az - Q_M * (vx * By - vy * Bx)
    L_r = torch.mean(fx**2 + fy**2 + fz**2)

    # Initial condition
    t0 = torch.zeros(1, 1, dtype=torch.float64, device=device, requires_grad=True)
    X0p = model(t0).squeeze()
    v0x = torch.autograd.grad(model(t0)[:, 0], t0, create_graph=True)[0].squeeze()
    v0y = torch.autograd.grad(model(t0)[:, 1], t0, create_graph=True)[0].squeeze()
    v0z = torch.autograd.grad(model(t0)[:, 2], t0, create_graph=True)[0].squeeze()
    Vp = torch.stack([v0x, v0y, v0z])
    L_ic = torch.sum((X0p - X0_t)**2) + torch.sum((Vp - V0_t)**2)

    # Energy conservation
    if use_energy:
        v0_sq = torch.sum(V0_t**2)
        L_E = torch.mean((vx**2 + vy**2 + vz**2 - v0_sq)**2)
    else:
        L_E = torch.tensor(0.0, device=device)

    # Canonical toroidal angular momentum conservation
    if use_pphi:
        r0 = torch.sqrt(X0_t[0]**2 + X0_t[1]**2)
        vphi0 = X0_t[0] * V0_t[1] - X0_t[1] * V0_t[0]
        psi0 = (B0 / q_safety) * (R0 * r0 - 0.5 * r0**2 - 0.5 * X0_t[2]**2)
        Pphi0 = vphi0 + Q_M * psi0

        rc = torch.sqrt(x**2 + y**2 + 1e-12)
        vphi = x * vy - y * vx
        psi = (B0 / q_safety) * (R0 * rc - 0.5 * rc**2 - 0.5 * z**2)
        L_P = torch.mean((vphi + Q_M * psi - Pphi0)**2)
    else:
        L_P = torch.tensor(0.0, device=device)

    return {'L_r': L_r, 'L_ic': L_ic, 'L_E': L_E, 'L_P': L_P}, X


def adaptive_weights(optimizer, model, losses_dict, lam_dict, alpha_aw=0.2):
    """Update adaptive loss weights based on gradient statistics."""
    active = [(k, v) for k, v in losses_dict.items() if v.item() != 0]
    grads = {}
    for nm, lv in active:
        optimizer.zero_grad()
        lv.backward(retain_graph=True)
        grads[nm] = torch.sqrt(sum(
            torch.sum(p.grad**2) for p in model.parameters() if p.grad is not None
        ) + 1e-20)
    optimizer.zero_grad()
    tg = sum(grads.values()) + 1e-12
    for nm in grads:
        lam_dict[nm] = alpha_aw * lam_dict[nm] + (1 - alpha_aw) * (tg / (grads[nm] + 1e-12)).item()


def train_pinn(state, state_name, model_factory, *,
               use_energy=True, use_pphi=True,
               max_epochs=100000, n_colloc=10000,
               track_omega=False, save_weights=False,
               output_dir='results', lr=1e-3):
    """Unified training loop: Adam + L-BFGS.

    Args:
        state: dict with 'X0', 'V0'
        state_name: label for logging
        model_factory: callable that returns an nn.Module
        use_energy, use_pphi: toggle conservation losses
        max_epochs: Adam epochs
        n_colloc: number of collocation points
        track_omega: record omega_c/omega_m during training
        save_weights: save model state_dict after training
        output_dir: directory for logs and saved weights
        lr: Adam learning rate
    Returns:
        X_pred, X_base, loss_history, t_eval
    """
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(123456)
    np.random.seed(123456)

    model = model_factory().to(device).double()
    optimizer_adam = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_adam, T_max=max_epochs, eta_min=1e-7)

    t_colloc = torch.linspace(0, t_max, n_colloc, device=device, dtype=torch.float64).view(-1, 1).requires_grad_(True)
    X0_t = torch.tensor(state['X0'], dtype=torch.float64, device=device)
    V0_t = torch.tensor(state['V0'], dtype=torch.float64, device=device)

    _, X_base = classical_trajectory(state, [0, t_max], n_colloc)
    X_base_t = torch.tensor(X_base, dtype=torch.float64, device=device)

    lam = {'L_r': 1.0, 'L_ic': 1.0, 'L_E': 1.0, 'L_P': 1.0}

    loss_history = {
        'Epochs': [], 'L': [], 'L_ic': [], 'L_r': [],
        'L_E': [], 'L_P': [], 'Error': [],
    }
    if track_omega:
        loss_history['omega_c'] = []
        loss_history['omega_m'] = []

    best_loss = float('inf')
    best_state = None

    print(f"  [Phase 1/2] Adam ({max_epochs} epochs, lr={lr}) for {state_name} ...")
    t0_start = time.time()

    # ---- Adam phase ----
    for epoch in range(1, max_epochs + 1):
        losses, X_pred = compute_losses(model, t_colloc, X0_t, V0_t, X_base_t,
                                        use_energy=use_energy, use_pphi=use_pphi)

        # Adaptive weighting every 1000 epochs
        if epoch % 1000 == 1:
            adaptive_weights(optimizer_adam, model, losses, lam)

        total_loss = lam['L_r'] * losses['L_r'] + lam['L_ic'] * losses['L_ic']
        if use_energy: total_loss = total_loss + lam['L_E'] * losses['L_E']
        if use_pphi:   total_loss = total_loss + lam['L_P'] * losses['L_P']

        optimizer_adam.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer_adam.step()
        scheduler.step()

        if epoch % 100 == 0:
            with torch.no_grad():
                err = torch.mean(torch.sum((X_pred - X_base_t)**2, dim=1)).item()
                loss_history['Epochs'].append(epoch)
                loss_history['L'].append(total_loss.item())
                loss_history['L_r'].append(losses['L_r'].item())
                loss_history['L_ic'].append(losses['L_ic'].item())
                loss_history['L_E'].append(losses['L_E'].item())
                loss_history['L_P'].append(losses['L_P'].item())
                loss_history['Error'].append(err)
                if track_omega:
                    loss_history['omega_c'].append(model.omega_c.item())
                    loss_history['omega_m'].append(model.omega_m.item())
                if total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_state = copy.deepcopy(model.state_dict())

        if epoch % 10000 == 0:
            print(f"     Ep {epoch:6d} | Loss={total_loss.item():.2e} | Err={err:.2e}")

    adam_elapsed = time.time() - t0_start
    print(f"  [Phase 1/2] Adam done ({adam_elapsed/60:.1f} min). Loading best weights...")
    if best_state:
        model.load_state_dict(best_state)

    # ---- L-BFGS phase ----
    final_loss = best_loss

    def closure():
        nonlocal final_loss
        optimizer_lbfgs.zero_grad()
        losses_c, _ = compute_losses(model, t_colloc, X0_t, V0_t, X_base_t,
                                     use_energy=use_energy, use_pphi=use_pphi)
        loss_c = lam['L_r'] * losses_c['L_r'] + lam['L_ic'] * losses_c['L_ic']
        if use_energy: loss_c = loss_c + lam['L_E'] * losses_c['L_E']
        if use_pphi:   loss_c = loss_c + lam['L_P'] * losses_c['L_P']
        loss_c.backward()
        final_loss = loss_c.item()
        return loss_c

    print(f"  [Phase 2/2] L-BFGS (max 5000 iter)...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), max_iter=5000,
                                  tolerance_grad=1e-12, tolerance_change=1e-12,
                                  history_size=100)
    try:
        optimizer_lbfgs.step(closure)
    except Exception as e:
        print(f"  Warning: L-BFGS failed ({e}), using Adam best.")
        model.load_state_dict(best_state)

    with torch.no_grad():
        X_final = model(t_colloc)
        final_err = torch.mean(torch.sum((X_final - X_base_t)**2, dim=1)).item()

    elapsed = time.time() - t0_start
    print(f"  Done: Loss={final_loss:.2e} | Err={final_err:.2e} | Time={elapsed/60:.1f} min")

    # Record final point
    loss_history['Epochs'].append(max_epochs + 1)
    loss_history['L'].append(final_loss)
    loss_history['Error'].append(final_err)

    # Save weights
    if save_weights:
        weight_path = os.path.join(output_dir, f"AF_PINN_State_{state_name}.pth")
        torch.save(model.state_dict(), weight_path)
        print(f"  Weights saved: {weight_path}")

    X_pred_np = X_final.cpu().numpy()
    t_np, _ = classical_trajectory(state, [0, t_max], n_colloc)

    return X_pred_np, X_base, loss_history, t_np


# Shared plotting utilities

def plot_6_panels(X_pred, X_base, history, state_name, output_dir='results'):
    """Standard 6-panel figure: XY trajectory, 3D baseline, 3D prediction,
    FFT spectrum, error evolution, loss evolution."""
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f'AF-PINN Forward Problem: State {state_name}',
                 fontsize=18, fontweight='bold', y=0.96)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X_pred[:, 0], X_pred[:, 1], '--', label='AF-PINN', color='tab:blue', linewidth=2, zorder=3)
    ax1.plot(X_base[:, 0], X_base[:, 1], '-', label='VPA Truth', color='tab:orange', alpha=0.7, zorder=2)
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.legend(); ax1.set_title('(1) X-Y Plane Trajectory')

    ax2 = fig.add_subplot(gs[0, 1], projection='3d')
    ax2.plot(X_base[:, 0], X_base[:, 1], X_base[:, 2], color='tab:orange', label='VPA Baseline', linewidth=2)
    ax2.set_title('(2) 3D VPA Baseline')

    ax3 = fig.add_subplot(gs[0, 2], projection='3d')
    ax3.plot(X_pred[:, 0], X_pred[:, 1], X_pred[:, 2], color='tab:blue', label='AF-PINN')
    ax3.set_title('(3) 3D AF-PINN Prediction')

    ax4 = fig.add_subplot(gs[1, 0])
    N, dt = len(X_base), t_max / (len(X_base) - 1)
    yf_vpa = np.fft.fft(X_base[:, 0] - np.mean(X_base[:, 0]))
    yf_pinn = np.fft.fft(X_pred[:, 0] - np.mean(X_pred[:, 0]))
    xf = np.fft.fftfreq(N, dt)[:N // 2]
    ax4.plot(xf, 2.0 / N * np.abs(yf_vpa[0:N // 2]), label='VPA Spectrum', color='tab:orange', linewidth=3, alpha=0.6)
    ax4.plot(xf, 2.0 / N * np.abs(yf_pinn[0:N // 2]), '--', label='AF-PINN Spectrum', color='darkblue', linewidth=1.5)
    ax4.set_xlim(0, 60); ax4.set_xlabel('Frequency (Hz)'); ax4.set_ylabel('Amplitude')
    ax4.set_title('(4) FFT Spectral Analysis'); ax4.legend()

    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history['Epochs'], history['Error'], color='tab:purple', linewidth=2)
    ax5.set_yscale('log'); ax5.set_xlabel('Epoch'); ax5.set_ylabel('Euclidean Error')
    ax5.set_title('(5) Trajectory Error Evolution')

    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(history['Epochs'], history['L'], '--', label='Total Loss', color='red', alpha=0.9, linewidth=1.5, zorder=10)
    ax6.plot(history['Epochs'], history['L_r'], label='Lorentz Res', color='tab:blue', alpha=0.8, linewidth=1)
    ax6.plot(history['Epochs'], history['L_ic'], label='Boundary', color='tab:orange', alpha=0.8, linewidth=1)
    ax6.plot(history['Epochs'], history['L_E'], label='Energy', color='tab:green', alpha=0.8, linewidth=1)
    ax6.plot(history['Epochs'], history['L_P'], label='P_phi', color='tab:purple', alpha=0.8, linewidth=1)
    ax6.set_yscale('log'); ax6.set_xlabel('Epoch'); ax6.set_ylabel('Loss')
    ax6.set_title('(6) Physical Loss Evolution'); ax6.legend()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'State_{state_name}_6Panels.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {save_path}")


def plot_error_analysis(X_pred, X_base, t_array, state_name, output_dir='results'):
    """Error evolution + poloidal-plane error distribution."""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['axes.linewidth'] = 1.0

    err_x = np.abs(X_pred[:, 0] - X_base[:, 0])
    err_y = np.abs(X_pred[:, 1] - X_base[:, 1])
    err_z = np.abs(X_pred[:, 2] - X_base[:, 2])
    total_err = np.sqrt(err_x**2 + err_y**2 + err_z**2)

    R_offset = np.sqrt(X_base[:, 0]**2 + X_base[:, 1]**2) - R0
    Z_coord = X_base[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax1 = axes[0]
    ax1.plot(t_array, err_x, label=r'$\Delta x$', color='#1f77b4', linewidth=1.5)
    ax1.plot(t_array, err_y, label=r'$\Delta y$', color='#d62728', linestyle='--', linewidth=1.5)
    ax1.plot(t_array, err_z, label=r'$\Delta z$', color='#2ca02c', linestyle='-.', linewidth=1.5)
    ax1.set_yscale('log'); ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('Time (s)', fontsize=14); ax1.set_ylabel('Absolute Error', fontsize=14)
    ax1.set_title(f'Error Evolution (State {state_name})', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='best')

    ax2 = axes[1]
    sc = ax2.scatter(R_offset, Z_coord, c=total_err, cmap='viridis', s=15, alpha=0.9, edgecolor='none')
    cbar = fig.colorbar(sc, ax=ax2); cbar.set_label('Position Error', fontsize=14)
    ax2.set_xlabel(r'$R - R_0$', fontsize=14); ax2.set_ylabel('Z', fontsize=14)
    ax2.set_title('Error Distribution in Poloidal Plane', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'Error_Analysis_{state_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Figure saved: {save_path}")
