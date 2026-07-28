"""PyTorch implementation of the requested self-consistency BSDE experiment.

The model is
    dk = [exp(z) k**alpha - delta*k - c(k,z)] dt,
    dz = kappa_z*(z_bar-z) dt + sigma_z dW,
    u(c) = (c**(1-gamma)-1)/(1-gamma).

For an outer iteration n, c^n is FROZEN.  The critic outputs (V,Z) and is
trained with the requested one-step loss

  [ V + (rho V-u(c)) dt + Z dW - V(k_next,z_next) ]**2.

After this critic evaluation, the policy is improved outside the BSDE by
  c_star=(V_k)^(-1/gamma),
and damped.  A picture is saved after every outer iteration.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator
from torch import nn

from toy_consumption_saving import howard_iteration, make_grid


@dataclass(frozen=True)
class Params:
    alpha: float = .36
    delta: float = .08
    gamma: float = 2.0
    rho: float = .05
    kappa_z: float = .30
    z_bar: float = .0
    sigma_z: float = .023
    dt: float = .005
    k_min: float = .05
    k_max: float = 30.0
    z_min: float = -.12
    z_max: float = .12
    nk: int = 180
    nz: int = 61


p = Params()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT = Path(__file__).resolve().parent

def utility(c):
    c = torch.clamp(c, min=1e-10)
    return (c.pow(1.0 - p.gamma) - 1.0) / (1.0 - p.gamma)


class Critic(nn.Module):
    """The same network produces both V(k,z) and its BSDE exposure Z(k,z)."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(2, 96), nn.Tanh(),
            nn.Linear(96, 96), nn.Tanh(),
            nn.Linear(96, 2),
        )

    def forward(self, k, z):
        # Normalization is numerical only.
        x = torch.stack(((k - 15.0) / 15.0, z / .06), dim=-1)
        out = self.body(x)
        return out[..., 0], , out[..., 1]


class FrozenGridPolicy:
    """A consumption rule frozen during each inner BSDE evaluation."""

    def __init__(self):
        self.k = np.linspace(p.k_min, p.k_max, p.nk)
        self.z = np.linspace(p.z_min, p.z_max, p.nz)
        kk, zz = np.meshgrid(self.k, self.z, indexing="ij")
        self.c = .10 * np.exp(zz) * kk**p.alpha
        self.refresh()

    def refresh(self):
        self.interp = RegularGridInterpolator(
            (self.k, self.z), self.c, bounds_error=False, fill_value=None
        )

    def evaluate(self, k, z):
        # Consumption is exogenous to the critic during its inner BSDE solve.
        points = np.column_stack((k.detach().cpu().numpy(), z.detach().cpu().numpy()))
        ans = self.interp(points)
        return torch.as_tensor(np.maximum(ans, 1e-10), dtype=torch.float32, device=DEVICE)

    def improve(self, critic, damping=1.0):
        kk, zz = np.meshgrid(self.k, self.z, indexing="ij")
        k = torch.tensor(kk.ravel(), dtype=torch.float32, device=DEVICE, requires_grad=True)
        z = torch.tensor(zz.ravel(), dtype=torch.float32, device=DEVICE)
        v, _ = critic(k, z)
        vk = torch.autograd.grad(v.sum(), k)[0].detach().cpu().numpy().reshape(kk.shape)
        # Concavity should imply vk>0.  The floor and trust region prevent an
        # untrained critic from causing an explosive one-shot policy update.
        c_star = np.maximum(vk, 1e-7) ** (-1.0 / p.gamma)
        c_star = np.clip(c_star, .25 * self.c, 4.0 * self.c)
        net_income = np.exp(zz) * kk**p.alpha - p.delta * kk
        c_star[0] = np.minimum(c_star[0], np.maximum(net_income[0], 1e-8))
        self.c = (1.0 - damping) * self.c + damping * c_star
        self.refresh()
        return vk


def draw_states(batch):
    """Uniform wealth coverage and stationary-OU productivity coverage."""
    k = torch.empty(batch, device=DEVICE).uniform_(p.k_min, p.k_max)
    sd_z = p.sigma_z / np.sqrt(2.0 * p.kappa_z)
    z = torch.randn(batch, device=DEVICE) * sd_z + p.z_bar
    return k, torch.clamp(z, p.z_min, p.z_max)

def train_fixed_policy(critic, policy, steps=20_000, batch=2048, lr=1e-3):
    """Self-consistency BSDE training with c held fixed for every one of steps."""
    optim = torch.optim.Adam(critic.parameters(), lr=lr)
    latest = np.nan
    for step in range(steps):
        k, z = draw_states(batch)
        c = policy.evaluate(k, z)
        dw = np.sqrt(p.dt) * torch.randn(batch, device=DEVICE)
        y = torch.exp(z) * k.pow(p.alpha)

        v, Z = critic(k, z)
        # This is exactly the BSDE Euler update specified in the request.
        v_bsde_next = v + (p.rho * v - utility(c)) * p.dt + Z * dw
        k_next = torch.clamp(k + (y - p.delta * k - c) * p.dt, p.k_min, p.k_max)
        z_next = torch.clamp(z + p.kappa_z * (p.z_bar - z) * p.dt + p.sigma_z * dw,
                             p.z_min, p.z_max)
        # Same critic at the next state: gradients deliberately flow through it.
        v_next, _ = critic(k_next, z_next)
        loss = (v_bsde_next - v_next).square().mean()

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        optim.step()
        latest = loss.item()
        if step % 2_500 == 0 or step == steps - 1:
            print(f"    inner step {step:5d}; one-step BSDE MSE={latest:.3e}")
    return latest

def curves(critic, policy):
    k = torch.tensor(policy.k, dtype=torch.float32, device=DEVICE, requires_grad=True)
    z = torch.zeros_like(k)
    v, Z = critic(k, z)
    vk = torch.autograd.grad(v.sum(), k)[0]
    return (v.detach().cpu().numpy(), Z.detach().cpu().numpy(),
            vk.detach().cpu().numpy(), policy.c[:, np.argmin(abs(policy.z))])


def save_round_figure(outer, critic, policy):
    v, Z, vk, c = curves(critic, policy)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    ax[0].plot(policy.k, v, lw=2); ax[0].set(title=fr"round {outer}: $V(k,0)$", xlabel="k")
    ax[1].plot(policy.k, c, lw=2, color="tab:green")
    ax[1].set(title=fr"round {outer}: $c(k,0)$", xlabel="k")
    ax[2].plot(policy.k, vk, label=r"$V_k$")
    ax[2].plot(policy.k, Z, label=r"$Z$")
    ax[2].set(title=fr"round {outer}: derivative and exposure", xlabel="k"); ax[2].legend()
    for a in ax: a.grid(alpha=.25)
    path = OUT / f"toy_bsde_round_{outer:02d}.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path

def save_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    for outer, k, v, c in history:
        ax[0].plot(k, v, label=f"round {outer}")
        ax[1].plot(k, c, label=f"round {outer}")
    ax[0].set(title=r"BSDE critic: $V(k,0)$", xlabel="k")
    ax[1].set(title=r"policy improvement: $c(k,0)$", xlabel="k")
    for a in ax:
        a.grid(alpha=.25); a.legend(ncol=2, fontsize=8)
    path = OUT / "toy_bsde_all_rounds.png"
    fig.savefig(path, dpi=180); plt.close(fig)
    return path


def save_hjb_bsde_overlay(k_hjb, z_hjb, c_hjb, v_hjb, k_bsde, c_bsde, v_bsde):
    """Plot the two solution methods on the same value and policy panels."""
    iz_hjb = int(np.argmin(np.abs(z_hjb)))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
    ax[0].plot(k_hjb, v_hjb[:, iz_hjb], lw=2.4, label="HJB / Howard")
    ax[0].plot(k_bsde, v_bsde, "--", lw=2.2, label="BSDE policy iteration")
    ax[0].set(title=r"Value $V(k,0)$", xlabel="capital $k$", ylabel="$V$")
    ax[1].plot(k_hjb, c_hjb[:, iz_hjb], lw=2.4, label="HJB / Howard")
    ax[1].plot(k_bsde, c_bsde, "--", lw=2.2, label="BSDE policy iteration")
    ax[1].set(title=r"Consumption policy $c(k,0)$", xlabel="capital $k$", ylabel="$c$")
    for a in ax:
        a.grid(alpha=.25)
        a.legend()
    path = OUT / "toy_hjb_bsde_overlay.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    torch.manual_seed(11)
    print("device:", DEVICE)
    policy = FrozenGridPolicy()              # c^0=.1*exp(z)*k^alpha
    critic = Critic().to(DEVICE)
    history = []
    outer_rounds = 20
    for outer in range(outer_rounds):
        print(f"\nouter policy iteration {outer}: freeze c^{outer}, train V and Z")
        mse = train_fixed_policy(critic, policy, steps=20_000)
        round_file = save_round_figure(outer, critic, policy)
        v, _Z, _vk, c = curves(critic, policy)
        history.append((outer, policy.k.copy(), v, c.copy()))
        print(f"  saved {round_file.name}; final inner MSE={mse:.3e}")
        policy.improve(critic, damping=1.0)
        print("  policy update: c <- (V_k)^(-1/gamma)")
    
    # Evaluate the final, already-improved policy before comparing it to HJB.
    print("\nfinal BSDE evaluation with the final frozen policy")
    train_fixed_policy(critic, policy, steps=20_000)
    v_bsde, _Z, _vk, c_bsde = curves(critic, policy)
    # Machine-readable output for the HJB / joint-residual comparison figure.
    np.savez(OUT / "toy_bsde_frozen_foc_20round.npz", k=policy.k, v=v_bsde, z=_Z, c=c_bsde)
    k_hjb, z_hjb, kk_hjb, zz_hjb = make_grid()
    c_hjb, v_hjb = howard_iteration(k_hjb, z_hjb, kk_hjb, zz_hjb)
    print("saved", save_history(history).name)
    print("saved", save_hjb_bsde_overlay(
        k_hjb, z_hjb, c_hjb, v_hjb, policy.k, c_bsde, v_bsde
    ).name)