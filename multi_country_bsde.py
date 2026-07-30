"""Three-country version of Huang (2026), Sections 3.1--3.5.

Network output:
    * three positive intermediate xi's, which are transformed into q's so
      that final-goods market clearing holds exactly at every state;
    * all nine price-volatility coefficients sigma_q[i,j].

The price/volatility network does NOT output r.  A separate bounded rate
network is trained in alternating outer blocks, while q and sigma_q remain
in their own split subnetworks.

The forward laws below retain the original 3c omega/zeta update.  mu_q is
defined by the price BSDE/Euler equation and is never a learned output.

Run:
    python 3c_subnet.py

Outputs go to ./three_country_output/.
"""

from __future__ import annotations

import csv
import argparse
import math
import os
import sys
import time
import types
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import Tensor, nn


@dataclass(frozen=True)
class Parameters:
    # Parameters used in the paper's Section 3.5 numerical example.
    a: float = 0.10
    delta: float = 0.05
    sigma: float = 0.023
    psi: float = 5.0
    rho: float = 0.03
    chi: float = 0.001
    dt: float = 0.005

    # Training parameters.  These are deliberately modest for a CPU/GPU
    # laptop. Increase steps after inspecting the diagnostics.
    sample_size: int = 200_000
    batch_size: int = 64
    epochs: int = 30
    shocks_per_state: int = 32
    learning_rate: float = 1e-3
    seed: int = 7

    r_fixed: float = 0.03


P = Parameters()
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "three_country_output"
# Single precision is ample at dt=1e-3 after residual scaling and is much
# faster for the repeated statewise scalar-rate solve on ordinary hardware.
DTYPE = torch.float32
M = sys.modules[__name__]


def unpack_state(state: Tensor) -> tuple[Tensor, Tensor]:
    """Return eta[B,3], zeta[B,3], with zeta_3 as the residual share."""
    eta = state[:, :3]
    zeta12 = state[:, 3:5]
    zeta3 = 1.0 - zeta12.sum(dim=1, keepdim=True)
    return eta, torch.cat((zeta12, zeta3), dim=1)


def prices_and_volatility(net: nn.Module, state: Tensor, p: Parameters) -> tuple[Tensor, Tensor]:
    """Hard-code equation (19), using the paper's xi parameterization."""
    raw_xi, sigq = net(state)
    _, zeta = unpack_state(state)
    Xi = (raw_xi * zeta).sum(dim=1, keepdim=True)
    xi = p.rho * raw_xi / Xi
    q = (p.a * p.psi + 1.0) / (p.psi * xi + 1.0)
    return q, sigq

def market_clearing_error(q: Tensor, state: Tensor, p: Parameters) -> Tensor:
    """Equation (19), written as sum zeta_i/q_i=(1+psi*rho)/(1+a*psi).

    The flow in (19) is ``(a-iota_i)/q_i``, not ``a-iota_i/q_i``.
    """
    _, zeta = unpack_state(state)
    return (zeta / q).sum(dim=1) - (1.0 + p.psi * p.rho) / (1.0 + p.a * p.psi)


def bsde_drift(eta: Tensor, q: Tensor, sigq: Tensor, r: Tensor, p: Parameters) -> Tensor:
    """mu_q = -h/q from equation (20), shape [B,3]."""
    eye = torch.eye(3, dtype=q.dtype, device=q.device).unsqueeze(0)
    sigqK = sigq + p.sigma * eye
    portfolio_risk = sigqK.square().sum(dim=2) / eta
    diag = sigq.diagonal(dim1=1, dim2=2)
    return (-(p.a * p.psi + 1.0) / (p.psi * q)
            - torch.log(q) / p.psi + (1.0 / p.psi + p.delta)
            - p.sigma * diag + portfolio_risk + r)


def relative_bsde_losses(drift_loss: Tensor, z_loss: Tensor, muq: Tensor,
                         sigq: Tensor, implied_sigq: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Scale the two BSDE conditions by their *current* economic magnitudes.
    No fixed coefficient is imposed on Z.  The detached denominators make the
    two terms relative errors and prevent the networks from manipulating their
    own normalizers.
    """
    eps = torch.as_tensor(1e-12, dtype=drift_loss.dtype, device=drift_loss.device)
    drift_scale2 = muq.detach().square().mean().clamp_min(eps)
    z_scale2 = (0.5 * (sigq.detach().square().mean() + implied_sigq.detach().square().mean())).clamp_min(eps)
    return drift_loss / drift_scale2 + z_loss / z_scale2, drift_loss / drift_scale2, z_loss / z_scale2


def forward_state(state: Tensor, q: Tensor, sigq: Tensor, r: Tensor,
                  dw: Tensor, p: Parameters) -> Tensor:
    """Euler step of the supplied paper's equations (21)--(22).

    ``dw`` has shape [B,D,3], while state/q/sigq have batch shape [B,...].
    The clipping only keeps simulated training states within the sampled
    compact domain; it does not change the formula used before projection.
    """
    eta, zeta = unpack_state(state)
    B, D, _ = dw.shape
    eye = torch.eye(3, dtype=state.dtype, device=state.device).unsqueeze(0)
    sigqK = sigq + p.sigma * eye
    risk = sigqK.square().sum(dim=2)

    # This is the omega update from the original 3c.py, written in vector
    # form.  In particular retain its chi term.  The expression is algebraically
    # the same as eta*(a-iota)/q + eta*(phi-1)*(premium-risk)-eta*(rho+chi).
    b_eta = (((p.a *  p.psi + 1.0) / (p.psi * q) - 1.0 / p.psi - p.rho - p.chi) * eta
             + (1.0 / eta - 1.0).square() * eta * risk)
    eta_next = (eta[:, None, :]
                + b_eta[:, None, :] * p.dt
                + (1.0 - eta)[:, None, :] * torch.einsum("bij,bdj->bdi", sigqK, dw))

    # equation (22)
    mu_qK = (-(p.a * p.psi + 1.0) / (p.psi * q) + 1.0 / p.psi
              + risk / eta + r)
    mu_H = (zeta * mu_qK).sum(dim=1, keepdim=True)
    sig_H = torch.einsum("bi,bij->bj", zeta, sigqK)
    mu_zeta = mu_qK - mu_H - (sig_H[:, None, :] * (sigqK - sig_H[:, None, :])).sum(dim=2)
    zeta_next = (zeta[:, None, :]
                 * (1.0 + mu_zeta[:, None, :] * p.dt
                    + torch.einsum("bij,bdj->bdi", sigqK - sig_H[:, None, :], dw)))

    # The state has only zeta_1,zeta_2.  Project shares softly back to the
    # simplex after Euler discretization, so evaluation is defined everywhere.
    eta_next = eta_next.clamp(0.03, 0.97)
    zeta_next = zeta_next.clamp_min(1e-5)
    zeta_next = zeta_next / zeta_next.sum(dim=2, keepdim=True)
    return torch.cat((eta_next, zeta_next[:, :, :2]), dim=2).reshape(B * D, 5)


def sample_states(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    # Financial amplification is strongest near low expert wealth.  Half the
    # draws cover the whole domain, while U^2 draws deliberately add density
    # close to the constrained/low-wealth region.
    uniform = torch.rand(n, 3, device=device, dtype=dtype)
    low_wealth = torch.rand(n, 3, device=device, dtype=dtype).square()
    use_low_wealth = torch.rand(n, 1, device=device, dtype=dtype) < 0.5
    eta = 0.02 + 0.93 * torch.where(use_low_wealth, low_wealth, uniform)
    # A Dirichlet-like draw bounded away from faces prevents a denominator
    # singularity while covering asymmetric wealth distributions.
    zeta = 0.10 + torch.rand(n, 3, device=device, dtype=dtype)
    zeta = zeta / zeta.sum(dim=1, keepdim=True)
    return torch.cat((eta, zeta[:, :2]), dim=1)


@dataclass(frozen=True)
class RateConfig:
    sample_size: int = 200_000
    batch_size: int = 64
    epochs: int = 10
    shocks_per_state: int = 32
    learning_rate: float = 1e-4
    r_center: float = 0.03
    r_half_width: float = 0.12
    seed: int = 19


class RateNet(nn.Module):
    """Bounded state-dependent risk-free-rate network."""
    def __init__(self, width: int = 96) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, state: Tensor) -> Tensor:
        return R.C.r_center + R.C.r_half_width * torch.tanh(self.body(state))


R = types.SimpleNamespace(C=RateConfig(), RateConfig=RateConfig, RateNet=RateNet)


class SplitPriceVolatilityNet(nn.Module):
    """Shared features with independent q and sigma_q subnetworks."""
    def __init__(self, width: int = 128) -> None:
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU())
        self.q_body = nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.sigma_body = nn.Sequential(nn.Linear(width, width), nn.SiLU())
        self.xi_head = nn.Linear(width, 3)
        self.sigq_head = nn.Linear(width, 9)
        nn.init.normal_(self.sigq_head.weight, std=.015)
        nn.init.zeros_(self.sigq_head.bias)
        nn.init.zeros_(self.xi_head.bias)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        h = self.shared(state)
        raw_xi = torch.nn.functional.softplus(self.xi_head(self.q_body(h))) + 1e-6
        sigq = .08 * torch.tanh(self.sigq_head(self.sigma_body(h))).reshape(-1, 3, 3)
        return raw_xi, sigq
@dataclass(frozen=True)
class Config:
    sample_size: int = 200_000
    batch_size: int = 64
    shocks_per_state: int = 192
    q_epochs_per_round: int = 3
    r_epochs_per_round: int = 1
    outer_rounds: int = 10_000           # wall-clock budget stops the run
    max_train_seconds: float = 3 * 60 * 60
    learning_rate_q: float = 1e-3
    learning_rate_r: float = 1e-4
    seed: int = 37


C = Config()
OUT = ROOT / "three_country_output" / "joint_alternating"


class TimeBudgetReached(RuntimeError):
    pass


def regression_loss(price_net: nn.Module, rate_net: nn.Module, state: Tensor, *,
                    config: Config | None = None, z_weight: float = 1.0,
                    loss_mode: str = "relative") -> tuple[Tensor, dict[str, float]]:
    """Conditional intercept/slope BSDE loss for arbitrary separate r(Omega)."""
    config = C if config is None else config
    B, D = state.shape[0], config.shocks_per_state
    q, sigq = M.prices_and_volatility(price_net, state, M.P)
    eta, _ = M.unpack_state(state)
    r = rate_net(state)
    muq = M.bsde_drift(eta, q, sigq, r, M.P)
    dw = math.sqrt(M.P.dt) * torch.randn(B, D, 3, dtype=state.dtype, device=state.device)
    next_state = M.forward_state(state, q, sigq, r, dw, M.P)
    q_next, _ = M.prices_and_volatility(price_net, next_state, M.P)
    q_next = q_next.reshape(B, D, 3)
    X = torch.cat((torch.ones(B, D, 1, dtype=state.dtype, device=state.device), dw), dim=2)
    coef = torch.linalg.solve(X.transpose(1, 2) @ X, X.transpose(1, 2) @ q_next)
    intercept = coef[:, 0, :]
    slopes = coef[:, 1:, :].transpose(1, 2)
    implied_sigq = slopes / q[:, :, None]
    drift = ((intercept - (q + q * muq * M.P.dt)) / (q * M.P.dt)).square().mean()
    z = (implied_sigq - sigq).square().mean()
    _, relative_drift, relative_z = M.relative_bsde_losses(drift, z, muq, sigq, implied_sigq)
    if loss_mode == "raw":
        loss = drift + z_weight * z
    elif loss_mode == "relative":
        loss = relative_drift + z_weight * relative_z
    else:
        raise ValueError(f"unknown loss mode: {loss_mode}")
    return loss, {"loss": float(loss.detach()), "drift": float(drift.detach()),
                  "z": float(z.detach()), "relative_drift": float(relative_drift.detach()),
                  "relative_z": float(relative_z.detach()), "r_mean": float(r.mean().detach()),
                  "r_std": float(r.std().detach()), "sigq_rms": float(sigq.square().mean().sqrt().detach())}


def set_trainable(net: nn.Module, value: bool) -> None:
    for p in net.parameters():
        p.requires_grad_(value)

@torch.no_grad()
def implied_rate_gap(price_net: nn.Module, rate_net: nn.Module, device: torch.device) -> np.ndarray:
    """Country-by-country RMS |r_implied(q_i)-r_net| on the symmetric slice."""
    eta = torch.linspace(0.03, 0.94, 61, dtype=M.DTYPE, device=device)
    state = torch.column_stack((eta, eta, eta, torch.full_like(eta, 1 / 3), torch.full_like(eta, 1 / 3)))
    q, sigq = M.prices_and_volatility(price_net, state, M.P)
    r = rate_net(state)
    d = 256
    dw = math.sqrt(M.P.dt) * torch.randn(eta.numel(), d, 3, dtype=M.DTYPE, device=device)
    q1, _ = M.prices_and_volatility(price_net, M.forward_state(state, q, sigq, r, dw, M.P), M.P)
    q1 = q1.reshape(eta.numel(), d, 3)
    x = torch.cat((torch.ones(eta.numel(), d, 1, dtype=M.DTYPE, device=device), dw), dim=2)
    coef = torch.linalg.solve(x.transpose(1, 2) @ x, x.transpose(1, 2) @ q1)
    mu_hat = (coef[:, 0, :] - q) / (q * M.P.dt)
    eye = torch.eye(3, dtype=M.DTYPE, device=device).unsqueeze(0)
    sigk = sigq + M.P.sigma * eye
    base = (-(1.0 + M.P.a * M.P.psi) / (M.P.psi * q) - torch.log(q) / M.P.psi
            + 1.0 / M.P.psi + M.P.delta - M.P.sigma * sigq.diagonal(dim1=1, dim2=2)
            + sigk.square().sum(dim=2) / eta[:, None])
    return (mu_hat - base - r).square().mean(dim=0).sqrt().cpu().numpy()


def phase(price_net: nn.Module, rate_net: nn.Module, states: Tensor, *, train_price: bool,
          epochs: int, optimizer: torch.optim.Optimizer, outer: int, label: str,
          deadline: float, config: Config | None = None, z_weight: float = 1.0,
          loss_mode: str = "relative") -> list[dict[str, float]]:
    target = price_net if train_price else rate_net
    config = C if config is None else config
    history: list[dict[str, float]] = []
    step = 0
    for epoch in range(epochs):
        for idx in torch.randperm(states.shape[0], device=states.device).split(config.batch_size):
            if time.monotonic() >= deadline:
                raise TimeBudgetReached
            step += 1
            optimizer.zero_grad(set_to_none=True)
            loss, stats = regression_loss(price_net, rate_net, states[idx], config=config,
                                          z_weight=z_weight, loss_mode=loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(target.parameters(), 5.0)
            optimizer.step()
            if step == 1 or step % 300 == 0:
                stats.update(outer=outer, phase=label, step=step)
                history.append(stats)
                print("outer={} {} step={:4d} total={:.3e} drift={:.3e} Z={:.3e} "
                      "r={:.4f}+/-{:.4f} sigq={:.2e}".format(
                        outer, label, step, stats["loss"], stats["drift"], stats["z"],
                        stats["r_mean"], stats["r_std"], stats["sigq_rms"]), flush=True)
    return history


@torch.no_grad()
def save_plots(price_net: nn.Module, rate_net: nn.Module, history: list[dict[str, float]],
               device: torch.device, output_dir: Path | None = None) -> None:
    output_dir = OUT if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eta = torch.linspace(0.02, 0.95, 201, dtype=M.DTYPE, device=device)
    state = torch.column_stack((eta, eta, eta, torch.full_like(eta, 1 / 3), torch.full_like(eta, 1 / 3)))
    q, sigq = M.prices_and_volatility(price_net, state, M.P)
    r = rate_net(state).squeeze(1)
    price_vol = sigq.square().sum(dim=2).sqrt()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for i in range(3):
        axes[0, 0].plot(eta.cpu(), q[:, i].cpu(), label=fr"$q_{i+1}$")
        axes[1, 0].plot(eta.cpu(), price_vol[:, i].cpu(), label=fr"$\|\sigma^q_{i+1}\|$")
    axes[0, 0].set(title="Prices on symmetric slice", xlabel=r"$\eta$", ylabel=r"$q$"); axes[0, 0].legend()
    axes[1, 0].set(title="Price-volatility norms", xlabel=r"$\eta$", ylabel=r"$\|\sigma^q\|$"); axes[1, 0].legend()
    axes[0, 1].plot(eta.cpu(), r.cpu(), label=r"$r(\eta)$")
    axes[0, 1].axhline(.03, c="k", ls="--", label="initial r")
    axes[0, 1].set(title="Alternating-improved rate", xlabel=r"$\eta$", ylabel="r"); axes[0, 1].legend()
    for label, color in [("q", "tab:blue"), ("r", "tab:orange")]:
        h = [x for x in history if x["phase"] == label]
        axes[1, 1].semilogy([x["outer"] for x in h], [x["loss"] for x in h], "o", color=color, label=f"{label} phase")
    axes[1, 1].set(title="Outer-round losses", xlabel="outer round", ylabel="loss"); axes[1, 1].legend()
    fig.savefig(output_dir / "joint_alternating.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11, 8), sharex=True, constrained_layout=True)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.plot(eta.cpu(), sigq[:, i, j].cpu(), lw=2)
            ax.axhline(0.0, color="black", lw=.7)
            ax.set_title(fr"$\sigma^q_{{{i+1},{j+1}}}(\eta)$")
            if i == 2:
                ax.set_xlabel(r"$\eta$")
            if j == 0:
                ax.set_ylabel("loading")
    fig.suptitle(r"All nine price-volatility loadings on the symmetric slice", fontsize=15)
    fig.savefig(output_dir / "sigma_q_matrix.png", dpi=180)
    plt.close(fig)

    eye = torch.eye(3, dtype=M.DTYPE, device=device).unsqueeze(0)
    total_asset_vol = (sigq + M.P.sigma * eye).square().sum(dim=2).sqrt()
    risk_premium = total_asset_vol.square() / eta[:, None]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    axes[0, 0].plot(eta.cpu(), r.cpu())
    axes[0, 0].set(title="Risk-free rate", xlabel=r"$\eta$", ylabel=r"$r$")
    for i in range(3):
        axes[0, 1].plot(eta.cpu(), price_vol[:, i].cpu(), label=fr"country {i+1}")
        axes[1, 0].plot(eta.cpu(), total_asset_vol[:, i].cpu(), label=fr"country {i+1}")
        axes[1, 1].plot(eta.cpu(), risk_premium[:, i].cpu(), label=fr"country {i+1}")
    axes[0, 1].set(title="Endogenous price volatility", xlabel=r"$\eta$", ylabel=r"$\|\sigma_q\|$")
    axes[1, 0].set(title="Total physical-asset return volatility", xlabel=r"$\eta$", ylabel=r"$\|\sigma e_i+\sigma^q_i\|$")
    axes[1, 1].set(title=r"Expert risk premium: $\eta^{-1}\|\sigma e_i+\sigma^q_i\|^2$", xlabel=r"$\eta$", ylabel="premium")
    axes[0, 1].legend(); axes[1, 0].legend(); axes[1, 1].legend()
    fig.savefig(output_dir / "financial_acceleration_diagnostics.png", dpi=180)
    plt.close(fig)

    # The price BSDE itself gives a separate implied common rate for each
    # country.  Estimate the local q drift by the same conditional bundle OLS
    # used in training, then invert the three Euler equations separately.
    # At an equilibrium all three curves equal the rate-network curve.
    bundle = 512
    dw = math.sqrt(M.P.dt) * torch.randn(eta.numel(), bundle, 3,
                                         dtype=M.DTYPE, device=device)
    next_state = M.forward_state(state, q, sigq, r[:, None], dw, M.P)
    q_next, _ = M.prices_and_volatility(price_net, next_state, M.P)
    q_next = q_next.reshape(eta.numel(),  bundle, 3)
    x = torch.cat((torch.ones(eta.numel(), bundle, 1, dtype=M.DTYPE, device=device), dw), dim=2)
    coef = torch.linalg.solve(x.transpose(1, 2) @ x, x.transpose(1, 2) @ q_next)
    mu_hat = (coef[:, 0, :] - q) / (q * M.P.dt)
    sigk = sigq + M.P.sigma * eye
    base = (-(1.0 + M.P.a * M.P.psi) / (M.P.psi * q)
            - torch.log(q) / M.P.psi + 1.0 / M.P.psi + M.P.delta
            - M.P.sigma * sigq.diagonal(dim1=1, dim2=2)
            + sigk.square().sum(dim=2) / eta[:, None])
    r_implied = mu_hat - base
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(eta.cpu(), r.cpu(), color="black", lw=2.5, label=r"network $r(\eta)$")
    for i, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
        ax.plot(eta.cpu(), r_implied[:, i].cpu(), color=color, lw=1.6,
                label=fr"$r^{{\mathrm{{implied}}}}_{i+1}(\eta)$ from $q_{i+1}$")
    ax.set(title="Rate consistency across the three price BSDEs", xlabel=r"$\eta$", ylabel=r"$r$")
    ax.legend(ncol=2)
    fig.savefig(output_dir / "implied_rate_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    global C, OUT
    parser = argparse.ArgumentParser(description="Joint alternating three-country BSDE training")
    parser.add_argument("--dt", type=float, default=M.P.dt)
    parser.add_argument("--a", type=float, default=M.P.a)
    parser.add_argument("--sigma", type=float, default=M.P.sigma)
    parser.add_argument("--outer-rounds", type=int, default=C.outer_rounds,
                        help="number of alternating outer rounds to run")
    parser.add_argument("--save-every", type=int, default=5,
                        help="save a versioned checkpoint and figures every N completed outer rounds")
    parser.add_argument("--hours", type=float, default=C.max_train_seconds / 3600,
                        help="wall-clock limit; set 0 for no limit")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--tag", type=str, default=None, help="subdirectory name under three_country_output")
    parser.add_argument("--price-init", type=Path, default=None, help="optional q/sigma checkpoint")
    parser.add_argument("--rate-init", type=Path, default=None, help="optional separate r checkpoint")
    parser.add_argument("--resume", type=Path, default=None,
                        help="latest_outer_checkpoint.pt from a previous segment")
    parser.add_argument("--fresh", action="store_true",
                        help="start random networks; do not load any prior checkpoint")
    parser.add_argument("--z-pretrain-epochs", type=int, default=30,
                        help="price-only epochs before joint training; set 0 to disable")
    parser.add_argument("--z-pretrain-states", type=int, default=100_000)
    parser.add_argument("--z-pretrain-shocks", type=int, default=32)
    parser.add_argument("--z-loss-weight", type=float, default=100.0,
                        help="relative Z-loss multiplier used only during price pretraining")
    parser.add_argument("--legacy-bootstrap", action="store_true",
                        help="reproduce the old raw-loss price/rate bootstrap before relative-loss joint training")
    parser.add_argument("--rate-half-width", type=float, default=0.05,
                        help="half-width in r=.03+width*tanh(h), used by legacy rate bootstrap and joint phase")
    args = parser.parse_args()

    M.P = replace(M.P, dt=args.dt, a=args.a, sigma=args.sigma)
    max_seconds = float("inf") if args.hours == 0 else args.hours * 3600
    C = replace(C, max_train_seconds=max_seconds, outer_rounds=args.outer_rounds)
    tag = args.tag or f"joint_dt{args.dt:.4f}_a{args.a:.3f}".replace(".", "p")
    OUT = ROOT / "three_country_output" / tag
    torch.manual_seed(C.seed)
    torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    price_net = M.PriceVolatilityNet().to(device=device, dtype=M.DTYPE)
    rate_net = R.RateNet().to(device=device, dtype=M.DTYPE)
    start_outer = 1
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        price_net.load_state_dict(resume["price_state_dict"])
        rate_net.load_state_dict(resume["rate_state_dict"])
        start_outer = int(resume["outer"]) + 1
        print(f"resuming from {args.resume} at outer round {start_outer}", flush=True)
    elif not args.fresh:
        ckpt = args.price_init or (ROOT / "three_country_output" / "three_country_model.pt")
        if not ckpt.exists():
            raise FileNotFoundError("Run 3c.py before joint alternating training, or pass --fresh.")
        price_net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["state_dict"])
        prior_rate = args.rate_init or (ROOT / "three_country_output" / "rate_improvement" / "rate_net.pt")
        if prior_rate.exists():
            rate_net.load_state_dict(torch.load(prior_rate, map_location=device, weights_only=False)["state_dict"])
    else:
        print("starting fresh random price and rate networks", flush=True)
    history: list[dict[str, float]] = []
    rate_gap_history: list[tuple[int, np.ndarray]] = []
    deadline = time.monotonic() + C.max_train_seconds

    if args.legacy_bootstrap:
        # Exact numerical bootstrap used to make three_country_model.pt and
        # rate_net.pt, except that dt remains the CLI-selected value.
        # The historic price and rate bootstrap were two independent programs
        # with seeds 7 and 19, respectively.
        torch.manual_seed(7)
        price_net = M.PriceVolatilityNet().to(device=device, dtype=M.DTYPE)
        legacy = replace(C, sample_size=100_000, shocks_per_state=32)
        R.C = replace(R.C, sample_size=100_000, shocks_per_state=32,
                      learning_rate=2e-4, r_half_width=args.rate_half_width)
        legacy_states = M.sample_states(legacy.sample_size, device, M.DTYPE)
        legacy_price_opt = torch.optim.Adam(price_net.parameters(), lr=2e-4)
        set_trainable(price_net, True); set_trainable(rate_net, False); price_net.train(); rate_net.eval()
        print("starting legacy raw bootstrap: price 30 epochs, r fixed at .03, loss=drift+50*Z", flush=True)
        history += phase(price_net, rate_net, legacy_states, train_price=True, epochs=30,
                         optimizer=legacy_price_opt, outer=0, label="legacy_price", deadline=deadline,
                         config=legacy, z_weight=50.0, loss_mode="raw")
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": price_net.state_dict(), "loss": "raw drift + 50 Z",
                    "config": legacy.__dict__, "model_parameters": M.P.__dict__},
                   OUT / "legacy_price_bootstrap.pt")
        torch.manual_seed(19)
        rate_net = R.RateNet().to(device=device, dtype=M.DTYPE)
        rate_states = M.sample_states(legacy.sample_size, device, M.DTYPE)
        legacy_rate_opt = torch.optim.Adam(rate_net.parameters(), lr=2e-4)
        set_trainable(price_net, False); set_trainable(rate_net, True); price_net.eval(); rate_net.train()
        print("continuing legacy raw bootstrap: rate 10 epochs, loss=drift+50*Z", flush=True)
        history += phase(price_net, rate_net, rate_states, train_price=False, epochs=10,
                         optimizer=legacy_rate_opt, outer=0, label="legacy_rate", deadline=deadline,
                         config=legacy, z_weight=50.0, loss_mode="raw")
        torch.save({"state_dict": rate_net.state_dict(), "loss": "raw drift + 50 Z",
                    "r_half_width": args.rate_half_width, "config": legacy.__dict__, "model_parameters": M.P.__dict__},
                   OUT / "legacy_rate_bootstrap.pt")
    elif args.z_pretrain_epochs:
        pre_config = replace(C, sample_size=args.z_pretrain_states, shocks_per_state=args.z_pretrain_shocks)
        pre_states = M.sample_states(pre_config.sample_size, device, M.DTYPE)
        set_trainable(price_net, True); set_trainable(rate_net, False); price_net.train(); rate_net.eval()
        print(f"starting Z-weighted price pretraining: epochs={args.z_pretrain_epochs}, "
              f"states={pre_config.sample_size}, shocks={pre_config.shocks_per_state}, "
              f"weight={args.z_loss_weight:g}", flush=True)
        history += phase(price_net, rate_net, pre_states, train_price=True,
                         epochs=args.z_pretrain_epochs, optimizer=price_optimizer, outer=0,
                         label="preZ", deadline=deadline, config=pre_config, z_weight=args.z_loss_weight)
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": price_net.state_dict(), "pretrain_epochs": args.z_pretrain_epochs,
                    "z_loss_weight": args.z_loss_weight, "config": pre_config.__dict__,
                    "model_parameters": M.P.__dict__}, OUT / "z_pretrained_price_model.pt")

    torch.manual_seed(C.seed)
    states = M.sample_states(C.sample_size, device, M.DTYPE)
    # New Adam states for the different, relative-loss joint objective; these
    # persist thereafter across every outer round.
    price_optimizer = torch.optim.Adam(price_net.parameters(), lr=C.learning_rate_q)
    rate_optimizer = torch.optim.Adam(rate_net.parameters(), lr=C.learning_rate_r)

    def save_checkpoint(outer: int, status: str) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save({"outer": outer, "status": status, "price_state_dict": price_net.state_dict(),
                    "rate_state_dict": rate_net.state_dict(), "config": C.__dict__, "model_parameters": M.P.__dict__},
                   OUT / "latest_outer_checkpoint.pt")

    def save_snapshot(outer: int) -> None:
        snapshot = OUT / "snapshots" / f"outer_{outer:03d}"
        snapshot.mkdir(parents=True, exist_ok=True)
        torch.save({"outer": outer, "status": "completed_round",
                    "price_state_dict": price_net.state_dict(),
                    "rate_state_dict": rate_net.state_dict(),
                    "config": C.__dict__, "model_parameters": M.P.__dict__},
                   snapshot / "checkpoint.pt")
        # This call occurs only after the r phase: q and sigma_q are frozen
        # while r is trained, so every saved picture is a complete outer round.
        save_plots(price_net, rate_net, history, device, snapshot)
        rate_gap_history.append((outer, implied_rate_gap(price_net, rate_net, device)))
        rounds = np.asarray([x[0] for x in rate_gap_history])
        gaps = np.stack([x[1] for x in rate_gap_history])
        np.savez(OUT / "implied_rate_gap_history.npz", outer=rounds, rms_gap=gaps)
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
        for i, color in enumerate(["tab:blue", "tab:orange", "tab:green"]):
            ax.semilogy(rounds, gaps[:, i], "o-", color=color, label=fr"$q_{i+1}$ BSDE")
        ax.set(title="Convergence of country-implied rates to network rate", xlabel="outer round",
               ylabel=r"RMS $|r_i^{\mathrm{implied}}-r_{\rm net}|$")
        ax.legend()
        fig.savefig(OUT / "implied_rate_gap_history.png", dpi=180)
        plt.close(fig)


    for outer in range(start_outer, start_outer + C.outer_rounds):
        try:
            set_trainable(price_net, True); set_trainable(rate_net, False); rate_net.eval(); price_net.train()
            history += phase(price_net, rate_net, states, train_price=True, epochs=C.q_epochs_per_round,
                             optimizer=price_optimizer, outer=outer, label="q", deadline=deadline)

            set_trainable(price_net, False); set_trainable(rate_net, True); price_net.eval();  rate_net.train()
            history += phase(price_net, rate_net, states, train_price=False, epochs=C.r_epochs_per_round,
                             optimizer=rate_optimizer, outer=outer, label="r", deadline=deadline)
            save_checkpoint(outer, "completed_round")
            if outer % args.save_every == 0:
                save_snapshot(outer)
        except TimeBudgetReached:
            save_checkpoint(outer, "time_budget_reached")
            print(f"two-hour budget reached during outer round {outer}; checkpoint saved", flush=True)
            break

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": price_net.state_dict(), "config": C.__dict__, "model_parameters": M.P.__dict__}, OUT / "joint_price_model.pt")
    torch.save({"state_dict": rate_net.state_dict(), "config": C.__dict__, "model_parameters": M.P.__dict__}, OUT / "joint_rate_model.pt")
    save_plots(price_net, rate_net, history, device)
    print(f"saved joint alternating result to {OUT}")




if __name__ == "__main__":
    # Exact defaults of the successful split-subnetwork run.
    M.P = replace(M.P, r_fixed=.03)
    R.C = replace(R.C, r_center=.03)
    C = replace(C, learning_rate_r=5e-4)
    M.PriceVolatilityNet = SplitPriceVolatilityNet
    _original_phase = phase
    def phase(*args, **kwargs):
        if not kwargs["train_price"] and kwargs["outer"] % 2:
            return []
        return _original_phase(*args, **kwargs)
    def _add_default(flag: str, value: str | None = None) -> None:
        if flag not in sys.argv and not any(x.startswith(f"{flag}=") for x in sys.argv[1:]):
            sys.argv.append(flag)
            if value is not None: sys.argv.append(value)
    _add_default("--fresh")
    _add_default("--legacy-bootstrap")
    _add_default("--a", ".10")
    _add_default("--sigma", ".023")
    _add_default("--dt", ".01")
    _add_default("--outer-rounds", "200")
    _add_default("--hours", "0")
    _add_default("--rate-half-width", ".12")
    _add_default("--tag", "3c_subnet")
    main()
  