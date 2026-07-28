"""Self-contained three-country fixed-drift / alternating-BSDE solver.

The state is (eta_1, eta_2, eta_3, zeta_1, zeta_2), with zeta_3 residual.
The price network outputs only q and sigma_q.  A separate network produces the
common rate r.  Every outer round freezes r while fitting q,sigma_q, then
freezes q,sigma_q while fitting r.
"""


from __future__ import annotations

import argparse
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parent
DTYPE = torch.float32


@dataclass(frozen=True)
class P:
    a: float = .10
    delta: float = .05
    sigma: float = .023
    psi: float = 5.
    rho: float = .03
    chi: float = .001
    dt: float = .005

@dataclass(frozen=True)
class Cfg:
    states: int = 200_000
    batch: int = 64
    shocks: int = 192
    q_epochs: int = 3
    r_epochs: int = 1
    outer_rounds: int = 100
    save_every: int = 5
    lr_q: float = 1e-3
    lr_r: float = 1e-4
    seed: int = 37


class PriceNet(nn.Module):
    """State -> positive xi intermediates and all 9 relative price loadings."""
    def __init__(self, width: int = 128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.xi = nn.Linear(width, 3)
        self.sq = nn.Linear(width, 9)
        nn.init.normal_(self.sq.weight, std=.015)
        nn.init.zeros_(self.sq.bias); nn.init.zeros_(self.xi.bias)

    def forward(self, s: Tensor) -> tuple[Tensor, Tensor]:
        h = self.body(s)
        return torch.nn.functional.softplus(self.xi(h)) + 1e-6, .08 * torch.tanh(self.sq(h)).reshape(-1, 3, 3)

class RateNet(nn.Module):
    """Separate common rate function; it never outputs q or sigma_q."""
    def __init__(self, width: int = 96, half_width: float = .12):
        super().__init__()
        self.half_width = half_width
        self.body = nn.Sequential(nn.Linear(5, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                  nn.Linear(width, 1))
        nn.init.zeros_(self.body[-1].weight); nn.init.zeros_(self.body[-1].bias)

    def forward(self, s: Tensor) -> Tensor:
        return .03 + self.half_width * torch.tanh(self.body(s))


def unpack(s: Tensor) -> tuple[Tensor, Tensor]:
    eta, z12 = s[:, :3], s[:, 3:5]
    return eta, torch.cat((z12, 1. - z12.sum(1, keepdim=True)), 1)


def q_and_sigma(net: PriceNet, s: Tensor, p: P) -> tuple[Tensor, Tensor]:
    raw, sq = net(s)
    _, zeta = unpack(s)
    xi = p.rho * raw / (raw * zeta).sum(1, keepdim=True)
    q = (1. + p.a * p.psi) / (1. + p.psi * xi)
    return q, sq


def mu_q(eta: Tensor, q: Tensor, sq: Tensor, r: Tensor, p: P) -> Tensor:
    eye = torch.eye(3, dtype=q.dtype, device=q.device).unsqueeze(0)
    sk = sq + p.sigma * eye
    risk = sk.square().sum(2) / eta
    return (-(1. + p.a * p.psi) / (p.psi * q) - torch.log(q) / p.psi + 1. / p.psi + p.delta
            - p.sigma * sq.diagonal(dim1=1, dim2=2) + risk + r)


def transition(s: Tensor, q: Tensor, sq: Tensor, r: Tensor, dw: Tensor, p: P) -> Tensor:
    """The original eta/zeta law of motion, Euler-discretized."""
    eta, zeta = unpack(s); eye = torch.eye(3, dtype=s.dtype, device=s.device).unsqueeze(0)
    sk = sq + p.sigma * eye; risk = sk.square().sum(2)
    b_eta = (((1. + p.a * p.psi) / (p.psi * q) - 1. / p.psi - p.rho - p.chi) * eta
             + (1. / eta - 1.).square() * eta * risk)
    eta1 = eta[:, None] + b_eta[:, None] * p.dt + (1. - eta)[:, None] * torch.einsum("bij,bdj->bdi", sk, dw)
    mu_k = (-(1. + p.a * p.psi) / (p.psi * q) + 1. / p.psi + risk / eta + r)
    mu_h = (zeta * mu_k).sum(1, keepdim=True)
    sig_h = torch.einsum("bi,bij->bj", zeta, sk)
    mu_z = mu_k - mu_h - (sig_h[:, None] * (sk - sig_h[:, None])).sum(2)
    zeta1 = zeta[:, None] * (1. + mu_z[:, None] * p.dt + torch.einsum("bij,bdj->bdi", sk - sig_h[:, None], dw))
    eta1 = eta1.clamp(.03, .97); zeta1 = zeta1.clamp_min(1e-5); zeta1 = zeta1 / zeta1.sum(2, keepdim=True)
    return torch.cat((eta1, zeta1[:, :, :2]), 2).reshape(-1, 5)

class EmaRelativeScales:
    """Stop-gradient EMA denominators for the two relative BSDE residuals."""
    def __init__(self, alpha: float = .05, min_scale: float = 1e-8): # alpha can be anything, I just try 0.05 or 0.1 or 0.2
        self.alpha = alpha
        self.min_scale = min_scale
        self.drift_scale: Tensor | None = None
        self.z_scale: Tensor | None = None

    def __call__(self, drift: Tensor, slope: Tensor, mu: Tensor, sq: Tensor,
                 implied: Tensor) -> tuple[Tensor, Tensor]:
        floor = torch.as_tensor(self.min_scale, dtype=drift.dtype, device=drift.device)
        current_drift = mu.detach().square().mean().clamp_min(floor)
        current_z = (.5 * (sq.detach().square().mean() + implied.detach().square().mean())).clamp_min(floor)
        if self.drift_scale, self.z_scale = current_drift, current_z
        else:
            self.drift_scale = ((1. - self.alpha) * self.drift_scale + self.alpha * current_drift).detach()
            self.z_scale = ((1. - self.alpha) * self.z_scale + self.alpha * current_z).detach()
        return drift / self.drift_scale.clamp_min(floor), slope / self.z_scale.clamp_min(floor)


relative_losses = EmaRelativeScales()


def regression_loss(price: PriceNet, rate: RateNet, state: Tensor, cfg: Cfg, p: P, *,
                    loss_mode: str = "relative", z_weight: float = 1.) -> tuple[Tensor, dict[str, float]]:
    """Conditional bundle regression: intercept=q+q*mu_q*dt, slope=q*sigma_q."""
    b, d = state.shape[0], cfg.shocks
    q, sq = q_and_sigma(price, state, p); eta, _ = unpack(state); r = rate(state)
    mu = mu_q(eta, q, sq, r, p)
    dw = math.sqrt(p.dt) * torch.randn(b, d, 3, dtype=state.dtype, device=state.device)
    q1, _ = q_and_sigma(price, transition(state, q, sq, r, dw, p), p); q1 = q1.reshape(b, d, 3)
    x = torch.cat((torch.ones(b, d, 1, dtype=state.dtype, device=state.device), dw), 2)
    coef = torch.linalg.solve(x.transpose(1, 2) @ x, x.transpose(1, 2) @ q1)
    intercept, slopes = coef[:, 0], coef[:, 1:].transpose(1, 2)
    implied = slopes / q[:, :, None]
    drift = ((intercept - q * (1. + mu * p.dt)) / (q * p.dt)).square().mean()
    slope = (implied - sq).square().mean()
    rd, rz = relative_losses(drift, slope, mu, sq, implied)
    if loss_mode == "raw": loss = drift + z_weight * slope
    elif loss_mode == "relative": loss = rd + z_weight * rz
    else: raise ValueError(f"unknown loss_mode {loss_mode}")
    return loss, {"loss": float(loss.detach()), "drift": float(drift.detach()), "slope": float(slope.detach())}


def draw_states(n: int, device: torch.device) -> Tensor:
    u = torch.rand(n, 3, device=device)
    eta = .02 + .93 * torch.where(torch.rand(n, 1, device=device) < .5, u.square(), u)
    # Match the joint solver: cover asymmetric wealth shares while avoiding
    # simplex faces where denominators become numerically fragile.
    zeta = .10 + torch.rand(n, 3, device=device); zeta = zeta / zeta.sum(1, keepdim=True)
    return torch.cat((eta, zeta[:, :2]), 1).to(DTYPE)

def set_grad(net: nn.Module, value: bool) -> None:
    for x in net.parameters(): x.requires_grad_(value)


def phase(price: PriceNet, rate: RateNet, states: Tensor, train_price: bool, epochs: int,
          optimizer: torch.optim.Optimizer, outer: int, cfg: Cfg, p: P, *, label: str | None = None,
          loss_mode: str = "relative", z_weight: float = 1.) -> list[dict[str, float]]:
    target = price if train_price else rate
    history, step = [], 0
    label = label or ("q" if train_price else "r")
    for _ in range(epochs):
        for ix in torch.randperm(len(states), device=states.device).split(cfg.batch):
            step += 1; optimizer.zero_grad(set_to_none=True)
            loss, stat = regression_loss(price, rate, states[ix], cfg, p, loss_mode=loss_mode, z_weight=z_weight)
            loss.backward(); torch.nn.utils.clip_grad_norm_(target.parameters(), 5.); optimizer.step()
            if step == 1 or step % 300 == 0:
                stat.update(outer=outer, phase=label, step=step); history.append(stat)
                print(f"outer={outer} {label} step={step:4d} loss={stat['loss']:.3e} "
                      f"drift={stat['drift']:.3e} Z={stat['slope']:.3e} r={stat['r']:.4f} sq={stat['sq']:.2e}", flush=True)
    return history


@torch.no_grad()
def plots(price: PriceNet, rate: RateNet, history: list[dict], p: P, out: Path, device: torch.device) -> None:
    out.mkdir(parents=True, exist_ok=True); eta = torch.linspace(.02, .95, 201, device=device, dtype=DTYPE)
    s = torch.column_stack((eta, eta, eta, torch.full_like(eta, 1/3), torch.full_like(eta, 1/3)))
    q, sq = q_and_sigma(price, s, p); r = rate(s).squeeze(1); eye = torch.eye(3, device=device).unsqueeze(0)
    av = (sq + p.sigma * eye).square().sum(2).sqrt(); rp = av.square() / eta[:, None]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for i in range(3):
        ax[0,0].plot(eta.cpu(), q[:,i].cpu(), label=f"q{i+1}"); ax[1,0].plot(eta.cpu(), sq[:,i].square().sum(1).sqrt()..cpu())
        ax[1,1].plot(eta.cpu(), rp[:,i].cpu(), label=f"country {i+1}")
    ax[0,0].set(title="Prices", xlabel="eta"); ax[0,0].legend(); ax[0,1].plot(eta.cpu(), r.cpu()); ax[0,1].set(title="Common rate", xlabel="eta")
    ax[1,0].set(title="Price-volatility norm", xlabel="eta"); ax[1,1].set(title="Euler risk term", xlabel="eta"); ax[1,1].legend()
    fig.savefig(out / "joint_alternating.png", dpi=180); plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, constrained_layout=True)
    for i in range(3):
        for j in range(3): axes[i,j].plot(eta.cpu(), sq[:,i,j].cpu()); axes[i,j].set_title(f"sigma_q,{i+1}{j+1}")
    fig.savefig(out / "sigma_q_matrix.png", dpi=180); plt.close(fig)

    # Separately invert each price BSDE for r.  Equality of these three curves
    # and the rate-net curve is the useful equilibrium consistency diagnostic.
    d = 512; dw = math.sqrt(p.dt) * torch.randn(eta.numel(), d, 3, dtype=DTYPE, device=device)
    q1, _ = q_and_sigma(price, transition(s, q, sq, r[:, None], dw, p), p); q1 = q1.reshape(eta.numel(), d, 3)
    x = torch.cat((torch.ones(eta.numel(), d, 1, dtype=DTYPE, device=device), dw), 2)
    coef = torch.linalg.solve(x.transpose(1, 2) @ x, x.transpose(1, 2) @ q1)
    muhat = (coef[:, 0] - q) / (q * p.dt)
    base = (-(1 + p.a * p.psi) / (p.psi * q) - torch.log(q) / p.psi + 1 / p.psi + p.delta
            - p.sigma * sq.diagonal(dim1=1, dim2=2) + (sq + p.sigma * eye).square().sum(2) / eta[:, None])
    ri = muhat - base
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True); ax.plot(eta.cpu(), r.cpu(), c="k", lw=2.5, label="network r")
    for i, color in enumerate(["tab:blue", "tab:orange", "tab:green"]): ax.plot(eta.cpu(), ri[:, i].cpu(), c=color, label=f"implied r from q{i+1}")
    ax.set(title="Rate consistency across price BSDEs", xlabel="eta", ylabel="r"); ax.legend(ncol=2)
    fig.savefig(out / "implied_rate_comparison.png", dpi=180); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--a", type=float, default=.10); ap.add_argument("--sigma", type=float, default=.023)
    ap.add_argument("--dt", type=float, default=.01); ap.add_argument("--outer-rounds", type=int, default=100); ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--tag", default="3c_toshare"); ap.add_argument("--resume", type=Path)
    ap.add_argument("--legacy-bootstrap", action="store_true", help="raw drift+50Z price/rate bootstrap before relative-loss joint training")
    ap.add_argument("--rate-half-width", type=float, default=.05,
                    help="bootstrap and joint r range: r=.03+width*tanh(h); .05 gives [-.02,.08], .12 gives [-.09,.15]")
    args = ap.parse_args(); torch.manual_seed(37); torch.set_num_threads(args.threads); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p, cfg = P(a=args.a, sigma=args.sigma, dt=args.dt), Cfg(outer_rounds=args.outer_rounds, save_every=args.save_every)
    out = ROOT / "three_country_output" / args.tag
    price = PriceNet().to(device)
    rate = RateNet(half_width=args.rate_half_width if args.legacy_bootstrap else .12).to(device)
    start = 1
    if args.resume:
        d = torch.load(args.resume, map_location=device, weights_only=False)
        price.load_state_dict(d["price_state_dict"])
        rate = RateNet(half_width=d.get("rate_half_width", rate.half_width)).to(device)
        rate.load_state_dict(d["rate_state_dict"]); start = d["outer"] + 1
    hist: list[dict] = []
    if args.legacy_bootstrap and not args.resume:
        # Historic procedure producing the old high-sigma_q branch, except the
        # user-selected dt is retained.  The raw loss is not the final loss.
        legacy = replace(cfg, states=100_000, shocks=32)
        torch.manual_seed(7); price = PriceNet().to(device)
        rate = RateNet(half_width=args.rate_half_width).to(device)
        legacy_price_states = draw_states(legacy.states, device)
        p_opt0 = torch.optim.Adam(price.parameters(), lr=2e-4)
        set_grad(price, True); set_grad(rate, False)
        print("legacy price bootstrap: 30 epochs, r=.03 fixed, raw drift + 50 Z", flush=True)
        hist += phase(price, rate, legacy_price_states, True, 30, p_opt0, 0, legacy, p,
                      label="legacy_price", loss_mode="raw", z_weight=50.)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": price.state_dict(), "loss": "raw drift + 50 Z", "parameters": asdict(p)}, out / "legacy_price_bootstrap.pt")
        torch.manual_seed(19); rate = RateNet(half_width=args.rate_half_width).to(device)
        legacy_rate_states = draw_states(legacy.states, device)
        r_opt0 = torch.optim.Adam(rate.parameters(), lr=2e-4)
        set_grad(price, False); set_grad(rate, True)
        print("legacy rate bootstrap: 10 epochs, q/sigma_q fixed, raw drift + 50 Z", flush=True)
        hist += phase(price, rate, legacy_rate_states, False, 10, r_opt0, 0, legacy, p,
                      label="legacy_rate", loss_mode="raw", z_weight=50.)
        torch.save({"state_dict": rate.state_dict(), "loss": "raw drift + 50 Z", "r_half_width": args.rate_half_width,
                    "parameters": asdict(p)}, out / "legacy_rate_bootstrap.pt")
    # New optimizer states begin only when the objective changes to the final
    # relative loss; these states then persist across every outer round.
    torch.manual_seed(cfg.seed)
    states = draw_states(cfg.states, device)
    price_opt = torch.optim.Adam(price.parameters(), lr=cfg.lr_q)
    rate_opt = torch.optim.Adam(rate.parameters(), lr=cfg.lr_r)
    for outer in range(start, start + cfg.outer_rounds):
        set_grad(price, True); set_grad(rate, False); hist += phase(price, rate, states, True, cfg.q_epochs, price_opt, outer, cfg, p)
        set_grad(price, False); set_grad(rate, True)
        hist += phase(price, rate, states, False, cfg.r_epochs, rate_opt, outer, cfg, p)
        d = {"outer": outer, "price_state_dict": price.state_dict(), "rate_state_dict": rate.state_dict(),
             "rate_half_width": rate.half_width, "parameters": asdict(p)}
        out.mkdir(parents=True, exist_ok=True); torch.save(d, out / "latest_outer_checkpoint.pt")
        if outer % cfg.save_every == 0:
            snap = out / "snapshots" / f"outer_{outer:03d}"; snap.mkdir(parents=True, exist_ok=True)
            torch.save(d, snap / "checkpoint.pt"); plots(price, rate, hist, p, snap, device)
    plots(price, rate, hist, p, out, device)


if __name__ == "__main__": main()

      
        
        
     
