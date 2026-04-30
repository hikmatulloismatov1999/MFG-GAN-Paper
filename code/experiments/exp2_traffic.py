"""Experiment 2: Traffic flow benchmark (MFG-LWR).

Compares Extended-MFGAN vs MFDGM-style baseline on the traffic flow MFG:
    H(x, rho, p) = 0.5*p^2 - (1-rho)*p
with bilinear coupling R(rho,p) = rho*p, eps=1, f0(rho)=lam*rho.

Metrics: outer convergence, HJB/FP residuals, density snapshots.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hamiltonians import TrafficHamiltonian
import extended_mfgan as emfgan
from metrics import l2_error, hjb_residual, fp_residual, wasserstein2_1d


def congested_initial(x, rho_min=0.1, bump_center=0.3, bump_height=0.4, sigma=0.08):
    """Initial density: uniform background with a density bump (traffic jam)."""
    bump = bump_height * np.exp(-0.5 * ((x - bump_center) / sigma) ** 2)
    rho  = rho_min + bump
    rho /= rho.sum() * (x[1] - x[0])  # normalise to unit mass
    return rho


def run(nu=0.0, lam=0.5, eps=1.0, K_outer=12, n_inner=40):
    print("=" * 60)
    print(f"Experiment 2: Traffic Flow  eps={eps}  lam={lam}  nu={nu}")
    print("=" * 60)

    Nx = 80
    T  = 1.0
    Nt = 80
    x  = np.linspace(0, 1, Nx, endpoint=False)
    dx = x[1] - x[0]
    dt = T / (Nt - 1)

    ham = TrafficHamiltonian(eps=eps, nu=nu, lam=lam)

    rho0 = congested_initial(x)
    g    = np.zeros(Nx)

    # Check GAN-tractable condition
    u_max   = np.max(np.abs(rho0)) * 2       # rough bound
    kappa_th = eps * u_max / lam
    print(f"  Theoretical kappa ~ eps*u_max/lam = {kappa_th:.3f}")
    if kappa_th < 1:
        print("  GAN-tractable: YES")
    else:
        print("  GAN-tractable: MARGINAL/NO (expect slow or no convergence)")

    # ---- Extended-MFGAN ----
    print("\n--- Extended-MFGAN ---")
    res_emfgan = emfgan.run(
        ham, rho0, g, dx, dt,
        K_outer=K_outer, n_inner=n_inner, verbose=True
    )

    # ---- MFDGM baseline ----
    print("\n--- MFDGM baseline ---")
    res_mfdgm = emfgan.mfdgm_run(
        ham, rho0, g, dx, dt,
        K_outer=K_outer, n_inner=n_inner, verbose=True
    )

    # ---- Metrics ----
    rho_em = res_emfgan["rho"]
    phi_em = res_emfgan["phi"]
    rho_mf = res_mfdgm["rho"]
    phi_mf = res_mfdgm["phi"]

    r_hjb_em = hjb_residual(phi_em, rho_em, ham, dx, dt, nu)
    r_fp_em  = fp_residual(rho_em, phi_em, ham, dx, dt, nu)
    r_hjb_mf = hjb_residual(phi_mf, rho_mf, ham, dx, dt, nu)
    r_fp_mf  = fp_residual(rho_mf, phi_mf, ham, dx, dt, nu)

    print("\n--- Final Residuals ---")
    print(f"{'Method':>12}  {'HJB res':>10}  {'FP res':>10}")
    print(f"{'Ext-MFGAN':>12}  {r_hjb_em:>10.3e}  {r_fp_em:>10.3e}")
    print(f"{'MFDGM':>12}  {r_hjb_mf:>10.3e}  {r_fp_mf:>10.3e}")

    # ---- Figure ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel 1: outer convergence
    ax = axes[0]
    ax.semilogy(range(1, len(res_emfgan["outer_errors"]) + 1),
                res_emfgan["outer_errors"], "b-o", label="Extended-MFGAN")
    ax.semilogy(range(1, len(res_mfdgm["outer_errors"]) + 1),
                res_mfdgm["outer_errors"],  "r-s", label="MFDGM")
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$\|\rho^{k+1}-\rho^k\|_{L^2}$")
    ax.set_title("Outer convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: final density — Extended-MFGAN
    ax = axes[1]
    ax.plot(x, rho0,          "k--", lw=1.5, label=r"$\rho_0$ (initial)")
    ax.plot(x, rho_em[Nt//2], "b-",  lw=1.5, label=r"$\rho(T/2)$")
    ax.plot(x, rho_em[-1],    "b:",  lw=1.5, label=r"$\rho(T)$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.set_title("Extended-MFGAN density")
    ax.legend(fontsize=8)

    # Panel 3: fundamental diagram rho vs flux q = rho * u
    ax = axes[2]
    from fd_solver import _grad_x
    p_em  = _grad_x(phi_em[Nt // 2], dx)
    v_em  = ham.grad_p_H_eff(rho_em[Nt // 2], p_em)
    q_em  = rho_em[Nt // 2] * v_em
    ax.plot(rho_em[Nt // 2], q_em, "b.", ms=4, label="Extended-MFGAN")
    ax.set_xlabel(r"density $\rho$")
    ax.set_ylabel(r"flux $q = \rho u$")
    ax.set_title("Fundamental diagram")
    ax.legend()

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "../../figures/exp2_traffic.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"\nFigure saved to {out}")

    return res_emfgan, res_mfdgm


if __name__ == "__main__":
    # Deterministic case (nu=0)
    run(nu=0.0, lam=0.5, eps=1.0)
    # Stochastic case (nu=0.1)
    run(nu=0.1, lam=0.5, eps=1.0)
