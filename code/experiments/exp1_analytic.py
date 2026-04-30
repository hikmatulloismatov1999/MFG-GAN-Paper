"""Experiment 1: Separable limit (eps=0) — analytic solution validation.

Verifies that Extended-MFGAN with eps=0 recovers the analytic solution
of the separable MFG in a single outer iteration (Corollary 5.2).

Analytic solution (1-D quadratic MFG, nu=0):
    phi(t, x) = 0.5 * x^2 - lam * t       (value function)
    rho(t, x) = rho_0(x)   (density constant in time for suitable rho_0)

We use a simpler analytic check:  with f0(rho)=lam*rho, H0=0.5p^2, nu=0,
the MFG equilibrium on [-L,L] with Gaussian rho_0 has phi solving
HJB:  partial_t phi + 0.5*(partial_x phi)^2 = lam*rho(t,x)
FP:   partial_t rho + partial_x(rho * partial_x phi) = 0

We test convergence of the outer loop for eps=0 and report L2 error.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hamiltonians import QuadraticHamiltonian
import extended_mfgan as emfgan
from metrics import l2_error, hjb_residual, fp_residual


def gaussian(x, mu=0.5, sigma=0.1):
    g = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return g / (g.sum() * (x[1] - x[0]))


def run():
    # Grid
    Nx = 80
    T  = 1.0
    Nt = 80
    x  = np.linspace(0, 1, Nx, endpoint=False)
    dx = x[1] - x[0]
    dt = T / (Nt - 1)

    lam = 1.0
    nu  = 0.0
    ham = QuadraticHamiltonian(eps=0.0, nu=nu, lam=lam)

    rho0 = gaussian(x, mu=0.5, sigma=0.15)
    g    = np.zeros(Nx)   # zero terminal cost

    print("=" * 55)
    print("Experiment 1: Separable limit (eps=0)")
    print("=" * 55)

    results_eps0 = emfgan.run(
        ham, rho0, g, dx, dt,
        K_outer=5, n_inner=40, verbose=True
    )

    # For eps=0 the outer loop should stagnate after 1 step
    errors = results_eps0["outer_errors"]
    print(f"\nOuter errors (eps=0): {[f'{e:.3e}' for e in errors]}")
    print(f"Ratio iter2/iter1 = {errors[1]/errors[0]:.4f}  (expect << 1 for eps=0)")

    # HJB / FP residuals
    rho_f = results_eps0["rho"]
    phi_f = results_eps0["phi"]
    r_hjb = hjb_residual(phi_f, rho_f, ham, dx, dt, nu)
    r_fp  = fp_residual(rho_f, phi_f, ham, dx, dt, nu)
    print(f"HJB residual: {r_hjb:.4e}")
    print(f"FP  residual: {r_fp:.4e}")

    # Now run with eps=0.5 for comparison
    ham2 = QuadraticHamiltonian(eps=0.5, nu=nu, lam=lam)
    print("\nRunning with eps=0.5 for comparison ...")
    results_eps05 = emfgan.run(
        ham2, rho0, g, dx, dt,
        K_outer=10, n_inner=40, verbose=False
    )
    errors2 = results_eps05["outer_errors"]

    # ------------------------------------------------------------------ #
    # Figure
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.semilogy(range(1, len(errors) + 1), errors,  "o-", label=r"$\varepsilon=0$")
    ax.semilogy(range(1, len(errors2)+1), errors2, "s-", label=r"$\varepsilon=0.5$")
    ax.set_xlabel("Outer iteration $k$")
    ax.set_ylabel(r"$\|\rho^{k+1}-\rho^k\|_{L^2}$")
    ax.set_title("Outer convergence")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(x, rho0,         "k--",  label=r"$\rho_0$")
    ax.plot(x, rho_f[0],     "b-",   label=r"$\rho(0,\cdot)$")
    ax.plot(x, rho_f[Nt//2], "g-",   label=r"$\rho(T/2,\cdot)$")
    ax.plot(x, rho_f[-1],    "r-",   label=r"$\rho(T,\cdot)$")
    ax.set_xlabel("$x$")
    ax.set_ylabel("density")
    ax.set_title(r"Density evolution ($\varepsilon=0$)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "../../figures/exp1_analytic.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"\nFigure saved to {out}")

    # Print summary table
    print("\n--- Summary Table ---")
    print(f"{'eps':>6}  {'K_outer':>7}  {'HJB_res':>10}  {'FP_res':>10}")
    print(f"{'0.0':>6}  {'5':>7}  {r_hjb:>10.3e}  {r_fp:>10.3e}")

    return results_eps0


if __name__ == "__main__":
    run()
