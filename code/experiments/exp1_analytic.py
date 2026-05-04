"""Experiment 1: Separable limit (eps = 0).

Validates the separable-limit sanity check: when eps = 0, the outer wrapper
reduces to a single inner solve because H^k = H_0 is independent
of rho_bar.

We use the same linearised setup as Experiments 2-3 (ergodic MFG on the
torus) and check that for eps = 0 the linearised outer map is the zero map,
so any perturbation collapses to zero in a single iteration.

For eps > 0 we recover the geometric rate kappa = eps L_R / lambda.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def linearised_step(da_bar, eps, LR, lam, nu, k_freqs):
    """Mode-wise factor (density-only coupling, c.f. exp3)."""
    da_hat = np.fft.fft(da_bar)
    factors = np.zeros_like(da_hat, dtype=complex)
    for i, k in enumerate(k_freqs):
        if k == 0:
            factors[i] = 0.0
        else:
            denom = lam + (nu ** 2) * (2 * np.pi * k) ** 2 / max(lam, 1e-12)
            factors[i] = (eps * LR) / denom
    return np.real(np.fft.ifft(factors * da_hat))


def run():
    print("=" * 60)
    print("Experiment 1: Separable limit  (eps = 0  =>  one-step convergence)")
    print("=" * 60)

    lam = 1.0
    LR  = 1.0
    nu  = 0.0
    Nx = 64
    K  = 8
    eps_values = [0.0, 0.1, 0.3, 0.5]

    x = np.linspace(0, 1, Nx, endpoint=False)
    dx = x[1] - x[0]
    k_freqs = np.fft.fftfreq(Nx, d=dx)

    # initial perturbation: bigger than usual, to see decay clearly
    delta0 = 0.2 * np.cos(2 * np.pi * x)

    print(f"\nInitial ||delta_rho||_L2 = {float(np.sqrt(np.mean(delta0**2))):.4f}")
    print(f"\n{'eps':>6}  {'kappa_th':>10}  {'iter1':>13}  {'iter K':>13}")
    print("-" * 50)

    err_curves = {}
    for eps in eps_values:
        delta = delta0.copy()
        errs = [float(np.sqrt(np.mean(delta ** 2)))]
        for _ in range(K):
            delta = linearised_step(delta, eps, LR, lam, nu, k_freqs)
            errs.append(float(np.sqrt(np.mean(delta ** 2))))
        err_curves[eps] = errs
        kappa_th = eps * LR / lam
        print(f"{eps:>6.2f}  {kappa_th:>10.3f}  "
              f"{errs[1]:>13.4e}  {errs[-1]:>13.4e}")

    # Sanity check: for eps=0, errs[1] should be machine zero
    eps0_err1 = err_curves[0.0][1]
    if eps0_err1 < 1e-12:
        print(f"\n[OK] eps=0 collapses to {eps0_err1:.2e} in ONE outer iteration.")
        print("     The outer wrapper reduces to one separable inner solve.")
    else:
        print(f"\n[FAIL] eps=0 should give zero in 1 iter, got {eps0_err1:.2e}")

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    cmap = plt.cm.plasma(np.linspace(0.1, 0.85, len(eps_values)))
    for i, eps in enumerate(eps_values):
        errs = err_curves[eps]
        # Avoid log(0) for the eps=0 case
        plot_errs = np.maximum(np.array(errs), 1e-16)
        ax.semilogy(range(len(plot_errs)), plot_errs,
                    color=cmap[i], lw=1.8, marker="o", ms=5,
                    label=rf"$\varepsilon={eps}$ ($\kappa={eps:.1f}$)")
    ax.set_xlabel("outer iteration $k$")
    ax.set_ylabel(r"$\|\rho^k - \rho^*\|_{L^2}$")
    ax.set_title("Separable-limit outer-map convergence at $\\varepsilon=0$")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.set_ylim(1e-16, 1)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__),
                        "../../figures/exp1_analytic.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"\nFigure saved to {out}")


if __name__ == "__main__":
    run()
