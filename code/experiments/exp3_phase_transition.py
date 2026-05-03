"""Experiment 3: Phase transition at eps * L_R = lambda.

Two complementary panels:
  (a) LINEARISED iteration on the ergodic MFG on the torus.  This is
      exactly the setting of Proposition 3.3 (necessity counterexample),
      where the outer fixed-point map has linear part T(δρ̄) = (εL_R/λ)·δρ̄.
      Iterating this map shows κ_emp = κ_th = ε exactly (up to a small
      diffusion correction when ν > 0).

  (b) Outer-error curves: ||δρ^k|| vs k for several ε, showing geometric
      decay below threshold and growth above.

Sweeps ε in [0.1, 1.5] with λ = L_R = 1.  Threshold ε* = 1.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def linearised_map_apply(da_bar, eps, LR, lam, nu, k_modes, dx):
    """Apply the linearised outer fixed-point map T to da_bar.

    Working in Fourier space on the periodic 1-D torus: at the uniform
    equilibrium rho* = 1, the stationary linearised system gives mode-wise
        δa_k  =  (eps * LR / (lam + nu^2 * (2 pi k)^2 / lam)) * δā_k
    For nu = 0 the factor is exactly eps*LR/lam.  Small nu introduces a
    O(nu^2) correction at high modes, which we keep for fidelity.
    """
    # FFT
    da_hat = np.fft.fft(da_bar)
    factors = np.zeros_like(da_hat)
    for i, k in enumerate(k_modes):
        if k == 0:
            factors[i] = 0.0          # zero-mean preserved
        else:
            denom = lam + (nu ** 2) * (2 * np.pi * k) ** 2 / max(lam, 1e-12)
            factors[i] = (eps * LR) / denom
    return np.real(np.fft.ifft(factors * da_hat))


def run_linearised(eps, LR=1.0, lam=1.0, nu=0.0, K=10, Nx=64, amp=0.05):
    """Iterate the linearised outer map K times for given eps.
    Returns (errors, kappa_emp).
    """
    x = np.linspace(0, 1, Nx, endpoint=False)
    dx = x[1] - x[0]
    k_modes = np.fft.fftfreq(Nx, d=dx)

    # Initial perturbation: low-mode cosine
    delta = amp * np.cos(2 * np.pi * x)

    errors = [float(np.sqrt(np.mean(delta ** 2)))]
    for _ in range(K):
        delta = linearised_map_apply(delta, eps, LR, lam, nu, k_modes, dx)
        errors.append(float(np.sqrt(np.mean(delta ** 2))))

    # Empirical contraction (median of consecutive ratios, skipping initial)
    ratios = [errors[k + 1] / errors[k]
              for k in range(len(errors) - 1) if errors[k] > 1e-15]
    kappa_emp = float(np.median(ratios)) if ratios else float("nan")
    return errors, kappa_emp


def run():
    print("=" * 60)
    print("Experiment 3: Phase Transition (linearised ergodic MFG)")
    print("=" * 60)

    LR = 1.0
    lam = 1.0
    nu  = 0.0       # exact rate; set to 0.05 for diffusion correction demo
    Nx = 64
    K  = 12
    eps_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]

    print(f"\nParameters: lam={lam}, LR={LR}, nu={nu}, threshold eps*=1.0\n")
    print(f"{'eps':>6}  {'kappa_th':>10}  {'kappa_emp':>10}  {'final_err':>13}  status")
    print("-" * 60)

    kappa_th_arr  = []
    kappa_emp_arr = []
    err_curves    = {}

    for eps in eps_values:
        errors, kappa_emp = run_linearised(
            eps, LR=LR, lam=lam, nu=nu, K=K, Nx=Nx
        )
        kappa_th = eps * LR / lam
        if   kappa_th < 1 - 1e-9: status = "converging"
        elif kappa_th > 1 + 1e-9: status = "diverging"
        else:                     status = "threshold"
        kappa_th_arr.append(kappa_th)
        kappa_emp_arr.append(kappa_emp)
        err_curves[eps] = errors
        print(f"{eps:>6.2f}  {kappa_th:>10.3f}  {kappa_emp:>10.3f}  "
              f"{errors[-1]:>13.4e}  {status}")

    # ----------------------------------------------------------------- #
    # Figure
    # ----------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    eps_arr   = np.array(eps_values)
    ax.plot(eps_arr, kappa_th_arr,
            "k--", lw=1.5, label=r"theory $\kappa = \varepsilon L_R/\lambda$")
    ax.plot(eps_arr, kappa_emp_arr,
            "bo-", ms=8, lw=1.5, label=r"empirical $\hat\kappa$")
    ax.axvline(x=1.0, color="red", lw=1, ls=":",
               label=r"$\varepsilon^* = \lambda/L_R = 1$")
    ax.axhline(y=1.0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel(r"coupling strength $\varepsilon$")
    ax.set_ylabel(r"contraction rate $\kappa$")
    ax.set_title("Phase transition at $\\varepsilon^*=1$")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(eps_values)))
    for i, eps in enumerate(eps_values):
        errs = err_curves[eps]
        ax.semilogy(range(len(errs)), errs,
                    color=cmap[i], lw=1.6,
                    label=rf"$\varepsilon={eps:.1f}$")
    ax.set_xlabel("outer iteration $k$")
    ax.set_ylabel(r"$\|\rho^k - \rho^*\|_{L^2}$")
    ax.set_title("Geometric decay/growth")
    ax.legend(fontsize=7, ncol=2, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__),
                        "../../figures/exp3_phase_transition.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"\nFigure saved to {out}")

    return eps_values, kappa_emp_arr, kappa_th_arr


if __name__ == "__main__":
    run()
