"""Experiment 2: Bilinear traffic flow benchmark (MFG-LWR).

Validates Proposition 3.2 (bilinear exactness): for R = rho*p with
H_0 = 0.5 p^2 - p, the linearised outer fixed-point map at the uniform
equilibrium has operator norm exactly  eps * L_R(p*) / lambda,
where L_R(p*) = |p*| is the Lipschitz constant of R in rho evaluated
at the equilibrium momentum.

Equilibrium analysis (stationary, periodic):
  Pick the no-drift equilibrium grad_p H_eff = 0:
    p* + eps*rho* - 1 = 0  with rho* = 1  =>  p* = 1 - eps.
  Then L_R(p*) = |1 - eps| and the linearised contraction rate is
    kappa_th = eps * |1 - eps| / lambda.

Threshold:  kappa_th = 1  <=>  eps*(1-eps)/lambda = 1
For lambda = 0.5:  eps_*(1-eps_*) = 0.5  =>  eps_* = (1 +/- sqrt(1-2))/2
                  no real root, so kappa_th < 0.5 for eps in [0,1].
For lambda = 0.2:  eps_*(1-eps_*) = 0.2  =>  eps_* = (1 - sqrt(0.2))/2
                  ~ 0.276 OR  (1 + sqrt(0.2))/2 ~ 0.724  (one transition each).

We use lambda = 0.2 to expose the phase transition within eps in (0, 1).
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def linearised_traffic_step(da_bar, eps, lam, nu, k_freqs):
    """Apply the linearised outer map for the bilinear MFG R=rho*p,
    H_0 = 0.5 p^2 - p, at the no-drift equilibrium (rho*=1, p*=1-eps).

    Mode-wise factor (derived in the paper notes):
        a_k = eps * [(1 - eps) - i * nu * (2 pi k)] / (lambda - nu^2 (2 pi k)^2)
              * a_bar_k.

    Returns the real-space update of da_bar.
    """
    da_hat = np.fft.fft(da_bar)
    factors = np.zeros_like(da_hat, dtype=complex)
    for i, k in enumerate(k_freqs):
        if k == 0:
            factors[i] = 0.0           # zero mean preserved
        else:
            kk = 2 * np.pi * k
            num = eps * ((1.0 - eps) - 1j * nu * kk)
            den = lam - (nu ** 2) * (kk ** 2)
            factors[i] = num / den
    return np.real(np.fft.ifft(factors * da_hat))


def run_linearised(eps, lam=0.2, nu=0.0, K=12, Nx=64, amp=0.05):
    x = np.linspace(0, 1, Nx, endpoint=False)
    dx = x[1] - x[0]
    k_freqs = np.fft.fftfreq(Nx, d=dx)

    delta = amp * np.cos(2 * np.pi * x)
    errors = [float(np.sqrt(np.mean(delta ** 2)))]
    for _ in range(K):
        delta = linearised_traffic_step(delta, eps, lam, nu, k_freqs)
        errors.append(float(np.sqrt(np.mean(delta ** 2))))

    ratios = [errors[k + 1] / errors[k]
              for k in range(len(errors) - 1) if errors[k] > 1e-15]
    kappa_emp = float(np.median(ratios)) if ratios else float("nan")
    return errors, kappa_emp


def run():
    print("=" * 60)
    print("Experiment 2: Bilinear traffic flow (MFG-LWR)")
    print("=" * 60)

    lam = 0.2
    nu  = 0.0
    Nx  = 64
    K   = 12
    eps_values = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    # Theoretical thresholds where kappa_th = eps*(1-eps)/lam = 1:
    disc = 1.0 - 4.0 * lam
    if disc > 0:
        eps_lo = 0.5 * (1 - np.sqrt(disc))
        eps_hi = 0.5 * (1 + np.sqrt(disc))
        print(f"Predicted thresholds for lam={lam}: "
              f"eps_lo={eps_lo:.3f}, eps_hi={eps_hi:.3f}")
    else:
        print(f"For lam={lam} there is no threshold in (0,1).")

    print(f"\n{'eps':>6}  {'kappa_th':>10}  {'kappa_emp':>10}  "
          f"{'final_err':>13}  status")
    print("-" * 62)

    kappa_th_arr  = []
    kappa_emp_arr = []
    err_curves    = {}

    for eps in eps_values:
        errors, kappa_emp = run_linearised(
            eps, lam=lam, nu=nu, K=K, Nx=Nx
        )
        kappa_th = eps * abs(1.0 - eps) / lam
        status = "converging" if kappa_th < 1 else ("threshold" if abs(kappa_th-1)<1e-2 else "diverging")
        kappa_th_arr.append(kappa_th)
        kappa_emp_arr.append(kappa_emp)
        err_curves[eps] = errors
        print(f"{eps:>6.2f}  {kappa_th:>10.3f}  {kappa_emp:>10.3f}  "
              f"{errors[-1]:>13.4e}  {status}")

    # ----------------------------------------------------------------- #
    # Figure
    # ----------------------------------------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    eps_dense = np.linspace(0.01, 0.99, 200)
    kappa_curve = eps_dense * np.abs(1 - eps_dense) / lam

    ax = axes[0]
    ax.plot(eps_dense, kappa_curve,
            "k--", lw=1.5, label=r"theory $\kappa = \varepsilon|1-\varepsilon|/\lambda$")
    ax.plot(eps_values, kappa_emp_arr,
            "ro", ms=8, label=r"empirical $\hat\kappa$")
    ax.axhline(y=1.0, color="gray", lw=0.7, ls="--")
    if disc > 0:
        ax.axvline(x=eps_lo, color="red", lw=1, ls=":", alpha=0.6)
        ax.axvline(x=eps_hi, color="red", lw=1, ls=":", alpha=0.6,
                   label=fr"thresholds $\varepsilon^*\in\{{{eps_lo:.2f},{eps_hi:.2f}\}}$")
    ax.set_xlabel(r"coupling strength $\varepsilon$")
    ax.set_ylabel(r"contraction rate $\kappa$")
    ax.set_title(rf"Bilinear $R=\rho p$, $\lambda={lam}$")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.95, len(eps_values)))
    for i, eps in enumerate(eps_values):
        errs = err_curves[eps]
        ax.semilogy(range(len(errs)), errs,
                    color=cmap[i], lw=1.4,
                    label=rf"$\varepsilon={eps}$")
    ax.set_xlabel("outer iteration $k$")
    ax.set_ylabel(r"$\|\rho^k - \rho^*\|_{L^2}$")
    ax.set_title("MFG-LWR convergence curves")
    ax.legend(fontsize=6, ncol=2, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__),
                        "../../figures/exp2_traffic.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches="tight")
    print(f"\nFigure saved to {out}")

    return eps_values, kappa_emp_arr, kappa_th_arr


if __name__ == "__main__":
    run()
