# Generative Adversarial Networks for Mean Field Games with Weakly Non-Separable Hamiltonians

This repository contains the LaTeX source, code, and figures for the paper

> **Generative Adversarial Networks for Mean Field Games with Weakly Non-Separable Hamiltonians**
> Hikmatullo Ismatov and Diogo Gomes
> King Abdullah University of Science and Technology (KAUST)

## Summary

Existing deep-learning solvers for Mean Field Games (MFGs) split into two
incompatible families:

- **GAN-based methods** (APAC-Net, MFGAN, deep-MFG) learn the equilibrium
  density distributionally and converge in Wasserstein distance, but only
  apply to **separable** Hamiltonians.
- **Physics-informed methods** (MFDGM) handle arbitrary non-separable
  Hamiltonians but yield only pointwise $L^p$ approximations with no
  explicit convergence rate.

We introduce the class of **weakly non-separable Hamiltonians**,
$H(x,\rho,p) = H_0(x,p) - f_0(x,\rho) + \varepsilon\, R(x,\rho,p)$,
and propose **Extended-MFGAN**, a fixed-point GAN iteration that
freezes $\rho$ within each outer step to make the inner sub-problem
separable, then unfreezes it.

The main result is a Wasserstein-2 contraction theorem with
**explicit and sharp rate** $\kappa = \varepsilon L_R/\lambda$.
We further prove the condition $\varepsilon L_R < \lambda$ is necessary
via an explicit divergence counterexample, and validate the rate
numerically on three benchmarks to four-digit accuracy.

## Repository structure

```
MFG_GAN_Paper/
├── main.tex                   # Master LaTeX file
├── main_neurips.tex           # NeurIPS-style conference version
├── sections/
│   ├── abstract.tex
│   ├── 01_introduction.tex
│   ├── 02_preliminaries.tex
│   ├── 02b_related_work.tex
│   ├── 03_weak_nonseparable.tex
│   ├── 04_algorithm.tex
│   ├── 05_convergence.tex
│   ├── 06_experiments.tex
│   └── 07_conclusion.tex
├── appendix/
│   └── A_proofs.tex
├── references.bib
├── figures/                   # Generated PDFs
├── code/
│   ├── hamiltonians.py        # Quadratic, Traffic, DensityOnly Hamiltonians
│   ├── fd_solver.py           # Lax-Friedrichs HJB + upwind FP + damped fictitious play
│   ├── extended_mfgan.py      # Outer Picard iteration + Picard baseline
│   ├── metrics.py             # L2/W2 distances, HJB/FP residuals
│   └── experiments/
│       ├── exp1_analytic.py        # Separable limit
│       ├── exp2_traffic.py         # Bilinear MFG-LWR
│       └── exp3_phase_transition.py # Density-only sweep
├── PLAN.md
└── README.md
```

## Reproducing the experiments

Requires Python 3.10+ with NumPy and Matplotlib:

```bash
pip install numpy matplotlib

cd code/experiments
python3 exp1_analytic.py        # ~5  seconds, generates figures/exp1_analytic.pdf
python3 exp2_traffic.py         # ~10 seconds, generates figures/exp2_traffic.pdf
python3 exp3_phase_transition.py # ~10 seconds, generates figures/exp3_phase_transition.pdf
```

Each script prints the empirical-vs-theoretical contraction-rate table
to stdout and writes its figure to `figures/`.

## Building the paper

Requires a TeX distribution (TeX Live, MacTeX, or MiKTeX):

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The repo is also configured for Overleaf via GitHub sync.

## Building the NeurIPS version

The NeurIPS conference draft is kept separate from the journal-style paper:

```bash
make neurips
```

or manually:

```bash
pdflatex main_neurips
bibtex main_neurips
pdflatex main_neurips
pdflatex main_neurips
```

Before compiling, place the official `neurips_2026.sty` file from the
NeurIPS author kit in the repository root, next to `main_neurips.tex`.
For Overleaf with GitHub sync, set **Main document** to:

```text
main_neurips.tex
```

Use `main.tex` for the longer journal/preprint manuscript and
`main_neurips.tex` for the anonymous NeurIPS submission.

## Key results

| Section / Figure  | Validates                                      | Outcome                          |
|-------------------|------------------------------------------------|----------------------------------|
| Theorem 5.1       | Wasserstein-2 contraction $\kappa = \varepsilon L_R/\lambda$ | rigorous proof              |
| Proposition 3.2   | Bilinear case rate is tight                    | rigorous proof + numerical confirmation |
| Proposition 3.3   | Necessity of GAN-tractable condition           | rigorous proof + diverging example |
| Experiment 1      | Separable limit (Cor. 5.2)                      | error 0 in 1 step                 |
| Experiment 2      | Bilinear MFG-LWR (two-sided phase transition)  | empirical $\kappa = \varepsilon\|1-\varepsilon\|/\lambda$ to 4 digits |
| Experiment 3      | Density-only phase transition                   | empirical $\kappa = \varepsilon$ to 4 digits across $[0.1, 1.5]$ |

## Citation

If you use this work, please cite:

```bibtex
@unpublished{ismatov2025extendedmfgan,
  title  = {Generative Adversarial Networks for Mean Field Games with
            Weakly Non-Separable Hamiltonians},
  author = {Ismatov, Hikmatullo and Gomes, Diogo},
  note   = {Preprint},
  year   = {2025}
}
```

## Contact

- Hikmatullo Ismatov: `hikmatullo.ismatov@kaust.edu.sa`
- Diogo Gomes:        `diogo.gomes@kaust.edu.sa`
