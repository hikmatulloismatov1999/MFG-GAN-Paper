# Project Checklist — Extended-MFGAN

> Last updated: see git log.  Always update this file when you finish a task.

---

## ✅ DONE — Core paper

### Theory
- [x] Definition of weakly non-separable Hamiltonians (Def 3.1)
- [x] Definition of GAN-tractable regime, simplified (Def 3.2) and full (Def 5.1)
- [x] Extended-MFGAN algorithm (Alg 4.1)
- [x] Bounded-momentum Assumption 5.4
- [x] Non-vacuum (`ρ_min > 0`) condition
- [x] Lasry–Lions stability Lemma 5.5 with 6-step proof
- [x] **Main Theorem 5.1 (`L²` contraction)** with rate `κ_ε`
- [x] Corollary 5.2 (`W₂` convergence at rate `κ^(k/2)`)
- [x] Proposition 3.2 (bilinear sharpness, correctly proved for `R = ρp`)
- [x] Proposition 3.3 (necessity counterexample on the torus)
- [x] Corollary 5.6 (density-only sub-case)
- [x] Corollary 5.7 (bilinear traffic flow)
- [x] Proposition 5.8 (inexact inner solves)
- [x] Equivalence of `L²` and `W₂` convergence (Appendix A.4)

### Writing
- [x] Abstract, introduction, conclusion all aligned with corrected math
- [x] Related Work section positioning vs APAC-Net, MFGAN, MFDGM, policy iteration, fictitious play
- [x] Algorithm box with cost analysis
- [x] All cross-references resolve
- [x] All bibliography entries cited
- [x] No `\todo` / `TODO` markers remain
- [x] No em-dashes (per author preference)
- [x] AMS Subject Codes added (49N80, 49L12, 49L25, 35Q89, 65M12, 68T07, 91A16)
- [x] SIAM keywords block

### Code
- [x] `code/hamiltonians.py` — Quadratic, Traffic, DensityOnly
- [x] `code/fd_solver.py` — Lax-Friedrichs HJB + upwind FP + damped fictitious play
- [x] `code/extended_mfgan.py` — outer Picard iteration + Picard baseline
- [x] `code/metrics.py` — L², W₂, residuals
- [x] `code/experiments/exp1_analytic.py` — separable limit (runs)
- [x] `code/experiments/exp2_traffic.py` — bilinear traffic (runs)
- [x] `code/experiments/exp3_phase_transition.py` — density-only sweep (runs)
- [x] All experiments match Section 6 tables to 4 digits

### Reviewer-issue fixes (deep review pass)
- [x] Issue 1: `W₂` rate corrected from `κ^k` to `κ^(k/2)`
- [x] Issue 2: bounded-momentum assumption added
- [x] Issue 3: factor-3 condition aligned between main text and appendix
- [x] Issue 4: Prop 3.2 proof rewritten for actual bilinear case
- [x] Issue 5: parameterised traffic family `H_ε` defined
- [x] Issue 6: Section 6 reframed as "linearised rate validation"
- [x] Minor: `huang2019mfg` reference corrected
- [x] Minor: exp3 `ε=1` labeled "threshold" not "diverging"
- [x] Minor: "GAN methods provably outperform" softened

### Submission package
- [x] `main_sicon.tex` (SIAM article class)
- [x] `cover_letter_sicon.tex` (with suggested editors / referees)
- [x] `SICON_SUBMISSION.md` (checklist + portal instructions)
- [x] `README.md` (repo overview)
- [x] `PROJECT_SUMMARY.md` (this brief)
- [x] All committed to GitHub `main` branch

### Presentations
- [x] 30-min presentation about our paper (`presentation.tex` + `speech.tex`)
- [x] Teaching presentation about MFDGM + our extension (`presentation_mfdgm.tex` + `speech_mfdgm.tex`, 38 slides, ~45 min)
- [x] Metropolis theme, KAUST colour palette, TikZ diagrams, tcolorbox highlight boxes

---

## 🟡 IN PROGRESS / OPEN — Before SICON submission

### Math
- [ ] **Diogo Gomes signoff** on:
  - the rewritten Proposition 3.2 (bilinear sharpness for `R=ρp`)
  - the new bounded-momentum Assumption 5.4
  - the factor-3 second GAN-tractable condition
  - the reframed Section 6 ("linearised rate validation")
- [ ] **Lavigne–Pfeiffer 2023** literature check: confirm whether they prove a *rate* for non-separable MFGs (affects our "first explicit rate" claim)
- [ ] **Carmona–Laurière 2021** check: their convergence rates for ergodic separable case; confirm our claim is for the non-separable case
- [ ] Optional: tighten the factor-3 to factor-1 via a sharper Young's split (small math project)

### Code / experiments
- [ ] Optional small finite-difference reference experiment on the bilinear MFG-LWR at one or two ε values, showing nonlinear behaviour matches the linearised prediction qualitatively (pre-empts likely referee request)
- [ ] **Full neural-network implementation** of the inner GAN (would address the main "what is not done here" caveat). Substantial engineering — defer to revision phase or v2.

### Polish
- [ ] Compile `main_sicon.tex` in Overleaf and verify no warnings
- [ ] Spell-check the entire PDF
- [ ] Run Turnitin / iThenticate plagiarism check via KAUST library
- [ ] Title decision: keep "Generative Adversarial Networks for…" or change to a more conservative title (e.g., "A fixed-point GAN scheme for…")?  Discuss with Diogo.

### Submission
- [ ] arXiv preprint (math.OC primary, cs.LG cross-list)
- [ ] SICON account creation
- [ ] SICON portal upload: PDF + source + cover letter
- [ ] Paste abstract / keywords / MSC codes into portal forms
- [ ] Submit, record submission ID and date

---

## 🔵 FUTURE WORK — Listed in Section 7

- [ ] Full neural-network implementation of the inner GAN solver
- [ ] Pipeline-level benchmark Extended-MFGAN vs MFDGM
- [ ] High-dimensional scaling (`d ≥ 50`)
- [ ] Hybrid algorithm: GAN inner solver + MFDGM fallback above the threshold
- [ ] Extension to mean field control problems
- [ ] Application to multi-agent reinforcement learning
- [ ] Application to congestion-type engineering systems

---

## 🟣 OPTIONAL NEXT-PAPER IDEAS (post-publication)

- [ ] Generalise to **time-fractional** HJB / FP (active topic at SICON)
- [ ] Extend to **major-minor MFGs** (where one big player breaks symmetry)
- [ ] **Anderson acceleration** of the outer Picard iteration
- [ ] **Stochastic** inner solves: analyse the bias introduced by SGD / Adam in the inner solver
- [ ] Application to **electricity markets** or **financial market making** (where MFGs are popular)

---

## How to update this checklist

- When you finish a task, change `[ ]` → `[x]` and add a date or git SHA.
- When you discover a new open item, add it under the right section.
- When in doubt, commit and push so the checklist is the single source of truth.
- Run `git log --oneline -20` to see the most recent activity if you lose context.
