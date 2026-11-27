# HGWO-VNS for the Capacitated Vehicle Routing Problem (CVRP)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![CVRPLIB Audited](https://img.shields.io/badge/CVRPLIB%20Audit-86%2F86%20Exact%20Match-brightgreen.svg)](https://galgos.inf.puc-rio.br/cvrplib/en/cvrp)
[![Execution](https://img.shields.io/badge/Execution-Single--Core%20Reproducible-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

Official Python implementation and reproducibility archive for the research article:

> **HGWO-VNS: An Embedded Hybrid Discrete Grey Wolf Optimizer with Variable Neighborhood Descent and Optimal Split Decoding for the Capacitated Vehicle Routing Problem**  
> *Mohammed El Amine Meziane* — Submitted to the [International Journal of Automotive Science and Technology (IJASTECH)](https://dergipark.org.tr/en/pub/ijastech/writing-rules) (`e-ISSN: 2587-0963`)

## Overview

**HGWO-VNS** is a discrete Hybrid Grey Wolf Optimizer designed for the static Capacitated Vehicle Routing Problem (CVRP) in automotive logistics and freight distribution. Rather than relying solely on permutation crossover and mutation, the solver embeds five synergistic components into a single-core Python framework:

1. **Multi-Elite Discrete Recombination ($\alpha, \beta, \delta$)**: Combines Order Crossover (OX) and Route-Based Crossover (RBX) guided by the linearly annealed Grey Wolf control parameter $a \in [2, 0]$.
2. **Fleet-Bounded 2D Optimal `Split` Decoding**: Computes the exact minimum-cost route partition for any fixed giant-tour permutation $\pi$ subject to fleet bound $K$ in $O(K \cdot n \cdot B)$ time (with 1D unbounded fallback on Golden 9–20 per [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/en/cvrp) rules).
3. **Selective Five-Operator Variable Neighborhood Descent (VND)**: Intensifies elite solutions using 2-opt, Or-opt, Relocate, Swap, and 2-opt\*, accelerated by granular nearest-neighbor candidate lists ($G = 28$) on large-scale instances ($240 \le n \le 483$).
4. **Heuristic BFD Fallback & Explicit Feasible-Incumbent Tracking ($X^*$)**: Applies a three-stage Best-Fit Decreasing repair when `Split` or local moves exceed $K$ routes, while independently tracking the best strictly feasible incumbent $X^*$ ($|R| \le K$).
5. **Shaw/Worst Destroy & Regret-2 Repair**: Escapes local-optima plateaus during stagnation without resetting search memory.

All **86 pure CVRP instances** reported in the manuscript are independently audited from raw `.vrp` coordinates and exported in standard CVRPLIB `.sol` format (**86/86 feasible, 86/86 exact cost match**).

## Repository Structure

```text
.
├── verify_all_86.py                 # Standalone independent verifier for all 86 .sol files
├── audit_86_report.csv              # Full 86-row verification log (86/86 OK & exact match)
├── code/
│   ├── hgwo_vns_cvrp.py             # Core HGWO-VNS solver (OX/RBX, 2D Split, VND, BFD, Regret-2)
│   ├── habc_baseline.py             # Same-hardware equal-iteration PyHABC baseline
│   ├── cvrp_validator.py            # Independent CVRPLIB feasibility & cost auditor
│   ├── run_all_set_p.py             # Augerat Set P benchmark runner (24 instances)
│   ├── benchmark_sets_a_b.py        # Augerat Sets A (27) & B (23) benchmark runner
│   ├── benchmark_golden.py          # Large-scale Golden 9–20 runner (granular G = 28)
│   ├── ablation_and_stats.py        # 4-variant ablation study (V1–V4) & Friedman/Wilcoxon tests
│   ├── run_sensitivity.py           # Granular neighborhood size (G) sensitivity sweep
│   └── test_hgwo_vns.py             # Smoke and unit tests for solver & validator
├── benchmarks/
│   ├── P/                           # 24 raw CVRPLIB Augerat Set P .vrp files
│   ├── A/                           # 27 raw CVRPLIB Augerat Set A .vrp files
│   ├── B/                           # 23 raw CVRPLIB Augerat Set B .vrp files
│   └── Golden/                      # 12 raw CVRPLIB Golden 9–20 .vrp files (pure CVRP)
├── solutions_sol/
│   ├── P/                           # 24 audited CVRPLIB .sol route certificates
│   ├── A/                           # 27 audited CVRPLIB .sol route certificates
│   ├── B/                           # 23 audited CVRPLIB .sol route certificates
│   └── Golden/                      # 12 audited CVRPLIB .sol route certificates
└── results/
    ├── ENVIRONMENT_AND_PROTOCOL.txt # Hardware, software versions, and paired-seed protocol
    ├── same_hardware_hgwo_vs_habc.csv
    ├── resultats_set_p_complet.csv
    ├── resultats_set_a_complet.csv
    ├── resultats_set_b_complet.csv
    ├── resultats_golden_cvrp.csv
    ├── audit_routes_set_p.csv
    └── parameter_sensitivity_results.csv
```

## Requirements & Installation

All reported experiments were executed single-threaded on **1 CPU core** using **Python 3.11.2**, **NumPy 2.1.3**, and **SciPy 1.15.2**.

```bash
git clone https://github.com/[GitHub-Username]/[Repository-Name].git
cd [Repository-Name]
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib
```

## Quick Verification (All 86 CVRPLIB Certificates)

To independently verify every exported `.sol` route certificate against its raw `.vrp` instance and cross-check every cost against the manuscript CSV tables:

```bash
python3 verify_all_86.py
```

Expected console summary:

```text
Total audited : 86/86
Feasible (OK) : 86/86
Exact matches : 86/86
```

Each instance is checked for: (1) exact partition of customers $\{1, \dots, n\}$ with depot $0$ excluded from route bodies, (2) strict capacity compliance $\sum_{i \in r} q_i \le Q$, (3) strict fleet bound $|R| \le K$ on Augerat Sets A, B, and P (unbounded fleet on Golden 9–20 per [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/en/cvrp)), and (4) exact independent cost recomputation from raw coordinates.

## Reproducing Experiments

Run any experimental suite from the repository root (single-core execution):

```bash
# 1. Quick unit & feasibility tests
python3 code/test_hgwo_vns.py

# 2. Augerat Set P benchmark & same-hardware PyHABC comparison (5 paired seeds)
python3 code/run_all_set_p.py
python3 code/habc_baseline.py

# 3. Augerat Sets A (27 instances) and B (23 instances)
python3 code/benchmark_sets_a_b.py

# 4. Large-scale Golden 9–20 benchmark (240 <= n <= 483, granular G = 28)
python3 code/benchmark_golden.py

# 5. Four-stage component ablation (V1–V4) & Friedman/Wilcoxon tests (60 paired runs)
python3 code/ablation_and_stats.py

# 6. Granular candidate-list size sensitivity analysis
python3 code/run_sensitivity.py
```

## Summary of Audited Benchmark Results

All results below are single-core Python 3.11.2 outcomes verified by `verify_all_86.py` against official [CVRPLIB](https://galgos.inf.puc-rio.br/cvrplib/en/cvrp) Best Known Solutions (BKS):

| Benchmark Family | Instances | Distance Metric | BKS Matched | Mean BKS Gap | Mean Single-Core Time | Audit Status |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Augerat Set P** (Table 4) | 23 (+1 extra) | `EUC_2D` rounded | 15 / 23 | 0.50% (0.26% on 20 comparable) | 1.09 s | 24 / 24 OK |
| **Augerat Set A** | 27 | `EUC_2D` rounded | 17 / 27 | 0.22% | 0.91 s | 27 / 27 OK |
| **Augerat Set B** | 23 | `EUC_2D` rounded | 14 / 23 | 0.23% | 1.20 s | 23 / 23 OK |
| **Golden 9–20** ($240 \le n \le 483$) | 12 | `EUC_2D` unrounded | — | 2.88% (+7.63% vs. CW) | 29.03 s | 12 / 12 OK |
| **Total Pure CVRP** | **86** | CVRPLIB standard | **46 / 73** (Augerat) | — | — | **86 / 86 OK** |

Key comparative highlights:

- **vs. Literature HABC** ([Djebbar & Kemmar, 2025](https://dergipark.org.tr/en/download/article-file/4714960)): **7.61%** mean distance reduction across the 20 comparable Set P instances (max **26.38%** on `P-n101-k4`, Wilcoxon $W = 0.0, p = 6.57 \times 10^{-5}$ one-sided).
- **vs. Same-Hardware PyHABC**: **32.59%** mean distance reduction under identical paired seeds $\{17, 34, 51, 68, 85\}$ and iteration budgets ($N_{\text{pop}} = 24, T_{\max} = 120, P_{\text{stop}} = 25$; Wilcoxon $p = 2.00 \times 10^{-5}$).
- **vs. Recent Discrete Swarm Hybrids** ([Fang et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11390905/)): Across 40 common Augerat instances (20 Set A + 20 comparable Set P), HGWO-VNS achieves **0.18%** mean gap and **28/40 BKS**, outperforming Hybrid FA (`1.41%`, $p = 1.55 \times 10^{-5}$) and DAEO (`0.54%`, $p = 1.13 \times 10^{-2}$), and matching DWHO (`0.26%`, $p = 0.138$) and DSRACO (`0.17%`, $p = 0.777$).

## Experimental Protocol Notes

- **Distance Conventions**: Augerat Sets A, B, and P use TSPLIB nearest-integer `EUC_2D` (`int(hypot(dx, dy) + 0.5)`). Golden 9–20 instances use unrounded double-precision Euclidean distances accumulated to the total tour cost, with Golden 1–8 excluded because they impose route-duration limits (DCVRP).
- **Blind Stopping Criterion**: No BKS value is ever used to stop the search; HGWO-VNS terminates strictly after $P_{\text{stop}} = 25$ consecutive iterations without global improvement (or at $T_{\max}$).
- **No Cross-Hardware Speedup Claims**: Published C++ runtimes from [Djebbar & Kemmar (2025)](https://dergipark.org.tr/en/download/article-file/4714960) are reported solely for context; all runtime comparisons are restricted to same-hardware single-core Python runs.

## Citation

If you use this code or the audited CVRPLIB route certificates in your research, please cite:

```bibtex
@article{hgwo_vns_cvrp_[Year],
  title   = {HGWO-VNS: An Embedded Hybrid Discrete Grey Wolf Optimizer with Variable Neighborhood Descent and Optimal Split Decoding for the Capacitated Vehicle Routing Problem},
  author  = {Meziane, Mohammed El Amine},
  journal = {International Journal of Automotive Science and Technology},
  year    = {[Year]},
  doi     = {[DOI]}
}
```

## Contact & License

- **Corresponding Author**: Mohammed El Amine Meziane (`amine.meziane@ese-oran.dz`)
- **License**: Released under the MIT License for academic and research reproducibility.
