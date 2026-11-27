"""
Étude d'ablation (V1 -> V4) et validation statistique non paramétrique (Wilcoxon & Friedman)
pour HGWO-VNS sur les benchmarks Augerat Set P.

Variantes évaluées sur les mêmes graines aléatoires (20 runs appariés par instance) :
  - V1 (GWO-Greedy)     : GWO discret seul (découpage séquentiel glouton, init aléatoire)
  - V2 (GWO-Split)      : V1 + décodage optimal Split borné par flotte K
  - V3 (GWO-Split-VND)  : V2 + croisement RBX + VND embarqué (2-opt, Relocate, Swap, 2-opt*, Or-opt)
  - V4 (HGWO-VNS)       : V3 + initialisation hybride Clarke-Wright + Destroy & Repair (Regret-2)
"""

from __future__ import annotations

import math
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon

from cvrp_validator import independent_audit
from hgwo_vns_cvrp import (
    CVRPInstance,
    DJEBBAR_2025_SET_P,
    HGWO_VNS,
    get_builtin_instances,
    mutation_inversion,
    order_crossover_ox,
    route_based_crossover_rbx,
    routes_to_giant_tour,
    split_optimal,
    vnd_local_search,
)


def greedy_sequential_decode(perm: Sequence[int], instance: CVRPInstance) -> Tuple[List[List[int]], int]:
    """Décodage séquentiel glouton (analogue à l'initialisation simple sans Split optimal)."""
    Q = instance.capacity
    dem = instance.demands
    routes: List[List[int]] = []
    cur: List[int] = []
    load = 0
    for c in perm:
        q = dem[c]
        if load + q > Q and cur:
            routes.append(cur)
            cur = [c]
            load = q
        else:
            cur.append(c)
            load += q
    if cur:
        routes.append(cur)
    return routes, instance.solution_cost(routes)


def run_ablation_variant(
    instance: CVRPInstance,
    variant: str,
    pop_size: int = 24,
    max_iter: int = 120,
    seed: int = 1,
) -> Tuple[int, float, List[int], bool]:
    """Exécute l'une des 4 variantes (V1, V2, V3, V4) avec le même budget et la même graine."""
    if variant == "V4":
        solver = HGWO_VNS(
            pop_size=pop_size,
            max_iter=max_iter,
            stagnation_limit=12,
            elite_ratio=0.35,
            seed=seed,
        )
        res = solver.solve(instance)
        cert = independent_audit(instance, res.best_routes, strict_fleet=True)
        # Pad convergence to max_iter + 1 for aligned averaging across runs
        conv = list(res.convergence)
        while len(conv) < max_iter + 1:
            conv.append(conv[-1])
        return res.best_cost, res.runtime_sec, conv[: max_iter + 1], cert.is_valid

    t0 = time.perf_counter()
    rng = random.Random(seed)
    n = instance.num_customers
    use_split = variant in ("V2", "V3")
    use_vnd = variant == "V3"

    def decode(perm: Sequence[int]) -> Tuple[List[List[int]], int]:
        if use_split:
            return split_optimal(perm, instance, enforce_fleet=True)
        return greedy_sequential_decode(perm, instance)

    def fitness(routes: Sequence[Sequence[int]], cost: int) -> int:
        excess = max(0, len([r for r in routes if r]) - instance.num_vehicles)
        return cost + excess * 10000

    # Initialisation 100 % aléatoire (pas de Clarke-Wright pour V1, V2, V3)
    perms: List[List[int]] = []
    routes_list: List[List[List[int]]] = []
    costs: List[int] = []
    fits: List[int] = []

    for _ in range(pop_size):
        perm = list(range(1, n + 1))
        rng.shuffle(perm)
        routes, cost = decode(perm)
        perms.append(perm)
        routes_list.append(routes)
        costs.append(cost)
        fits.append(fitness(routes, cost))

    n_elite = max(3, int(pop_size * 0.35))
    if use_vnd:
        init_order = sorted(range(pop_size), key=lambda i: fits[i])
        for idx in init_order[:n_elite]:
            r_imp, c_imp = vnd_local_search(routes_list[idx], instance)
            routes_list[idx] = r_imp
            perms[idx] = routes_to_giant_tour(r_imp)
            costs[idx] = c_imp
            fits[idx] = fitness(r_imp, c_imp)

    order = sorted(range(pop_size), key=lambda i: fits[i])
    best_routes = [list(r) for r in routes_list[order[0]]]
    best_cost = costs[order[0]]
    best_fit = fits[order[0]]
    conv = [best_cost]

    for it in range(1, max_iter + 1):
        a = 2.0 * (1.0 - it / max_iter)
        order = sorted(range(pop_size), key=lambda i: fits[i])
        alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
        elite_set = set(order[:n_elite])

        for i in range(pop_size):
            A = 2.0 * a * rng.random() - a
            u = rng.random()
            leader_idx = alpha_idx if u < 0.50 else (beta_idx if u < 0.80 else delta_idx)

            if not use_vnd or abs(A) >= 1.0:
                cand_perm = order_crossover_ox(perms[i], perms[leader_idx], rng)
                if rng.random() < 0.25:
                    cand_perm = mutation_inversion(cand_perm, rng)
                cand_routes, cand_cost = decode(cand_perm)
            else:
                cand_routes = route_based_crossover_rbx(routes_list[i], routes_list[leader_idx], instance, rng)
                cand_cost = instance.solution_cost(cand_routes)
                cand_perm = routes_to_giant_tour(cand_routes)

            cand_fit = fitness(cand_routes, cand_cost)

            if use_vnd and (i in elite_set or cand_fit <= int(fits[i] * 1.06)):
                cand_routes, cand_cost = vnd_local_search(cand_routes, instance)
                cand_perm = routes_to_giant_tour(cand_routes)
                cand_fit = fitness(cand_routes, cand_cost)

            if cand_fit < fits[i]:
                perms[i] = cand_perm
                routes_list[i] = cand_routes
                costs[i] = cand_cost
                fits[i] = cand_fit

            if fits[i] < best_fit:
                best_fit = fits[i]
                best_cost = costs[i]
                best_routes = [list(r) for r in routes_list[i]]

        conv.append(best_cost)

    elapsed = time.perf_counter() - t0
    cert = independent_audit(instance, best_routes, strict_fleet=True)
    return best_cost, elapsed, conv, cert.is_valid


def safe_wilcoxon(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Test de rangs signés de Wilcoxon apparié (gère le cas où toutes les différences sont nulles)."""
    diffs = [a - b for a, b in zip(x, y)]
    if all(d == 0 for d in diffs):
        return 0.0, 1.0
    stat, pval = wilcoxon(x, y, alternative="less")  # H1 : x (V4) < y (baseline)
    return float(stat), float(pval)


def plot_ablation_results(
    costs_p40: Dict[str, List[int]],
    convs_p40: Dict[str, List[List[int]]],
    output_png: str,
) -> None:
    """Génère une figure double panneau (Boxplots + Courbes de convergence) sur P-n40-k5."""
    labels = {
        "V1": "V1: GWO-Greedy",
        "V2": "V2: GWO+Split",
        "V3": "V3: GWO+Split+VND",
        "V4": "V4: HGWO-VNS",
    }
    colors = {
        "V1": "#7f7f7f",
        "V2": "#1f77b4",
        "V3": "#ff7f0e",
        "V4": "#2ca02c",
    }
    order = ["V1", "V2", "V3", "V4"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=220)

    # Panneau A : Boxplots sur 20 runs (P-n40-k5)
    data = [costs_p40[v] for v in order]
    bp = ax1.boxplot(
        data,
        patch_artist=True,
        tick_labels=[labels[v].replace(": ", "\n") for v in order],
        widths=0.52,
        medianprops={"color": "#111111", "linewidth": 1.8},
    )
    for patch, v in zip(bp["boxes"], order):
        patch.set_facecolor(colors[v])
        patch.set_alpha(0.75)

    ax1.axhline(505, color="#d62728", linestyle="--", linewidth=1.4, label="HABC Djebbar (2025) = 505")
    ax1.axhline(458, color="#111111", linestyle=":", linewidth=1.5, label="BKS CVRPLIB = 458")
    ax1.set_title("A. Étude d'ablation sur P-n40-k5 (20 runs appariés)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Distance totale (EUC_2D)", fontsize=9.5)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    # Panneau B : Courbes de convergence moyennes (P-n40-k5)
    iters = np.arange(len(convs_p40["V1"][0]))
    for v in order:
        arr = np.array(convs_p40[v], dtype=float)
        mean_curve = arr.mean(axis=0)
        std_curve = arr.std(axis=0)
        ax2.plot(iters, mean_curve, color=colors[v], linewidth=2.0, label=labels[v])
        ax2.fill_between(iters, mean_curve - std_curve, mean_curve + std_curve, color=colors[v], alpha=0.14)

    ax2.axhline(505, color="#d62728", linestyle="--", linewidth=1.3, label="HABC (2025) = 505")
    ax2.axhline(458, color="#111111", linestyle=":", linewidth=1.4, label="BKS = 458")
    ax2.set_title("B. Convergence moyenne ± écart-type sur P-n40-k5", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Itérations", fontsize=9.5)
    ax2.set_ylabel("Meilleure distance moyenne", fontsize=9.5)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper right", fontsize=8.0, frameon=True)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    instances = get_builtin_instances()
    variants = ["V1", "V2", "V3", "V4"]
    num_runs = 20
    seeds = [17 * (i + 1) for i in range(num_runs)]

    all_costs: Dict[str, Dict[str, List[int]]] = {inst.name: {v: [] for v in variants} for inst in instances}
    all_times: Dict[str, Dict[str, List[float]]] = {inst.name: {v: [] for v in variants} for inst in instances}
    all_valid: Dict[str, Dict[str, int]] = {inst.name: {v: 0 for v in variants} for inst in instances}
    convs_p40: Dict[str, List[List[int]]] = {v: [] for v in variants}

    for inst in instances:
        for v in variants:
            for s in seeds:
                cost, dt, conv, is_val = run_ablation_variant(inst, v, pop_size=24, max_iter=120, seed=s)
                all_costs[inst.name][v].append(cost)
                all_times[inst.name][v].append(dt)
                if is_val:
                    all_valid[inst.name][v] += 1
                if inst.name == "P-n40-k5":
                    convs_p40[v].append(conv)

    # 1. Tableau récapitulatif d'ablation
    print("=== ÉTUDE D'ABLATION (20 RUNS APPARIÉS PAR INSTANCE) ===")
    print(f"{'Instance':<9} | {'Var':<3} | {'Valide':>6} | {'Best':>5} | {'Mean':>7} | {'Std':>6} | {'Gap Mean':>8} | {'Temps(s)':>8}")
    print("-" * 74)
    for inst in instances:
        bks = inst.bks or 1
        for v in variants:
            cs = all_costs[inst.name][v]
            ts = all_times[inst.name][v]
            mean_c = statistics.mean(cs)
            std_c = statistics.stdev(cs)
            gap_m = (mean_c - bks) / bks * 100.0
            print(
                f"{inst.name:<9} | {v:<3} | {all_valid[inst.name][v]:>2d}/{num_runs:<3d} | "
                f"{min(cs):>5d} | {mean_c:>7.2f} | {std_c:>6.2f} | {gap_m:>7.2f}% | {statistics.mean(ts):>8.3f}"
            )
        print("-" * 74)

    # 2. Tests statistiques de Wilcoxon appariés (V4 vs V1, V2, V3) et Friedman global
    print("\n=== TESTS STATISTIQUES DE WILCOXON (unilatéral H1 : V4 < Vi) ===")
    pooled: Dict[str, List[float]] = {v: [] for v in variants}
    for inst in instances:
        print(f"\nInstance {inst.name} (n={num_runs} paires) :")
        v4_c = all_costs[inst.name]["V4"]
        for v in ("V1", "V2", "V3"):
            stat, pval = safe_wilcoxon(v4_c, all_costs[inst.name][v])
            sig = "SIGNIFICATIF (p < 0.05)" if pval < 0.05 else "non significatif"
            print(f"  V4 vs {v} : W = {stat:6.1f}, p-value = {pval:.6e} -> {sig}")
        for v in variants:
            # Normalisation par BKS pour l'analyse globale inter-instances
            pooled[v].extend([c / inst.bks for c in all_costs[inst.name][v]])

    print("\n=== ANALYSE GLOBALE SUR LES 60 RUNS APPARIÉS (3 INSTANCES x 20 SEEDS) ===")
    for v in ("V1", "V2", "V3"):
        stat, pval = safe_wilcoxon(pooled["V4"], pooled[v])
        print(f"  Global V4 vs {v} : W = {stat:6.1f}, p-value = {pval:.6e}")

    f_stat, f_pval = friedmanchisquare(pooled["V1"], pooled["V2"], pooled["V3"], pooled["V4"])
    # Calcul des rangs moyens (1 = meilleur)
    rank_sums = {v: 0.0 for v in variants}
    for idx in range(len(pooled["V1"])):
        vals = [pooled[v][idx] for v in variants]
        # Rangs moyens avec gestion des ex-aequo
        sorted_vals = sorted(vals)
        for v, val in zip(variants, vals):
            positions = [p + 1 for p, sv in enumerate(sorted_vals) if sv == val]
            rank_sums[v] += sum(positions) / len(positions)
    avg_ranks = {v: rank_sums[v] / len(pooled["V1"]) for v in variants}
    print(f"  Test de Friedman : chi2 = {f_stat:.3f}, p-value = {f_pval:.6e}")
    print(
        "  Rangs moyens (1=meilleur) : "
        + ", ".join(f"{v}={avg_ranks[v]:.2f}" for v in variants)
    )

    plot_ablation_results(all_costs["P-n40-k5"], convs_p40, "ablation_convergence_boxplot.png")
    print("\nGraphique généré : ablation_convergence_boxplot.png")


if __name__ == "__main__":
    main()
