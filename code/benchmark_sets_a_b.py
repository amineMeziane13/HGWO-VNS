"""Exécute HGWO-VNS et l'audit indépendant CVRPLIB sur les 50 instances des Sets A (27) et B (23)."""

import csv
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath("../output"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cvrp_validator import independent_audit
from hgwo_vns_cvrp import HGWO_VNS, load_vrp_file

BKS_A = {
    "A-n32-k5": 784, "A-n33-k5": 661, "A-n33-k6": 742, "A-n34-k5": 778, "A-n36-k5": 799,
    "A-n37-k5": 669, "A-n37-k6": 949, "A-n38-k5": 730, "A-n39-k5": 822, "A-n39-k6": 831,
    "A-n44-k6": 937, "A-n45-k6": 944, "A-n45-k7": 1146, "A-n46-k7": 914, "A-n48-k7": 1073,
    "A-n53-k7": 1010, "A-n54-k7": 1167, "A-n55-k9": 1073, "A-n60-k9": 1354, "A-n61-k9": 1034,
    "A-n62-k8": 1288, "A-n63-k9": 1616, "A-n63-k10": 1314, "A-n64-k9": 1401, "A-n65-k9": 1174,
    "A-n69-k9": 1159, "A-n80-k10": 1763,
}

BKS_B = {
    "B-n31-k5": 672, "B-n34-k5": 788, "B-n35-k5": 955, "B-n38-k6": 805, "B-n39-k5": 549,
    "B-n41-k6": 829, "B-n43-k6": 742, "B-n44-k7": 909, "B-n45-k5": 751, "B-n45-k6": 678,
    "B-n50-k7": 741, "B-n50-k8": 1312, "B-n51-k7": 1032, "B-n52-k7": 747, "B-n56-k7": 707,
    "B-n57-k7": 1153, "B-n57-k9": 1598, "B-n63-k10": 1496, "B-n64-k9": 861, "B-n66-k9": 1316,
    "B-n67-k10": 1032, "B-n68-k9": 1272, "B-n78-k10": 1221,
}


def run_set(label: str, folder: str, bks_map: dict, out_csv: str, num_runs: int = 3):
    files = sorted(glob.glob(os.path.join(folder, "**", "*.vrp"), recursive=True))
    # Tri par dimension puis nom
    insts = []
    for fp in files:
        stem = os.path.splitext(os.path.basename(fp))[0]
        inst = load_vrp_file(fp, bks=bks_map.get(stem))
        insts.append(inst)
    insts.sort(key=lambda x: (x.num_customers, x.num_vehicles, x.name))

    rows = []
    print(f"\n=== SET {label} ({len(insts)} instances, {num_runs} runs) ===", flush=True)
    for inst in insts:
        costs, times = [], []
        all_ok = True
        for s in range(1, num_runs + 1):
            solver = HGWO_VNS(pop_size=20, max_iter=100, stagnation_limit=12, elite_ratio=0.35, seed=s * 17)
            res = solver.solve(inst)
            cert = independent_audit(inst, res.best_routes, strict_fleet=True)
            if not cert.is_valid:
                all_ok = False
            costs.append(res.best_cost)
            times.append(res.runtime_sec)

        best_c = min(costs)
        mean_c = statistics.mean(costs)
        std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0
        mean_t = statistics.mean(times)
        bks = inst.bks
        gap = (best_c - bks) / bks * 100.0
        row = {
            "Instance": inst.name,
            "n": inst.num_customers,
            "K": inst.num_vehicles,
            "Q": inst.capacity,
            "BKS": bks,
            "HGWO_Best": best_c,
            "HGWO_Mean": round(mean_c, 2),
            "HGWO_Std": round(std_c, 2),
            "Gap_BKS_pct": round(gap, 2),
            "Time_s": round(mean_t, 3),
            "Valid": "OK" if all_ok else "ERR",
        }
        rows.append(row)
        print(
            f"  {inst.name:<10} | BKS={bks:>4} | Best={best_c:>4} | Mean={mean_c:>7.2f} | "
            f"Gap={gap:>5.2f}% | t={mean_t:>5.2f}s | {row['Valid']}",
            flush=True,
        )

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    n_opt = sum(1 for r in rows if r["Gap_BKS_pct"] <= 0.0)
    n_ok = sum(1 for r in rows if r["Valid"] == "OK")
    avg_gap = statistics.mean(r["Gap_BKS_pct"] for r in rows)
    avg_t = statistics.mean(r["Time_s"] for r in rows)
    print(
        f"--> RÉSUMÉ SET {label} : Valides={n_ok}/{len(rows)} | Optimums BKS={n_opt}/{len(rows)} "
        f"| Gap moyen={avg_gap:.2f}% | Temps moyen={avg_t:.2f}s",
        flush=True,
    )
    return rows


if __name__ == "__main__":
    rows_a = run_set("A", "benchmarks/A", BKS_A, "../output/resultats_set_a_complet.csv", num_runs=3)
    rows_b = run_set("B", "benchmarks/B", BKS_B, "../output/resultats_set_b_complet.csv", num_runs=3)

    # Graphique de synthèse Set A & Set B (Gap % au BKS par instance)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.2), dpi=220)
    for ax, rows, title, color in (
        (ax1, rows_a, "Augerat Set A (27 instances) — Écart HGWO-VNS au BKS CVRPLIB (%)", "#1f77b4"),
        (ax2, rows_b, "Augerat Set B (23 instances) — Écart HGWO-VNS au BKS CVRPLIB (%)", "#2ca02c"),
    ):
        names = [r["Instance"] for r in rows]
        gaps = [r["Gap_BKS_pct"] for r in rows]
        x = np.arange(len(names))
        ax.bar(x, gaps, color=color, alpha=0.85, width=0.62)
        ax.axhline(statistics.mean(gaps), color="#d62728", linestyle="--", linewidth=1.3,
                   label=f"Gap moyen = {statistics.mean(gaps):.2f}%")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Gap au BKS (%)", fontsize=9.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    plt.tight_layout(pad=2.0)
    plt.savefig("../output/benchmark_sets_a_b_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("Graphique généré : ../output/benchmark_sets_a_b_summary.png")
