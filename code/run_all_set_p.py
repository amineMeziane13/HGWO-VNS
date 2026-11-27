"""
Script clé en main pour exécuter HGWO-VNS, l'audit indépendant CVRPLIB et le test de Wilcoxon
sur l'ensemble des 23 instances du Set P d'Augerat (Djebbar & Kemmar, 2025 — Table 5).

Utilisation :
  1. Depuis un dossier contenant les fichiers .vrp du Set P :
       python3 run_all_set_p.py /chemin/vers/dossier_set_P
  2. Depuis l'archive ZIP téléchargée sur CVRPLIB (ex. Vrp-Set-P.zip) :
       python3 run_all_set_p.py /chemin/vers/Vrp-Set-P.zip
  3. Sans argument (exécute les 3 instances officielles intégrées : P-n16-k8, P-n19-k2, P-n40-k5) :
       python3 run_all_set_p.py

Sorties générées :
  - resultats_set_p_complet.csv   (tableau comparatif complet HGWO-VNS vs HABC vs BKS)
  - solutions_sol/<instance>.sol  (certificats officiels vérifiables sur CVRPLIB)
"""

from __future__ import annotations

import csv
import os
import statistics
import sys
import zipfile
from typing import Dict, List, Sequence

from scipy.stats import wilcoxon

from cvrp_validator import independent_audit
from hgwo_vns_cvrp import (
    CVRPInstance,
    DJEBBAR_2025_SET_P,
    HGWO_VNS,
    get_builtin_instances,
    load_vrp_file,
    parse_vrp_text,
)


def load_instances_from_path(path: str) -> List[CVRPInstance]:
    """Charge toutes les instances .vrp depuis un dossier ou une archive .zip."""
    instances: List[CVRPInstance] = []

    if os.path.isfile(path) and path.lower().endswith(".zip"):
        with zipfile.ZipFile(path, "r") as zf:
            vrp_names = sorted(
                n for n in zf.namelist()
                if n.lower().endswith(".vrp") and not os.path.basename(n).startswith(".")
            )
            for member in vrp_names:
                stem = os.path.splitext(os.path.basename(member))[0]
                ref = DJEBBAR_2025_SET_P.get(stem, {})
                text = zf.read(member).decode("utf-8", errors="replace")
                instances.append(
                    parse_vrp_text(text, num_vehicles=ref.get("k"), bks=ref.get("bks"))
                )
    elif os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fname in sorted(files):
                if fname.lower().endswith(".vrp") and not fname.startswith("."):
                    stem = os.path.splitext(fname)[0]
                    ref = DJEBBAR_2025_SET_P.get(stem, {})
                    instances.append(
                        load_vrp_file(
                            os.path.join(root, fname),
                            num_vehicles=ref.get("k"),
                            bks=ref.get("bks"),
                        )
                    )
    else:
        raise FileNotFoundError(f"Chemin introuvable ou format non supporté : {path}")

    # Tri selon l'ordre standard de la Table 5 de Djebbar & Kemmar (2025)
    order_keys = list(DJEBBAR_2025_SET_P.keys())
    instances.sort(
        key=lambda inst: (
            order_keys.index(inst.name) if inst.name in order_keys else 999,
            inst.name,
        )
    )
    return instances


def run_full_suite(
    instances: Sequence[CVRPInstance],
    num_runs: int = 10,
    pop_size: int = 24,
    max_iter: int = 120,
    out_csv: str = "resultats_set_p_complet.csv",
    sol_dir: str = "solutions_sol",
) -> None:
    os.makedirs(sol_dir, exist_ok=True)
    rows: List[Dict[str, object]] = []
    habc_paired: List[float] = []
    hgwo_paired: List[float] = []

    header = (
        f"{'Instance':<10} | {'K':>2} | {'BKS':>4} | {'HABC':>5} | {'HABC(s)':>7} | "
        f"{'Best':>5} | {'Mean':>7} | {'Std':>5} | {'Temps(s)':>8} | {'Gap BKS':>7} | {'Gain/HABC':>9} | {'Valide':>6}"
    )
    print(header)
    print("-" * len(header))

    for inst in instances:
        costs: List[int] = []
        times: List[float] = []
        best_routes = None
        best_cost = 10**12
        all_ok = True

        for r in range(1, num_runs + 1):
            solver = HGWO_VNS(
                pop_size=pop_size,
                max_iter=max_iter,
                stagnation_limit=12,
                elite_ratio=0.35,
                seed=r * 17,
            )
            res = solver.solve(inst)
            cert = independent_audit(inst, res.best_routes, strict_fleet=True)
            if not cert.is_valid:
                all_ok = False
            costs.append(res.best_cost)
            times.append(res.runtime_sec)
            if res.best_cost < best_cost:
                best_cost = res.best_cost
                best_routes = res.best_routes

        # Sauvegarde du certificat officiel .sol de la meilleure solution
        assert best_routes is not None
        best_cert = independent_audit(inst, best_routes, strict_fleet=True)
        with open(os.path.join(sol_dir, f"{inst.name}.sol"), "w", encoding="utf-8") as f:
            f.write(best_cert.cvrplib_sol_text + "\n")

        ref = DJEBBAR_2025_SET_P.get(inst.name, {})
        bks = inst.bks or ref.get("bks")
        habc_cost = ref.get("habc_cost")
        habc_time = ref.get("habc_time")

        mean_c = statistics.mean(costs)
        std_c = statistics.stdev(costs) if len(costs) > 1 else 0.0
        mean_t = statistics.mean(times)
        gap_bks = ((best_cost - bks) / bks * 100.0) if bks else None
        gain_habc = ((habc_cost - best_cost) / habc_cost * 100.0) if habc_cost else None

        if habc_cost is not None:
            habc_paired.append(float(habc_cost))
            hgwo_paired.append(float(best_cost))

        row = {
            "Instance": inst.name,
            "K": inst.num_vehicles,
            "Q": inst.capacity,
            "BKS": bks,
            "HABC_2025": habc_cost,
            "HABC_Time_s": habc_time,
            "HGWO_Best": best_cost,
            "HGWO_Mean": round(mean_c, 2),
            "HGWO_Std": round(std_c, 2),
            "HGWO_Time_s": round(mean_t, 3),
            "Gap_BKS_pct": round(gap_bks, 2) if gap_bks is not None else "",
            "Gain_vs_HABC_pct": round(gain_habc, 2) if gain_habc is not None else "",
            "Valid": "OK" if all_ok else "ERR",
        }
        rows.append(row)

        print(
            f"{inst.name:<10} | {inst.num_vehicles:>2d} | {str(bks):>4} | {str(habc_cost):>5} | "
            f"{str(habc_time):>7} | {best_cost:>5d} | {mean_c:>7.2f} | {std_c:>5.2f} | "
            f"{mean_t:>8.3f} | {f'{gap_bks:.2f}%' if gap_bks is not None else '-':>7} | "
            f"{f'{gain_habc:+.2f}%' if gain_habc is not None else '-':>9} | {row['Valid']:>6}"
        )

    # Export CSV
    if rows:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    # Test de Wilcoxon apparié HGWO-VNS vs HABC (si >= 5 instances avec différences non nulles)
    diffs = [a - b for a, b in zip(hgwo_paired, habc_paired)]
    non_zero = [d for d in diffs if d != 0]
    print("\n" + "=" * 78)
    print(f"Instances évaluées : {len(rows)} | Export CSV : {out_csv} | Certificats .sol : {sol_dir}/")
    if len(non_zero) >= 5:
        stat, pval = wilcoxon(hgwo_paired, habc_paired, alternative="less")
        print(f"Test de Wilcoxon (HGWO-VNS < HABC sur {len(hgwo_paired)} instances) : W = {stat:.1f}, p = {pval:.6e}")
    else:
        print(f"Test de Wilcoxon global : requiert >= 5 instances avec différences non nulles (actuellement {len(non_zero)}).")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        loaded_instances = load_instances_from_path(sys.argv[1])
    else:
        loaded_instances = get_builtin_instances()
    run_full_suite(loaded_instances, num_runs=10)
