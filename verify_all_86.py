#!/usr/bin/env python3
"""
Independent 86-instance CVRPLIB verifier for the HGWO-VNS IJASTECH manuscript.
Recomputes every route's demand, fleet count, customer coverage, and distance
directly from raw official .vrp files and .sol certificates, and cross-checks
against the manuscript CSV tables.
"""
import csv
import math
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, "code"))
from hgwo_vns_cvrp import load_vrp_file
from cvrp_validator import independent_audit
from benchmark_golden import load_golden_vrp


def read_sol(path):
    routes, cost_decl = [], None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Route #"):
                rhs = line.split(":", 1)[1].strip()
                if rhs:
                    routes.append([int(x) for x in rhs.split()])
            elif line.startswith("Cost"):
                cost_decl = float(line.split()[1])
    return routes, cost_decl


def audit_golden_independent(vrp_path, routes, k_ref=None):
    inst = load_golden_vrp(vrp_path)
    k_max = getattr(inst, "num_vehicles", getattr(inst, "k_ref", k_ref))
    visited = []
    total_dist = 0.0
    errors = []
    # Golden 9-20 CVRPLIB .vrp files have no hard fleet cap (unbounded fleet);
    # k_max is recorded for reference against BKS fleet size only.
    for r_idx, r in enumerate(routes, 1):
        load = sum(inst.demands[u] for u in r)
        if load > inst.capacity:
            errors.append(f"Route {r_idx} capacity exceeded: {load} > {inst.capacity}")
        prev = 0
        for u in r:
            visited.append(u)
            dx = inst.coords[prev][0] - inst.coords[u][0]
            dy = inst.coords[prev][1] - inst.coords[u][1]
            total_dist += math.hypot(dx, dy)
            prev = u
        dx = inst.coords[prev][0] - inst.coords[0][0]
        dy = inst.coords[prev][1] - inst.coords[0][1]
        total_dist += math.hypot(dx, dy)
    expected_nodes = set(range(1, inst.n + 1))
    if len(visited) != inst.n or set(visited) != expected_nodes:
        errors.append(f"Coverage mismatch: visited={len(visited)}, unique={len(set(visited))}, expected={inst.n}")
    return len(errors) == 0, round(total_dist, 2), len(routes), k_max, "; ".join(errors)


def main():
    report_rows = []
    mismatches = []

    # 1. Set P (23 Table 4 instances + P-n55-k8)
    with open(os.path.join(BASE, "results", "same_hardware_hgwo_vs_habc.csv"), encoding="utf-8") as f:
        p_rows = list(csv.DictReader(f))
    expected_p = {r["Instance"]: int(float(r["HGWO_Best"])) for r in p_rows}
    expected_p["P-n55-k8"] = 576

    for name, exp_cost in sorted(expected_p.items()):
        vrp = os.path.join(BASE, "benchmarks", "P", f"{name}.vrp")
        sol = os.path.join(BASE, "solutions_sol", "P", f"{name}.sol")
        inst = load_vrp_file(vrp)
        routes, cost_decl = read_sol(sol)
        cert = independent_audit(inst, routes, strict_fleet=True)
        match = (cert.is_valid and cert.total_cost_euc2d_int == exp_cost and int(round(cost_decl)) == exp_cost)
        report_rows.append(["P", name, inst.num_customers, inst.num_vehicles, inst.num_vehicles, len(routes), exp_cost, cert.total_cost_euc2d_int, "OK" if cert.is_valid else "FAIL", "YES" if match else "NO"])
        if not match:
            mismatches.append(("P", name, exp_cost, cost_decl, cert.total_cost_euc2d_int, cert.is_valid))

    # 2. Set A & Set B
    for suite, csv_name in [("A", "resultats_set_a_complet.csv"), ("B", "resultats_set_b_complet.csv")]:
        with open(os.path.join(BASE, "results", csv_name), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            name = r["Instance"]
            exp_cost = int(float(r["HGWO_Best"]))
            vrp = os.path.join(BASE, "benchmarks", suite, f"{name}.vrp")
            sol = os.path.join(BASE, "solutions_sol", suite, f"{name}.sol")
            inst = load_vrp_file(vrp)
            routes, cost_decl = read_sol(sol)
            cert = independent_audit(inst, routes, strict_fleet=True)
            match = (cert.is_valid and cert.total_cost_euc2d_int == exp_cost and int(round(cost_decl)) == exp_cost)
            report_rows.append([suite, name, inst.num_customers, inst.num_vehicles, inst.num_vehicles, len(routes), exp_cost, cert.total_cost_euc2d_int, "OK" if cert.is_valid else "FAIL", "YES" if match else "NO"])
            if not match:
                mismatches.append((suite, name, exp_cost, cost_decl, cert.total_cost_euc2d_int, cert.is_valid))

    # 3. Golden 9-20
    with open(os.path.join(BASE, "results", "resultats_golden_cvrp.csv"), encoding="utf-8") as f:
        g_rows = list(csv.DictReader(f))
    for r in g_rows:
        name = r["Instance"]
        exp_cost = round(float(r["HGWO_Best"]), 2)
        vrp = os.path.join(BASE, "benchmarks", "Golden", f"{name}.vrp")
        sol = os.path.join(BASE, "solutions_sol", "Golden", f"{name}.sol")
        routes, cost_decl = read_sol(sol)
        valid, recomp, k_used, k_max, err = audit_golden_independent(vrp, routes, k_ref=int(r['K_ref']))
        match = (valid and abs(recomp - exp_cost) <= 0.02 and abs(cost_decl - exp_cost) <= 0.02)
        report_rows.append(["Golden", name, int(r["n"]), "unbounded", k_max, k_used, f"{exp_cost:.2f}", f"{recomp:.2f}", "OK" if valid else "FAIL", "YES" if match else "NO"])
        if not match:
            mismatches.append(("Golden", name, exp_cost, cost_decl, recomp, valid, err))

    rep_csv = os.path.join(BASE, "audit_86_report.csv")
    with open(rep_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Suite", "Instance", "n", "K_cap", "K_ref", "K_used", "Manuscript_Cost", "Recomputed_Cost", "Feasible", "Exact_Match"])
        w.writerows(report_rows)

    n_total = len(report_rows)
    n_feas = sum(1 for r in report_rows if r[8] == "OK")
    n_match = sum(1 for r in report_rows if r[9] == "YES")
    print(f"=== INDEPENDENT 86-INSTANCE AUDIT SUMMARY ===")
    print(f"Total audited : {n_total}/86")
    print(f"Feasible (OK) : {n_feas}/{n_total}")
    print(f"Exact matches : {n_match}/{n_total}")
    if mismatches:
        print("Mismatches:")
        for m in mismatches:
            print(" ", m)
    else:
        print("ZERO mismatches: all 86 .sol files are 100% feasible and match the manuscript exactly!")


if __name__ == "__main__":
    main()
