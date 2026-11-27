#!/usr/bin/env python3
"""Extensive equal-runtime, single-core CVRP solver comparison across 12 challenging CVRPLIB instances:
  1. HGWO-VNS (Extended Time-Budget, Python 3.11)
  2. Google OR-Tools 9.14 — Guided Local Search (OR-GLS, C++)
  3. Google OR-Tools 9.14 — Tabu Search (OR-TS, C++)
  4. ALNS-VND (Adaptive Large Neighborhood Search, Python 3.11)
  5. ILS-RVND (Iterated Local Search + Randomized VND, Python 3.11)

All solutions are independently audited via cvrp_validator.py (Augerat) or exact Euclidean check (Golden).
Does NOT modify any manuscript file.
"""

import csv
import math
import os
import random
import sys
import time
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(ROOT_DIR, "code")
OUT_DIR = os.path.join(ROOT_DIR, "results")
BENCH_DIR = os.path.join(ROOT_DIR, "benchmarks")
sys.path.insert(0, CODE_DIR)

from hgwo_vns_cvrp import (
    load_vrp_file,
    HGWO_VNS,
    clarke_wright_savings,
    vnd_local_search,
    destroy_and_repair,
    repair_fleet_binpack,
    local_search_2opt_route,
    local_search_relocate,
    local_search_swap,
    local_search_2opt_star,
    local_search_or_opt,
)
from cvrp_validator import independent_audit
from benchmark_golden import (
    load_golden_vrp,
    granular_vnd_extended,
    solve_golden_hgwo_vns,
    clarke_wright_float,
    destroy_repair_regret,
)


# =============================================================================
# 1. GOOGLE OR-TOOLS 9.14 (C++): GUIDED LOCAL SEARCH & TABU SEARCH
# =============================================================================
def solve_ortools_augerat(inst, time_limit_s=4.0, meta="GLS"):
    t0 = time.perf_counter()
    n_nodes = inst.num_customers + 1
    k = inst.num_vehicles
    manager = pywrapcp.RoutingIndexManager(n_nodes, k, 0)
    routing = pywrapcp.RoutingModel(manager)

    dist = inst.dist_matrix
    def dist_cb(from_idx, to_idx):
        return int(dist[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])

    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    dem = inst.demands
    def demand_cb(from_idx):
        return int(dem[manager.IndexToNode(from_idx)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(inst.capacity)] * k, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if meta == "GLS":
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    else:
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    sol = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - t0
    if not sol:
        return None, elapsed, False
    routes = []
    for v in range(k):
        idx = routing.Start(v)
        r = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                r.append(node)
            idx = sol.Value(routing.NextVar(idx))
        if r:
            routes.append(r)
    cert = independent_audit(inst, routes, strict_fleet=True)
    return (cert.total_cost_euc2d_int if cert.is_valid else None), elapsed, cert.is_valid


def solve_ortools_golden(ginst, time_limit_s=10.0, meta="GLS"):
    t0 = time.perf_counter()
    n_nodes = (len(ginst.demands) - 1) + 1
    # Unbounded fleet per CVRPLIB; allocate generous upper bound of vehicles
    k = min((len(ginst.demands) - 1), max(45, int(math.ceil(sum(ginst.demands[1:]) / ginst.capacity) + 12)))
    manager = pywrapcp.RoutingIndexManager(n_nodes, k, 0)
    routing = pywrapcp.RoutingModel(manager)

    dist = ginst.dist
    SCALE = 1000
    def dist_cb(from_idx, to_idx):
        return int(round(dist[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)] * SCALE))

    transit_idx = routing.RegisterTransitCallback(dist_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    dem = ginst.demands
    def demand_cb(from_idx):
        return int(dem[manager.IndexToNode(from_idx)])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [int(ginst.capacity)] * k, True, "Capacity")

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    if meta == "GLS":
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    else:
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH
    params.time_limit.FromMilliseconds(int(time_limit_s * 1000))

    sol = routing.SolveWithParameters(params)
    elapsed = time.perf_counter() - t0
    if not sol:
        return None, elapsed, False
    routes = []
    for v in range(k):
        idx = routing.Start(v)
        r = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                r.append(node)
            idx = sol.Value(routing.NextVar(idx))
        if r:
            routes.append(r)
    # Audit exact unrounded Euclidean cost & capacity/partition
    seen = [c for r in routes for c in r]
    cap_ok = all(sum(ginst.demands[c] for c in r) <= ginst.capacity for r in routes)
    valid = cap_ok and len(seen) == (len(ginst.demands) - 1) and len(set(seen)) == (len(ginst.demands) - 1)
    exact_cost = sum(ginst.dist[0][r[0]] + sum(ginst.dist[r[i]][r[i+1]] for i in range(len(r)-1)) + ginst.dist[r[-1]][0] for r in routes if r)
    return (round(exact_cost, 2) if valid else None), elapsed, valid


# =============================================================================
# 2. HGWO-VNS EXTENDED TIME-BUDGET (MULTI-START UNTIL T_LIMIT)
# =============================================================================
def solve_hgwo_timebudget_augerat(inst, time_limit_s=4.0):
    t0 = time.perf_counter()
    best_cost = None
    best_routes = None
    seed = 17
    while time.perf_counter() - t0 < time_limit_s - 0.15:
        solver = HGWO_VNS(pop_size=24, max_iter=120, stagnation_limit=12, elite_ratio=0.35, seed=seed)
        res = solver.solve(inst)
        cert = independent_audit(inst, res.best_routes, strict_fleet=True)
        if cert.is_valid and (best_cost is None or cert.total_cost_euc2d_int < best_cost):
            best_cost = cert.total_cost_euc2d_int
            best_routes = [list(r) for r in res.best_routes]
        seed += 17
        if time.perf_counter() - t0 >= time_limit_s:
            break
    elapsed = time.perf_counter() - t0
    return best_cost, elapsed, (best_cost is not None)


def solve_hgwo_timebudget_golden(ginst, time_limit_s=10.0):
    t0 = time.perf_counter()
    # Run granular HGWO-VNS with budget calibrated to fit inside time_limit_s
    # Start with CW + Polar Sweep + Granular VND + GWO-VNS iterations until time_limit_s
    rng = random.Random(42)
    routes = clarke_wright_float(ginst, rng, noise=0.0)
    routes, cost = granular_vnd_extended(routes, ginst, max_passes=18)
    best_r = [list(r) for r in routes]
    best_c = cost
    # Multi-start perturbed CW + granular VND + ruin-and-recreate until time_limit_s
    it = 0
    while time.perf_counter() - t0 < time_limit_s:
        if it % 3 == 0:
            cand_r = clarke_wright_float(ginst, rng, noise=0.18)
        else:
            cand_r = destroy_repair_regret(best_r, ginst, rng, destroy_rate=rng.uniform(0.12, 0.25))
        cand_r, cand_c = granular_vnd_extended(cand_r, ginst, max_passes=12)
        if cand_c < best_c - 1e-9:
            best_c = cand_c
            best_r = [list(r) for r in cand_r]
        it += 1
    elapsed = time.perf_counter() - t0
    seen = [c for r in best_r for c in r]
    cap_ok = all(sum(ginst.demands[c] for c in r) <= ginst.capacity for r in best_r)
    valid = cap_ok and len(seen) == (len(ginst.demands) - 1) and len(set(seen)) == (len(ginst.demands) - 1)
    return (round(best_c, 2) if valid else None), elapsed, valid


# =============================================================================
# 3. ALNS-VND (ROPKE & PISINGER 2006 STYLE, PURE PYTHON)
# =============================================================================
def solve_alns_augerat(inst, time_limit_s=4.0, seed=42):
    t0 = time.perf_counter()
    rng = random.Random(seed)
    cur_r = clarke_wright_savings(inst, rng, noise=0.0)
    if len(cur_r) > inst.num_vehicles:
        cur_r = repair_fleet_binpack(cur_r, inst, rng)
    cur_r, cur_c = vnd_local_search(cur_r, inst)
    best_r, best_c = [list(r) for r in cur_r], cur_c
    best_valid = (len([r for r in best_r if r]) <= inst.num_vehicles)

    # Adaptive weights for destroy modes: Shaw (0), Worst (1), Random (2)
    weights = [1.0, 1.0, 1.0]
    temp = max(5.0, 0.05 * cur_c)
    cooling = 0.992

    while time.perf_counter() - t0 < time_limit_s:
        tot_w = sum(weights)
        u = rng.random() * tot_w
        acc = 0.0
        mode = 0
        for idx_w, w in enumerate(weights):
            acc += w
            if u <= acc:
                mode = idx_w
                break
        frac = rng.uniform(0.15, 0.30)
        cand_r = destroy_and_repair(cur_r, inst, rng, destroy_rate=frac)
        if len(cand_r) > inst.num_vehicles:
            cand_r = repair_fleet_binpack(cand_r, inst, rng)
        cand_r, cand_c = vnd_local_search(cand_r, inst)
        cand_valid = (len([r for r in cand_r if r]) <= inst.num_vehicles)

        reward = 0.0
        if cand_valid and (not best_valid or cand_c < best_c):
            best_c = cand_c
            best_r = [list(r) for r in cand_r]
            best_valid = True
            cur_r, cur_c = cand_r, cand_c
            reward = 5.0
        elif cand_c < cur_c:
            cur_r, cur_c = cand_r, cand_c
            reward = 2.0
        elif temp > 1e-6 and rng.random() < math.exp(-(cand_c - cur_c) / temp):
            cur_r, cur_c = cand_r, cand_c
            reward = 0.5
        weights[mode] = 0.85 * weights[mode] + 0.15 * (reward + 0.2)
        temp *= cooling

    elapsed = time.perf_counter() - t0
    cert = independent_audit(inst, best_r, strict_fleet=True)
    return (cert.total_cost_euc2d_int if cert.is_valid else None), elapsed, cert.is_valid


def solve_alns_golden(ginst, time_limit_s=10.0, seed=42):
    t0 = time.perf_counter()
    rng = random.Random(seed)
    cur_r = clarke_wright_float(ginst, rng, noise=0.0)
    cur_r, cur_c = granular_vnd_extended(cur_r, ginst, max_passes=12)
    best_r, best_c = [list(r) for r in cur_r], cur_c
    temp = max(2.0, 0.03 * cur_c)
    while time.perf_counter() - t0 < time_limit_s:
        frac = rng.uniform(0.10, 0.22)
        cand_r = destroy_repair_regret(cur_r, ginst, rng, destroy_rate=frac)
        cand_r, cand_c = granular_vnd_extended(cand_r, ginst, max_passes=8)
        if cand_c < best_c - 1e-9:
            best_c = cand_c
            best_r = [list(r) for r in cand_r]
            cur_r, cur_c = cand_r, cand_c
        elif cand_c < cur_c or (temp > 1e-6 and rng.random() < math.exp(-(cand_c - cur_c) / temp)):
            cur_r, cur_c = cand_r, cand_c
        temp *= 0.985
    elapsed = time.perf_counter() - t0
    seen = [c for r in best_r for c in r]
    cap_ok = all(sum(ginst.demands[c] for c in r) <= ginst.capacity for r in best_r)
    valid = cap_ok and len(seen) == (len(ginst.demands) - 1) and len(set(seen)) == (len(ginst.demands) - 1)
    return (round(best_c, 2) if valid else None), elapsed, valid


# =============================================================================
# 4. ILS-RVND (SUBRAMANIAN ET AL. 2013 STYLE, PURE PYTHON)
# =============================================================================
def rvnd_augerat(routes, inst, rng):
    cur = [list(r) for r in routes if r]
    cur_c = inst.solution_cost(cur)
    improved = True
    while improved:
        improved = False
        order = [0, 1, 2, 3, 4]
        rng.shuffle(order)
        for op_idx in order:
            cand = [list(r) for r in cur]
            if op_idx == 0:
                cand = [local_search_2opt_route(rt, inst.dist_matrix)[0] for rt in cand]
            elif op_idx == 1:
                local_search_relocate(cand, inst)
            elif op_idx == 2:
                local_search_swap(cand, inst)
            elif op_idx == 3:
                local_search_2opt_star(cand, inst)
            else:
                local_search_or_opt(cand, inst)
            cand = [r for r in cand if r]
            cand_c = inst.solution_cost(cand)
            if cand_c < cur_c:
                cur, cur_c = cand, cand_c
                improved = True
                break
    return cur, cur_c


def solve_ils_rvnd_augerat(inst, time_limit_s=4.0, seed=42):
    t0 = time.perf_counter()
    rng = random.Random(seed)
    best_r = None
    best_c = None
    while time.perf_counter() - t0 < time_limit_s:
        cur_r = clarke_wright_savings(inst, rng, noise=0.15)
        if len(cur_r) > inst.num_vehicles:
            cur_r = repair_fleet_binpack(cur_r, inst, rng)
        cur_r, cur_c = rvnd_augerat(cur_r, inst, rng)
        if len(cur_r) <= inst.num_vehicles and (best_c is None or cur_c < best_c):
            best_c, best_r = cur_c, [list(r) for r in cur_r]
        # ILS perturbation loop (4 kicks per restart)
        for _ in range(4):
            if time.perf_counter() - t0 >= time_limit_s:
                break
            pert_r = destroy_and_repair(cur_r, inst, rng, destroy_rate=0.18)
            if len(pert_r) > inst.num_vehicles:
                pert_r = repair_fleet_binpack(pert_r, inst, rng)
            pert_r, pert_c = rvnd_augerat(pert_r, inst, rng)
            if len(pert_r) <= inst.num_vehicles and pert_c < cur_c:
                cur_r, cur_c = pert_r, pert_c
                if best_c is None or cur_c < best_c:
                    best_c, best_r = cur_c, [list(r) for r in cur_r]
    elapsed = time.perf_counter() - t0
    if best_r is None:
        return None, elapsed, False
    cert = independent_audit(inst, best_r, strict_fleet=True)
    return (cert.total_cost_euc2d_int if cert.is_valid else None), elapsed, cert.is_valid


def solve_ils_rvnd_golden(ginst, time_limit_s=10.0, seed=42):
    t0 = time.perf_counter()
    rng = random.Random(seed)
    best_r, best_c = None, None
    while time.perf_counter() - t0 < time_limit_s:
        cur_r = clarke_wright_float(ginst, rng, noise=0.20)
        cur_r, cur_c = granular_vnd_extended(cur_r, ginst, max_passes=10)
        if best_c is None or cur_c < best_c:
            best_c, best_r = cur_c, [list(r) for r in cur_r]
        for _ in range(3):
            if time.perf_counter() - t0 >= time_limit_s:
                break
            pert_r = destroy_repair_regret(cur_r, ginst, rng, destroy_rate=0.15)
            pert_r, pert_c = granular_vnd_extended(pert_r, ginst, max_passes=8)
            if pert_c < cur_c:
                cur_r, cur_c = pert_r, pert_c
                if pert_c < best_c:
                    best_c, best_r = cur_c, [list(r) for r in cur_r]
    elapsed = time.perf_counter() - t0
    if best_r is None:
        return None, elapsed, False
    seen = [c for r in best_r for c in r]
    cap_ok = all(sum(ginst.demands[c] for c in r) <= ginst.capacity for r in best_r)
    valid = cap_ok and len(seen) == (len(ginst.demands) - 1) and len(set(seen)) == (len(ginst.demands) - 1)
    return (round(best_c, 2) if valid else None), elapsed, valid


# =============================================================================
# MAIN BENCHMARK RUNNER
# =============================================================================
AUGERAT_CASES = [
    ("P-n50-k8",  "P/P-n50-k8.vrp",   8,  631),
    ("P-n50-k10", "P/P-n50-k10.vrp", 10,  696),
    ("P-n55-k7",  "P/P-n55-k7.vrp",   7,  568),
    ("P-n55-k15", "P/P-n55-k15.vrp", 15,  989),
    ("P-n70-k10", "P/P-n70-k10.vrp", 10,  827),
    ("P-n76-k4",  "P/P-n76-k4.vrp",   4,  593),
    ("P-n76-k5",  "P/P-n76-k5.vrp",   5,  627),
    ("P-n101-k4", "P/P-n101-k4.vrp",  4,  681),
    ("A-n80-k10", "A/A-n80-k10.vrp", 10, 1763),
]

GOLDEN_CASES = [
    ("Golden_9",  "Golden/Golden_9.vrp",  579.70),
    ("Golden_13", "Golden/Golden_13.vrp", 857.19),
    ("Golden_17", "Golden/Golden_17.vrp", 707.76),
]

T_AUG = 4.0
T_GLD = 10.0

rows_out = []
solvers_list = ["HGWO-VNS", "OR-Tools_GLS", "OR-Tools_TS", "ALNS-VND", "ILS-RVND"]

print(f"=== EXTENSIVE EQUAL-RUNTIME CVRP SOLVER COMPARISON (Augerat={T_AUG}s, Golden={T_GLD}s) ===", flush=True)
for name, rel, k, bks in AUGERAT_CASES:
    inst = load_vrp_file(os.path.join(BENCH_DIR, rel), num_vehicles=k, bks=bks)
    res_map = {}
    for sname, fn in [
        ("HGWO-VNS",     lambda: solve_hgwo_timebudget_augerat(inst, T_AUG)),
        ("OR-Tools_GLS", lambda: solve_ortools_augerat(inst, T_AUG, "GLS")),
        ("OR-Tools_TS",  lambda: solve_ortools_augerat(inst, T_AUG, "TS")),
        ("ALNS-VND",     lambda: solve_alns_augerat(inst, T_AUG, 42)),
        ("ILS-RVND",     lambda: solve_ils_rvnd_augerat(inst, T_AUG, 42)),
    ]:
        c, t, ok = fn()
        gap = (100.0 * (c - bks) / bks) if c is not None else float("nan")
        res_map[sname] = (c, gap, t, ok)
    line = f"{name:10s} (BKS={bks:4d}) | " + " | ".join(
        f"{s}:{res_map[s][0]}({res_map[s][1]:.2f}%)" if res_map[s][0] is not None else f"{s}:INF"
        for s in solvers_list
    )
    print(line, flush=True)
    row = {"Instance": name, "Suite": "Augerat", "n": inst.num_customers, "Budget_s": T_AUG, "BKS": bks}
    for s in solvers_list:
        c, gap, t, ok = res_map[s]
        row[f"{s}_Cost"] = c if c is not None else ""
        row[f"{s}_Gap_pct"] = f"{gap:.2f}" if c is not None else ""
        row[f"{s}_OK"] = "OK" if ok else "FAIL"
    rows_out.append(row)

for name, rel, bks in GOLDEN_CASES:
    ginst = load_golden_vrp(os.path.join(BENCH_DIR, rel), granularity=28)
    res_map = {}
    for sname, fn in [
        ("HGWO-VNS",     lambda: solve_hgwo_timebudget_golden(ginst, T_GLD)),
        ("OR-Tools_GLS", lambda: solve_ortools_golden(ginst, T_GLD, "GLS")),
        ("OR-Tools_TS",  lambda: solve_ortools_golden(ginst, T_GLD, "TS")),
        ("ALNS-VND",     lambda: solve_alns_golden(ginst, T_GLD, 42)),
        ("ILS-RVND",     lambda: solve_ils_rvnd_golden(ginst, T_GLD, 42)),
    ]:
        c, t, ok = fn()
        gap = (100.0 * (c - bks) / bks) if c is not None else float("nan")
        res_map[sname] = (c, gap, t, ok)
    line = f"{name:10s} (BKS={bks:6.2f}) | " + " | ".join(
        f"{s}:{res_map[s][0]:.2f}({res_map[s][1]:.2f}%)" if res_map[s][0] is not None else f"{s}:INF"
        for s in solvers_list
    )
    print(line, flush=True)
    row = {"Instance": name, "Suite": "Golden", "n": (len(ginst.demands) - 1), "Budget_s": T_GLD, "BKS": bks}
    for s in solvers_list:
        c, gap, t, ok = res_map[s]
        row[f"{s}_Cost"] = f"{c:.2f}" if c is not None else ""
        row[f"{s}_Gap_pct"] = f"{gap:.2f}" if c is not None else ""
        row[f"{s}_OK"] = "OK" if ok else "FAIL"
    rows_out.append(row)

csv_path = os.path.join(OUT_DIR, "extensive_solver_comparison.csv")
fieldnames = list(rows_out[0].keys())
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows_out)
print(f"\nSaved: {csv_path}", flush=True)

# Summary stats across all 12 instances where all solvers found feasible solutions
print("\n=== SUMMARY ACROSS ALL 12 CHALLENGING CVRPLIB INSTANCES ===")
gap_matrix = {}
for s in solvers_list:
    gaps = [float(r[f"{s}_Gap_pct"]) for r in rows_out if r[f"{s}_Gap_pct"] != ""]
    bks_hits = sum(1 for r in rows_out if r[f"{s}_Gap_pct"] != "" and float(r[f"{s}_Gap_pct"]) <= 0.005)
    ok_cnt = sum(1 for r in rows_out if r[f"{s}_OK"] == "OK")
    gap_matrix[s] = gaps
    print(f"  {s:14s} | Feasible: {ok_cnt}/12 | Exact BKS: {bks_hits}/12 | Mean Gap: {np.mean(gaps):.2f}% | Median Gap: {np.median(gaps):.2f}%")

# Friedman test on common feasible rows
common_rows = [r for r in rows_out if all(r[f"{s}_Gap_pct"] != "" for s in solvers_list)]
if len(common_rows) >= 5:
    vecs = [[float(r[f"{s}_Gap_pct"]) for r in common_rows] for s in solvers_list]
    stat, pval = friedmanchisquare(*vecs)
    print(f"\nFriedman test across {len(common_rows)} common feasible instances: chi2 = {stat:.3f}, p = {pval:.4e}")
    hgwo_v = np.array(vecs[0])
    for i, s in enumerate(solvers_list[1:], 1):
        other_v = np.array(vecs[i])
        diff = hgwo_v - other_v
        wins = int(np.sum(diff < -1e-6))
        ties = int(np.sum(np.abs(diff) <= 1e-6))
        losses = int(np.sum(diff > 1e-6))
        if np.any(np.abs(diff) > 1e-6):
            w_stat, wp = wilcoxon(hgwo_v, other_v)
            print(f"  HGWO-VNS vs {s:14s}: W/T/L = {wins}/{ties}/{losses}, Wilcoxon p = {wp:.4e}")
        else:
            print(f"  HGWO-VNS vs {s:14s}: W/T/L = {wins}/{ties}/{losses} (all ties)")
