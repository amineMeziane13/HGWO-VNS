import csv
import os
import statistics
import sys
import time

OUT_DIR = "/working_dir/c_4d84fd73234e9417/artifacts/file_generation/ttl=63d/output"
BENCH_DIR = "/working_dir/c_4d84fd73234e9417/artifacts/file_generation/ttl=63d/intermediate/benchmarks"
sys.path.insert(0, OUT_DIR)

from hgwo_vns_cvrp import HGWO_VNS, load_vrp_file, validate_solution
from benchmark_golden import load_golden_vrp, solve_golden_hgwo_vns, independent_validate_float

SET_P_INSTS = [
    ("P-n40-k5", 5, 458),
    ("P-n55-k7", 7, 568),
    ("P-n76-k4", 4, 593),
]
GOLDEN_INSTS = [
    ("Golden_9", 579.70),
    ("Golden_17", 707.76),
]
SEEDS_P = [17, 34, 51]
SEEDS_G = [17, 34]

def eval_set_p(param_name, param_val, **kwargs):
    gaps = []
    times = []
    valid_all = True
    for name, k, bks in SET_P_INSTS:
        inst = load_vrp_file(os.path.join(BENCH_DIR, "P", f"{name}.vrp"), num_vehicles=k, bks=bks)
        for s in SEEDS_P:
            solver = HGWO_VNS(seed=s, **kwargs)
            t0 = time.perf_counter()
            res = solver.solve(inst)
            el = time.perf_counter() - t0
            ok, _ = validate_solution(inst, res.best_routes, strict_fleet=True)
            if not ok:
                valid_all = False
            gaps.append(100.0 * (res.best_cost - bks) / bks)
            times.append(el)
    return {
        "Parameter": param_name,
        "Value": str(param_val),
        "Suite": "Set P (3 inst x 3 seeds)",
        "Mean_Gap_pct": round(statistics.mean(gaps), 3),
        "Std_Gap_pct": round(statistics.stdev(gaps), 3),
        "Mean_Time_s": round(statistics.mean(times), 3),
        "Valid": "OK" if valid_all else "FAIL",
    }

def eval_golden(g_val):
    gaps = []
    times = []
    valid_all = True
    for name, bks in GOLDEN_INSTS:
        inst = load_golden_vrp(os.path.join(BENCH_DIR, "Golden", f"{name}.vrp"), granularity=g_val)
        for s in SEEDS_G:
            routes, cost, _, elapsed = solve_golden_hgwo_vns(inst, pop_size=12, max_iter=40, seed=s)
            ok, vcost, _ = independent_validate_float(inst, routes)
            if not ok:
                valid_all = False
            gaps.append(100.0 * (vcost - bks) / bks)
            times.append(elapsed)
    return {
        "Parameter": "Granularity G",
        "Value": str(g_val),
        "Suite": "Golden (2 inst x 2 seeds)",
        "Mean_Gap_pct": round(statistics.mean(gaps), 3),
        "Std_Gap_pct": round(statistics.stdev(gaps), 3),
        "Mean_Time_s": round(statistics.mean(times), 3),
        "Valid": "OK" if valid_all else "FAIL",
    }

rows = []
t_start = time.perf_counter()

for n_pop in [12, 18, 24, 36]:
    r = eval_set_p("Pack size N_pop", n_pop, pop_size=n_pop, elite_ratio=0.30, stagnation_limit=15)
    rows.append(r)
    print(r, flush=True)

for pe in [0.15, 0.30, 0.50, 1.00]:
    r = eval_set_p("Elite VND ratio p_elite", pe, pop_size=24, elite_ratio=pe, stagnation_limit=15)
    rows.append(r)
    print(r, flush=True)

for lmax in [6, 10, 15, 20]:
    r = eval_set_p("Stagnation limit L_max", lmax, pop_size=24, elite_ratio=0.30, stagnation_limit=lmax)
    rows.append(r)
    print(r, flush=True)

for g in [12, 20, 28, 40]:
    r = eval_golden(g)
    rows.append(r)
    print(r, flush=True)

csv_path = os.path.join(OUT_DIR, "parameter_sensitivity_results.csv")
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"DONE in {time.perf_counter() - t_start:.2f}s -> {csv_path}")
