"""
Évaluation grande échelle optimisée de HGWO-VNS (VND granulaire étendue + Sweep/Split)
sur les 12 instances CVRP pures du benchmark Golden et al. (1998) : Golden_9 à Golden_20 (240 à 483 clients).

Améliorations pour une meilleure optimalité grande échelle :
  - Initialisation tri-modale : Clarke & Wright randomisé + Sweep polaire multi-angles + Split 1D optimal.
  - VND granulaire étendue (G = 28 voisins) : 2-opt intra, Relocate (avant/après v),
    Or-opt (chaîne de 2 clients), Swap 1-1 et 2-opt* (deux orientations).
  - Paramètres renforcés : pop_size=14, max_iter=45, destroy_rate adaptatif (10%–20%), 2 runs.
"""

from __future__ import annotations

import csv
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GOLDEN_PURE_BKS: Dict[str, Tuple[int, float]] = {
    "Golden_9":  (14, 579.70),
    "Golden_10": (16, 735.43),
    "Golden_11": (17, 911.98),
    "Golden_12": (19, 1100.67),
    "Golden_13": (26, 857.19),
    "Golden_14": (29, 1080.55),
    "Golden_15": (33, 1337.27),
    "Golden_16": (36, 1611.28),
    "Golden_17": (22, 707.76),
    "Golden_18": (27, 995.13),
    "Golden_19": (33, 1365.60),
    "Golden_20": (38, 1817.59),
}


@dataclass
class FloatCVRPInstance:
    name: str
    capacity: int
    num_vehicles: int
    demands: List[int]
    coords: List[Tuple[float, float]]
    dist: List[List[float]]
    neighbors: List[List[int]]
    angles: List[float]
    bks: float

    @property
    def n(self) -> int:
        return len(self.demands) - 1

    def route_cost(self, r: Sequence[int]) -> float:
        if not r:
            return 0.0
        d = self.dist
        c = d[0][r[0]]
        for i in range(len(r) - 1):
            c += d[r[i]][r[i + 1]]
        c += d[r[-1]][0]
        return c

    def solution_cost(self, routes: Sequence[Sequence[int]]) -> float:
        return sum(self.route_cost(r) for r in routes if r)


def load_golden_vrp(filepath: str, granularity: int = 28) -> FloatCVRPInstance:
    name = os.path.splitext(os.path.basename(filepath))[0]
    capacity = 0
    dim = 0
    coords_map: Dict[int, Tuple[float, float]] = {}
    demands_map: Dict[int, int] = {}
    section = None

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line or line == "EOF":
                continue
            up = line.upper()
            if up.startswith("CAPACITY"):
                capacity = int(float(line.split(":", 1)[1].strip()))
            elif up.startswith("DIMENSION"):
                dim = int(line.split(":", 1)[1].strip())
            elif up.startswith("NODE_COORD_SECTION"):
                section = "COORD"
            elif up.startswith("DEMAND_SECTION"):
                section = "DEMAND"
            elif up.startswith("DEPOT_SECTION"):
                section = "DEPOT"
            elif section == "COORD":
                p = line.split()
                if len(p) >= 3:
                    coords_map[int(p[0])] = (float(p[1]), float(p[2]))
            elif section == "DEMAND":
                p = line.split()
                if len(p) >= 2:
                    demands_map[int(p[0])] = int(float(p[1]))

    coords = [coords_map[i] for i in range(1, dim + 1)]
    demands = [demands_map[i] for i in range(1, dim + 1)]
    dist = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        xi, yi = coords[i]
        di = dist[i]
        for j in range(i + 1, dim):
            xj, yj = coords[j]
            val = math.hypot(xi - xj, yi - yj)
            di[j] = val
            dist[j][i] = val

    x0, y0 = coords[0]
    angles = [0.0] + [math.atan2(coords[i][1] - y0, coords[i][0] - x0) for i in range(1, dim)]

    neighbors: List[List[int]] = [[]]
    for i in range(1, dim):
        di = dist[i]
        cand = sorted((j for j in range(1, dim) if j != i), key=lambda j: di[j])[:granularity]
        neighbors.append(cand)

    k_ref, bks = GOLDEN_PURE_BKS[name]
    return FloatCVRPInstance(
        name=name,
        capacity=capacity,
        num_vehicles=k_ref,
        demands=demands,
        coords=coords,
        dist=dist,
        neighbors=neighbors,
        angles=angles,
        bks=bks,
    )


def independent_validate_float(inst: FloatCVRPInstance, routes: Sequence[Sequence[int]]) -> Tuple[bool, float, str]:
    non_empty = [list(r) for r in routes if r]
    seen: List[int] = []
    total_cost = 0.0
    for idx, r in enumerate(non_empty, 1):
        load = 0
        for c in r:
            if c < 1 or c > inst.n:
                return False, 0.0, f"Client invalide {c}"
            load += inst.demands[c]
            seen.append(c)
        if load > inst.capacity:
            return False, 0.0, f"Capacité dépassée route {idx} ({load}>{inst.capacity})"
        seq = [0] + r + [0]
        for a, b in zip(seq[:-1], seq[1:]):
            xa, ya = inst.coords[a]
            xb, yb = inst.coords[b]
            total_cost += math.hypot(xa - xb, ya - yb)

    if len(seen) != inst.n or len(set(seen)) != inst.n:
        return False, total_cost, f"Couverture clients invalide ({len(set(seen))}/{inst.n})"
    return True, total_cost, "OK"


def clarke_wright_float(inst: FloatCVRPInstance, rng: random.Random, noise: float = 0.0) -> List[List[int]]:
    n = inst.n
    Q = inst.capacity
    dem = inst.demands
    dist = inst.dist
    routes: Dict[int, List[int]] = {i: [i] for i in range(1, n + 1)}
    loads: Dict[int, int] = {i: dem[i] for i in range(1, n + 1)}
    owner: Dict[int, int] = {i: i for i in range(1, n + 1)}

    savings: List[Tuple[float, int, int]] = []
    for i in range(1, n + 1):
        d0i = dist[0][i]
        di = dist[i]
        for j in range(i + 1, n + 1):
            s = d0i + dist[0][j] - di[j]
            if s > 0:
                if noise > 0.0:
                    s *= rng.uniform(1.0 - noise, 1.0 + noise)
                savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    for _, i, j in savings:
        ri, rj = owner[i], owner[j]
        if ri == rj or loads[ri] + loads[rj] > Q:
            continue
        r_i, r_j = routes[ri], routes[rj]
        if r_i[-1] == i and r_j[0] == j:
            merged = r_i + r_j
        elif r_i[0] == i and r_j[-1] == j:
            merged = r_j + r_i
        elif r_i[-1] == i and r_j[-1] == j:
            merged = r_i + r_j[::-1]
        elif r_i[0] == i and r_j[0] == j:
            merged = r_i[::-1] + r_j
        else:
            continue
        routes[ri] = merged
        loads[ri] += loads[rj]
        for c in r_j:
            owner[c] = ri
        del routes[rj]
        del loads[rj]

    return list(routes.values())


def split_1d_float(perm: Sequence[int], inst: FloatCVRPInstance) -> Tuple[List[List[int]], float]:
    n = len(perm)
    Q = inst.capacity
    dem = inst.demands
    dist = inst.dist
    INF = 1e18
    v = [INF] * (n + 1)
    p = [-1] * (n + 1)
    v[0] = 0.0
    for i in range(n):
        base = v[i]
        if base >= INF:
            continue
        load = 0
        cost = 0.0
        prev = 0
        for j in range(i, n):
            node = perm[j]
            load += dem[node]
            if load > Q:
                break
            cost += dist[prev][node]
            cand = base + cost + dist[node][0]
            if cand < v[j + 1]:
                v[j + 1] = cand
                p[j + 1] = i
            prev = node
    routes: List[List[int]] = []
    pos = n
    while pos > 0:
        prev_pos = p[pos]
        routes.append(list(perm[prev_pos:pos]))
        pos = prev_pos
    routes.reverse()
    return routes, v[n]


def polar_sweep_init(inst: FloatCVRPInstance, rng: random.Random, reverse_dir: bool = False) -> List[List[int]]:
    """Initialisation Sweep polaire à partir d'un angle aléatoire + découpage optimal Split 1D."""
    offset = rng.uniform(-math.pi, math.pi)
    two_pi = 2.0 * math.pi
    nodes = list(range(1, inst.n + 1))
    nodes.sort(key=lambda c: (inst.angles[c] - offset) % two_pi, reverse=reverse_dir)
    routes, _ = split_1d_float(nodes, inst)
    return routes


def two_opt_route_float(r: List[int], dist: List[List[float]]) -> bool:
    m = len(r)
    if m < 2:
        return False
    any_imp = False
    improved = True
    while improved:
        improved = False
        for i in range(m - 1):
            a = 0 if i == 0 else r[i - 1]
            b = r[i]
            dab = dist[a][b]
            for j in range(i + 1, m):
                c = r[j]
                d = 0 if j == m - 1 else r[j + 1]
                if (dist[a][c] + dist[b][d]) < (dab + dist[c][d]) - 1e-8:
                    r[i : j + 1] = reversed(r[i : j + 1])
                    improved = True
                    any_imp = True
                    break
            if improved:
                break
    return any_imp


def granular_vnd_extended(
    routes: Sequence[Sequence[int]],
    inst: FloatCVRPInstance,
    max_passes: int = 14,
) -> Tuple[List[List[int]], float]:
    """
    VND granulaire étendue :
      1. 2-opt intra-tournée
      2. Relocate granulaire (insertion avant ET après chaque voisin v)
      3. Or-opt granulaire (déplacement d'une chaîne de 2 clients (u, u_next) après v)
      4. Swap 1-1 granulaire inter-tournées
      5. 2-opt* granulaire (orientation directe ET inversée)
    """
    cur = [list(r) for r in routes if r]
    dist = inst.dist
    dem = inst.demands
    Q = inst.capacity
    nbrs = inst.neighbors

    for r in cur:
        two_opt_route_float(r, dist)

    for _ in range(max_passes):
        any_improved = False
        r_of = [0] * (inst.n + 1)
        p_of = [0] * (inst.n + 1)
        loads = [0] * len(cur)
        for ri, r in enumerate(cur):
            ld = 0
            for pi, c in enumerate(r):
                r_of[c] = ri
                p_of[c] = pi
                ld += dem[c]
            loads[ri] = ld

        dirty = set()

        # 1. Relocate granulaire (avant j ET après j+1)
        for u in range(1, inst.n + 1):
            r1_idx = r_of[u]
            r1 = cur[r1_idx]
            i = p_of[u]
            if i >= len(r1) or r1[i] != u:
                continue
            len1 = len(r1)
            qu = dem[u]
            p1 = 0 if i == 0 else r1[i - 1]
            n1 = 0 if i == len1 - 1 else r1[i + 1]
            rem_delta = dist[p1][n1] - dist[p1][u] - dist[u][n1]

            best_delta = -1e-6
            best_target = None
            for v in nbrs[u]:
                r2_idx = r_of[v]
                if r1_idx != r2_idx and loads[r2_idx] + qu > Q:
                    continue
                r2 = cur[r2_idx]
                j = p_of[v]
                if j >= len(r2) or r2[j] != v:
                    continue
                len2 = len(r2)
                # Position après v (j + 1)
                ins_pos = j + 1
                if not (r1_idx == r2_idx and (ins_pos == i or ins_pos == i + 1)):
                    n2 = 0 if ins_pos == len2 else r2[ins_pos]
                    tot = rem_delta + dist[v][u] + dist[u][n2] - dist[v][n2]
                    if tot < best_delta:
                        best_delta = tot
                        best_target = (r2_idx, ins_pos)
                # Position avant v (j)
                ins_pos = j
                if not (r1_idx == r2_idx and (ins_pos == i or ins_pos == i + 1)):
                    p2 = 0 if ins_pos == 0 else r2[ins_pos - 1]
                    tot = rem_delta + dist[p2][u] + dist[u][v] - dist[p2][v]
                    if tot < best_delta:
                        best_delta = tot
                        best_target = (r2_idx, ins_pos)

            if best_target is not None:
                r2_idx, ins_pos = best_target
                r1.pop(i)
                if r1_idx == r2_idx:
                    if ins_pos > i:
                        ins_pos -= 1
                    r1.insert(ins_pos, u)
                    for idx_p, node_p in enumerate(r1):
                        p_of[node_p] = idx_p
                    dirty.add(r1_idx)
                else:
                    cur[r2_idx].insert(ins_pos, u)
                    loads[r1_idx] -= qu
                    loads[r2_idx] += qu
                    for idx_p, node_p in enumerate(r1):
                        p_of[node_p] = idx_p
                    for idx_p, node_p in enumerate(cur[r2_idx]):
                        r_of[node_p] = r2_idx
                        p_of[node_p] = idx_p
                    dirty.add(r1_idx)
                    dirty.add(r2_idx)
                any_improved = True

        # 2. Or-opt granulaire (chaîne de 2 clients consécutifs (u, w) insérée après v)
        for u in range(1, inst.n + 1):
            r1_idx = r_of[u]
            r1 = cur[r1_idx]
            i = p_of[u]
            if i + 1 >= len(r1) or r1[i] != u:
                continue
            w = r1[i + 1]
            q_seg = dem[u] + dem[w]
            p1 = 0 if i == 0 else r1[i - 1]
            n1 = 0 if i + 1 == len(r1) - 1 else r1[i + 2]
            rem_delta = dist[p1][n1] - dist[p1][u] - dist[w][n1]

            best_delta = -1e-6
            best_move = None
            for v in nbrs[u]:
                if v == w:
                    continue
                r2_idx = r_of[v]
                if r1_idx != r2_idx and loads[r2_idx] + q_seg > Q:
                    continue
                r2 = cur[r2_idx]
                j = p_of[v]
                if j >= len(r2) or r2[j] != v:
                    continue
                ins_pos = j + 1
                if r1_idx == r2_idx and i <= ins_pos <= i + 2:
                    continue
                n2 = 0 if ins_pos == len(r2) else r2[ins_pos]
                # Direct (u, w)
                tot_fwd = rem_delta + dist[v][u] + dist[w][n2] - dist[v][n2]
                if tot_fwd < best_delta:
                    best_delta = tot_fwd
                    best_move = (r2_idx, ins_pos, False)
                # Inversé (w, u)
                tot_rev = rem_delta + dist[v][w] + dist[u][n2] - dist[v][n2]
                if tot_rev < best_delta:
                    best_delta = tot_rev
                    best_move = (r2_idx, ins_pos, True)

            if best_move is not None:
                r2_idx, ins_pos, rev = best_move
                seg = [w, u] if rev else [u, w]
                del r1[i : i + 2]
                if r1_idx == r2_idx:
                    if ins_pos > i:
                        ins_pos -= 2
                    r1[ins_pos:ins_pos] = seg
                    for idx_p, node_p in enumerate(r1):
                        p_of[node_p] = idx_p
                    dirty.add(r1_idx)
                else:
                    cur[r2_idx][ins_pos:ins_pos] = seg
                    loads[r1_idx] -= q_seg
                    loads[r2_idx] += q_seg
                    for idx_p, node_p in enumerate(r1):
                        p_of[node_p] = idx_p
                    for idx_p, node_p in enumerate(cur[r2_idx]):
                        r_of[node_p] = r2_idx
                        p_of[node_p] = idx_p
                    dirty.add(r1_idx)
                    dirty.add(r2_idx)
                any_improved = True

        # 3. Swap 1-1 granulaire inter-tournées
        for u in range(1, inst.n + 1):
            r1_idx = r_of[u]
            r1 = cur[r1_idx]
            i = p_of[u]
            if i >= len(r1) or r1[i] != u:
                continue
            len1 = len(r1)
            qu = dem[u]
            p1 = 0 if i == 0 else r1[i - 1]
            n1 = 0 if i == len1 - 1 else r1[i + 1]
            old1 = dist[p1][u] + dist[u][n1]

            for v in nbrs[u]:
                r2_idx = r_of[v]
                if r2_idx <= r1_idx:
                    continue
                qv = dem[v]
                if loads[r1_idx] - qu + qv > Q or loads[r2_idx] - qv + qu > Q:
                    continue
                r2 = cur[r2_idx]
                j = p_of[v]
                if j >= len(r2) or r2[j] != v:
                    continue
                p2 = 0 if j == 0 else r2[j - 1]
                n2 = 0 if j == len(r2) - 1 else r2[j + 1]
                old2 = dist[p2][v] + dist[v][n2]
                new1 = dist[p1][v] + dist[v][n1]
                new2 = dist[p2][u] + dist[u][n2]
                if (new1 + new2) < (old1 + old2) - 1e-6:
                    r1[i], r2[j] = v, u
                    r_of[u], p_of[u] = r2_idx, j
                    r_of[v], p_of[v] = r1_idx, i
                    loads[r1_idx] = loads[r1_idx] - qu + qv
                    loads[r2_idx] = loads[r2_idx] - qv + qu
                    dirty.add(r1_idx)
                    dirty.add(r2_idx)
                    any_improved = True
                    break

        # 4. 2-opt* granulaire (deux orientations : directe + inversée)
        prefs = []
        for r in cur:
            pr = [0] * (len(r) + 1)
            for idx_p, c in enumerate(r):
                pr[idx_p + 1] = pr[idx_p] + dem[c]
            prefs.append(pr)

        star_done = False
        for u in range(1, inst.n + 1):
            r1_idx = r_of[u]
            r1 = cur[r1_idx]
            if not r1:
                continue
            i = p_of[u] + 1
            if i > len(r1) or r1[i - 1] != u:
                continue
            len1 = len(r1)
            b = 0 if i == len1 else r1[i]
            h1 = prefs[r1_idx][i]
            t1 = prefs[r1_idx][len1] - h1

            for v in nbrs[u]:
                r2_idx = r_of[v]
                if r2_idx == r1_idx:
                    continue
                r2 = cur[r2_idx]
                if not r2:
                    continue
                j = p_of[v]
                if j >= len(r2) or r2[j] != v:
                    continue
                len2 = len(r2)
                c = 0 if j == 0 else r2[j - 1]
                h2 = prefs[r2_idx][j]
                t2 = prefs[r2_idx][len2] - h2

                # Orientation 1 : head1 + tail2 et head2 + tail1
                if h1 + t2 <= Q and h2 + t1 <= Q:
                    if (dist[u][v] + dist[c][b]) < (dist[u][b] + dist[c][v]) - 1e-6:
                        cur[r1_idx] = r1[:i] + r2[j:]
                        cur[r2_idx] = r2[:j] + r1[i:]
                        dirty.add(r1_idx)
                        dirty.add(r2_idx)
                        any_improved = True
                        star_done = True
                        break
                # Orientation 2 : head1 + rev(head2) et rev(tail1) + tail2
                if h1 + h2 <= Q and t1 + t2 <= Q and j > 0 and i < len1:
                    if (dist[u][c] + dist[b][v]) < (dist[u][b] + dist[c][v]) - 1e-6:
                        cur[r1_idx] = r1[:i] + list(reversed(r2[:j]))
                        cur[r2_idx] = list(reversed(r1[i:])) + r2[j:]
                        dirty.add(r1_idx)
                        dirty.add(r2_idx)
                        any_improved = True
                        star_done = True
                        break
            if star_done:
                break

        for ri in dirty:
            if ri < len(cur) and cur[ri]:
                two_opt_route_float(cur[ri], dist)
        cur = [r for r in cur if r]

        if not any_improved:
            break

    return cur, inst.solution_cost(cur)


def destroy_repair_regret(
    routes: Sequence[Sequence[int]],
    inst: FloatCVRPInstance,
    rng: random.Random,
    destroy_rate: float = 0.14,
) -> List[List[int]]:
    cur = [list(r) for r in routes if r]
    num_rem = max(8, min(45, int(round(inst.n * destroy_rate))))
    seed = rng.randint(1, inst.n)
    removed = [seed] + inst.neighbors[seed][: num_rem - 1]
    removed_set = set(removed)
    partial = [[c for c in r if c not in removed_set] for r in cur]
    partial = [r for r in partial if r]

    loads = [sum(inst.demands[c] for c in r) for r in partial]
    dist = inst.dist
    Q = inst.capacity
    dem = inst.demands

    # Insertion bruitée par plus forte demande / proximité
    rem_list = list(removed)
    if rng.random() < 0.5:
        rem_list.sort(key=lambda c: dem[c] * rng.uniform(0.85, 1.15), reverse=True)
    else:
        rng.shuffle(rem_list)

    for c in rem_list:
        qc = dem[c]
        best_delta = 1e18
        best_loc = None
        for ri, r in enumerate(partial):
            if loads[ri] + qc > Q:
                continue
            rlen = len(r)
            for pos in range(rlen + 1):
                p = 0 if pos == 0 else r[pos - 1]
                nx = 0 if pos == rlen else r[pos]
                delta = dist[p][c] + dist[c][nx] - dist[p][nx]
                if delta < best_delta:
                    best_delta = delta
                    best_loc = (ri, pos)
        if best_loc is not None:
            ri, pos = best_loc
            partial[ri].insert(pos, c)
            loads[ri] += qc
        else:
            partial.append([c])
            loads.append(qc)
    return partial


def rbx_float(r_parent: Sequence[Sequence[int]], r_leader: Sequence[Sequence[int]], rng: random.Random) -> List[List[int]]:
    lead = [list(r) for r in r_leader if r]
    k_take = max(1, rng.randint(1, max(1, len(lead) // 3)))
    chosen = [list(r) for r in rng.sample(lead, k_take)]
    covered = {c for r in chosen for c in r}
    child = chosen
    for r in r_parent:
        rem = [c for c in r if c not in covered]
        if rem:
            child.append(rem)
            covered.update(rem)
    return child


def solve_golden_hgwo_vns(
    inst: FloatCVRPInstance,
    pop_size: int = 14,
    max_iter: int = 100,
    seed: int = 17,
) -> Tuple[List[List[int]], float, float, float]:
    t0 = time.perf_counter()
    rng = random.Random(seed)

    cw_routes = clarke_wright_float(inst, rng, noise=0.0)
    cw_cost = inst.solution_cost(cw_routes)

    routes_list: List[List[List[int]]] = []
    costs: List[float] = []
    stag = [0] * pop_size

    # Initialisation tri-modale : Clarke-Wright (40%), Sweep polaire + Split (50%), aléatoire + Split (10%)
    for idx in range(pop_size):
        if idx == 0:
            r0 = cw_routes
        elif idx < max(3, int(pop_size * 0.40)):
            r0 = clarke_wright_float(inst, rng, noise=0.14)
        elif idx < int(pop_size * 0.90):
            r0 = polar_sweep_init(inst, rng, reverse_dir=(idx % 2 == 1))
        else:
            perm = list(range(1, inst.n + 1))
            rng.shuffle(perm)
            r0, _ = split_1d_float(perm, inst)
        r_imp, c_imp = granular_vnd_extended(r0, inst, max_passes=18)
        routes_list.append(r_imp)
        costs.append(c_imp)

    order = sorted(range(pop_size), key=lambda i: costs[i])
    best_routes = [list(r) for r in routes_list[order[0]]]
    best_cost = costs[order[0]]
    global_stag = 0

    for it in range(1, max_iter + 1):
        a = 2.0 * (1.0 - it / max_iter)
        order = sorted(range(pop_size), key=lambda i: costs[i])
        alpha, beta, delta = order[0], order[1], order[2]
        improved_global = False

        for i in range(pop_size):
            A = 2.0 * a * rng.random() - a
            u = rng.random()
            leader = alpha if u < 0.50 else (beta if u < 0.80 else delta)

            if abs(A) >= 1.0:
                cand_r = destroy_repair_regret(routes_list[leader], inst, rng, destroy_rate=0.12)
            else:
                cand_r = rbx_float(routes_list[i], routes_list[leader], rng)

            cand_r, cand_c = granular_vnd_extended(cand_r, inst, max_passes=16)
            if cand_c < costs[i] - 1e-6:
                routes_list[i] = cand_r
                costs[i] = cand_c
                stag[i] = 0
                if cand_c < best_cost - 1e-6:
                    best_cost = cand_c
                    best_routes = [list(r) for r in cand_r]
                    improved_global = True
            else:
                stag[i] += 1

            if stag[i] >= 6:
                pert = destroy_repair_regret(routes_list[i], inst, rng, destroy_rate=0.18)
                pert, pc = granular_vnd_extended(pert, inst, max_passes=16)
                if pc < costs[i] - 1e-6 or i != alpha:
                    routes_list[i] = pert
                    costs[i] = pc
                if pc < best_cost - 1e-6:
                    best_cost = pc
                    best_routes = [list(r) for r in pert]
                    improved_global = True
                stag[i] = 0

        if improved_global:
            global_stag = 0
        else:
            global_stag += 1
            if global_stag >= 25:
                break

    elapsed = time.perf_counter() - t0
    return best_routes, best_cost, cw_cost, elapsed


def main() -> None:
    folder = "../intermediate/benchmarks/Golden"
    ordered_names = [f"Golden_{i}" for i in range(9, 21)]
    rows = []

    header = (
        f"{'Instance':<10} | {'n':>3} | {'K_ref':>5} | {'K_obt':>5} | {'BKS':>8} | "
        f"{'CW':>8} | {'HGWO Best':>9} | {'Gap BKS':>7} | {'Gain/CW':>8} | {'Temps(s)':>8} | {'Valide':>6}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for name in ordered_names:
        fp = os.path.join(folder, f"{name}.vrp")
        inst = load_golden_vrp(fp, granularity=28)
        best_cost = 1e18
        best_routes = None
        times = []
        cw_c = 0.0

        for seed in (17, 34):
            r, c, cw_cost, dt = solve_golden_hgwo_vns(inst, pop_size=14, max_iter=100, seed=seed)
            cw_c = cw_cost
            times.append(dt)
            if c < best_cost:
                best_cost = c
                best_routes = r

        assert best_routes is not None
        ok, exact_cost, msg = independent_validate_float(inst, best_routes)
        assert ok, f"Invalide sur {name}: {msg}"
        sol_dir = "../intermediate/repro_bundle/solutions_sol/Golden"
        os.makedirs(sol_dir, exist_ok=True)
        with open(os.path.join(sol_dir, f"{name}.sol"), "w", encoding="utf-8") as sf:
            for r_i, rt in enumerate([r for r in best_routes if r], 1):
                sf.write(f"Route #{r_i}: " + " ".join(str(x) for x in rt) + "\n")
            sf.write(f"Cost {exact_cost:.2f}\n")
        gap_bks = (exact_cost - inst.bks) / inst.bks * 100.0
        gain_cw = (cw_c - exact_cost) / cw_c * 100.0
        mean_t = statistics.mean(times)
        k_obt = len(best_routes)

        row = {
            "Instance": name,
            "n": inst.n,
            "K_ref": inst.num_vehicles,
            "K_obt": k_obt,
            "BKS": inst.bks,
            "CW_Cost": round(cw_c, 2),
            "HGWO_Best": round(exact_cost, 2),
            "Gap_BKS_pct": round(gap_bks, 2),
            "Gain_vs_CW_pct": round(gain_cw, 2),
            "Time_s": round(mean_t, 2),
            "Valid": "OK" if ok else "ERR",
        }
        rows.append(row)
        print(
            f"{name:<10} | {inst.n:>3d} | {inst.num_vehicles:>5d} | {k_obt:>5d} | {inst.bks:>8.2f} | "
            f"{cw_c:>8.2f} | {exact_cost:>9.2f} | {gap_bks:>6.2f}% | {gain_cw:>+7.2f}% | {mean_t:>8.2f} | {row['Valid']:>6}",
            flush=True,
        )

    with open("resultats_golden_cvrp.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    avg_gap = statistics.mean(r["Gap_BKS_pct"] for r in rows)
    avg_gain = statistics.mean(r["Gain_vs_CW_pct"] for r in rows)
    avg_time = statistics.mean(r["Time_s"] for r in rows)
    print("-" * len(header), flush=True)
    print(
        f"MOYENNE (12 instances Golden 240–483 clients) | Gap BKS = {avg_gap:.2f}% | "
        f"Gain vs CW = +{avg_gain:.2f}% | Temps moyen = {avg_time:.2f} s",
        flush=True,
    )

    names = [r["Instance"].replace("Golden_", "G") for r in rows]
    cw_gaps = [(r["CW_Cost"] - r["BKS"]) / r["BKS"] * 100.0 for r in rows]
    hgwo_gaps = [r["Gap_BKS_pct"] for r in rows]
    ns = [r["n"] for r in rows]
    ts = [r["Time_s"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2), dpi=220)
    x = np.arange(len(names))
    w_bar = 0.36
    ax1.bar(x - w_bar / 2, cw_gaps, width=w_bar, color="#7f7f7f", alpha=0.82, label="Clarke & Wright (baseline)")
    ax1.bar(x + w_bar / 2, hgwo_gaps, width=w_bar, color="#2ca02c", alpha=0.90, label=f"HGWO-VNS optimisé (moy. {avg_gap:.2f}%)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=8.5)
    ax1.set_ylabel("Écart au BKS CVRPLIB (%)", fontsize=9.5)
    ax1.set_title("A. Qualité grande échelle (Golden 9–20 : 240 à 483 clients)", fontsize=10.5, fontweight="bold")
    ax1.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper right", fontsize=8.5, frameon=True)

    order_n = sorted(range(len(ns)), key=lambda i: ns[i])
    ax2.plot([ns[i] for i in order_n], [ts[i] for i in order_n], marker="s", color="#1f77b4", linewidth=2.0, label="HGWO-VNS (VND granulaire étendue)")
    ax2.set_xlabel("Nombre de clients (n)", fontsize=9.5)
    ax2.set_ylabel("Temps CPU moyen (s)", fontsize=9.5)
    ax2.set_title("B. Passage à l'échelle : Temps CPU vs taille n", fontsize=10.5, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="upper left", fontsize=8.5, frameon=True)

    plt.tight_layout(pad=2.0)
    plt.savefig("benchmark_golden_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Graphique mis à jour : benchmark_golden_comparison.png", flush=True)


if __name__ == "__main__":
    main()
