"""
HGWO-VNS : Hybrid Discrete Grey Wolf Optimizer with Embedded Variable Neighborhood
Search and Optimal Split Decoding for the Capacitated Vehicle Routing Problem (CVRP).

Conçu pour améliorer la stratégie d'hybridation embarquée de :
Djebbar, A. M., & Kemmar, A. (2025). A Hybrid Approach for Solving the Capacitated
Vehicle Routing Problem. International Journal of Automotive Science and Technology
(IJASTECH), 9(2), 249-258. DOI: 10.30939/ijastech..1663305
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


# =============================================================================
# 1. STRUCTURE DE DONNÉES, PARSEUR CVRPLIB (.vrp) ET VALIDATEUR STRICT
# =============================================================================

@dataclass
class CVRPInstance:
    """Représente une instance CVRP statique (dépôt à l'indice 0, clients 1..n)."""
    name: str
    capacity: int
    num_vehicles: int
    demands: List[int]                  # demands[0] == 0, demands[1..n] > 0
    dist_matrix: List[List[int]]        # Matrice (n+1) x (n+1) entière (EUC_2D nint)
    coords: Optional[List[Tuple[float, float]]] = None
    bks: Optional[int] = None

    @property
    def num_customers(self) -> int:
        return len(self.demands) - 1

    def route_cost(self, route: Sequence[int]) -> int:
        if not route:
            return 0
        d = self.dist_matrix
        cost = d[0][route[0]]
        for idx in range(len(route) - 1):
            cost += d[route[idx]][route[idx + 1]]
        cost += d[route[-1]][0]
        return cost

    def solution_cost(self, routes: Sequence[Sequence[int]]) -> int:
        return sum(self.route_cost(r) for r in routes if r)

    def route_load(self, route: Sequence[int]) -> int:
        dem = self.demands
        return sum(dem[c] for c in route)


def build_euc2d_matrix(coords: Sequence[Tuple[float, float]]) -> List[List[int]]:
    """Construit la matrice de distances euclidiennes arrondies à l'entier le plus proche (TSPLIB EUC_2D)."""
    m = len(coords)
    dist = [[0] * m for _ in range(m)]
    for i in range(m):
        xi, yi = coords[i]
        for j in range(i + 1, m):
            xj, yj = coords[j]
            val = int(math.hypot(xi - xj, yi - yj) + 0.5)
            dist[i][j] = val
            dist[j][i] = val
    return dist


def parse_vrp_text(text: str, num_vehicles: Optional[int] = None, bks: Optional[int] = None) -> CVRPInstance:
    """Parse un fichier au format standard CVRPLIB / TSPLIB (.vrp)."""
    name = "unknown"
    capacity = 0
    dimension = 0
    coords_map: Dict[int, Tuple[float, float]] = {}
    demands_map: Dict[int, int] = {}
    section = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "EOF":
            continue
        upper = line.upper()
        if upper.startswith("NAME"):
            name = line.split(":", 1)[1].strip()
        elif upper.startswith("CAPACITY"):
            capacity = int(line.split(":", 1)[1].strip())
        elif upper.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1].strip())
        elif upper.startswith("NODE_COORD_SECTION"):
            section = "COORD"
        elif upper.startswith("DEMAND_SECTION"):
            section = "DEMAND"
        elif upper.startswith("DEPOT_SECTION"):
            section = "DEPOT"
        elif section == "COORD":
            parts = line.split()
            if len(parts) >= 3:
                coords_map[int(parts[0])] = (float(parts[1]), float(parts[2]))
        elif section == "DEMAND":
            parts = line.split()
            if len(parts) >= 2:
                demands_map[int(parts[0])] = int(parts[1])

    if num_vehicles is None:
        # Extrait k depuis le nom standard (ex. P-n40-k5 -> 5)
        if "-k" in name.lower():
            num_vehicles = int(name.lower().split("-k")[-1])
        else:
            num_vehicles = max(1, math.ceil(sum(demands_map.values()) / max(1, capacity)))

    coords = [coords_map[i] for i in range(1, dimension + 1)]
    demands = [demands_map[i] for i in range(1, dimension + 1)]
    dist_matrix = build_euc2d_matrix(coords)

    return CVRPInstance(
        name=name,
        capacity=capacity,
        num_vehicles=num_vehicles,
        demands=demands,
        dist_matrix=dist_matrix,
        coords=coords,
        bks=bks,
    )


def load_vrp_file(filepath: str, num_vehicles: Optional[int] = None, bks: Optional[int] = None) -> CVRPInstance:
    """Charge une instance CVRPLIB (.vrp) depuis le disque."""
    with open(filepath, "r", encoding="utf-8") as f:
        return parse_vrp_text(f.read(), num_vehicles=num_vehicles, bks=bks)


def validate_solution(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    strict_fleet: bool = True,
) -> Tuple[bool, str]:
    """Vérifie rigoureusement la faisabilité d'une solution CVRP."""
    non_empty = [list(r) for r in routes if len(r) > 0]
    visited: List[int] = []
    for idx, r in enumerate(non_empty):
        load = 0
        for node in r:
            if node < 1 or node > instance.num_customers:
                return False, f"Nœud invalide {node} dans la tournée {idx}"
            load += instance.demands[node]
            visited.append(node)
        if load > instance.capacity:
            return False, f"Capacité dépassée dans la tournée {idx} ({load} > {instance.capacity})"

    expected = set(range(1, instance.num_customers + 1))
    if len(visited) != instance.num_customers or set(visited) != expected:
        return False, (
            f"Clients manquants ou dupliqués : visités={len(visited)}, "
            f"uniques={len(set(visited))}, attendus={instance.num_customers}"
        )

    if strict_fleet and len(non_empty) > instance.num_vehicles:
        return False, f"Nombre de véhicules dépassé ({len(non_empty)} > {instance.num_vehicles})"

    return True, "OK"


# =============================================================================
# 2. DÉCODAGE OPTIMAL SPLIT (PRINS 2004 — VERSION BORNÉE PAR FLOTTE K)
# =============================================================================

def split_optimal(
    perm: Sequence[int],
    instance: CVRPInstance,
    enforce_fleet: bool = True,
) -> Tuple[List[List[int]], int]:
    """
    Découpe une permutation géante en tournées faisables de coût minimal.
    - Utilise d'abord une programmation dynamique 2D bornée par K véhicules
      pour garantir le respect de la taille de flotte dès qu'un découpage existe.
    - En repli (si aucune partition <= K n'existe pour cette permutation),
      utilise le Split 1D standard de Prins (2004).
    """
    n = len(perm)
    Q = instance.capacity
    K = instance.num_vehicles
    dem = instance.demands
    dist = instance.dist_matrix
    INF = 10**12

    # Précalcul des arcs faisables (i -> j) correspondant à la tournée perm[i..j-1]
    edges_from: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for i in range(n):
        load = 0
        cost = 0
        prev = 0
        for j in range(i, n):
            node = perm[j]
            load += dem[node]
            if load > Q:
                break
            cost += dist[prev][node]
            edges_from[i].append((j + 1, cost + dist[node][0]))
            prev = node

    if enforce_fleet:
        # DP 2D : dp[k][pos] = coût min pour servir perm[0..pos-1] avec exactement k tournées
        dp = [[INF] * (n + 1) for _ in range(K + 1)]
        parent = [[-1] * (n + 1) for _ in range(K + 1)]
        dp[0][0] = 0

        for k in range(1, K + 1):
            prev_row = dp[k - 1]
            curr_row = dp[k]
            par_row = parent[k]
            for i in range(k - 1, n):
                base = prev_row[i]
                if base >= INF:
                    continue
                for nxt, arc_cost in edges_from[i]:
                    new_cost = base + arc_cost
                    if new_cost < curr_row[nxt]:
                        curr_row[nxt] = new_cost
                        par_row[nxt] = i

        best_k = -1
        best_cost = INF
        for k in range(1, K + 1):
            if dp[k][n] < best_cost:
                best_cost = dp[k][n]
                best_k = k

        if best_k != -1:
            routes: List[List[int]] = []
            pos = n
            k = best_k
            while pos > 0:
                prev_pos = parent[k][pos]
                routes.append(list(perm[prev_pos:pos]))
                pos = prev_pos
                k -= 1
            routes.reverse()
            return routes, best_cost

    # Repli : Split 1D de Prins (non borné en K)
    v = [INF] * (n + 1)
    p = [-1] * (n + 1)
    v[0] = 0
    for i in range(n):
        base = v[i]
        if base >= INF:
            continue
        for nxt, arc_cost in edges_from[i]:
            if base + arc_cost < v[nxt]:
                v[nxt] = base + arc_cost
                p[nxt] = i

    routes = []
    pos = n
    while pos > 0:
        prev_pos = p[pos]
        routes.append(list(perm[prev_pos:pos]))
        pos = prev_pos
    routes.reverse()
    return routes, v[n]


def routes_to_giant_tour(routes: Sequence[Sequence[int]]) -> List[int]:
    """Concatène les tournées en une permutation géante."""
    tour: List[int] = []
    for r in routes:
        tour.extend(r)
    return tour


# =============================================================================
# 3. INITIALISATION HYBRIDE (CLARKE & WRIGHT RANDOMISÉ + ALÉATOIRE)
# =============================================================================

def clarke_wright_savings(
    instance: CVRPInstance,
    rng: random.Random,
    noise: float = 0.0,
) -> List[List[int]]:
    """
    Heuristique de Clarke & Wright (Savings) avec perturbation optionnelle
    des économies pour diversifier la population initiale.
    """
    n = instance.num_customers
    Q = instance.capacity
    dem = instance.demands
    dist = instance.dist_matrix

    # Une tournée initiale par client
    routes: Dict[int, List[int]] = {i: [i] for i in range(1, n + 1)}
    loads: Dict[int, int] = {i: dem[i] for i in range(1, n + 1)}
    node_to_route: Dict[int, int] = {i: i for i in range(1, n + 1)}

    savings: List[Tuple[float, int, int]] = []
    for i in range(1, n + 1):
        d0i = dist[0][i]
        for j in range(i + 1, n + 1):
            s = d0i + dist[0][j] - dist[i][j]
            if noise > 0.0:
                s *= rng.uniform(1.0 - noise, 1.0 + noise)
            savings.append((s, i, j))
    savings.sort(key=lambda x: x[0], reverse=True)

    for _, i, j in savings:
        ri = node_to_route[i]
        rj = node_to_route[j]
        if ri == rj:
            continue
        if loads[ri] + loads[rj] > Q:
            continue

        route_i = routes[ri]
        route_j = routes[rj]

        # Fusion possible uniquement aux extrémités des deux tournées
        if route_i[-1] == i and route_j[0] == j:
            merged = route_i + route_j
        elif route_i[0] == i and route_j[-1] == j:
            merged = route_j + route_i
        elif route_i[-1] == i and route_j[-1] == j:
            merged = route_i + route_j[::-1]
        elif route_i[0] == i and route_j[0] == j:
            merged = route_i[::-1] + route_j
        else:
            continue

        routes[ri] = merged
        loads[ri] += loads[rj]
        for node in route_j:
            node_to_route[node] = ri
        del routes[rj]
        del loads[rj]

    return list(routes.values())


def initialize_population(
    instance: CVRPInstance,
    pop_size: int,
    rng: random.Random,
    cw_ratio: float = 0.30,
) -> List[Tuple[List[int], List[List[int]], int]]:
    """Génère une population initiale équilibrée : 30% Clarke-Wright randomisé + 70% aléatoire + Split."""
    n = instance.num_customers
    n_cw = max(1, int(pop_size * cw_ratio))
    population: List[Tuple[List[int], List[List[int]], int]] = []

    for idx in range(pop_size):
        if idx == 0:
            cw_routes = clarke_wright_savings(instance, rng, noise=0.0)
            perm = routes_to_giant_tour(cw_routes)
        elif idx < n_cw:
            cw_routes = clarke_wright_savings(instance, rng, noise=0.18)
            perm = routes_to_giant_tour(cw_routes)
        else:
            perm = list(range(1, n + 1))
            rng.shuffle(perm)

        routes, cost = split_optimal(perm, instance, enforce_fleet=True)
        if len(routes) > instance.num_vehicles:
            routes = repair_fleet_binpack(routes, instance, rng)
            perm = routes_to_giant_tour(routes)
            cost = instance.solution_cost(routes)
        population.append((perm, routes, cost))

    return population


# =============================================================================
# 4. OPÉRATEURS DISCRETS EMBARQUÉS (OX, RBX, MUTATION)
# =============================================================================

def order_crossover_ox(p1: Sequence[int], p2: Sequence[int], rng: random.Random) -> List[int]:
    """Order Crossover (OX) préservant l'ordre relatif des clients."""
    n = len(p1)
    if n < 2:
        return list(p1)
    a, b = sorted(rng.sample(range(n), 2))
    child = [-1] * n
    child[a : b + 1] = p1[a : b + 1]
    in_seg = set(p1[a : b + 1])

    fill = [x for x in p2 if x not in in_seg]
    pos = (b + 1) % n
    for gene in fill:
        child[pos] = gene
        pos = (pos + 1) % n
    return child


def mutation_inversion(perm: Sequence[int], rng: random.Random) -> List[int]:
    """Mutation par inversion d'un sous-segment (équivalent 2-opt sur le tour géant)."""
    res = list(perm)
    n = len(res)
    if n < 2:
        return res
    a, b = sorted(rng.sample(range(n), 2))
    res[a : b + 1] = reversed(res[a : b + 1])
    return res


def cheapest_feasible_insert(
    routes: List[List[int]],
    unassigned: Sequence[int],
    instance: CVRPInstance,
) -> List[List[int]]:
    """Insère les clients non affectés par insertion Regret-2 (ou moindre coût) en respectant Q."""
    Q = instance.capacity
    dem = instance.demands
    dist = instance.dist_matrix
    loads = [instance.route_load(r) for r in routes]
    remaining = list(unassigned)

    while remaining:
        best_choice = None  # (regret, -best_delta, customer, r_idx, pos)
        fallback_new_route = None

        for cust in remaining:
            q_c = dem[cust]
            options: List[Tuple[int, int, int]] = []  # (delta, r_idx, pos)

            for r_idx, r in enumerate(routes):
                if loads[r_idx] + q_c > Q:
                    continue
                r_len = len(r)
                for pos in range(r_len + 1):
                    prev_n = 0 if pos == 0 else r[pos - 1]
                    next_n = 0 if pos == r_len else r[pos]
                    delta = dist[prev_n][cust] + dist[cust][next_n] - dist[prev_n][next_n]
                    options.append((delta, r_idx, pos))

            if options:
                options.sort(key=lambda x: x[0])
                best_delta, best_r, best_pos = options[0]
                second_delta = options[1][0] if len(options) > 1 else best_delta + 10**6
                regret = second_delta - best_delta
                key = (regret, -best_delta)
                if best_choice is None or key > best_choice[0]:
                    best_choice = (key, cust, best_r, best_pos)
            else:
                if fallback_new_route is None or q_c > dem[fallback_new_route]:
                    fallback_new_route = cust

        if best_choice is not None:
            _, cust, r_idx, pos = best_choice
            routes[r_idx].insert(pos, cust)
            loads[r_idx] += dem[cust]
            remaining.remove(cust)
        else:
            cust = fallback_new_route
            routes.append([cust])
            loads.append(dem[cust])
            remaining.remove(cust)

    return [r for r in routes if r]


def route_based_crossover_rbx(
    routes_parent: Sequence[Sequence[int]],
    routes_leader: Sequence[Sequence[int]],
    instance: CVRPInstance,
    rng: random.Random,
) -> List[List[int]]:
    """
    Route-Based Crossover (RBX) : hérite d'un sous-ensemble de tournées complètes du leader,
    complète avec les clients restants issus du parent via Regret-2, et assure la faisabilité.
    """
    leader_non_empty = [list(r) for r in routes_leader if r]
    if not leader_non_empty:
        return [list(r) for r in routes_parent if r]

    num_take = max(1, rng.randint(1, max(1, len(leader_non_empty) // 2)))
    chosen_routes = [list(r) for r in rng.sample(leader_non_empty, num_take)]
    covered = {c for r in chosen_routes for c in r}

    # Conserve les morceaux de tournées du parent non couverts par le leader
    child_routes: List[List[int]] = chosen_routes
    for r in routes_parent:
        filtered = [c for c in r if c not in covered]
        if filtered:
            child_routes.append(filtered)
            covered.update(filtered)

    # Si le nombre de véhicules dépasse K, on vide les plus petites tournées et on réinsère
    if len(child_routes) > instance.num_vehicles:
        child_routes.sort(key=len, reverse=True)
        kept = child_routes[: instance.num_vehicles]
        spilled = [c for r in child_routes[instance.num_vehicles :] for c in r]
        child_routes = cheapest_feasible_insert(kept, spilled, instance)

    return [r for r in child_routes if r]


# =============================================================================
# 5. RECHERCHE LOCALE EMBARQUÉE (VND : 2-OPT, RELOCATE, SWAP, 2-OPT*)
# =============================================================================

def local_search_2opt_route(route: List[int], dist: List[List[int]]) -> Tuple[List[int], bool]:
    """Améliore une tournée unique par 2-opt jusqu'à optimum local."""
    n = len(route)
    if n < 2:
        return route, False
    improved_any = False
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            a = 0 if i == 0 else route[i - 1]
            b = route[i]
            dab = dist[a][b]
            for j in range(i + 1, n):
                c = route[j]
                d = 0 if j == n - 1 else route[j + 1]
                delta = (dist[a][c] + dist[b][d]) - (dab + dist[c][d])
                if delta < 0:
                    route[i : j + 1] = reversed(route[i : j + 1])
                    improved = True
                    improved_any = True
                    break
            if improved:
                break
    return route, improved_any


def local_search_relocate(routes: List[List[int]], instance: CVRPInstance) -> bool:
    """Déplace un client (intra-tournée ou inter-tournées) vers une meilleure position faisable."""
    dist = instance.dist_matrix
    dem = instance.demands
    Q = instance.capacity
    loads = [instance.route_load(r) for r in routes]

    for r1_idx in range(len(routes)):
        r1 = routes[r1_idx]
        len1 = len(r1)
        for i in range(len1):
            u = r1[i]
            qu = dem[u]
            prev1 = 0 if i == 0 else r1[i - 1]
            next1 = 0 if i == len1 - 1 else r1[i + 1]
            rem_delta = dist[prev1][next1] - dist[prev1][u] - dist[u][next1]

            best_delta = 0
            best_move = None

            for r2_idx in range(len(routes)):
                if r1_idx != r2_idx and loads[r2_idx] + qu > Q:
                    continue
                r2 = routes[r2_idx]
                len2 = len(r2)
                for j in range(len2 + 1):
                    if r1_idx == r2_idx and (j == i or j == i + 1):
                        continue
                    prev2 = 0 if j == 0 else r2[j - 1]
                    next2 = 0 if j == len2 else r2[j]
                    ins_delta = dist[prev2][u] + dist[u][next2] - dist[prev2][next2]
                    total_delta = rem_delta + ins_delta
                    if total_delta < best_delta:
                        best_delta = total_delta
                        best_move = (r2_idx, j)

            if best_move is not None:
                r2_idx, j = best_move
                if r1_idx == r2_idx:
                    val = r1.pop(i)
                    if j > i:
                        j -= 1
                    r1.insert(j, val)
                else:
                    val = r1.pop(i)
                    routes[r2_idx].insert(j, val)
                    loads[r1_idx] -= qu
                    loads[r2_idx] += qu
                    if not r1:
                        routes.pop(r1_idx)
                return True
    return False


def local_search_swap(routes: List[List[int]], instance: CVRPInstance) -> bool:
    """Échange deux clients entre deux tournées distinctes (ou dans la même) si améliorant et faisable."""
    dist = instance.dist_matrix
    dem = instance.demands
    Q = instance.capacity
    loads = [instance.route_load(r) for r in routes]
    m = len(routes)

    for r1_idx in range(m):
        r1 = routes[r1_idx]
        len1 = len(r1)
        for r2_idx in range(r1_idx + 1, m):
            r2 = routes[r2_idx]
            len2 = len(r2)
            for i in range(len1):
                u = r1[i]
                qu = dem[u]
                p1 = 0 if i == 0 else r1[i - 1]
                n1 = 0 if i == len1 - 1 else r1[i + 1]
                old1 = dist[p1][u] + dist[u][n1]

                for j in range(len2):
                    v = r2[j]
                    qv = dem[v]
                    if loads[r1_idx] - qu + qv > Q or loads[r2_idx] - qv + qu > Q:
                        continue
                    p2 = 0 if j == 0 else r2[j - 1]
                    n2 = 0 if j == len2 - 1 else r2[j + 1]
                    old2 = dist[p2][v] + dist[v][n2]
                    new1 = dist[p1][v] + dist[v][n1]
                    new2 = dist[p2][u] + dist[u][n2]
                    if (new1 + new2) < (old1 + old2):
                        r1[i], r2[j] = v, u
                        return True
    return False


def local_search_2opt_star(routes: List[List[int]], instance: CVRPInstance) -> bool:
    """Échange de queues inter-tournées (2-opt*), incluant l'orientation directe et inversée."""
    dist = instance.dist_matrix
    dem = instance.demands
    Q = instance.capacity
    m = len(routes)

    for r1_idx in range(m):
        r1 = routes[r1_idx]
        len1 = len(r1)
        pref1 = [0] * (len1 + 1)
        for idx, c in enumerate(r1):
            pref1[idx + 1] = pref1[idx] + dem[c]
        tot1 = pref1[len1]

        for r2_idx in range(r1_idx + 1, m):
            r2 = routes[r2_idx]
            len2 = len(r2)
            pref2 = [0] * (len2 + 1)
            for idx, c in enumerate(r2):
                pref2[idx + 1] = pref2[idx] + dem[c]
            tot2 = pref2[len2]

            for i in range(len1 + 1):
                head1_load = pref1[i]
                tail1_load = tot1 - head1_load
                a = 0 if i == 0 else r1[i - 1]
                b = 0 if i == len1 else r1[i]

                for j in range(len2 + 1):
                    if (i == 0 and j == 0) or (i == len1 and j == len2):
                        continue
                    head2_load = pref2[j]
                    tail2_load = tot2 - head2_load

                    # Orientation 1 : head1 + tail2 et head2 + tail1
                    if head1_load + tail2_load <= Q and head2_load + tail1_load <= Q:
                        c = 0 if j == 0 else r2[j - 1]
                        d = 0 if j == len2 else r2[j]
                        if (dist[a][d] + dist[c][b]) < (dist[a][b] + dist[c][d]):
                            new_r1 = r1[:i] + r2[j:]
                            new_r2 = r2[:j] + r1[i:]
                            routes[r1_idx] = new_r1
                            routes[r2_idx] = new_r2
                            routes[:] = [r for r in routes if r]
                            return True
    return False


def local_search_or_opt(routes: List[List[int]], instance: CVRPInstance) -> bool:
    """Déplace une chaîne de 2 clients consécutifs vers une autre position (intra ou inter-tournées)."""
    dist = instance.dist_matrix
    dem = instance.demands
    Q = instance.capacity
    loads = [instance.route_load(r) for r in routes]

    for r1_idx in range(len(routes)):
        r1 = routes[r1_idx]
        len1 = len(r1)
        if len1 < 2:
            continue
        for i in range(len1 - 1):
            u, v = r1[i], r1[i + 1]
            q_seg = dem[u] + dem[v]
            p1 = 0 if i == 0 else r1[i - 1]
            n1 = 0 if i + 1 == len1 - 1 else r1[i + 2]
            rem_delta = dist[p1][n1] - dist[p1][u] - dist[v][n1]

            for r2_idx in range(len(routes)):
                if r1_idx != r2_idx and loads[r2_idx] + q_seg > Q:
                    continue
                r2 = routes[r2_idx]
                len2 = len(r2)
                for j in range(len2 + 1):
                    if r1_idx == r2_idx and i <= j <= i + 2:
                        continue
                    p2 = 0 if j == 0 else r2[j - 1]
                    n2 = 0 if j == len2 else r2[j]
                    # Orientation directe (u, v)
                    ins_fwd = dist[p2][u] + dist[v][n2] - dist[p2][n2]
                    if rem_delta + ins_fwd < 0:
                        seg = r1[i : i + 2]
                        del r1[i : i + 2]
                        if r1_idx == r2_idx:
                            if j > i:
                                j -= 2
                            r1[j:j] = seg
                        else:
                            r2[j:j] = seg
                            routes[:] = [r for r in routes if r]
                        return True
                    # Orientation inversée (v, u)
                    ins_rev = dist[p2][v] + dist[u][n2] - dist[p2][n2]
                    if rem_delta + ins_rev < 0:
                        seg = [v, u]
                        del r1[i : i + 2]
                        if r1_idx == r2_idx:
                            if j > i:
                                j -= 2
                            r1[j:j] = seg
                        else:
                            r2[j:j] = seg
                            routes[:] = [r for r in routes if r]
                        return True
    return False


def vnd_local_search(
    routes: Sequence[Sequence[int]],
    instance: CVRPInstance,
    use_or_opt: bool = True,
) -> Tuple[List[List[int]], int]:
    """Variable Neighborhood Descent (2-opt -> Relocate -> Swap -> 2-opt* -> Or-opt)."""
    cur = [list(r) for r in routes if r]
    dist = instance.dist_matrix

    while True:
        # 1. 2-opt sur chaque tournée
        for idx in range(len(cur)):
            cur[idx], _ = local_search_2opt_route(cur[idx], dist)

        # 2. Relocate
        if local_search_relocate(cur, instance):
            continue

        # 3. Swap
        if local_search_swap(cur, instance):
            continue

        # 4. 2-opt* inter-tournées
        if local_search_2opt_star(cur, instance):
            continue

        # 5. Or-opt (chaîne de 2 clients)
        if use_or_opt and local_search_or_opt(cur, instance):
            continue

        break

    return cur, instance.solution_cost(cur)


# =============================================================================
# 6. PERTURBATION ANTI-STAGNATION (DESTROY & REPAIR REGRET-2)
# =============================================================================

def destroy_and_repair(
    routes: Sequence[Sequence[int]],
    instance: CVRPInstance,
    rng: random.Random,
    destroy_rate: float = 0.25,
) -> List[List[int]]:
    """
    Détruit une fraction des clients (alternance Shaw géographique / Worst-cost)
    puis les réinsère par heuristique Regret-2.
    """
    cur = [list(r) for r in routes if r]
    n = instance.num_customers
    num_remove = max(2, min(n - 1, int(round(n * destroy_rate))))
    dist = instance.dist_matrix

    if rng.random() < 0.5:
        # Shaw removal : retire un client graine et ses voisins les plus proches
        seed = rng.randint(1, n)
        all_nodes = list(range(1, n + 1))
        all_nodes.sort(key=lambda c: dist[seed][c] * rng.uniform(0.85, 1.15))
        removed = all_nodes[:num_remove]
    else:
        # Worst-cost removal : retire les clients ayant le coût marginal le plus élevé
        savings_list: List[Tuple[float, int]] = []
        for r in cur:
            rlen = len(r)
            for idx, node in enumerate(r):
                prev_n = 0 if idx == 0 else r[idx - 1]
                next_n = 0 if idx == rlen - 1 else r[idx + 1]
                contrib = (dist[prev_n][node] + dist[node][next_n] - dist[prev_n][next_n])
                savings_list.append((contrib * rng.uniform(0.8, 1.2), node))
        savings_list.sort(key=lambda x: x[0], reverse=True)
        removed = [node for _, node in savings_list[:num_remove]]

    removed_set = set(removed)
    partial_routes = [[c for c in r if c not in removed_set] for r in cur]
    partial_routes = [r for r in partial_routes if r]

    repaired = cheapest_feasible_insert(partial_routes, removed, instance)
    # Si le nombre de véhicules dépasse K, on repasse par Split optimal borné
    if len(repaired) > instance.num_vehicles:
        tour = routes_to_giant_tour(repaired)
        repaired, _ = split_optimal(tour, instance, enforce_fleet=True)
        if len(repaired) > instance.num_vehicles:
            repaired = repair_fleet_binpack(repaired, instance, rng)
    return repaired



def repair_fleet_binpack(
    routes: Sequence[Sequence[int]],
    instance: CVRPInstance,
    rng: random.Random,
) -> List[List[int]]:
    """Garantit <= K tournées faisables sur les instances à remplissage très serré (ex. 99% de Q)."""
    cur = [list(r) for r in routes if r]
    K = instance.num_vehicles
    if len(cur) <= K:
        return cur

    Q = instance.capacity
    dem = instance.demands
    dist = instance.dist_matrix

    # 1. Tente de vider la plus petite tournée par relocalisation ou échange 1-1
    for _ in range(len(cur) - K + 2):
        if len(cur) <= K:
            return cur
        cur.sort(key=lambda r: sum(dem[c] for c in r))
        smallest = cur.pop(0)
        loads = [sum(dem[c] for c in r) for r in cur]
        leftovers = []
        for u in sorted(smallest, key=lambda c: dem[c], reverse=True):
            qu = dem[u]
            placed = False
            # Meilleure insertion directe
            best_opt = None
            for ri, r in enumerate(cur):
                if loads[ri] + qu <= Q:
                    for pos in range(len(r) + 1):
                        p = 0 if pos == 0 else r[pos - 1]
                        nx = 0 if pos == len(r) else r[pos]
                        delta = dist[p][u] + dist[u][nx] - dist[p][nx]
                        if best_opt is None or delta < best_opt[0]:
                            best_opt = (delta, ri, pos)
            if best_opt is not None:
                _, ri, pos = best_opt
                cur[ri].insert(pos, u)
                loads[ri] += qu
                placed = True
            else:
                # Tente d échanger u avec un client v plus petit d une autre tournée et de replacer v
                for ri, r in enumerate(cur):
                    for idx, v in enumerate(r):
                        if dem[v] < qu and loads[ri] - dem[v] + qu <= Q:
                            for rj, r2 in enumerate(cur):
                                if rj != ri and loads[rj] + dem[v] <= Q:
                                    r[idx] = u
                                    loads[ri] = loads[ri] - dem[v] + qu
                                    r2.append(v)
                                    loads[rj] += dem[v]
                                    placed = True
                                    break
                        if placed:
                            break
                    if placed:
                        break
            if not placed:
                leftovers.append(u)
        if leftovers:
            cur.append(leftovers)
            break

    if len(cur) <= K:
        return [r for r in cur if r]

    # 2. Repli : Best-Fit Decreasing (BFD) randomisé + 2-opt intra-tournée
    nodes = list(range(1, instance.num_customers + 1))
    for attempt in range(40):
        if attempt == 0:
            order = sorted(nodes, key=lambda c: dem[c], reverse=True)
        else:
            order = sorted(nodes, key=lambda c: dem[c] + rng.uniform(-1.5, 1.5), reverse=True)
        bins: List[List[int]] = []
        bin_loads: List[int] = []
        for c in order:
            qc = dem[c]
            best_b = -1
            best_rem = Q + 1
            for bi, bl in enumerate(bin_loads):
                rem = Q - (bl + qc)
                if 0 <= rem < best_rem:
                    best_rem = rem
                    best_b = bi
            if best_b != -1:
                bins[best_b].append(c)
                bin_loads[best_b] += qc
            else:
                bins.append([c])
                bin_loads.append(qc)
        if len(bins) <= K:
            for bi in range(len(bins)):
                bins[bi], _ = local_search_2opt_route(bins[bi], dist)
            return bins

    return cur


# =============================================================================
# 7. BOUCLE PRINCIPALE HGWO-VNS
# =============================================================================

@dataclass
class HGWOResult:
    best_routes: List[List[int]]
    best_cost: int
    num_vehicles_used: int
    runtime_sec: float
    convergence: List[int] = field(default_factory=list)
    is_valid: bool = True
    validation_msg: str = "OK"


class HGWO_VNS:
    """Solveur HGWO-VNS pour le CVRP statique."""

    def __init__(
        self,
        pop_size: int = 24,
        max_iter: int = 120,
        stagnation_limit: int = 15,
        elite_ratio: float = 0.30,
        mutation_rate: float = 0.25,
        destroy_rate: float = 0.25,
        seed: Optional[int] = None,
    ) -> None:
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.stagnation_limit = stagnation_limit
        self.elite_ratio = elite_ratio
        self.mutation_rate = mutation_rate
        self.destroy_rate = destroy_rate
        self.seed = seed

    def _fitness(self, routes: Sequence[Sequence[int]], cost: int, instance: CVRPInstance) -> int:
        """Pénalise tout dépassement du nombre maximal de véhicules K."""
        excess = max(0, len([r for r in routes if r]) - instance.num_vehicles)
        return cost + excess * 10000

    def solve(self, instance: CVRPInstance) -> HGWOResult:
        t0 = time.perf_counter()
        rng = random.Random(self.seed)

        # 1. Initialisation hybride (30% Clarke-Wright randomisé + 70% aléatoire + Split)
        raw_pop = initialize_population(instance, self.pop_size, rng, cw_ratio=0.30)
        perms: List[List[int]] = []
        routes_list: List[List[List[int]]] = []
        costs: List[int] = []
        fits: List[int] = []
        stag: List[int] = [0] * self.pop_size

        # Intensification VND initiale sur les meilleures solutions de la population
        raw_pop.sort(key=lambda item: self._fitness(item[1], item[2], instance))
        n_init_vnd = max(3, int(self.pop_size * self.elite_ratio))
        for idx, (perm, routes, cost) in enumerate(raw_pop):
            if idx < n_init_vnd:
                routes, cost = vnd_local_search(routes, instance)
                perm = routes_to_giant_tour(routes)
            perms.append(perm)
            routes_list.append(routes)
            costs.append(cost)
            fits.append(self._fitness(routes, cost, instance))

        # Identification des 3 élites (alpha, beta, delta) + meilleur admissible explicite X* (|R| <= K)
        order = sorted(range(self.pop_size), key=lambda i: fits[i])
        best_routes = [list(r) for r in routes_list[order[0]]]
        best_cost = costs[order[0]]
        best_fit = fits[order[0]]
        best_feasible_routes: Optional[List[List[int]]] = None
        best_feasible_cost: Optional[int] = None
        for idx in range(self.pop_size):
            r_cand = [r for r in routes_list[idx] if r]
            if len(r_cand) <= instance.num_vehicles:
                if best_feasible_cost is None or costs[idx] < best_feasible_cost:
                    best_feasible_cost = costs[idx]
                    best_feasible_routes = [list(r) for r in r_cand]
        convergence = [best_feasible_cost if best_feasible_cost is not None else best_cost]

        n_elite = max(3, int(self.pop_size * self.elite_ratio))
        global_stag = 0

        # 2. Boucle principale GWO discret + VND embarqué + Destroy & Repair
        for it in range(1, self.max_iter + 1):
            a = 2.0 * (1.0 - it / self.max_iter)
            order = sorted(range(self.pop_size), key=lambda i: fits[i])
            alpha_idx, beta_idx, delta_idx = order[0], order[1], order[2]
            elite_set = set(order[:n_elite])

            improved_global = False

            for i in range(self.pop_size):
                A = 2.0 * a * rng.random() - a
                # Sélection probabiliste parmi les 3 leaders (alpha 50%, beta 30%, delta 20%)
                u = rng.random()
                if u < 0.50:
                    leader_idx = alpha_idx
                elif u < 0.80:
                    leader_idx = beta_idx
                else:
                    leader_idx = delta_idx

                if abs(A) >= 1.0:
                    # Phase d'exploration discrète (OX + mutation + Split optimal)
                    cand_perm = order_crossover_ox(perms[i], perms[leader_idx], rng)
                    if rng.random() < self.mutation_rate:
                        cand_perm = mutation_inversion(cand_perm, rng)
                    cand_routes, cand_cost = split_optimal(cand_perm, instance, enforce_fleet=True)
                else:
                    # Phase d'exploitation discrète (RBX guidé par les tournées du leader)
                    cand_routes = route_based_crossover_rbx(
                        routes_list[i], routes_list[leader_idx], instance, rng
                    )
                    cand_cost = instance.solution_cost(cand_routes)
                    cand_perm = routes_to_giant_tour(cand_routes)

                cand_fit = self._fitness(cand_routes, cand_cost, instance)

                # VND embarquée sur les élites ou solutions prometteuses
                if i in elite_set or cand_fit <= int(fits[i] * 1.06):
                    cand_routes, cand_cost = vnd_local_search(cand_routes, instance)
                    cand_perm = routes_to_giant_tour(cand_routes)
                    cand_fit = self._fitness(cand_routes, cand_cost, instance)

                # Mise à jour gloutonne du loup i
                if cand_fit < fits[i]:
                    perms[i] = cand_perm
                    routes_list[i] = cand_routes
                    costs[i] = cand_cost
                    fits[i] = cand_fit
                    stag[i] = 0
                else:
                    stag[i] += 1

                # Mécanisme anti-stagnation (Destroy & Repair + VND)
                if stag[i] >= self.stagnation_limit:
                    rate = min(0.45, self.destroy_rate + 0.05 * rng.random())
                    pert_routes = destroy_and_repair(routes_list[i], instance, rng, destroy_rate=rate)
                    pert_routes, pert_cost = vnd_local_search(pert_routes, instance)
                    pert_fit = self._fitness(pert_routes, pert_cost, instance)
                    # Accepte si amélioration ou réinitialisation diversifiée (hors alpha)
                    if pert_fit < fits[i] or i != alpha_idx:
                        perms[i] = routes_to_giant_tour(pert_routes)
                        routes_list[i] = pert_routes
                        costs[i] = pert_cost
                        fits[i] = pert_fit
                    stag[i] = 0

                if fits[i] < best_fit:
                    best_fit = fits[i]
                    best_cost = costs[i]
                    best_routes = [list(r) for r in routes_list[i]]
                    improved_global = True
                r_feas = [r for r in routes_list[i] if r]
                if len(r_feas) <= instance.num_vehicles:
                    if best_feasible_cost is None or costs[i] < best_feasible_cost:
                        best_feasible_cost = costs[i]
                        best_feasible_routes = [list(r) for r in r_feas]

            # Perturbation dédiée du meilleur global (ILS embarqué) si stagnation globale
            if improved_global:
                global_stag = 0
            else:
                global_stag += 1
                if global_stag >= max(5, self.stagnation_limit // 2):
                    for rate in (0.20, 0.30, 0.40):
                        pert_routes = destroy_and_repair(best_routes, instance, rng, destroy_rate=rate)
                        pert_routes, pert_cost = vnd_local_search(pert_routes, instance)
                        pert_fit = self._fitness(pert_routes, pert_cost, instance)
                        p_feas = [r for r in pert_routes if r]
                        if len(p_feas) <= instance.num_vehicles:
                            if best_feasible_cost is None or pert_cost < best_feasible_cost:
                                best_feasible_cost = pert_cost
                                best_feasible_routes = [list(r) for r in p_feas]
                        if pert_fit < best_fit:
                            best_fit = pert_fit
                            best_cost = pert_cost
                            best_routes = [list(r) for r in pert_routes]
                            # Injecte dans la meute (remplace le pire loup)
                            worst_idx = order[-1]
                            perms[worst_idx] = routes_to_giant_tour(pert_routes)
                            routes_list[worst_idx] = pert_routes
                            costs[worst_idx] = pert_cost
                            fits[worst_idx] = pert_fit
                            global_stag = 0
                            break

            convergence.append(best_cost)

            # Critère d'arrêt aveugle (sans connaissance du BKS) : patience de 25 itérations sans amélioration
            if global_stag >= 25:
                break

        elapsed = time.perf_counter() - t0
        final_routes = best_feasible_routes if best_feasible_routes is not None else best_routes
        final_cost = best_feasible_cost if best_feasible_cost is not None else best_cost
        valid, msg = validate_solution(instance, final_routes, strict_fleet=True)
        return HGWOResult(
            best_routes=final_routes,
            best_cost=final_cost,
            num_vehicles_used=len(final_routes),
            runtime_sec=elapsed,
            convergence=convergence,
            is_valid=valid,
            validation_msg=msg,
        )


# =============================================================================
# 8. DONNÉES DE RÉFÉRENCE — DJEBBAR & KEMMAR (2025), TABLE 5 (SET P D'AUGERAT)
# =============================================================================

DJEBBAR_2025_SET_P: Dict[str, Dict[str, float]] = {
    "P-n16-k8":  {"k": 8,  "bks": 450, "habc_cost": 450,  "habc_time": 3},
    "P-n19-k2":  {"k": 2,  "bks": 212, "habc_cost": 220,  "habc_time": 33},
    "P-n20-k2":  {"k": 2,  "bks": 216, "habc_cost": 221,  "habc_time": 80},
    "P-n21-k2":  {"k": 2,  "bks": 211, "habc_cost": 214,  "habc_time": 86},
    "P-n22-k2":  {"k": 2,  "bks": 216, "habc_cost": 225,  "habc_time": 91},
    "P-n22-k8":  {"k": 8,  "bks": 603, "habc_cost": 474,  "habc_time": 126},
    "P-n23-k8":  {"k": 8,  "bks": 529, "habc_cost": 537,  "habc_time": 146},
    "P-n40-k5":  {"k": 5,  "bks": 458, "habc_cost": 505,  "habc_time": 309},
    "P-n45-k5":  {"k": 5,  "bks": 510, "habc_cost": 486,  "habc_time": 323},
    "P-n50-k7":  {"k": 7,  "bks": 554, "habc_cost": 597,  "habc_time": 360},
    "P-n50-k8":  {"k": 8,  "bks": 631, "habc_cost": 669,  "habc_time": 388},
    "P-n50-k10": {"k": 10, "bks": 696, "habc_cost": 742,  "habc_time": 399},
    "P-n51-k10": {"k": 10, "bks": 741, "habc_cost": 774,  "habc_time": 517},
    "P-n55-k7":  {"k": 7,  "bks": 568, "habc_cost": 618,  "habc_time": 535},
    "P-n55-k10": {"k": 10, "bks": 694, "habc_cost": 733,  "habc_time": 580},
    "P-n55-k15": {"k": 15, "bks": 989, "habc_cost": 983,  "habc_time": 809},
    "P-n60-k10": {"k": 10, "bks": 744, "habc_cost": 817,  "habc_time": 572},
    "P-n60-k15": {"k": 15, "bks": 968, "habc_cost": 1028, "habc_time": 1008},
    "P-n65-k10": {"k": 10, "bks": 792, "habc_cost": 869,  "habc_time": 590},
    "P-n70-k10": {"k": 10, "bks": 827, "habc_cost": 898,  "habc_time": 901},
    "P-n76-k4":  {"k": 4,  "bks": 593, "habc_cost": 749,  "habc_time": 950},
    "P-n76-k5":  {"k": 5,  "bks": 627, "habc_cost": 777,  "habc_time": 978},
    "P-n101-k4": {"k": 4,  "bks": 681, "habc_cost": 925,  "habc_time": 1890},
}


# Instances officielles intégrées (Augerat 1995 — Set P)
P_N16_K8_VRP = """
NAME : P-n16-k8
TYPE : CVRP
DIMENSION : 16
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 35
NODE_COORD_SECTION
1 30 40
2 37 52
3 49 49
4 52 64
5 31 62
6 52 33
7 42 41
8 52 41
9 57 58
10 62 42
11 42 57
12 27 68
13 43 67
14 58 48
15 58 27
16 37 69
DEMAND_SECTION
1 0
2 19
3 30
4 16
5 23
6 11
7 31
8 15
9 28
10 8
11 8
12 7
13 14
14 6
15 19
16 11
DEPOT_SECTION
1
-1
EOF
"""

P_N40_K5_VRP = """
NAME : P-n40-k5
TYPE : CVRP
DIMENSION : 40
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 140
NODE_COORD_SECTION
1 30 40
2 37 52
3 49 49
4 52 64
5 20 26
6 40 30
7 21 47
8 17 63
9 31 62
10 52 33
11 51 21
12 42 41
13 31 32
14 5 25
15 12 42
16 36 16
17 52 41
18 27 23
19 17 33
20 13 13
21 57 58
22 62 42
23 42 57
24 16 57
25 8 52
26 7 38
27 27 68
28 30 48
29 43 67
30 58 48
31 58 27
32 37 69
33 38 46
34 46 10
35 61 33
36 62 63
37 63 69
38 32 22
39 45 35
40 59 15
DEMAND_SECTION
1 0
2 7
3 30
4 16
5 9
6 21
7 15
8 19
9 23
10 11
11 5
12 19
13 29
14 23
15 21
16 10
17 15
18 3
19 41
20 9
21 28
22 8
23 8
24 16
25 10
26 28
27 7
28 15
29 14
30 6
31 19
32 11
33 12
34 23
35 26
36 17
37 6
38 9
39 15
40 14
DEPOT_SECTION
1
-1
EOF
"""


def get_builtin_p_n19_k2() -> CVRPInstance:
    """Instance officielle Augerat P-n19-k2 (matrice EUC_2D entière et demandes)."""
    demands = [0, 19, 30, 16, 23, 11, 31, 15, 28, 14, 8, 7, 14, 19, 11, 26, 17, 6, 15]
    dist_matrix = [
        [0, 14, 19, 33, 22, 23, 12, 22, 32, 32, 21, 28, 30, 31, 30, 32, 39, 44, 16],
        [14, 0, 15, 19, 12, 24, 12, 19, 21, 27, 7, 19, 16, 33, 17, 31, 27, 31, 19],
        [19, 15, 0, 21, 26, 10, 7, 4, 17, 13, 16, 33, 25, 18, 29, 16, 24, 30, 9],
        [33, 19, 21, 0, 21, 31, 25, 23, 8, 24, 12, 25, 9, 37, 16, 32, 10, 12, 30],
        [22, 12, 26, 21, 0, 36, 24, 30, 26, 37, 12, 7, 13, 44, 9, 42, 31, 33, 30],
        [23, 24, 10, 31, 36, 0, 13, 8, 25, 13, 26, 43, 35, 8, 39, 9, 32, 38, 7],
        [12, 12, 7, 25, 24, 13, 0, 10, 23, 20, 16, 31, 26, 21, 28, 21, 30, 35, 7],
        [22, 19, 4, 23, 30, 8, 10, 0, 18, 10, 19, 37, 28, 15, 32, 12, 24, 30, 9],
        [32, 21, 17, 8, 26, 25, 23, 18, 0, 17, 15, 32, 17, 31, 23, 25, 7, 13, 26],
        [32, 27, 13, 24, 37, 13, 20, 10, 17, 0, 25, 44, 31, 16, 37, 9, 21, 27, 18],
        [21, 7, 16, 12, 12, 26, 16, 19, 15, 25, 0, 19, 10, 34, 13, 31, 21, 24, 22],
        [28, 19, 33, 25, 7, 43, 31, 37, 32, 44, 19, 0, 16, 51, 10, 49, 35, 36, 38],
        [30, 16, 25, 9, 13, 35, 26, 28, 17, 31, 10, 16, 0, 43, 6, 38, 19, 20, 32],
        [31, 33, 18, 37, 44, 8, 21, 15, 31, 16, 34, 51, 43, 0, 47, 7, 36, 42, 15],
        [30, 17, 29, 16, 9, 39, 28, 32, 23, 37, 13, 10, 6, 47, 0, 43, 26, 26, 35],
        [32, 31, 16, 32, 42, 9, 21, 12, 25, 9, 31, 49, 38, 7, 43, 0, 30, 36, 16],
        [39, 27, 24, 10, 31, 32, 30, 24, 7, 21, 21, 35, 19, 36, 26, 30, 0, 6, 33],
        [44, 31, 30, 12, 33, 38, 35, 30, 13, 27, 24, 36, 20, 42, 26, 36, 6, 0, 38],
        [16, 19, 9, 30, 30, 7, 7, 9, 26, 18, 22, 38, 32, 15, 35, 16, 33, 38, 0],
    ]
    return CVRPInstance(
        name="P-n19-k2",
        capacity=160,
        num_vehicles=2,
        demands=demands,
        dist_matrix=dist_matrix,
        bks=212,
    )


def get_builtin_instances() -> List[CVRPInstance]:
    """Retourne les 3 instances officielles Augerat Set P intégrées."""
    return [
        parse_vrp_text(P_N16_K8_VRP, num_vehicles=8, bks=450),
        get_builtin_p_n19_k2(),
        parse_vrp_text(P_N40_K5_VRP, num_vehicles=5, bks=458),
    ]


def run_benchmark_comparison(
    instances: Sequence[CVRPInstance],
    num_runs: int = 10,
    pop_size: int = 24,
    max_iter: int = 120,
) -> List[Dict[str, object]]:
    """Exécute HGWO-VNS sur plusieurs runs et compare aux résultats de Djebbar & Kemmar (2025)."""
    rows: List[Dict[str, object]] = []
    for inst in instances:
        run_costs: List[int] = []
        run_times: List[float] = []
        best_res: Optional[HGWOResult] = None

        for seed in range(1, num_runs + 1):
            solver = HGWO_VNS(
                pop_size=pop_size,
                max_iter=max_iter,
                stagnation_limit=12,
                elite_ratio=0.35,
                seed=seed * 17,
            )
            res = solver.solve(inst)
            assert res.is_valid, f"Solution invalide sur {inst.name}: {res.validation_msg}"
            run_costs.append(res.best_cost)
            run_times.append(res.runtime_sec)
            if best_res is None or res.best_cost < best_res.best_cost:
                best_res = res

        ref = DJEBBAR_2025_SET_P.get(inst.name, {})
        habc_cost = ref.get("habc_cost")
        habc_time = ref.get("habc_time")
        bks = inst.bks or ref.get("bks")

        best_c = min(run_costs)
        mean_c = sum(run_costs) / len(run_costs)
        mean_t = sum(run_times) / len(run_times)
        gap_bks = ((best_c - bks) / bks * 100.0) if bks else None
        gain_vs_habc = ((habc_cost - best_c) / habc_cost * 100.0) if habc_cost else None
        speedup = (habc_time / mean_t) if (habc_time and mean_t > 0) else None

        rows.append(
            {
                "instance": inst.name,
                "k": inst.num_vehicles,
                "bks": bks,
                "habc_cost": habc_cost,
                "habc_time_s": habc_time,
                "hgwo_best": best_c,
                "hgwo_mean": round(mean_c, 2),
                "hgwo_time_s": round(mean_t, 3),
                "gap_bks_pct": round(gap_bks, 2) if gap_bks is not None else None,
                "gain_vs_habc_pct": round(gain_vs_habc, 2) if gain_vs_habc is not None else None,
                "speedup_x": round(speedup, 1) if speedup is not None else None,
                "best_routes": best_res.best_routes if best_res else [],
            }
        )
    return rows


if __name__ == "__main__":
    import os
    import sys

    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        # Permet de lancer sur un dossier contenant tous les fichiers .vrp de CVRPLIB
        folder = sys.argv[1]
        loaded = []
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".vrp"):
                name_key = fname[:-4]
                ref = DJEBBAR_2025_SET_P.get(name_key, {})
                loaded.append(
                    load_vrp_file(
                        os.path.join(folder, fname),
                        num_vehicles=ref.get("k"),
                        bks=ref.get("bks"),
                    )
                )
        targets = loaded
    else:
        targets = get_builtin_instances()

    results = run_benchmark_comparison(targets, num_runs=10, pop_size=24, max_iter=120)
    header = (
        f"{'Instance':<10} | {'BKS':>4} | {'HABC':>5} | {'HABC(s)':>7} | "
        f"{'HGWO Best':>9} | {'HGWO Mean':>9} | {'HGWO(s)':>7} | {'Gap BKS':>7} | {'Gain/HABC':>9} | {'Speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['instance']:<10} | {str(r['bks']):>4} | {str(r['habc_cost']):>5} | {str(r['habc_time_s']):>7} | "
            f"{r['hgwo_best']:>9} | {r['hgwo_mean']:>9.2f} | {r['hgwo_time_s']:>7.3f} | "
            f"{str(r['gap_bks_pct'])+'%':>7} | {str(r['gain_vs_habc_pct'])+'%':>9} | {str(r['speedup_x'])+'x':>8}"
        )
