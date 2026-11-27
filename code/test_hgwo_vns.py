"""
Suite de tests unitaires étape par étape pour HGWO-VNS (CVRP).
Chaque brique de l'algorithme est vérifiée indépendamment avant l'évaluation globale.
"""

import random
import unittest

from hgwo_vns_cvrp import (
    CVRPInstance,
    HGWO_VNS,
    P_N16_K8_VRP,
    P_N40_K5_VRP,
    clarke_wright_savings,
    destroy_and_repair,
    get_builtin_p_n19_k2,
    initialize_population,
    local_search_2opt_route,
    local_search_2opt_star,
    local_search_or_opt,
    local_search_relocate,
    local_search_swap,
    mutation_inversion,
    order_crossover_ox,
    parse_vrp_text,
    route_based_crossover_rbx,
    routes_to_giant_tour,
    split_optimal,
    validate_solution,
    vnd_local_search,
)


class TestHGWOVNSStepByStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inst16 = parse_vrp_text(P_N16_K8_VRP, num_vehicles=8, bks=450)
        cls.inst19 = get_builtin_p_n19_k2()
        cls.inst40 = parse_vrp_text(P_N40_K5_VRP, num_vehicles=5, bks=458)

    # Étape 1 : Parseur CVRPLIB et validateur strict
    def test_step1_parser_and_validator(self) -> None:
        self.assertEqual(self.inst16.num_customers, 15)
        self.assertEqual(self.inst16.capacity, 35)
        self.assertEqual(self.inst40.num_customers, 39)
        self.assertEqual(self.inst40.capacity, 140)
        # Symétrie et diagonale nulle de la matrice de distances
        for inst in (self.inst16, self.inst19, self.inst40):
            m = len(inst.dist_matrix)
            for i in range(m):
                self.assertEqual(inst.dist_matrix[i][i], 0)
                for j in range(m):
                    self.assertEqual(inst.dist_matrix[i][j], inst.dist_matrix[j][i])

        # Détection d'une solution invalide (capacité dépassée)
        ok, _ = validate_solution(self.inst16, [list(range(1, 16))])
        self.assertFalse(ok)
        # Détection d'un client manquant
        ok, _ = validate_solution(self.inst16, [[1, 2], [3]])
        self.assertFalse(ok)

    # Étape 2 : Décodage Split optimal (borné par K) vs découpage séquentiel glouton
    def test_step2_split_optimal(self) -> None:
        rng = random.Random(42)
        for inst in (self.inst16, self.inst19, self.inst40):
            perm = list(range(1, inst.num_customers + 1))
            rng.shuffle(perm)
            routes, cost = split_optimal(perm, inst, enforce_fleet=True)
            self.assertEqual(cost, inst.solution_cost(routes))
            ok, msg = validate_solution(inst, routes, strict_fleet=False)
            self.assertTrue(ok, msg)

            # Comparaison au découpage glouton séquentiel sur la même permutation
            greedy_routes = []
            cur_r, cur_load = [], 0
            for c in perm:
                if cur_load + inst.demands[c] > inst.capacity:
                    greedy_routes.append(cur_r)
                    cur_r, cur_load = [c], inst.demands[c]
                else:
                    cur_r.append(c)
                    cur_load += inst.demands[c]
            if cur_r:
                greedy_routes.append(cur_r)
            greedy_cost = inst.solution_cost(greedy_routes)
            # Split non borné doit toujours être <= glouton séquentiel
            _, split_unbounded_cost = split_optimal(perm, inst, enforce_fleet=False)
            self.assertLessEqual(split_unbounded_cost, greedy_cost)

    # Étape 3 : Initialisation hybride (Clarke-Wright randomisé + aléatoire)
    def test_step3_hybrid_initialization(self) -> None:
        rng = random.Random(7)
        cw_routes = clarke_wright_savings(self.inst40, rng, noise=0.0)
        ok, msg = validate_solution(self.inst40, cw_routes, strict_fleet=False)
        self.assertTrue(ok, msg)

        pop = initialize_population(self.inst40, pop_size=20, rng=rng, cw_ratio=0.30)
        self.assertEqual(len(pop), 20)
        unique_costs = set()
        for perm, routes, cost in pop:
            self.assertEqual(len(perm), self.inst40.num_customers)
            self.assertEqual(cost, self.inst40.solution_cost(routes))
            ok, msg = validate_solution(self.inst40, routes, strict_fleet=True)
            self.assertTrue(ok, msg)
            unique_costs.add(cost)
        self.assertGreater(len(unique_costs), 5, "La population initiale manque de diversité")

    # Étape 4 : Opérateurs discrets OX, Mutation Inversion et RBX
    def test_step4_discrete_operators(self) -> None:
        rng = random.Random(99)
        n = self.inst40.num_customers
        p1 = list(range(1, n + 1))
        p2 = list(range(1, n + 1))
        rng.shuffle(p1)
        rng.shuffle(p2)

        # OX
        child_ox = order_crossover_ox(p1, p2, rng)
        self.assertEqual(sorted(child_ox), list(range(1, n + 1)))

        # Mutation Inversion
        mut = mutation_inversion(child_ox, rng)
        self.assertEqual(sorted(mut), list(range(1, n + 1)))

        # RBX
        r1, _ = split_optimal(p1, self.inst40, enforce_fleet=True)
        r2, _ = split_optimal(p2, self.inst40, enforce_fleet=True)
        child_rbx = route_based_crossover_rbx(r1, r2, self.inst40, rng)
        ok, msg = validate_solution(self.inst40, child_rbx, strict_fleet=False)
        self.assertTrue(ok, msg)

    # Étape 5 : Recherche locale VND (2-opt, Relocate, Swap, 2-opt*, Or-opt)
    def test_step5_vnd_local_search(self) -> None:
        rng = random.Random(123)
        perm = list(range(1, self.inst40.num_customers + 1))
        rng.shuffle(perm)
        routes, init_cost = split_optimal(perm, self.inst40, enforce_fleet=True)

        # Test individuel 2-opt sur une tournée
        r_copy = list(routes[0])
        c_before = self.inst40.route_cost(r_copy)
        r_after, _ = local_search_2opt_route(r_copy, self.inst40.dist_matrix)
        self.assertLessEqual(self.inst40.route_cost(r_after), c_before)

        # Test VND complet
        improved_routes, improved_cost = vnd_local_search(routes, self.inst40)
        self.assertEqual(improved_cost, self.inst40.solution_cost(improved_routes))
        self.assertLess(improved_cost, init_cost, "VND doit strictement améliorer une solution aléatoire")
        ok, msg = validate_solution(self.inst40, improved_routes, strict_fleet=True)
        self.assertTrue(ok, msg)

    # Étape 6 : Perturbation Destroy & Repair (Shaw / Worst + Regret-2)
    def test_step6_destroy_and_repair(self) -> None:
        rng = random.Random(321)
        cw_routes = clarke_wright_savings(self.inst40, rng, noise=0.0)
        for _ in range(10):
            pert = destroy_and_repair(cw_routes, self.inst40, rng, destroy_rate=0.25)
            ok, msg = validate_solution(self.inst40, pert, strict_fleet=False)
            self.assertTrue(ok, msg)

    # Étape 7 : Intégration complète HGWO-VNS (convergence monotone et faisabilité stricte)
    def test_step7_full_hgwo_vns(self) -> None:
        solver = HGWO_VNS(pop_size=16, max_iter=40, seed=42)
        res = solver.solve(self.inst19)
        self.assertTrue(res.is_valid, res.validation_msg)
        # La courbe de convergence doit être décroissante (monotone)
        for a, b in zip(res.convergence, res.convergence[1:]):
            self.assertLessEqual(b, a)
        # Doit battre Djebbar & Kemmar (HABC = 220 sur P-n19-k2)
        self.assertLessEqual(res.best_cost, 212)


if __name__ == "__main__":
    unittest.main(verbosity=2)
