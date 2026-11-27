"""
Validateur indépendant et générateur de certificats de faisabilité CVRP (norme CVRPLIB / TSPLIB).

Vérifie de manière totalement indépendante du solveur :
  1. Convention de distance : EUC_2D arrondie à l'entier le plus proche (TSPLIB : int(hypot(dx, dy) + 0.5))
     avec comparaison à la distance euclidienne exacte (flottante).
  2. Couverture exacte des clients : chaque client 1..n visité exactement une fois, dépôt 0 absent du corps.
  3. Contrainte de capacité : somme des demandes de chaque tournée <= Q.
  4. Contrainte de flotte : nombre de tournées actives <= K.
  5. Export au format standard CVRPLIB (.sol) et visualisation 2D publication-ready des tournées.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hgwo_vns_cvrp import (
    CVRPInstance,
    HGWO_VNS,
    get_builtin_instances,
)


@dataclass
class RouteAudit:
    route_id: int
    nodes: List[int]
    load: int
    capacity: int
    cost_euc2d_int: int
    cost_euc2d_float: Optional[float]
    feasible_load: bool


@dataclass
class SolutionCertificate:
    instance_name: str
    is_valid: bool
    errors: List[str]
    num_customers: int
    num_vehicles_used: int
    max_vehicles: int
    capacity: int
    total_demand: int
    total_cost_euc2d_int: int
    total_cost_euc2d_float: Optional[float]
    bks: Optional[int]
    gap_bks_pct: Optional[float]
    route_audits: List[RouteAudit]
    cvrplib_sol_text: str


def euc2d_nint(p1: Tuple[float, float], p2: Tuple[float, float]) -> int:
    """Convention TSPLIB / CVRPLIB EUC_2D : arrondi à l'entier le plus proche (nint)."""
    return int(math.hypot(p1[0] - p2[0], p1[1] - p2[1]) + 0.5)


def euc2d_exact(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Distance euclidienne exacte en virgule flottante (non arrondie)."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def independent_audit(
    instance: CVRPInstance,
    routes: Sequence[Sequence[int]],
    strict_fleet: bool = True,
) -> SolutionCertificate:
    """
    Audite une solution CVRP indépendamment des fonctions internes du solveur
    et produit un certificat complet + texte au format officiel CVRPLIB (.sol).
    """
    errors: List[str] = []
    non_empty = [list(r) for r in routes if len(r) > 0]
    n = instance.num_customers
    Q = instance.capacity
    K = instance.num_vehicles

    # 1. Vérification de la flotte
    if strict_fleet and len(non_empty) > K:
        errors.append(f"Flotte dépassée : {len(non_empty)} véhicules utilisés > K={K}.")

    # 2. Vérification de la couverture, des doublons, du dépôt et des capacités
    seen: List[int] = []
    route_audits: List[RouteAudit] = []
    total_int = 0
    total_float = 0.0 if instance.coords is not None else None

    for idx, r in enumerate(non_empty, start=1):
        load = 0
        for node in r:
            if node == 0:
                errors.append(f"Route #{idx} : le dépôt (0) apparaît à l'intérieur de la tournée.")
            elif node < 1 or node > n:
                errors.append(f"Route #{idx} : identifiant client hors bornes ({node}).")
            else:
                load += instance.demands[node]
                seen.append(node)

        feasible_load = load <= Q
        if not feasible_load:
            errors.append(f"Route #{idx} : capacité dépassée ({load} > Q={Q}).")

        # Recalcul indépendant du coût (depuis coords si disponibles, sinon dist_matrix)
        seq = [0] + r + [0]
        c_int = 0
        c_flt = 0.0 if instance.coords is not None else None
        for a, b in zip(seq[:-1], seq[1:]):
            if instance.coords is not None:
                c_int += euc2d_nint(instance.coords[a], instance.coords[b])
                c_flt += euc2d_exact(instance.coords[a], instance.coords[b])
            else:
                c_int += instance.dist_matrix[a][b]

        total_int += c_int
        if total_float is not None and c_flt is not None:
            total_float += c_flt

        route_audits.append(
            RouteAudit(
                route_id=idx,
                nodes=r,
                load=load,
                capacity=Q,
                cost_euc2d_int=c_int,
                cost_euc2d_float=round(c_flt, 4) if c_flt is not None else None,
                feasible_load=feasible_load,
            )
        )

    expected_set = set(range(1, n + 1))
    seen_set = set(seen)
    missing = sorted(expected_set - seen_set)
    duplicates = sorted({x for x in seen if seen.count(x) > 1})
    if missing:
        errors.append(f"Clients non visités : {missing}.")
    if duplicates:
        errors.append(f"Clients visités plusieurs fois : {duplicates}.")

    # 3. Export au format officiel CVRPLIB (.sol)
    sol_lines = [
        f"Route #{ra.route_id}: {' '.join(str(c) for c in ra.nodes)}"
        for ra in route_audits
    ]
    sol_lines.append(f"Cost {total_int}")
    sol_text = "\n".join(sol_lines)

    gap_pct = None
    if instance.bks is not None and instance.bks > 0:
        gap_pct = round((total_int - instance.bks) / instance.bks * 100.0, 2)

    return SolutionCertificate(
        instance_name=instance.name,
        is_valid=(len(errors) == 0),
        errors=errors,
        num_customers=n,
        num_vehicles_used=len(non_empty),
        max_vehicles=K,
        capacity=Q,
        total_demand=sum(instance.demands[1:]),
        total_cost_euc2d_int=total_int,
        total_cost_euc2d_float=round(total_float, 2) if total_float is not None else None,
        bks=instance.bks,
        gap_bks_pct=gap_pct,
        route_audits=route_audits,
        cvrplib_sol_text=sol_text,
    )


def plot_validated_routes(
    pairs: Sequence[Tuple[CVRPInstance, SolutionCertificate]],
    output_png: str,
) -> None:
    """Trace les tournées validées en 2D (style publication scientifique)."""
    plottable = [(inst, cert) for inst, cert in pairs if inst.coords is not None]
    if not plottable:
        return

    fig, axes = plt.subplots(1, len(plottable), figsize=(7.2 * len(plottable), 6.2), dpi=220)
    if len(plottable) == 1:
        axes = [axes]

    palette = [
        "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
        "#9467bd", "#17becf", "#e377c2", "#8c564b",
        "#bcbd22", "#393b79",
    ]

    for ax, (inst, cert) in zip(axes, plottable):
        coords = inst.coords
        assert coords is not None

        for idx, ra in enumerate(cert.route_audits):
            color = palette[idx % len(palette)]
            xs = [coords[0][0]] + [coords[c][0] for c in ra.nodes] + [coords[0][0]]
            ys = [coords[0][1]] + [coords[c][1] for c in ra.nodes] + [coords[0][1]]
            ax.plot(
                xs, ys,
                color=color,
                linewidth=1.9,
                alpha=0.88,
                label=f"R{ra.route_id} ({ra.load}/{ra.capacity}, d={ra.cost_euc2d_int})",
                zorder=2,
            )
            ax.scatter(
                [coords[c][0] for c in ra.nodes],
                [coords[c][1] for c in ra.nodes],
                color=color,
                s=34,
                edgecolors="white",
                linewidths=0.6,
                zorder=3,
            )

        # Dépôt en carré noir bien visible
        ax.scatter(
            [coords[0][0]], [coords[0][1]],
            color="#111111", marker="s", s=110, edgecolors="white", linewidths=1.0,
            label="Dépôt (0)", zorder=5,
        )

        # Numéros des nœuds pour les petites instances (<= 40 clients)
        if inst.num_customers <= 40:
            for node_id in range(1, inst.num_customers + 1):
                x, y = coords[node_id]
                ax.annotate(
                    str(node_id),
                    (x, y),
                    textcoords="offset points",
                    xytext=(3, 3),
                    fontsize=6.5,
                    color="#222222",
                    zorder=4,
                )

        status = "VALIDE" if cert.is_valid else "INVALIDE"
        ax.set_title(
            f"{inst.name} — HGWO-VNS [{status}]\n"
            f"Coût EUC_2D = {cert.total_cost_euc2d_int} (BKS = {cert.bks}, Gap = {cert.gap_bks_pct:.2f}% | "
            f"Exact = {cert.total_cost_euc2d_float})",
            fontsize=10.5,
            fontweight="bold",
            pad=10,
        )
        ax.set_xlabel("Coordonnée X", fontsize=9)
        ax.set_ylabel("Coordonnée Y", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.30)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.13),
            fontsize=7.5,
            frameon=True,
            facecolor="white",
            framealpha=0.95,
            ncol=3 if len(cert.route_audits) > 5 else 2,
        )

    plt.tight_layout(pad=2.0)
    plt.savefig(output_png, bbox_inches="tight")
    plt.close(fig)


def print_certificate(cert: SolutionCertificate) -> None:
    print("=" * 78)
    print(f"CERTIFICAT DE FAISABILITÉ — {cert.instance_name} : {'VALIDE' if cert.is_valid else 'INVALIDE'}")
    print("=" * 78)
    print(f"Clients : {cert.num_customers} | Demande totale : {cert.total_demand} | Capacité Q : {cert.capacity}")
    print(f"Véhicules utilisés : {cert.num_vehicles_used} / {cert.max_vehicles}")
    print(
        f"Coût EUC_2D (nint, norme CVRPLIB) : {cert.total_cost_euc2d_int} "
        f"| BKS : {cert.bks} | Gap : {cert.gap_bks_pct}%"
    )
    if cert.total_cost_euc2d_float is not None:
        print(f"Coût euclidien exact (flottant)   : {cert.total_cost_euc2d_float:.2f}")
    if cert.errors:
        print("ERREURS :", "; ".join(cert.errors))
    print("-" * 78)
    for ra in cert.route_audits:
        fill_pct = ra.load / ra.capacity * 100.0
        flt_str = f" | exact={ra.cost_euc2d_float:.2f}" if ra.cost_euc2d_float is not None else ""
        print(
            f"  Route #{ra.route_id:02d} | Charge {ra.load:3d}/{ra.capacity} ({fill_pct:5.1f}%) "
            f"| Coût={ra.cost_euc2d_int:3d}{flt_str} | 0 -> {' -> '.join(map(str, ra.nodes))} -> 0"
        )
    print("-" * 78)
    print("Format officiel CVRPLIB (.sol) :")
    print(cert.cvrplib_sol_text)
    print()


if __name__ == "__main__":
    instances = get_builtin_instances()
    audited_pairs: List[Tuple[CVRPInstance, SolutionCertificate]] = []

    for inst in instances:
        # Exécute HGWO-VNS (graine fixe reproductible)
        solver = HGWO_VNS(pop_size=24, max_iter=120, stagnation_limit=12, elite_ratio=0.35, seed=17)
        res = solver.solve(inst)
        cert = independent_audit(inst, res.best_routes, strict_fleet=True)
        assert cert.is_valid, f"Échec de validation sur {inst.name}: {cert.errors}"
        assert cert.total_cost_euc2d_int == res.best_cost, "Incohérence entre solveur et validateur !"
        print_certificate(cert)
        audited_pairs.append((inst, cert))

    plot_validated_routes(audited_pairs, "routes_validation_set_p.png")
    print("Graphique de validation généré : routes_validation_set_p.png")
