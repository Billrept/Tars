"""Grey Wolf Optimizer (GWO) for trajectory optimization.

A simple, working metaheuristic optimizer. Designed with a clean interface
so it can be swapped out for a research-paper-based algorithm later.

The optimizer searches over (departure_jd, tof_days) to minimize total delta-v.

---------------------------------------------------------------------------
To replace this optimizer, implement a class with the same
interface as GreyWolfOptimizer (especially the `run()` generator method).
The dispatcher only cares about receiving OptimizationProgress yields.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Generator

import numpy as np

from ephemeris.spline_cache import EphemerisCache
from optimizer.objective import delta_v_objective
from optimizer.multileg_objective import multileg_objective, multileg_objective_full

logger = logging.getLogger("tars.optimizer")


@dataclass
class OptimizationProgress:
    """Yielded by the optimizer at each iteration improvement."""
    iteration: int
    max_iterations: int
    best_dv_total: float
    best_departure_jd: float
    best_tof_days: float
    best_dv_departure: float
    best_dv_arrival: float
    converged: bool
    population_best_dvs: list[float] = field(default_factory=list)  # top-3
    population_positions: list[list[float]] = field(default_factory=list)  # [[dep_jd, tof], ...]
    pareto_front: list[dict] = field(default_factory=list)  # [{dv, tof, dep_jd, ...}]
    mode: str = "min_dv"


@dataclass
class OptimizationRequest:
    """Input parameters for an optimization run."""
    origin_id: int
    target_id: int
    dep_start_jd: float
    dep_end_jd: float
    tof_min_days: float
    tof_max_days: float
    population_size: int = 30
    max_iterations: int = 200
    prograde: bool = True
    mode: str = "pareto"  # "min_dv", "min_tof", "pareto"


class ParetoArchive:
    """Maintains a set of non-dominated solutions (Pareto front)."""

    def __init__(self, max_size: int = 50):
        self.solutions: list[dict] = []
        self.max_size = max_size

    def add(self, sol: dict) -> bool:
        dv = sol.get("dv_total", float("inf"))
        tof = sol.get("tof_days", sol.get("total_tof_days", 0))
        if dv == float("inf") or tof <= 0:
            return False

        to_remove = []
        for existing in self.solutions:
            e_dv = existing.get("dv_total", float("inf"))
            e_tof = existing.get("tof_days", existing.get("total_tof_days", 0))
            if e_dv <= dv and e_tof <= tof and (e_dv < dv or e_tof < tof):
                return False
            if dv <= e_dv and tof <= e_tof and (dv < e_dv or tof < e_tof):
                to_remove.append(existing)

        for r in to_remove:
            self.solutions.remove(r)

        self.solutions.append(sol)
        self.solutions.sort(key=lambda x: x.get("dv_total", float("inf")))

        if len(self.solutions) > self.max_size:
            self.solutions = self.solutions[::2]
        return True

    def to_list(self) -> list[dict]:
        return [
            {
                "dv_total": s.get("dv_total"),
                "tof_days": s.get("tof_days", s.get("total_tof_days")),
                "departure_jd": s.get("departure_jd"),
            }
            for s in self.solutions
        ]


class GreyWolfOptimizer:
    """Grey Wolf Optimizer (GWO) for minimum delta-v trajectory search.

    Yields OptimizationProgress after each iteration so the caller
    can stream intermediate results to the frontend.

    To replace with your own algorithm:
    1. Accept the same OptimizationRequest + EphemerisCache
    2. Implement a `run()` method that yields OptimizationProgress
    3. Update the import in dispatcher.py
    """

    def __init__(self, request: OptimizationRequest, cache: EphemerisCache) -> None:
        self.req = request
        self.cache = cache

    def run(self) -> Generator[OptimizationProgress, None, None]:
        """Run the GWO optimization, yielding progress at each improvement."""
        req = self.req
        n_wolves = req.population_size
        max_iter = req.max_iterations
        mode = req.mode

        lb = np.array([req.dep_start_jd, req.tof_min_days])
        ub = np.array([req.dep_end_jd, req.tof_max_days])

        rng = np.random.default_rng()
        positions = rng.uniform(lb, ub, size=(n_wolves, 2))

        pareto_archive = ParetoArchive() if mode == "pareto" else None

        results_full = [self._evaluate_full(pos) for pos in positions]
        fitness = np.array([self._get_fitness(r) for r in results_full])

        if mode == "pareto":
            for r in results_full:
                pareto_archive.add(r)

        sorted_idx = np.argsort(fitness)
        alpha_pos = positions[sorted_idx[0]].copy()
        alpha_fit = fitness[sorted_idx[0]]
        beta_pos = positions[sorted_idx[1]].copy()
        beta_fit = fitness[sorted_idx[1]]
        delta_pos = positions[sorted_idx[2]].copy()
        delta_fit = fitness[sorted_idx[2]]

        best_fit = alpha_fit
        best_result = results_full[sorted_idx[0]]

        yield self._make_progress(
            0, max_iter, best_result,
            [float(alpha_fit), float(beta_fit), float(delta_fit)],
            positions.tolist(),
            pareto_archive.to_list() if pareto_archive else [],
            mode,
        )

        for iteration in range(1, max_iter + 1):
            a = 2.0 * (1.0 - iteration / max_iter)

            for i in range(n_wolves):
                for dim in range(2):
                    r1, r2 = rng.random(), rng.random()
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_alpha = abs(C1 * alpha_pos[dim] - positions[i, dim])
                    X1 = alpha_pos[dim] - A1 * D_alpha

                    r1, r2 = rng.random(), rng.random()
                    A2 = 2.0 * a * r1 - a
                    C2 = 2.0 * r2
                    D_beta = abs(C2 * beta_pos[dim] - positions[i, dim])
                    X2 = beta_pos[dim] - A2 * D_beta

                    r1, r2 = rng.random(), rng.random()
                    A3 = 2.0 * a * r1 - a
                    C3 = 2.0 * r2
                    D_delta = abs(C3 * delta_pos[dim] - positions[i, dim])
                    X3 = delta_pos[dim] - A3 * D_delta

                    positions[i, dim] = (X1 + X2 + X3) / 3.0

                positions[i] = np.clip(positions[i], lb, ub)

            results_full = [self._evaluate_full(pos) for pos in positions]
            fitness = np.array([self._get_fitness(r) for r in results_full])

            if mode == "pareto":
                for r in results_full:
                    pareto_archive.add(r)

            sorted_idx = np.argsort(fitness)
            if fitness[sorted_idx[0]] < alpha_fit:
                alpha_pos = positions[sorted_idx[0]].copy()
                alpha_fit = fitness[sorted_idx[0]]
            if fitness[sorted_idx[1]] < beta_fit:
                beta_pos = positions[sorted_idx[1]].copy()
                beta_fit = fitness[sorted_idx[1]]
            if fitness[sorted_idx[2]] < delta_fit:
                delta_pos = positions[sorted_idx[2]].copy()
                delta_fit = fitness[sorted_idx[2]]

            if alpha_fit < best_fit:
                best_fit = alpha_fit
                best_result = results_full[sorted_idx[0]]
                yield self._make_progress(
                    iteration, max_iter, best_result,
                    [float(alpha_fit), float(beta_fit), float(delta_fit)],
                    positions.tolist(),
                    pareto_archive.to_list() if pareto_archive else [],
                    mode,
                )
            elif iteration % 10 == 0:
                yield self._make_progress(
                    iteration, max_iter, results_full[sorted_idx[0]],
                    [float(alpha_fit), float(beta_fit), float(delta_fit)],
                    positions.tolist(),
                    pareto_archive.to_list() if pareto_archive else [],
                    mode,
                )

        best_result = self._evaluate_full(alpha_pos)
        yield self._make_progress(
            max_iter, max_iter, best_result,
            [float(alpha_fit), float(beta_fit), float(delta_fit)],
            positions.tolist(),
            pareto_archive.to_list() if pareto_archive else [],
            mode,
        )

    def _get_fitness(self, result: dict) -> float:
        """Get fitness value based on optimization mode."""
        if self.req.mode == "min_tof":
            return result["tof_days"] if result["tof_days"] > 0 else 1e12
        return result["dv_total"]

    def _evaluate_full(self, pos: np.ndarray) -> dict:
        """Get full result dict for a wolf position."""
        dep_jd, tof_days = pos[0], pos[1]
        return delta_v_objective(
            dep_jd, tof_days,
            self.req.origin_id, self.req.target_id,
            self.cache, prograde=self.req.prograde,
        )

    @staticmethod
    def _make_progress(
        iteration: int,
        max_iter: int,
        result: dict,
        top3: list[float],
        population_positions: list[list[float]] | None = None,
        pareto_front: list[dict] | None = None,
        mode: str = "min_dv",
    ) -> OptimizationProgress:
        return OptimizationProgress(
            iteration=iteration,
            max_iterations=max_iter,
            best_dv_total=result["dv_total"],
            best_departure_jd=result["departure_jd"],
            best_tof_days=result["tof_days"],
            best_dv_departure=result["dv_departure"],
            best_dv_arrival=result["dv_arrival"],
            converged=result["converged"],
            population_best_dvs=top3,
            population_positions=population_positions or [],
            pareto_front=pareto_front or [],
            mode=mode,
        )


# --------------------------------------------------------------------------- #
#  Multi-leg optimizer
# --------------------------------------------------------------------------- #


@dataclass
class MultiLegOptimizationProgress:
    """Yielded by the multi-leg optimizer at each iteration improvement."""
    iteration: int
    max_iterations: int
    best_dv_total: float
    best_departure_jd: float
    best_leg_tof_days: list[float]
    best_total_tof_days: float
    best_dv_departure: float
    best_dv_arrival: float
    best_dv_flyby: float
    converged: bool
    body_sequence: list[str]
    population_best_dvs: list[float] = field(default_factory=list)
    population_positions: list[list[float]] = field(default_factory=list)
    pareto_front: list[dict] = field(default_factory=list)
    mode: str = "min_dv"


@dataclass
class MultiLegOptimizationRequest:
    """Input parameters for a multi-leg optimization run.

    The search space is (N+1)-dimensional:
        x[0] = departure Julian Date
        x[1..N] = TOF in days for each of the N legs
    """
    body_names: list[str]
    dep_start_jd: float
    dep_end_jd: float
    leg_tof_bounds: list[tuple[float, float]]
    population_size: int = 40
    max_iterations: int = 300
    max_c3: float | None = None
    mode: str = "pareto"  # "min_dv", "min_tof", "pareto"


class MultiLegGreyWolfOptimizer:
    """Grey Wolf Optimizer for multi-leg trajectory search.

    Searches over [departure_jd, tof_0, tof_1, ..., tof_{N-1}] to minimize
    total delta-v (departure + powered flybys + arrival) or other objectives.
    """

    def __init__(
        self,
        request: MultiLegOptimizationRequest,
        cache: EphemerisCache,
    ) -> None:
        self.req = request
        self.cache = cache
        self.n_legs = len(request.body_names) - 1
        self.n_dims = 1 + self.n_legs

    def run(self) -> Generator[MultiLegOptimizationProgress, None, None]:
        """Run the GWO optimization for multi-leg trajectories."""
        req = self.req
        n_wolves = req.population_size
        max_iter = req.max_iterations
        mode = req.mode

        lb = np.zeros(self.n_dims)
        ub = np.zeros(self.n_dims)
        lb[0] = req.dep_start_jd
        ub[0] = req.dep_end_jd
        for i, (tmin, tmax) in enumerate(req.leg_tof_bounds):
            lb[1 + i] = tmin
            ub[1 + i] = tmax

        rng = np.random.default_rng()
        positions = rng.uniform(lb, ub, size=(n_wolves, self.n_dims))

        pareto_archive = ParetoArchive() if mode == "pareto" else None

        results_full = [self._evaluate_full(pos) for pos in positions]
        fitness = np.array([self._get_fitness(r) for r in results_full])

        if mode == "pareto":
            for r in results_full:
                pareto_archive.add(r)

        sorted_idx = np.argsort(fitness)
        alpha_pos = positions[sorted_idx[0]].copy()
        alpha_fit = fitness[sorted_idx[0]]
        beta_pos = positions[sorted_idx[1]].copy()
        beta_fit = fitness[sorted_idx[1]]
        delta_pos = positions[sorted_idx[2]].copy()
        delta_fit = fitness[sorted_idx[2]]

        best_fit = alpha_fit
        best_result = results_full[sorted_idx[0]]

        yield self._make_progress(
            0, max_iter, alpha_pos, best_result.get("dv_total", float("inf")),
            [float(alpha_fit), float(beta_fit), float(delta_fit)],
            positions.tolist(),
            pareto_archive.to_list() if pareto_archive else [],
            mode,
        )

        for iteration in range(1, max_iter + 1):
            a = 2.0 * (1.0 - iteration / max_iter)

            for i in range(n_wolves):
                for dim in range(self.n_dims):
                    r1, r2 = rng.random(), rng.random()
                    A1 = 2.0 * a * r1 - a
                    C1 = 2.0 * r2
                    D_alpha = abs(C1 * alpha_pos[dim] - positions[i, dim])
                    X1 = alpha_pos[dim] - A1 * D_alpha

                    r1, r2 = rng.random(), rng.random()
                    A2 = 2.0 * a * r1 - a
                    C2 = 2.0 * r2
                    D_beta = abs(C2 * beta_pos[dim] - positions[i, dim])
                    X2 = beta_pos[dim] - A2 * D_beta

                    r1, r2 = rng.random(), rng.random()
                    A3 = 2.0 * a * r1 - a
                    C3 = 2.0 * r2
                    D_delta = abs(C3 * delta_pos[dim] - positions[i, dim])
                    X3 = delta_pos[dim] - A3 * D_delta

                    positions[i, dim] = (X1 + X2 + X3) / 3.0

                positions[i] = np.clip(positions[i], lb, ub)

            results_full = [self._evaluate_full(pos) for pos in positions]
            fitness = np.array([self._get_fitness(r) for r in results_full])

            if mode == "pareto":
                for r in results_full:
                    pareto_archive.add(r)

            sorted_idx = np.argsort(fitness)
            if fitness[sorted_idx[0]] < alpha_fit:
                alpha_pos = positions[sorted_idx[0]].copy()
                alpha_fit = fitness[sorted_idx[0]]
            if fitness[sorted_idx[1]] < beta_fit:
                beta_pos = positions[sorted_idx[1]].copy()
                beta_fit = fitness[sorted_idx[1]]
            if fitness[sorted_idx[2]] < delta_fit:
                delta_pos = positions[sorted_idx[2]].copy()
                delta_fit = fitness[sorted_idx[2]]

            if alpha_fit < best_fit:
                best_fit = alpha_fit
                best_result = results_full[sorted_idx[0]]
                yield self._make_progress(
                    iteration, max_iter, alpha_pos, best_result.get("dv_total", float("inf")),
                    [float(alpha_fit), float(beta_fit), float(delta_fit)],
                    positions.tolist(),
                    pareto_archive.to_list() if pareto_archive else [],
                    mode,
                )
            elif iteration % 10 == 0:
                yield self._make_progress(
                    iteration, max_iter, alpha_pos, results_full[sorted_idx[0]].get("dv_total", float("inf")),
                    [float(alpha_fit), float(beta_fit), float(delta_fit)],
                    positions.tolist(),
                    pareto_archive.to_list() if pareto_archive else [],
                    mode,
                )

        best_result = self._evaluate_full(alpha_pos)
        yield self._make_progress(
            max_iter, max_iter, alpha_pos, best_result.get("dv_total", float("inf")),
            [float(alpha_fit), float(beta_fit), float(delta_fit)],
            positions.tolist(),
            pareto_archive.to_list() if pareto_archive else [],
            mode,
        )

    def _get_fitness(self, result: dict) -> float:
        """Get fitness value based on optimization mode."""
        if self.req.mode == "min_tof":
            return result.get("total_tof_days", 1e12)
        return result.get("dv_total", 1e12)

    def _evaluate_full(self, pos: np.ndarray) -> dict:
        """Get full result dict for a wolf position."""
        result = multileg_objective_full(pos, self.req.body_names, self.cache, n_traj_points=0)
        if result is None:
            return {"dv_total": float("inf"), "total_tof_days": 1e12, "departure_jd": float(pos[0])}
        return {
            "dv_total": result.total_dv_km_s,
            "total_tof_days": result.total_tof_days,
            "departure_jd": result.departure_jd,
            "departure_dv_km_s": result.departure_dv_km_s,
            "arrival_dv_km_s": result.arrival_dv_km_s,
            "flyby_dv_km_s": result.flyby_dv_km_s,
            "legs": result.legs,
            "flybys": result.flybys,
        }

    def _make_progress(
        self,
        iteration: int,
        max_iter: int,
        pos: np.ndarray,
        dv_total: float,
        top3: list[float],
        population_positions: list[list[float]] | None = None,
        pareto_front: list[dict] | None = None,
        mode: str = "min_dv",
    ) -> MultiLegOptimizationProgress:
        dep_jd = float(pos[0])
        leg_tofs = [float(pos[1 + i]) for i in range(self.n_legs)]
        total_tof = sum(leg_tofs)

        result = multileg_objective_full(pos, self.req.body_names, self.cache, n_traj_points=0)

        if result is not None:
            dv_departure = result.departure_dv_km_s
            dv_arrival = result.arrival_dv_km_s
            dv_flyby = result.flyby_dv_km_s
            converged = True
        else:
            dv_departure = float("inf")
            dv_arrival = float("inf")
            dv_flyby = 0.0
            converged = False

        return MultiLegOptimizationProgress(
            iteration=iteration,
            max_iterations=max_iter,
            best_dv_total=dv_total if dv_total < 1e11 else float("inf"),
            best_departure_jd=dep_jd,
            best_leg_tof_days=leg_tofs,
            best_total_tof_days=total_tof,
            best_dv_departure=dv_departure,
            best_dv_arrival=dv_arrival,
            best_dv_flyby=dv_flyby,
            converged=converged,
            body_sequence=self.req.body_names,
            population_best_dvs=top3,
            population_positions=population_positions or [],
            pareto_front=pareto_front or [],
            mode=mode,
        )
