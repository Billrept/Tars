"""Non-dominated Sorting Genetic Algorithm II (NSGA-II) for trajectory optimization.

Implements multi-objective evolutionary optimization as requested, directly
supporting generation of a Pareto front trading off Time of Flight (minimum time)
and Delta-V (minimum fuel).
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np

from ephemeris.spline_cache import EphemerisCache
from optimizer.gmpa import (
    OptimizationProgress,
    OptimizationRequest,
    MultiLegOptimizationProgress,
    MultiLegOptimizationRequest,
    ParetoArchive,
)
from optimizer.objective import delta_v_objective
from optimizer.multileg_objective import multileg_objective_full

logger = logging.getLogger("tars.optimizer.nsga2")

def fast_non_dominated_sort(fitnesses: np.ndarray) -> list[list[int]]:
    """O(N^2) non-dominated sorting."""
    N = len(fitnesses)
    S = [[] for _ in range(N)]
    fronts = [[]]
    n = np.zeros(N, dtype=int)

    for p in range(N):
        for q in range(N):
            if p == q: continue
            if (fitnesses[p] <= fitnesses[q]).all() and (fitnesses[p] < fitnesses[q]).any():
                S[p].append(q)
            elif (fitnesses[q] <= fitnesses[p]).all() and (fitnesses[q] < fitnesses[p]).any():
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        if len(next_front) > 0:
            fronts.append(next_front)
        else:
            break

    return fronts

def calculate_crowding_distance(fitnesses: np.ndarray, front: list[int]) -> np.ndarray:
    """Calculate crowding distance for a single Pareto front."""
    L = len(front)
    distances = np.zeros(L)
    if L == 0:
        return distances
    if L <= 2:
        distances[:] = np.inf
        return distances

    num_obj = fitnesses.shape[1]
    for m in range(num_obj):
        sorted_indices = np.argsort([fitnesses[f, m] for f in front])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        f_max = fitnesses[front[sorted_indices[-1]], m]
        f_min = fitnesses[front[sorted_indices[0]], m]
        if f_max == f_min:
            continue

        for i in range(1, L - 1):
            if distances[sorted_indices[i]] != np.inf:
                val_next = fitnesses[front[sorted_indices[i+1]], m]
                val_prev = fitnesses[front[sorted_indices[i-1]], m]
                distances[sorted_indices[i]] += (val_next - val_prev) / (f_max - f_min)

    return distances

def sbx_crossover(p1: np.ndarray, p2: np.ndarray, lb: np.ndarray, ub: np.ndarray, eta_c: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover."""
    if np.random.rand() > 0.9:
        return np.copy(p1), np.copy(p2)
    
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5, (2 * u) ** (1.0 / (eta_c + 1)), (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1)))
    
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    
    return np.clip(c1, lb, ub), np.clip(c2, lb, ub)

def polynomial_mutation(p: np.ndarray, lb: np.ndarray, ub: np.ndarray, eta_m: float = 20.0, prob: float | None = None) -> np.ndarray:
    """Polynomial Mutation."""
    if prob is None:
        prob = 1.0 / len(p)
    
    mutated = np.copy(p)
    for i in range(len(p)):
        if np.random.rand() < prob:
            u = np.random.rand()
            delta_max = ub[i] - lb[i]
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (eta_m + 1)) - 1.0
            else:
                delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta_m + 1))
            mutated[i] += delta * delta_max
            mutated[i] = np.clip(mutated[i], lb[i], ub[i])
    return mutated


class NSGA2Optimizer:
    """NSGA-II for single-leg trajectory optimization."""

    def __init__(self, request: OptimizationRequest, cache: EphemerisCache) -> None:
        self.req = request
        self.cache = cache
        self.rng = np.random.default_rng()
        self.initial_best = None

    def _get_fitness(self, result: dict) -> np.ndarray:
        dv = result.get("dv_total", 1e12)
        tof = result.get("tof_days", result.get("total_tof_days", 1e12))
        
        if dv > 1e11 or tof <= 0:
            dv, tof = 1e12, 1e12
            
        if self.req.mode == "min_dv":
            return np.array([dv, dv])
        elif self.req.mode == "min_tof":
            return np.array([tof, tof])
        else: # pareto
            return np.array([dv, tof])

    def _evaluate_full(self, pos: np.ndarray) -> dict:
        dep_jd, tof_days = pos[0], pos[1]
        return delta_v_objective(
            dep_jd, tof_days,
            self.req.origin_id, self.req.target_id,
            self.cache, prograde=self.req.prograde,
        )

    def _generate_insight(self, best_result: dict) -> str:
        if not self.initial_best or not best_result.get("converged"):
            return "Finding initial feasible solutions..."
        
        dv_saved = self.initial_best["dv_total"] - best_result["dv_total"]
        tof_diff = best_result["tof_days"] - self.initial_best["tof_days"]
        dep_diff = best_result["departure_jd"] - self.initial_best["departure_jd"]
        
        parts = []
        if abs(dv_saved) > 0.01:
            parts.append(f"Reduced ΔV by {dv_saved:.2f} km/s")
        
        if abs(dep_diff) > 0.1:
            dir_str = "later" if dep_diff > 0 else "earlier"
            parts.append(f"shifted departure {abs(dep_diff):.1f} days {dir_str}")
            
        if abs(tof_diff) > 0.1:
            dir_str = "longer" if tof_diff > 0 else "shorter"
            parts.append(f"trip is {abs(tof_diff):.1f} days {dir_str}")
            
        if not parts:
            return "Refining trajectory precision..."
            
        return "Optimized: " + ", ".join(parts) + "."

    def run(self) -> Generator[OptimizationProgress, None, None]:
        req = self.req
        N = req.population_size
        if N % 2 != 0: N += 1 # Ensure even population size
        
        max_iter = req.max_iterations
        lb = np.array([req.dep_start_jd, req.tof_min_days])
        ub = np.array([req.dep_end_jd, req.tof_max_days])

        # Initialize population
        positions = self.rng.uniform(lb, ub, size=(N, 2))
        results_full = [self._evaluate_full(pos) for pos in positions]
        fitnesses = np.array([self._get_fitness(r) for r in results_full])

        # Capture initial best for insight baseline
        valid_idx = np.where(fitnesses[:, 0] < 1e11)[0]
        if len(valid_idx) > 0:
            best_init_idx = valid_idx[np.argmin(fitnesses[valid_idx, 0])]
            self.initial_best = results_full[best_init_idx]

        pareto_archive = ParetoArchive()
        for r in results_full:
            pareto_archive.add(r)

        # Initial yield
        best_idx = np.argmin(fitnesses[:, 0])
        yield self._make_progress(0, max_iter, results_full[best_idx], fitnesses, positions, pareto_archive)

        for iteration in range(1, max_iter + 1):
            # Tournament selection to create offspring
            fronts = fast_non_dominated_sort(fitnesses)
            rank = np.zeros(N, dtype=int)
            global_distances = np.zeros(N)
            for i, f in enumerate(fronts):
                for idx in f: rank[idx] = i
                dists = calculate_crowding_distance(fitnesses, f)
                for j, idx in enumerate(f): global_distances[idx] = dists[j]
            
            selected = []
            for _ in range(N):
                i1, i2 = self.rng.choice(N, 2, replace=False)
                if rank[i1] < rank[i2]: selected.append(i1)
                elif rank[i1] > rank[i2]: selected.append(i2)
                else:
                    if global_distances[i1] > global_distances[i2]: selected.append(i1)
                    else: selected.append(i2)
            
            # Crossover and Mutation
            offspring = np.zeros((N, 2))
            for i in range(0, N, 2):
                p1, p2 = positions[selected[i]], positions[selected[i+1]]
                c1, c2 = sbx_crossover(p1, p2, lb, ub)
                offspring[i] = polynomial_mutation(c1, lb, ub)
                offspring[i+1] = polynomial_mutation(c2, lb, ub)
                
            # Evaluate offspring
            offspring_results = [self._evaluate_full(pos) for pos in offspring]
            offspring_fitnesses = np.array([self._get_fitness(r) for r in offspring_results])
            
            for r in offspring_results:
                pareto_archive.add(r)
                
            # Merge and sort
            R_pos = np.vstack([positions, offspring])
            R_res = results_full + offspring_results
            R_fit = np.vstack([fitnesses, offspring_fitnesses])
            
            R_fronts = fast_non_dominated_sort(R_fit)
            next_pop_indices = []
            
            for f in R_fronts:
                if len(next_pop_indices) + len(f) <= N:
                    next_pop_indices.extend(f)
                else:
                    dists = calculate_crowding_distance(R_fit, f)
                    sorted_f = [x for _, x in sorted(zip(dists, f), reverse=True)]
                    needed = N - len(next_pop_indices)
                    next_pop_indices.extend(sorted_f[:needed])
                    break
                    
            positions = R_pos[next_pop_indices]
            results_full = [R_res[idx] for idx in next_pop_indices]
            fitnesses = R_fit[next_pop_indices]
            
            if iteration % 10 == 0 or iteration == max_iter:
                best_idx = np.argmin(fitnesses[:, 0])
                yield self._make_progress(iteration, max_iter, results_full[best_idx], fitnesses, positions, pareto_archive)

    def _make_progress(self, iteration: int, max_iter: int, best_result: dict, fitnesses: np.ndarray, positions: np.ndarray, archive: ParetoArchive) -> OptimizationProgress:
        # Top 3 based on primary objective (dv or tof)
        sorted_idx = np.argsort(fitnesses[:, 0])
        top3 = [float(fitnesses[i, 0]) for i in sorted_idx[:min(3, len(fitnesses))]]
        
        return OptimizationProgress(
            iteration=iteration,
            max_iterations=max_iter,
            best_dv_total=best_result.get("dv_total", float("inf")),
            best_departure_jd=best_result.get("departure_jd", 0.0),
            best_tof_days=best_result.get("tof_days", 0.0),
            best_dv_departure=best_result.get("dv_departure", float("inf")),
            best_dv_arrival=best_result.get("dv_arrival", float("inf")),
            converged=best_result.get("converged", False),
            population_best_dvs=top3,
            population_positions=positions.tolist(),
            pareto_front=archive.to_list() if self.req.mode == "pareto" else [],
            mode=self.req.mode,
            insight=self._generate_insight(best_result),
            orbit_elements=best_result.get("orbit_elements"),
        )


class MultiLegNSGA2Optimizer:
    """NSGA-II for multi-leg trajectory optimization."""

    def __init__(self, request: MultiLegOptimizationRequest, cache: EphemerisCache) -> None:
        self.req = request
        self.cache = cache
        self.n_legs = len(request.body_names) - 1
        self.n_dims = 1 + self.n_legs
        self.rng = np.random.default_rng()
        self.initial_best = None

    def _get_fitness(self, result: dict) -> np.ndarray:
        dv = result.get("dv_total", 1e12)
        tof = result.get("total_tof_days", 1e12)
        if dv > 1e11 or tof <= 0:
            dv, tof = 1e12, 1e12
            
        if self.req.mode == "min_dv":
            return np.array([dv, dv])
        elif self.req.mode == "min_tof":
            return np.array([tof, tof])
        else: # pareto
            return np.array([dv, tof])

    def _evaluate_full(self, pos: np.ndarray) -> dict:
        result = multileg_objective_full(pos, self.req.body_names, self.cache, n_traj_points=0)
        if result is None:
            return {"dv_total": float("inf"), "total_tof_days": 1e12, "departure_jd": float(pos[0])}
        
        elements_list = []
        for leg in result.legs:
            elements_list.append(leg.orbit_elements if hasattr(leg, "orbit_elements") else None)

        return {
            "dv_total": result.total_dv_km_s,
            "total_tof_days": result.total_tof_days,
            "departure_jd": result.departure_jd,
            "departure_dv_km_s": result.departure_dv_km_s,
            "arrival_dv_km_s": result.arrival_dv_km_s,
            "flyby_dv_km_s": result.flyby_dv_km_s,
            "legs": result.legs,
            "flybys": result.flybys,
            "orbit_elements_list": elements_list,
            "leg_tof_days": [leg.tof_days for leg in result.legs],
            "converged": result.total_dv_km_s < 1e11
        }

    def _generate_insight(self, best_result: dict) -> str:
        if not self.initial_best or not best_result.get("converged"):
            return "Finding gravity assist windows..."
            
        dv_saved = self.initial_best["dv_total"] - best_result["dv_total"]
        if abs(dv_saved) < 0.01:
            return "Fine-tuning arrival geometry..."
            
        # Analyze which leg changed most
        old_tofs = self.initial_best["leg_tof_days"]
        new_tofs = best_result["leg_tof_days"]
        
        max_change = 0
        leg_idx = 0
        for i in range(len(old_tofs)):
            diff = abs(new_tofs[i] - old_tofs[i])
            if diff > max_change:
                max_change = diff
                leg_idx = i
        
        body_prev = self.req.body_names[leg_idx]
        body_next = self.req.body_names[leg_idx+1]
        
        return f"Saved {dv_saved:.2f} km/s by adjusting {body_prev}→{body_next} leg by {max_change:.1f} days."

    def run(self) -> Generator[MultiLegOptimizationProgress, None, None]:
        req = self.req
        N = req.population_size
        if N % 2 != 0: N += 1
        max_iter = req.max_iterations

        lb = np.zeros(self.n_dims)
        ub = np.zeros(self.n_dims)
        lb[0] = req.dep_start_jd
        ub[0] = req.dep_end_jd
        for i, (tmin, tmax) in enumerate(req.leg_tof_bounds):
            lb[1 + i] = tmin
            ub[1 + i] = tmax

        positions = self.rng.uniform(lb, ub, size=(N, self.n_dims))
        results_full = [self._evaluate_full(pos) for pos in positions]
        fitnesses = np.array([self._get_fitness(r) for r in results_full])

        valid_idx = np.where(fitnesses[:, 0] < 1e11)[0]
        if len(valid_idx) > 0:
            best_init_idx = valid_idx[np.argmin(fitnesses[valid_idx, 0])]
            self.initial_best = results_full[best_init_idx]

        pareto_archive = ParetoArchive()
        for r in results_full:
            pareto_archive.add(r)

        best_idx = np.argmin(fitnesses[:, 0])
        yield self._make_progress(0, max_iter, positions[best_idx], results_full[best_idx], fitnesses, positions, pareto_archive)

        for iteration in range(1, max_iter + 1):
            fronts = fast_non_dominated_sort(fitnesses)
            rank = np.zeros(N, dtype=int)
            global_distances = np.zeros(N)
            for i, f in enumerate(fronts):
                for idx in f: rank[idx] = i
                dists = calculate_crowding_distance(fitnesses, f)
                for j, idx in enumerate(f): global_distances[idx] = dists[j]
            
            selected = []
            for _ in range(N):
                i1, i2 = self.rng.choice(N, 2, replace=False)
                if rank[i1] < rank[i2]: selected.append(i1)
                elif rank[i1] > rank[i2]: selected.append(i2)
                else:
                    if global_distances[i1] > global_distances[i2]: selected.append(i1)
                    else: selected.append(i2)
            
            offspring = np.zeros((N, self.n_dims))
            for i in range(0, N, 2):
                p1, p2 = positions[selected[i]], positions[selected[i+1]]
                c1, c2 = sbx_crossover(p1, p2, lb, ub)
                offspring[i] = polynomial_mutation(c1, lb, ub)
                offspring[i+1] = polynomial_mutation(c2, lb, ub)
                
            offspring_results = [self._evaluate_full(pos) for pos in offspring]
            offspring_fitnesses = np.array([self._get_fitness(r) for r in offspring_results])
            
            for r in offspring_results:
                pareto_archive.add(r)
                
            R_pos = np.vstack([positions, offspring])
            R_res = results_full + offspring_results
            R_fit = np.vstack([fitnesses, offspring_fitnesses])
            
            R_fronts = fast_non_dominated_sort(R_fit)
            next_pop_indices = []
            
            for f in R_fronts:
                if len(next_pop_indices) + len(f) <= N:
                    next_pop_indices.extend(f)
                else:
                    dists = calculate_crowding_distance(R_fit, f)
                    sorted_f = [x for _, x in sorted(zip(dists, f), reverse=True)]
                    needed = N - len(next_pop_indices)
                    next_pop_indices.extend(sorted_f[:needed])
                    break
                    
            positions = R_pos[next_pop_indices]
            results_full = [R_res[idx] for idx in next_pop_indices]
            fitnesses = R_fit[next_pop_indices]
            
            if iteration % 10 == 0 or iteration == max_iter:
                best_idx = np.argmin(fitnesses[:, 0])
                yield self._make_progress(iteration, max_iter, positions[best_idx], results_full[best_idx], fitnesses, positions, pareto_archive)

    def _make_progress(self, iteration: int, max_iter: int, pos: np.ndarray, best_result: dict, fitnesses: np.ndarray, positions: np.ndarray, archive: ParetoArchive) -> MultiLegOptimizationProgress:
        dep_jd = float(pos[0])
        leg_tofs = [float(pos[1 + i]) for i in range(self.n_legs)]
        total_tof = sum(leg_tofs)

        dv_total = best_result.get("dv_total", float("inf"))
        dv_departure = best_result.get("departure_dv_km_s", float("inf"))
        dv_arrival = best_result.get("arrival_dv_km_s", float("inf"))
        dv_flyby = best_result.get("flyby_dv_km_s", 0.0)
        converged = dv_total < 1e11

        sorted_idx = np.argsort(fitnesses[:, 0])
        top3 = [float(fitnesses[i, 0]) for i in sorted_idx[:min(3, len(fitnesses))]]

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
            population_positions=positions.tolist(),
            pareto_front=archive.to_list() if self.req.mode == "pareto" else [],
            mode=self.req.mode,
            insight=self._generate_insight(best_result),
            orbit_elements_list=best_result.get("orbit_elements_list"),
        )
