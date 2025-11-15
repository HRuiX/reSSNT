"""
NSGA-III Optimization Algorithm
"""

from typing import List, Tuple
import random
from rich.console import Console
import numpy as np
import itertools
from tqdm import tqdm
import time


console = Console()

try:
    from ..core.chromosome import Chromosome
    from .operators import UnifiedGeneticOperators
except (ImportError, ValueError):
    from core.chromosome import Chromosome
    from optimization.operators import UnifiedGeneticOperators

def generate_reference_points(num_objectives: int, divisions: int = 4) -> np.ndarray:
    def generate_combinations(n_obj, div):
        if n_obj == 1:
            yield [div]
        else:
            for i in range(div + 1):
                for combo in generate_combinations(n_obj - 1, div - i):
                    yield [i] + combo

    ref_points = []
    for combo in generate_combinations(num_objectives, divisions):
        point = np.array(combo, dtype=float) / divisions
        ref_points.append(point)

    ref_points = np.array(ref_points)

    print(f"Generated {len(ref_points)} reference points for {num_objectives} objectives")

    return ref_points


def associate_to_reference_point(objectives: np.ndarray, reference_points: np.ndarray) -> np.ndarray:
    n_individuals = objectives.shape[0]
    n_points = reference_points.shape[0]

    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range < 1e-8] = 1.0

    normalized_obj = (objectives - obj_min) / obj_range

    obj_expanded = normalized_obj[:, np.newaxis, :]
    ref_expanded = reference_points[np.newaxis, :, :]

    ref_norms = np.linalg.norm(reference_points, axis=1)  # (n_points,)
    ref_norms[ref_norms < 1e-8] = 1.0

    dots = np.sum(normalized_obj[:, np.newaxis, :] * reference_points[np.newaxis, :, :], axis=2)
    projections = dots / (ref_norms[np.newaxis, :] ** 2 + 1e-8)

    projected_points = projections[:, :, np.newaxis] * reference_points[np.newaxis, :, :]
    distances = np.linalg.norm(normalized_obj[:, np.newaxis, :] - projected_points, axis=2)

    associations = np.argmin(distances, axis=1)

    return associations


class NSGA3:

    def __init__(self, mostest_config=None):
        self.population_size = mostest_config.population_size
        self.max_generations = mostest_config.max_generations
        self.num_objectives = mostest_config.num_objectives
        self.tournament_size = mostest_config.tournament_size

        self.reference_points = generate_reference_points(mostest_config.num_objectives, mostest_config.ref_point_divisions)

        self.genetic_ops = UnifiedGeneticOperators(
            max_transforms = mostest_config.max_transforms,
            crossover_prob=mostest_config.crossover_prob
        )

    def fast_non_dominated_sort(self, population: List[Chromosome]) -> List[List[Chromosome]]:
        n = len(population)
        objectives_array = np.array([ind.objectives for ind in population])
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]

        for i in range(n):
            obj_i = objectives_array[i]

            # i支配j: obj_i >= obj_j (所有维度) 且至少一个 obj_i > obj_j
            # 从j的角度: obj_j <= obj_i (所有维度) 且至少一个 obj_j < obj_i
            all_leq = np.all(objectives_array <= obj_i, axis=1)
            any_less = np.any(objectives_array < obj_i, axis=1)
            dominated_by_i = all_leq & any_less & (np.arange(n) != i)

            # j支配i: obj_j >= obj_i (所有维度) 且至少一个 obj_j > obj_i
            all_geq = np.all(objectives_array >= obj_i, axis=1)
            any_greater = np.any(objectives_array > obj_i, axis=1)
            dominates_i = all_geq & any_greater & (np.arange(n) != i)

            dominated_solutions[i] = np.where(dominated_by_i)[0].tolist()
            domination_count[i] = np.sum(dominates_i)

        first_front_mask = (domination_count == 0)
        for i in np.where(first_front_mask)[0]:
            population[i].rank = 0
            fronts[0].append(population[i])

        current_front = 0
        while fronts[current_front]:
            next_front = []

            for p in fronts[current_front]:
                p_idx = population.index(p)

                for q_idx in dominated_solutions[p_idx]:
                    domination_count[q_idx] -= 1

                    if domination_count[q_idx] == 0:
                        q = population[q_idx]
                        q.rank = current_front + 1
                        next_front.append(q)

            current_front += 1
            fronts.append(next_front)

        if not fronts[-1]:
            fronts.pop()

        return fronts

    def _dominates(self, obj1: List[float], obj2: List[float]) -> bool:
        better_in_any = False

        for o1, o2 in zip(obj1, obj2):
            if o1 < o2:
                return False
            elif o1 > o2:
                better_in_any = True

        return better_in_any

    def environmental_selection(self, combined_population: List[Chromosome], target_size: int) -> List[Chromosome]:
        fronts = self.fast_non_dominated_sort(combined_population)

        console.print(
            f"[dim]  environmental Selection: {len(combined_population)} → {target_size} | {len(fronts)} fronts[/dim]")

        new_population = []
        last_front_idx = 0

        for front_idx, front in enumerate(fronts):
            if len(new_population) + len(front) <= target_size:
                new_population.extend(front)
                last_front_idx = front_idx
            else:
                break

        if len(new_population) == target_size:
            return new_population

        last_front = fronts[last_front_idx + 1] if last_front_idx + 1 < len(fronts) else []
        k = target_size - len(new_population)

        console.print(f"[dim]    Need {k} more from front {last_front_idx + 1} (size: {len(last_front)})[/dim]")

        if k > 0 and last_front:
            selected_from_last = self._niche_preserving_selection(
                last_front, new_population, k
            )
            new_population.extend(selected_from_last)
        elif k > 0 and not last_front:
            console.print(f"  [yellow]⚠[/yellow] Warning: Need {k} but no more fronts, final: {len(new_population)}")

        return new_population

    def _niche_preserving_selection(self, last_front: List[Chromosome], current_population: List[Chromosome], k: int) -> List[Chromosome]:
        all_individuals = current_population + last_front
        objectives = np.array([ind.objectives for ind in all_individuals])

        obj_min = objectives.min(axis=0)
        obj_max = objectives.max(axis=0)
        obj_range = obj_max - obj_min
        obj_range[obj_range < 1e-8] = 1.0

        normalized_obj = (objectives - obj_min) / obj_range

        last_front_obj = normalized_obj[len(current_population):]
        associations = associate_to_reference_point(
            last_front_obj, self.reference_points
        )

        ref_point_counts = np.zeros(len(self.reference_points), dtype=int)

        for ind in current_population:
            ind_obj = np.array(ind.objectives).reshape(1, -1)
            ind_obj_norm = (ind_obj - obj_min) / obj_range
            assoc = associate_to_reference_point(ind_obj_norm, self.reference_points)[0]
            ref_point_counts[assoc] += 1

        selected = []

        for _ in range(k):
            min_count = ref_point_counts.min()
            min_refs = np.where(ref_point_counts == min_count)[0]

            candidates = []
            candidate_distances = []

            for i, ind in enumerate(last_front):
                if ind in selected:
                    continue

                if associations[i] in min_refs:
                    # Compute distance to reference point
                    ref_point = self.reference_points[associations[i]]
                    ind_obj = last_front_obj[i]

                    if np.linalg.norm(ref_point) < 1e-8:
                        distance = np.linalg.norm(ind_obj)
                    else:
                        projection = np.dot(ind_obj, ref_point) / np.dot(ref_point, ref_point)
                        projected_point = projection * ref_point
                        distance = np.linalg.norm(ind_obj - projected_point)

                    candidates.append(ind)
                    candidate_distances.append(distance)

            if not candidates:
                remaining = [ind for ind in last_front if ind not in selected]
                if remaining:
                    selected.append(random.choice(remaining))
            else:
                # Select individual with minimum distance
                best_idx = np.argmin(candidate_distances)
                selected_ind = candidates[best_idx]
                selected.append(selected_ind)

                # Update reference point count
                selected_ind_idx = last_front.index(selected_ind)
                ref_point_counts[associations[selected_ind_idx]] += 1

        return selected

    def tournament_selection(self, population: List[Chromosome]) -> Chromosome:
        """
        Single tournament selection - kept for backward compatibility
        For better performance, use batch_tournament_selection when selecting multiple parents
        """
        tournament = random.sample(population, self.tournament_size)
        best = min(tournament, key=lambda x: x.rank)
        return best

    def batch_tournament_selection(self, population: List[Chromosome], num_selections: int) -> List[Chromosome]:
        """
        Batch tournament selection with sampling without replacement
        For each tournament, randomly selects tournament_size individuals without replacement
        This ensures consistency with the original tournament_selection logic
        """
        pop_size = len(population)

        # Extract ranks to numpy array for faster access
        ranks = np.array([ind.rank for ind in population], dtype=np.int64)

        # Generate tournament indices without replacement for each selection
        tournament_indices = np.zeros((num_selections, self.tournament_size), dtype=np.int64)
        for i in range(num_selections):
            tournament_indices[i] = np.random.choice(pop_size, size=self.tournament_size, replace=False)

        # Use NumPy vectorization (still much faster than original loop)
        tournament_ranks = ranks[tournament_indices]  # shape: (num_selections, tournament_size)
        best_positions = np.argmin(tournament_ranks, axis=1)  # shape: (num_selections,)
        selected_indices = tournament_indices[np.arange(num_selections), best_positions]

        # Return selected individuals
        return [population[idx] for idx in selected_indices]

    def create_offspring(self, population: List[Chromosome], generation: int, datalist, psnr_threshold) -> List[Chromosome]:
        """
        Create offspring population with guaranteed size
        Continues generating until reaching population_size or max attempts
        """
        offspring = []
        pabr = tqdm(total=self.population_size, desc="Creating Offspring")

        mutation_rate = self.genetic_ops.adaptive_mutation_rate(
            generation, self.max_generations
        )

        # Maximum attempts to prevent infinite loops
        max_attempts = self.population_size * 3
        attempts = 0

        while len(offspring) < self.population_size and attempts < max_attempts:
            # Calculate how many more offspring we need
            remaining = self.population_size - len(offspring)
            num_pairs = (remaining + 1) // 2
            num_parents_needed = num_pairs * 2

            # Select parents
            selected_parents = self.batch_tournament_selection(population, num_parents_needed)

            # Create offspring in pairs
            for i in range(num_pairs):
                if len(offspring) >= self.population_size:
                    break

                parent1 = selected_parents[2 * i]
                parent2 = selected_parents[2 * i + 1]

                child1, child2 = self.genetic_ops.sbx_crossover(parent1, parent2)
                child1 = self.genetic_ops.polynomial_mutation(child1, mutation_rate)
                child2 = self.genetic_ops.polynomial_mutation(child2, mutation_rate)

                valid1 = child1.check_chromosome_validity(datalist, psnr_threshold)
                valid2 = child2.check_chromosome_validity(datalist, psnr_threshold)

                if valid1:
                    offspring.append(child1)
                    pabr.update(1)

                if valid2:
                    offspring.append(child2)
                    pabr.update(1)

                attempts += 1

        pabr.close()

        # Apply transform distribution constraints to offspring
        if offspring and len(offspring) > 0:
            # Get transform_configs from first chromosome
            transform_configs = offspring[0].transform_configs
            offspring = Chromosome._enforce_transform_distribution(
                offspring, transform_configs, datalist, psnr_threshold
            )

        return offspring