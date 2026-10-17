"""Categorical cross-entropy search over VLM-proposed stacking orders."""

import math
import random

from vlm_stacking.planner import StackPlanError, validate_plan
from vlm_stacking.search_planner import (
    _geometry_evaluation,
    _order_agreement,
)


class CEMStackPlanner:
    """Keep the VLM proposal, then optimize discrete orders with CEM.

    The CEM distribution is a position-by-object categorical matrix. Samples
    are constructed without replacement so every candidate remains a
    permutation. Candidate scoring uses the same feasibility evaluator as the
    beam-search planner, or the centered-footprint heuristic when none is
    supplied.
    """

    def __init__(self, vlm_planner, evaluator=None, population_size=12,
                 elite_fraction=0.25, iterations=3, smoothing=0.7, seed=0):
        self.vlm_planner = vlm_planner
        self.evaluator = evaluator
        self.population_size = max(2, int(population_size))
        self.elite_fraction = min(1.0, max(0.01, float(elite_fraction)))
        self.iterations = max(1, int(iterations))
        self.smoothing = min(1.0, max(0.0, float(smoothing)))
        self.seed = seed

    def plan(self, instruction, image_bytes, objects):
        proposal = self.vlm_planner.plan(instruction, image_bytes, objects)
        preferred = [str(item) for item in proposal["stack_order"]]
        by_id = {str(item["object_id"]): item for item in objects}
        if set(preferred) != set(by_id):
            raise StackPlanError("VLM proposal does not match the scene catalog.")

        rng = random.Random(self.seed)
        probabilities = _initial_distribution(len(preferred))
        cache = {}
        iteration_counts = []

        def evaluate(order):
            key = tuple(order)
            if key in cache:
                return cache[key]
            candidate = {"stack_order": list(order),
                         "summary": proposal.get("summary", "")}
            try:
                validate_plan(candidate, objects)
                result = (self.evaluator(candidate, objects)
                          if self.evaluator else
                          _geometry_evaluation(candidate, objects))
            except (ValueError, RuntimeError) as error:
                result = {"feasible": False, "stability_score": 0.0,
                          "reason": str(error)}
            if result.get("feasible", False):
                agreement = _order_agreement(order, preferred)
                score = float(result.get("stability_score", 0.0)) + 0.25 * agreement
                cache[key] = {"order": list(order), "evaluation": result,
                              "agreement": agreement, "score": score}
            else:
                cache[key] = {"order": list(order), "evaluation": result,
                              "agreement": 0.0, "score": None}
            return cache[key]

        evaluate(preferred)
        for _ in range(self.iterations):
            population = []
            for _ in range(self.population_size):
                order = _sample_permutation(preferred, probabilities, rng)
                population.append(evaluate(order))
            valid = [item for item in population if item["score"] is not None]
            valid.sort(key=lambda item: item["score"], reverse=True)
            iteration_counts.append(len(valid))
            if not valid:
                continue

            elite_count = max(1, int(math.ceil(
                self.population_size * self.elite_fraction)))
            elites = valid[:elite_count]
            elite_distribution = [[0.0 for _ in preferred]
                                  for _ in preferred]
            index = {object_id: i for i, object_id in enumerate(preferred)}
            for item in elites:
                for position, object_id in enumerate(item["order"]):
                    elite_distribution[position][index[object_id]] += 1.0
            for position in range(len(preferred)):
                total = float(len(elites))
                for object_index in range(len(preferred)):
                    frequency = elite_distribution[position][object_index] / total
                    probabilities[position][object_index] = (
                        (1.0 - self.smoothing) * probabilities[position][object_index]
                        + self.smoothing * frequency + 1e-6)
                normalizer = sum(probabilities[position])
                probabilities[position] = [value / normalizer
                                            for value in probabilities[position]]

        valid_results = [item for item in cache.values()
                         if item["score"] is not None]
        if not valid_results:
            raise StackPlanError("CEM found no feasible stacking order.")
        valid_results.sort(key=lambda item: item["score"], reverse=True)
        best = valid_results[0]
        return {
            "stack_order": best["order"],
            "summary": proposal.get("summary", "") +
                       " Selected by categorical CEM search.",
            "search": {
                "method": "categorical_cem",
                "population_size": self.population_size,
                "elite_fraction": self.elite_fraction,
                "iterations": self.iterations,
                "smoothing": self.smoothing,
                "unique_evaluated_candidates": len(cache),
                "feasible_samples_by_iteration": iteration_counts,
                "proposal_order_agreement": best["agreement"],
                "evaluation": best["evaluation"],
                "score": best["score"],
            },
        }


def _initial_distribution(size):
    if size == 0:
        return []
    distribution = []
    for position in range(size):
        if size == 1:
            distribution.append([1.0])
        else:
            row = [0.2 / (size - 1) for _ in range(size)]
            row[position] = 0.8
            distribution.append(row)
    return distribution


def _sample_permutation(preferred, probabilities, rng):
    remaining = list(range(len(preferred)))
    order = []
    for position, row in enumerate(probabilities):
        weights = [row[index] if index in remaining else 0.0
                   for index in range(len(preferred))]
        total = sum(weights)
        if total <= 0.0:
            selected = rng.choice(remaining)
        else:
            threshold = rng.random() * total
            selected = remaining[-1]
            for index in remaining:
                threshold -= weights[index]
                if threshold <= 0.0:
                    selected = index
                    break
        remaining.remove(selected)
        order.append(preferred[selected])
    return order
