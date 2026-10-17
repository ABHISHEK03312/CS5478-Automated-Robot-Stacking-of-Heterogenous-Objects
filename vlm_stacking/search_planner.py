"""VLM-guided beam search for physically safer stacking orders."""

from vlm_stacking.planner import StackPlanError, validate_plan
from vlm_stacking.ravens_sim import RavensSimulationStackEvaluator


class HybridStackPlanner:
    """Use an Ollama plan as a semantic proposal, then rank beam candidates.

    The search is over discrete bottom-to-top object orders. An optional
    evaluator can perform simulator or robot-specific feasibility checks and
    return ``{"feasible": bool, "stability_score": float, "reason": str}``.
    Without one, the planner uses centered-footprint support geometry only.
    """

    def __init__(self, vlm_planner, evaluator: RavensSimulationStackEvaluator=None, beam_width=12):
        self.vlm_planner = vlm_planner
        self.evaluator = evaluator
        self.beam_width = max(1, int(beam_width))

    def plan(self, instruction, image_bytes, objects):
        proposal = self.vlm_planner.plan(instruction, image_bytes, objects)
        preferred = [str(item) for item in proposal["stack_order"]]
        by_id = {str(item["object_id"]): item for item in objects}
        if set(preferred) != set(by_id):
            raise StackPlanError("VLM proposal does not match the scene catalog.")

        # Each beam item is (partial_order, accumulated heuristic score).
        beam = [([], 0.0)]
        rank = {object_id: index for index, object_id in enumerate(preferred)}
        for depth in range(len(preferred)):
            expanded = []
            for partial, score in beam:
                for object_id in preferred:
                    if object_id in partial:
                        continue
                    next_score = score - 0.15 * abs(rank[object_id] - depth)
                    if partial:
                        support = _support_score(by_id[partial[-1]],
                                                 by_id[object_id])
                        if support is None:
                            continue
                        next_score += support
                    expanded.append((partial + [object_id], next_score))
            expanded.sort(key=lambda candidate: candidate[1], reverse=True)
            beam = expanded[:self.beam_width]
            if not beam:
                raise StackPlanError(
                    "Beam search found no order with sufficient adjacent "
                    "footprint support.")

        candidate_orders = [order for order, _ in beam]
        # Always include the VLM proposal when it passes geometric checks.
        if preferred not in candidate_orders and _order_supported(preferred, by_id):
            candidate_orders.append(preferred)

        scored = []
        failures = []
        for order in candidate_orders:
            candidate = {"stack_order": order, "summary": proposal.get("summary", "")}
            try:
                validate_plan(candidate, objects)
                evaluation = (self.evaluator(candidate, objects)
                             if self.evaluator else
                             _geometry_evaluation(candidate, objects))
            except (ValueError, RuntimeError) as error:
                failures.append(str(error))
                continue
            if not evaluation.get("feasible", False):
                failures.append(evaluation.get("reason", "candidate infeasible"))
                continue
            agreement = _order_agreement(order, preferred)
            total = float(evaluation.get("stability_score", 0.0)) + 0.25 * agreement
            scored.append((total, order, evaluation, agreement))

        if not scored:
            detail = failures[0] if failures else "no candidates were evaluated"
            raise StackPlanError("No feasible stacking plan found: %s" % detail)
        scored.sort(key=lambda item: item[0], reverse=True)
        total, order, evaluation, agreement = scored[0]
        return {
            "stack_order": order,
            "summary": proposal.get("summary", "") +
                       " Selected by VLM-guided beam search.",
            "search": {
                "method": "beam_search",
                "beam_width": self.beam_width,
                "evaluated_candidates": len(candidate_orders),
                "proposal_order_agreement": agreement,
                "evaluation": evaluation,
                "score": total,
            },
        }


def _footprint(item):
    dimensions = item.get("dimensions")
    if dimensions is None or len(dimensions) < 2:
        return None
    return sorted(float(value) for value in dimensions[:2])


def _support_score(lower, upper, tolerance=0.005):
    lower_xy, upper_xy = _footprint(lower), _footprint(upper)
    if lower_xy is None or upper_xy is None:
        return None
    margins = [lower_xy[index] - upper_xy[index] for index in range(2)]
    if min(margins) < -tolerance:
        return None
    # Centered placements are assumed by the current Ravens target provider.
    return min(margins) / max(lower_xy)


def _order_supported(order, by_id):
    return all(_support_score(by_id[lower], by_id[upper]) is not None
               for lower, upper in zip(order, order[1:]))


def _geometry_evaluation(plan, objects):
    by_id = {str(item["object_id"]): item for item in objects}
    margins = [_support_score(by_id[lower], by_id[upper])
               for lower, upper in zip(plan["stack_order"],
                                       plan["stack_order"][1:])]
    if any(value is None for value in margins):
        return {"feasible": False, "stability_score": 0.0,
                "reason": "insufficient support footprint"}
    return {"feasible": True,
            "stability_score": sum(margins) / max(1, len(margins)),
            "evaluation": "centered footprint heuristic; no physics rollout"}


def _order_agreement(order, preferred):
    if len(order) < 2:
        return 1.0
    preferred_rank = {object_id: i for i, object_id in enumerate(preferred)}
    displacement = sum(abs(preferred_rank[item] - index)
                       for index, item in enumerate(order))
    maximum = len(order) * (len(order) - 1)
    return 1.0 - float(displacement) / maximum
