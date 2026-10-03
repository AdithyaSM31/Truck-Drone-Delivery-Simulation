"""
Non-learning policies to measure the RL agent against.

A learned policy that beats a random one has proved almost nothing. The
comparison that matters is against a competent hand-written heuristic, because
that is what a logistics company would actually deploy. Four references live
here:

``TruckOnlyPolicy``
    Never launches the drone. The truck drives the full OR-Tools TSP tour.
    This is the status quo -- the number every improvement is quoted against.

``RandomPolicy``
    Uniform over legal actions. A floor: any agent that cannot beat this has
    not learned anything.

``AlwaysNearestPolicy``
    Launch constantly, always at the closest customer. Crude, and a
    surprisingly strong opponent.

``GreedyPolicy``
    A cost-benefit dispatcher written by hand: launch when the road detour
    saved outweighs the truck idling created. No lookahead past the current
    decision, and that limitation costs it more than it looks.

All four expose the same ``predict(obs, env)`` signature as a trained
Stable-Baselines3 model, so ``evaluate.py`` scores every policy through one
code path and no policy gets an accidental advantage.
"""

import numpy as np


class BasePolicy:
    name = "base"

    def predict(self, obs, env, deterministic=True):
        raise NotImplementedError

    def reset(self):
        pass


class TruckOnlyPolicy(BasePolicy):
    """Always advance the truck. The drone never leaves the vehicle."""

    name = "truck_only"

    def predict(self, obs, env, deterministic=True):
        return 0


class RandomPolicy(BasePolicy):
    """Uniform over the legal actions in the current state."""

    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, env, deterministic=True):
        legal = np.flatnonzero(env.action_masks())
        return int(self.rng.choice(legal))


class AlwaysNearestPolicy(BasePolicy):
    """
    Launch a drone at every opportunity, always to the nearest unserved
    customer. No cost-benefit reasoning at all.

    This looks too naive to be worth reporting, and it is exactly the reason
    it is here: it beats the cost-benefit heuristic below on delivery time.
    Because it always picks the closest customer, its sorties are short, the
    drone is nearly always back before the truck arrives, and the truck almost
    never idles. The "smarter" rule chases expensive detours, flies further,
    and leaves the truck waiting. Any learned policy has to beat this floor
    before it has demonstrated anything.
    """

    name = "always_nearest"

    def predict(self, obs, env, deterministic=True):
        # Actions are ordered slot-major, earliest rendezvous first, so the
        # lowest legal index is the nearest customer met at the soonest stop.
        legal = np.flatnonzero(env.action_masks()[1:])
        return int(legal[0]) + 1 if len(legal) else 0


class GreedyPolicy(BasePolicy):
    """
    Launch the sortie with the best saved-minus-waited trade-off, otherwise
    drive on.

    For every legal (customer, rendezvous) pair it weighs:

      saved   the road detour the truck avoids by not visiting that customer,
              as driving time at the current congestion, plus the doorstep
              service time the truck no longer spends there
      wait    how long the truck is expected to idle at the meeting point
              before the drone arrives

    and launches the pair maximising ``saved - wait`` if that is positive. It
    reads the same wind- and traffic-aware estimates the agent observes, so it
    is a competent opponent rather than a strawman.

    What it cannot do is look past the current decision: it will spend a
    drone on a merely good sortie now and have none free for a much better
    one two stops later, and it judges each rendezvous in isolation rather
    than against the rest of the tour.
    """

    name = "greedy"

    def predict(self, obs, env, deterministic=True):
        speed = env.free_flow_ms if env.stochastic else env.static_truck_ms
        speed /= env._congestion(env.total_time_s)
        best_action, best_value = 0, 0.0
        for (slot, r), entry in env._plans().items():
            if entry["plan"] is None:
                continue
            saved = entry["detour_m"] / speed + env.service_s
            value = saved - max(0.0, entry["margin_s"])
            if value > best_value:
                best_value, best_action = value, 1 + slot * env.R + r
        return best_action


POLICIES = {
    "truck_only": TruckOnlyPolicy,
    "random": RandomPolicy,
    "always_nearest": AlwaysNearestPolicy,
    "greedy": GreedyPolicy,
}


# ======================================================================
# OR-Tools reference solver
# ======================================================================

def solve_tsp(distance_matrix, depot=0, time_limit_s=2):
    """
    Solve a travelling-salesman tour over ``distance_matrix`` with OR-Tools,
    using guided local search. Returns (route, total_distance_m).

    The truck routes cached in the scenario file were produced by this
    function; it is kept here so the baseline can be reproduced or rerun on a
    new customer set rather than taken on trust.
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n = len(distance_matrix)
    scaled = (np.asarray(distance_matrix) * 100).astype(np.int64)

    manager = pywrapcp.RoutingIndexManager(n, 1, depot)
    routing = pywrapcp.RoutingModel(manager)

    def cost(from_index, to_index):
        return int(scaled[manager.IndexToNode(from_index)][
            manager.IndexToNode(to_index)])

    transit = routing.RegisterTransitCallback(cost)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    params.time_limit.FromSeconds(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no solution for the truck tour.")

    route, index, total = [], routing.Start(0), 0.0
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        nxt = solution.Value(routing.NextVar(index))
        total += scaled[manager.IndexToNode(index)][
            manager.IndexToNode(nxt)] / 100.0
        index = nxt
    route.append(manager.IndexToNode(index))
    return route, float(total)
