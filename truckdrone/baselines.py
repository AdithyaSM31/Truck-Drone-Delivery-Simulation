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
    leg, and that limitation costs it more than it looks.

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
        legal = np.flatnonzero(env.action_masks()[1:])
        return int(legal[0]) + 1 if len(legal) else 0


class GreedyPolicy(BasePolicy):
    """
    Launch to the feasible customer with the best detour-saved-per-minute-waited
    trade-off, otherwise drive on.

    For each legal sortie the policy computes two quantities:

      saved   the road detour the truck avoids by not visiting that customer
              itself, expressed as driving time
      wait    how long the truck would sit idle at the recovery point waiting
              for the drone to catch up

    It launches the candidate maximising ``saved - wait``, and only if that
    value is positive. That single rule captures the core economics of the
    problem: a sortie is worth flying exactly when the driving it removes
    exceeds the waiting it creates.

    What it cannot do is look past the current leg. The detour it credits to a
    customer assumes that customer sits between the truck's current stop and
    its next one. When the drone is sent somewhere further down the tour, the
    truck's saving is only realised later and by a different amount, so the
    rule systematically over-values distant customers -- which is why it flies
    longer sorties, idles the truck, and loses to ``AlwaysNearestPolicy`` on
    delivery time despite removing more kilometres. Fixing that requires
    reasoning about the whole remaining tour, which is what the RL agent gets
    to learn.
    """

    name = "greedy"

    def predict(self, obs, env, deterministic=True):
        mask = env.action_masks()
        candidates = np.flatnonzero(mask[1:])
        if len(candidates) == 0:
            return 0

        truck_node = env.truck_route[min(env.truck_stop_idx,
                                         len(env.truck_route) - 1)]
        nearest = env._get_nearest_unserved(truck_node)

        # The dispatcher sends the drone with the fullest pack, swapping to it
        # if need be, so cost the sortie against that pack.
        busy = {d["drone_idx"] for d in env.current_dispatches}
        free = [i for i in range(env.n_drones) if i not in busy]
        soc = max((max(env.drone_packs[i]) for i in free), default=0.0)

        best_action, best_value = 0, 0.0
        for slot in candidates:
            if slot >= len(nearest):
                continue
            cust_local_idx = nearest[slot]
            cust_node = env.customer_indices[cust_local_idx]

            recovery_idx = env._next_required_stop(
                env.truck_stop_idx, extra_skip=(cust_local_idx,))
            recovery_node = env.truck_route[recovery_idx]

            # Driving time the truck avoids by skipping this customer.
            detour_m = max(0.0, float(
                env.road_matrix[truck_node, cust_node]
                + env.road_matrix[cust_node, recovery_node]
                - env.road_matrix[truck_node, recovery_node]))
            saved_s = env.drone.truck_travel_time_s(detour_m)

            # Idle time the truck would spend waiting at the recovery point.
            dist_out = float(env.air_matrix[truck_node, cust_node])
            dist_back = float(env.air_matrix[cust_node, recovery_node])
            profile = env.drone.sortie_profile(
                dist_out, dist_back, payload_kg=env.payload_kg, soc_pct=soc)
            leg_m = float(env.road_matrix[truck_node, recovery_node])
            wait_s = max(0.0, profile["time_s"]
                         - env.drone.truck_travel_time_s(leg_m))

            value = saved_s - wait_s
            if value > best_value:
                best_value, best_action = value, int(slot) + 1

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
