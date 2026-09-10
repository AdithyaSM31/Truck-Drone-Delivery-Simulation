"""
TruckDroneEnv -- a Gymnasium environment for truck-drone collaborative
delivery.

THE PROBLEM
-----------
A delivery truck must serve N customers scattered across a city road network
and return to the depot. It carries a drone. At any stop the truck can launch
the drone to a nearby customer; the drone flies straight there (roads do not
constrain it), drops the parcel, then flies ahead to meet the truck at its
*next* stop. The customer the drone served is struck off the truck's route, so
the truck genuinely drives fewer kilometres.

That last point is the mechanism the whole project rests on. A sortie is only
worth flying if the road detour it saves the truck outweighs the drone's
energy and the risk of the truck having to idle while it waits.

THE AGENT
---------
The agent is the *dispatcher*, not the pilot. At each truck stop it makes one
discrete decision:

    action 0          advance the truck to its next required stop, recovering
                      any drone that is currently in flight
    action 1..K       launch the drone to the i-th nearest unserved customer

Observation (dimension 7 + 4K for a single drone):
    truck (x, y) in a normalised map frame, then per drone its fullest pack,
    an in-flight flag and its mean charge in reserve; fraction of customers
    still unserved, normalised elapsed time; then four features for each of
    the K nearest unserved customers: air distance, bearing, the road detour
    the truck would save by skipping it, and how the sortie duration compares
    with the truck's leg time.

    The last two matter more than they look. Whether a sortie pays is a *road*
    question -- how much driving it removes -- and two customers equally close
    as the crow flies can differ several-fold in road detour. An observation
    carrying only air distance and bearing does not contain the answer, and a
    policy trained on one plateaus below a two-line heuristic that computes the
    detour directly.

BATTERIES
---------
Each drone carries several interchangeable packs -- by default two, one fitted
and one spare. The drone always takes up its fullest pack; if that is not the
one fitted, the launch costs ``battery_swap_time_s`` of ground time while the
crew changes it over.

Packs charge only while they are on the truck, and this is where the spare
earns its place: the pack that flew goes on the charger for whatever is left
of the leg after it lands, but a spare has been charging for the *whole* leg,
including all the time the drone was away. So a drone's turnaround is set by
how fast a pack can be swapped rather than how fast one refills, which is
exactly why real delivery operations swap rather than charge.

ACTION MASKING
--------------
``action_masks()`` reports which actions are legal in the current state: a
sortie is illegal if no drone is free, if the slot points past the end of the
unserved list, or if the drone lacks the battery to fly out, deliver, and
reach the recovery point with its safety reserve intact.

This matters more than it sounds. Without masking, an agent spends most of
training discovering the feasibility rules by collecting penalties, and a
value-based learner can get stuck in a loop of illegal dispatches that burns
the step budget while the truck never moves. With masking, every action the
agent can take is a real choice, and the only thing left to learn is which
choice is *good*.

REWARD
------
The agent is charged for every minute the fleet spends (``penalty_per_minute``)
and paid for every package delivered, with a flat bonus for finishing the route
and a real penalty for abandoning customers. Charging elapsed time is what
makes the reward agree with the metric the results table reports, and it prices
sorties without any special-case term: a sortie that lets the truck skip a stop
pays for itself in shorter legs, while one that leaves the truck idling at the
recovery point charges for every minute it waits.

A potential-based shaping term accelerates learning without changing the
optimal policy (Ng, Harada & Russell, 1999).

HONEST TERMINATION
------------------
An episode ends when the truck returns to the depot. If customers are left
unserved the run is marked ``route_complete=False`` and carries a penalty
scaled to how many were abandoned. Evaluation reports completion rate
separately and never compares delivery time against the baseline on a run
that failed to deliver everything -- a route that "finished early" because it
skipped a third of its customers is not a faster route, and averaging it in
would manufacture an improvement that does not exist.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .physics import DronePhysics

_MDS_CACHE = {}


class TruckDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        scenario,
        instance_ids=None,
        n_drones=1,
        drone_config=None,
        max_visible_customers=5,
        max_steps=None,
        time_limit_s=28800,        # an 8-hour driver shift
        payload_kg=1.0,
        recharge_pct_per_min=1.0,  # onboard charger, per battery pack
        batteries_per_drone=2,     # one installed + one spare (hot-swap)
        battery_swap_time_s=60.0,  # ground time to change a pack over
        # --- reward terms ---
        reward_delivery=10.0,
        reward_drone_delivery_bonus=2.0,
        reward_completion_bonus=100.0,
        penalty_per_minute=-2.0,
        penalty_per_step=-0.1,
        penalty_failed_sortie=-15.0,
        penalty_battery_death=-30.0,
        penalty_per_unserved=-30.0,
        # --- potential-based shaping ---
        use_reward_shaping=True,
        shaping_omega=5.0,
        shaping_gamma=0.99,
        record_trace=False,
    ):
        super().__init__()

        self.road_matrix = np.asarray(scenario["road_matrix"], dtype=np.float64)
        self.air_matrix = np.asarray(scenario["air_matrix"], dtype=np.float64)
        self.coords = np.asarray(scenario["coords"], dtype=np.float64)
        self.n_nodes = len(self.road_matrix)

        all_instances = scenario["instances"]
        if instance_ids is None:
            instance_ids = list(range(len(all_instances)))
        self.instance_ids = list(instance_ids)
        self.instances = [all_instances[i] for i in self.instance_ids]
        if not self.instances:
            raise ValueError("No instances selected for TruckDroneEnv.")

        self.n_drones = n_drones
        self.drone = drone_config or DronePhysics()
        self.K = max_visible_customers
        self.time_limit_s = time_limit_s
        self.payload_kg = payload_kg
        self.recharge_pct_per_min = recharge_pct_per_min
        self.batteries_per_drone = max(1, int(batteries_per_drone))
        self.battery_swap_time_s = battery_swap_time_s

        self.R_delivery = reward_delivery
        self.R_drone_bonus = reward_drone_delivery_bonus
        self.R_completion = reward_completion_bonus
        self.P_minute = penalty_per_minute
        self.P_step = penalty_per_step
        self.P_failed = penalty_failed_sortie
        self.P_battery_dead = penalty_battery_death
        self.P_unserved = penalty_per_unserved

        self.use_reward_shaping = use_reward_shaping
        self.omega = shaping_omega
        self.gamma = shaping_gamma
        self.record_trace = record_trace

        self.action_space = spaces.Discrete(self.K + 1)
        # 2 truck xy + 3 per drone + 2 progress/time + 4 per candidate customer
        obs_size = 4 + 3 * self.n_drones + 4 * self.K
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        finite = self.road_matrix[np.isfinite(self.road_matrix)]
        self.max_dist = float(finite.max()) or 1.0

        self.node_xy = self._embed_positions()
        self._max_steps_override = max_steps
        self._instance_cursor = 0

        self.reset()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _embed_positions(self):
        """
        Normalised 2-D node positions for the observation vector.

        These come from multidimensional scaling of the *road* distance
        matrix, not from raw map coordinates. Two intersections can be close
        as the crow flies yet far apart by road; MDS lays the nodes out so
        that Euclidean distance in this frame approximates driving distance,
        which is what the truck actually experiences. Cached per matrix so
        constructing many envs does not re-run MDS each time.
        """
        key = (self.n_nodes, float(np.nansum(self.road_matrix)))
        if key in _MDS_CACHE:
            return _MDS_CACHE[key]

        from sklearn.manifold import MDS

        d = self.road_matrix.copy()
        d[~np.isfinite(d)] = self.max_dist * 3.0
        mds = MDS(n_components=2, dissimilarity="precomputed",
                  random_state=42, normalized_stress="auto")
        xy = mds.fit_transform(d)
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        span = np.where(hi - lo == 0, 1.0, hi - lo)
        xy = 2.0 * (xy - lo) / span - 1.0
        _MDS_CACHE[key] = xy
        return xy

    def _load_instance(self, index):
        pos = index % len(self.instances)
        inst = self.instances[pos]
        self.instance_index = pos
        self.instance_global_id = self.instance_ids[pos]
        self.depot_index = int(inst["depot"])
        self.customer_indices = [int(c) for c in inst["customers"]]
        self.n_customers = len(self.customer_indices)
        self.truck_route = [int(n) for n in inst["truck_route"]]
        self.truck_only_distance_m = float(inst["truck_route_distance_m"])
        self.max_steps = (self._max_steps_override
                          or len(self.truck_route) * (self.n_drones + 3))

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if options and "instance" in options:
            index = int(options["instance"])
        elif seed is not None:
            index = int(seed)
        else:
            index = self._instance_cursor
            self._instance_cursor += 1
        self._load_instance(index)

        self.current_step = 0
        self.truck_stop_idx = 0
        # Each drone carries `batteries_per_drone` packs. One is installed;
        # the rest ride on the truck charging, including while the drone is
        # away. That is the whole point of a spare: the drone is limited by
        # how fast a pack can be swapped, not by how fast one recharges.
        self.drone_packs = [[100.0] * self.batteries_per_drone
                            for _ in range(self.n_drones)]
        self.drone_installed = [0] * self.n_drones
        self.battery_swaps = 0
        self.min_landing_pct = 100.0
        self.served = np.zeros(self.n_customers, dtype=bool)
        self.total_time_s = 0.0
        self.drone_energy_wh = 0.0
        self.truck_energy_wh = 0.0
        self.truck_distance_m = 0.0
        self.drone_distance_m = 0.0
        self.drone_deliveries = 0
        self.truck_deliveries = 0
        self.truck_wait_time_s = 0.0
        self.drone_wait_time_s = 0.0
        self.failed_sorties = 0
        self.battery_deaths = 0
        self.current_dispatches = []
        self.route_complete = False
        self.trace = []

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        reward = self.P_step
        phi_before = self._potential()
        terminated = False
        truncated = False

        if action == 0:
            reward += self._advance_truck()
        else:
            reward += self._dispatch_drone(int(action) - 1)

        if any(soc <= 0 for packs in self.drone_packs for soc in packs):
            reward += self.P_battery_dead
            self.battery_deaths += 1
            self.drone_packs = [[max(0.0, soc) for soc in packs]
                                for packs in self.drone_packs]

        all_served = bool(np.all(self.served))
        at_route_end = self.truck_stop_idx >= len(self.truck_route) - 1
        out_of_time = self.total_time_s >= self.time_limit_s

        # The episode ends only once the truck is back at the depot, so the
        # hybrid route includes the same return leg as the truck-only baseline
        # it is measured against. Once every customer is served, the next
        # advance goes straight there.
        if at_route_end or out_of_time:
            n_unserved = int(self.n_customers - np.sum(self.served))
            # "Complete" means every package delivered *and* the truck back at
            # the depot. Running out of shift with the last parcel dropped but
            # the truck still out is not comparable to a baseline that always
            # includes its return leg.
            if all_served and at_route_end:
                self.route_complete = True
                # A flat bonus. An earlier version scaled this by the fraction
                # of customers the drone served, which rewarded flying for its
                # own sake -- the agent could raise its return by using drones
                # even where doing so made the route slower. Whether a sortie
                # was worth it is already answered by the per-minute cost.
                reward += self.R_completion
            else:
                # Abandoning customers is a failure, not a partial success.
                # Give it a real cost so quitting early is never the cheapest
                # way to end an episode.
                reward += self.P_unserved * n_unserved
            terminated = True
        elif self.current_step >= self.max_steps:
            n_unserved = int(self.n_customers - np.sum(self.served))
            reward += self.P_unserved * n_unserved
            truncated = True

        # Potential-based shaping. Terminal states take potential 0 so the
        # transformation stays policy-invariant (Ng, Harada & Russell, 1999):
        # shaping changes how fast the agent learns, never which policy is
        # optimal.
        if self.use_reward_shaping:
            phi_after = 0.0 if (terminated or truncated) else self._potential()
            reward += self.gamma * phi_after - phi_before

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ------------------------------------------------------------------
    # Action masking
    # ------------------------------------------------------------------

    def action_masks(self):
        """
        Boolean legality mask over the K+1 actions, consumed by MaskablePPO.

        Advancing the truck is always legal -- it is what guarantees the
        episode can always make progress. A sortie is legal only if there is a
        free drone, the slot names a real unserved customer, and the physics
        model says the round trip fits in the remaining battery.
        """
        mask = np.zeros(self.K + 1, dtype=bool)
        mask[0] = True

        truck_node = self.truck_route[min(self.truck_stop_idx,
                                          len(self.truck_route) - 1)]

        for slot, cust_local_idx in enumerate(
                self._get_nearest_unserved(truck_node)):
            candidate = self._candidate(cust_local_idx, truck_node)
            if candidate is None:      # every drone already in the air
                break
            # Exactly the test _dispatch_drone applies, so the mask stays a
            # promise rather than an approximation: the launch is legal only
            # if the new sortie *and* every sortie already in flight can reach
            # the rendezvous this launch would move them to.
            recovery_node = self._rendezvous_node(extra_skip=(cust_local_idx,))
            mask[slot + 1] = self._all_feasible_at(
                self.current_dispatches + [candidate], recovery_node)
        return mask

    # ------------------------------------------------------------------
    # Truck movement
    # ------------------------------------------------------------------

    def _pending_customer_at(self, node):
        """Local index of an unserved, undispatched customer at ``node``."""
        dispatched = {d["cust_local_idx"] for d in self.current_dispatches}
        for local_idx, cust_node in enumerate(self.customer_indices):
            if cust_node == node and not self.served[local_idx] \
                    and local_idx not in dispatched:
                return local_idx
        return None

    def _next_required_stop(self, from_idx, extra_skip=()):
        """
        Index of the next stop the truck must actually visit.

        Stops whose customer is already served, or is currently assigned to a
        drone, get skipped -- that skipping is exactly how a sortie removes
        heavy-vehicle mileage. The final depot return is never skipped.
        """
        last = len(self.truck_route) - 1
        for j in range(from_idx + 1, last):
            local_idx = self._pending_customer_at(self.truck_route[j])
            if local_idx is not None and local_idx not in extra_skip:
                return j
        return last

    def _advance_truck(self):
        reward = 0.0
        from_node = self.truck_route[self.truck_stop_idx]
        next_idx = self._next_required_stop(self.truck_stop_idx)
        to_node = self.truck_route[next_idx]
        t_start = self.total_time_s

        travel_dist = float(self.road_matrix[from_node, to_node])
        truck_time = self.drone.truck_travel_time_s(travel_dist)

        # Recover every drone in flight at the truck's new position.
        max_sortie_time = 0.0
        sortie_records = []
        airborne_s = {}          # drone index -> seconds spent flying this leg
        for d in self.current_dispatches:
            max_sortie_time = max(max_sortie_time, d["sortie_time"])
            airborne_s[d["drone_idx"]] = d["sortie_time"]
            pack = d["pack_idx"]
            battery_before = self.drone_packs[d["drone_idx"]][pack]
            self.drone_packs[d["drone_idx"]][pack] -= d["battery_pct_cost"]
            self.min_landing_pct = min(
                self.min_landing_pct, self.drone_packs[d["drone_idx"]][pack])
            self.drone_energy_wh += d["energy_wh"]
            self.drone_distance_m += d["distance_m"]
            self.served[d["cust_local_idx"]] = True
            self.drone_deliveries += 1
            reward += self.R_delivery + self.R_drone_bonus

            # The road detour the truck would have driven to serve this
            # customer itself -- reported, not rewarded.
            cust_node = self.customer_indices[d["cust_local_idx"]]
            detour = max(0.0, float(
                self.road_matrix[from_node, cust_node]
                + self.road_matrix[cust_node, to_node]
                - self.road_matrix[from_node, to_node]))

            if self.record_trace:
                sortie_records.append({
                    "drone_idx": d["drone_idx"],
                    "customer_node": cust_node,
                    "launch_node": from_node,
                    "recovery_node": to_node,
                    "time_s": d["sortie_time"],
                    "distance_m": d["distance_m"],
                    "energy_wh": d["energy_wh"],
                    "battery_before": battery_before,
                    "battery_after": battery_before - d["battery_pct_cost"],
                    "swapped": d["swap_time_s"] > 0,
                    "road_detour_saved_m": detour,
                })

        # Truck and drone fly concurrently, so the leg costs the longer of the
        # two, not their sum.
        parallel_time = max(truck_time, max_sortie_time)

        # THE OBJECTIVE. Every minute the fleet spends costs the agent, which
        # is what makes the reward agree with the metric the results table
        # reports. An earlier version penalised each *decision* instead, at
        # -0.1 per step -- but a decision can be two minutes of driving or
        # twenty, so nothing in the reward was proportional to delivery time
        # and the agent was never actually asked to minimise it. It learned to
        # collect drone-delivery bonuses instead and lost to a heuristic that
        # simply launches at the nearest customer every chance it gets.
        #
        # Charging real elapsed time here also prices sorties correctly with
        # no special-case term: a sortie that lets the truck skip a stop pays
        # for itself in shorter legs later, while one that strands the truck
        # waiting at the recovery point charges for every minute it idles.
        reward += self.P_minute * (parallel_time / 60.0)

        # The truck idles if the slowest drone lands after it arrives; each
        # drone idles for whatever is left of the leg once it is back aboard.
        if max_sortie_time > truck_time:
            self.truck_wait_time_s += max_sortie_time - truck_time
        for sortie_s in airborne_s.values():
            self.drone_wait_time_s += parallel_time - sortie_s

        self.total_time_s += parallel_time
        self.truck_distance_m += travel_dist
        self.truck_energy_wh += self.drone.truck_energy_wh(travel_dist)
        self.truck_stop_idx = next_idx
        self.current_dispatches = []

        # Charging is per pack, and a pack only charges while it is on the
        # truck. The pack that flew is on the charger for whatever is left of
        # the leg after it lands; every spare has been on the charger for the
        # whole leg, including the entire time the drone was away.
        #
        # That asymmetry is the case for carrying a spare. Without one, a
        # drone's next sortie is gated by how fast a single pack refills. With
        # one, the spare has been filling the whole time the drone was flying,
        # so the drone turns round in a swap instead of a recharge.
        for i in range(self.n_drones):
            flew_s = airborne_s.get(i, 0.0)
            installed = self.drone_installed[i]
            for p in range(len(self.drone_packs[i])):
                on_truck_s = (max(0.0, parallel_time - flew_s)
                              if p == installed else parallel_time)
                gain = (on_truck_s / 60.0) * self.recharge_pct_per_min
                self.drone_packs[i][p] = min(100.0,
                                             self.drone_packs[i][p] + gain)

        # The truck serves the customer at the stop it has just reached.
        truck_served_node = None
        local_idx = self._pending_customer_at(to_node)
        if local_idx is not None:
            self.served[local_idx] = True
            self.truck_deliveries += 1
            truck_served_node = to_node
            reward += self.R_delivery

        if self.record_trace:
            self.trace.append({
                "type": "leg",
                "step": self.current_step,
                "truck_from": from_node,
                "truck_to": to_node,
                "truck_distance_m": travel_dist,
                "truck_time_s": truck_time,
                "parallel_time_s": parallel_time,
                "t_start_s": t_start,
                "t_end_s": self.total_time_s,
                "sorties": sortie_records,
                "truck_served_node": truck_served_node,
                "packs_after": [[round(x, 1) for x in packs]
                                for packs in self.drone_packs],
                "served_count": int(np.sum(self.served)),
            })

        return reward

    # ------------------------------------------------------------------
    # Drone dispatch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Rendezvous accounting
    #
    # Every drone in flight is recovered where the truck next stops, so they
    # all share one rendezvous -- and that rendezvous MOVES each time another
    # drone is launched, because the truck can then skip one more stop.
    #
    # Getting this wrong is subtle and expensive. Costing a sortie once at
    # launch and never revisiting it leaves the first drone budgeted to fly to
    # a pickup point the truck no longer visits: it is charged for the short
    # return leg and flown on the long one. Measured over the held-out set,
    # that under-charged 9% of sorties by 1.9 km on average, understating
    # energy and battery drain and letting the feasibility check pass sorties
    # that could not actually be flown.
    #
    # So the rendezvous is resolved for *all* in-flight drones together
    # whenever it moves, and a launch that would strand a drone already in the
    # air is refused.
    # ------------------------------------------------------------------

    def _rendezvous_node(self, extra_skip=()):
        """Where the truck will next stop, and therefore recover its drones."""
        return self.truck_route[
            self._next_required_stop(self.truck_stop_idx, extra_skip=extra_skip)]

    def _sortie_legs(self, dispatch, recovery_node):
        cust_node = self.customer_indices[dispatch["cust_local_idx"]]
        return (float(self.air_matrix[dispatch["launch_node"], cust_node]),
                float(self.air_matrix[cust_node, recovery_node]))

    def _best_pack(self, drone_idx):
        """Index of the fullest pack this drone could take up, and its charge."""
        packs = self.drone_packs[drone_idx]
        idx = max(range(len(packs)), key=lambda p: packs[p])
        return idx, packs[idx]

    def _all_feasible_at(self, dispatches, recovery_node):
        """True if every one of these sorties can reach ``recovery_node``."""
        for d in dispatches:
            out, back = self._sortie_legs(d, recovery_node)
            if not self.drone.is_sortie_feasible(
                    out, back, current_battery_pct=d["soc"],
                    payload_kg=self.payload_kg):
                return False
        return True

    def _recost(self, dispatches, recovery_node):
        """Re-price every sortie against the rendezvous the truck will use."""
        for d in dispatches:
            out, back = self._sortie_legs(d, recovery_node)
            profile = self.drone.sortie_profile(
                out, back, payload_kg=self.payload_kg, soc_pct=d["soc"])
            # A pack change is ground time before the drone can leave, so it
            # lengthens the sortie the truck has to wait out.
            d["sortie_time"] = profile["time_s"] + d["swap_time_s"]
            d["battery_pct_cost"] = profile["battery_pct_cost"]
            d["energy_wh"] = profile["energy_wh"]
            d["distance_m"] = out + back
            d["recovery_node"] = recovery_node

    def _candidate(self, cust_local_idx, truck_node):
        """
        A prospective dispatch, or None if every drone is already airborne.

        The drone flies on its fullest pack. If that is not the one currently
        fitted, the launch costs a swap: the crew pulls the flat pack, drops in
        the charged spare and the flat one goes on the charger. This is what
        breaks the dependency on recharge rate -- a drone can turn round in the
        time it takes to change a battery instead of waiting for one to fill.
        """
        busy = {d["drone_idx"] for d in self.current_dispatches}
        free = [i for i in range(self.n_drones) if i not in busy]
        if not free:
            return None

        drone_idx = max(free, key=lambda i: self._best_pack(i)[1])
        pack_idx, soc = self._best_pack(drone_idx)
        needs_swap = pack_idx != self.drone_installed[drone_idx]
        return {"drone_idx": drone_idx,
                "cust_local_idx": cust_local_idx,
                "launch_node": truck_node,
                "pack_idx": pack_idx,
                "soc": soc,
                "swap_time_s": self.battery_swap_time_s if needs_swap else 0.0}

    def _dispatch_drone(self, slot):
        truck_node = self.truck_route[self.truck_stop_idx]
        nearest = self._get_nearest_unserved(truck_node)

        if slot >= len(nearest):
            self.failed_sorties += 1
            return self.P_failed

        cust_local_idx = nearest[slot]
        candidate = self._candidate(cust_local_idx, truck_node)
        if candidate is None:
            self.failed_sorties += 1
            return self.P_failed

        # Launching this drone lets the truck skip one more stop, so the shared
        # rendezvous moves. Check the whole flight -- the newcomer and everyone
        # already airborne -- against the point the truck will actually reach.
        recovery_node = self._rendezvous_node(extra_skip=(cust_local_idx,))
        trial = self.current_dispatches + [candidate]
        if not self._all_feasible_at(trial, recovery_node):
            self.failed_sorties += 1
            return self.P_failed

        # Accepted: fit the chosen pack. Only now, because action_masks() calls
        # _candidate() purely to test legality and must not change any state.
        if candidate["swap_time_s"] > 0:
            self.drone_installed[candidate["drone_idx"]] = candidate["pack_idx"]
            self.battery_swaps += 1

        self._recost(trial, recovery_node)
        self.current_dispatches = trial
        return 0.0

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_nearest_unserved(self, truck_node):
        dispatched = {d["cust_local_idx"] for d in self.current_dispatches}
        unserved = [i for i in range(self.n_customers)
                    if not self.served[i] and i not in dispatched]
        if not unserved:
            return []
        unserved.sort(
            key=lambda i: self.air_matrix[truck_node, self.customer_indices[i]])
        return unserved[:self.K]

    def _potential(self):
        """
        Shaping potential: Phi(s) = omega / (1 + d(truck, nearest target)).

        Phi rises as the fleet closes on its next target, so the shaping term
        rewards progress and penalises wandering. The target is the nearest
        unserved customer, or the depot once everything is served.
        """
        truck_node = self.truck_route[min(self.truck_stop_idx,
                                          len(self.truck_route) - 1)]
        p = self.node_xy[truck_node]
        pending = [self.customer_indices[i] for i in range(self.n_customers)
                   if not self.served[i]]
        goal_node = (min(pending, key=lambda c: self.air_matrix[truck_node, c])
                     if pending else self.depot_index)
        distance = float(np.linalg.norm(self.node_xy[goal_node] - p))
        return self.omega / (1.0 + distance)

    def _get_obs(self):
        truck_node = self.truck_route[min(self.truck_stop_idx,
                                          len(self.truck_route) - 1)]
        tx, ty = self.node_xy[truck_node]
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        obs[0] = tx
        obs[1] = ty

        in_flight = {d["drone_idx"] for d in self.current_dispatches}
        for i in range(self.n_drones):
            # What it could launch on right now (its fullest pack), whether
            # it is away, and how much charge the whole drone has in
            # reserve. The first drives the next decision; the third is what
            # lets the agent plan a sequence of sorties rather than one.
            packs = self.drone_packs[i]
            obs[2 + 3 * i] = max(packs) / 100.0
            obs[3 + 3 * i] = 1.0 if i in in_flight else 0.0
            obs[4 + 3 * i] = (sum(packs) / len(packs)) / 100.0

        base = 2 + 3 * self.n_drones
        obs[base] = 1.0 - np.sum(self.served) / max(1, self.n_customers)
        obs[base + 1] = min(1.0, self.total_time_s / self.time_limit_s)

        # Per-candidate features. The first two locate the customer; the last
        # two are what the decision actually turns on.
        #
        # An agent given only air distance and bearing cannot tell a good
        # sortie from a bad one, because the payoff is a *road* quantity: how
        # much driving the truck avoids by skipping that stop. Two customers
        # the same distance away as the crow flies can differ several-fold in
        # road detour. Without this feature the policy plateaued below a
        # two-line heuristic that computes the detour explicitly.
        #
        #   detour  road distance the truck saves, normalised
        #   margin  sortie duration against the truck's leg time. Negative
        #           means the drone lands before the truck arrives; positive
        #           means the truck idles waiting for it.
        for k, local_idx in enumerate(self._get_nearest_unserved(truck_node)):
            cust_node = self.customer_indices[local_idx]
            obs[base + 2 + 4 * k] = min(
                1.0, self.air_matrix[truck_node, cust_node] / self.max_dist)
            cx, cy = self.node_xy[cust_node]
            obs[base + 3 + 4 * k] = np.arctan2(cy - ty, cx - tx) / np.pi

            recovery_idx = self._next_required_stop(
                self.truck_stop_idx, extra_skip=(local_idx,))
            recovery_node = self.truck_route[recovery_idx]

            leg_m = float(self.road_matrix[truck_node, recovery_node])
            detour_m = max(0.0, float(
                self.road_matrix[truck_node, cust_node]
                + self.road_matrix[cust_node, recovery_node] - leg_m))
            obs[base + 4 + 4 * k] = min(1.0, detour_m / self.max_dist)

            sortie_s = self.drone.sortie_time_s(
                float(self.air_matrix[truck_node, cust_node]),
                float(self.air_matrix[cust_node, recovery_node]))
            leg_s = self.drone.truck_travel_time_s(leg_m)
            obs[base + 5 + 4 * k] = float(np.clip(
                (sortie_s - leg_s) / 1800.0, -1.0, 1.0))
        return obs

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def truck_only_metrics(self):
        """
        The OR-Tools truck-only baseline for the current instance: the full
        precomputed route, no drone, no skipped stops.
        """
        distance = self.truck_only_distance_m
        return {
            "distance_m": distance,
            "time_s": self.drone.truck_travel_time_s(distance),
            "energy_wh": self.drone.truck_energy_wh(distance),
        }

    def _get_info(self):
        return {
            "instance": self.instance_index,
            "instance_global_id": self.instance_global_id,
            "total_time_s": self.total_time_s,
            "truck_distance_m": self.truck_distance_m,
            "drone_distance_m": self.drone_distance_m,
            "truck_energy_wh": self.truck_energy_wh,
            "drone_energy_wh": self.drone_energy_wh,
            "total_energy_wh": self.truck_energy_wh + self.drone_energy_wh,
            "drone_packs": [list(packs) for packs in self.drone_packs],
            "drone_batteries": [packs[self.drone_installed[i]]
                                for i, packs in enumerate(self.drone_packs)],
            "battery_swaps": self.battery_swaps,
            # The lowest charge any pack has come home with. A spare
            # sitting half-full on the charger is not a safety margin
            # problem; a drone landing near empty is.
            "min_battery_pct": self.min_landing_pct,
            "served": int(np.sum(self.served)),
            "total_customers": self.n_customers,
            "drone_deliveries": self.drone_deliveries,
            "truck_deliveries": self.truck_deliveries,
            "failed_sorties": self.failed_sorties,
            "battery_deaths": self.battery_deaths,
            "truck_stop": self.truck_stop_idx,
            "completion_pct": np.sum(self.served) / self.n_customers * 100.0,
            "route_complete": self.route_complete,
            "truck_wait_time_s": self.truck_wait_time_s,
            "drone_wait_time_s": self.drone_wait_time_s,
            "truck_only": self.truck_only_metrics(),
        }

    def render(self, mode="human"):
        truck_node = self.truck_route[min(self.truck_stop_idx,
                                          len(self.truck_route) - 1)]
        status = (
            "Step {:3d} | Truck @ node {:3d} (stop {}/{}) | Bat {}% | "
            "Flying {} | Served {}/{} | {:6.1f} min | drone/truck {}/{}".format(
                self.current_step, truck_node, self.truck_stop_idx,
                len(self.truck_route) - 1,
                [[round(x) for x in packs] for packs in self.drone_packs],
                len(self.current_dispatches), int(np.sum(self.served)),
                self.n_customers, self.total_time_s / 60.0,
                self.drone_deliveries, self.truck_deliveries))
        if mode == "human":
            print(status)
        return status

    def close(self):
        pass
