"""
TruckDroneEnv -- a Gymnasium environment for truck-drone collaborative
delivery on a real street network.

THE PROBLEM
-----------
A delivery truck must serve N customers and return to the depot. It carries
drones. At any stop the truck can launch a drone with one parcel; the drone
flies straight to its customer (roads do not constrain it), drops the parcel,
then flies on to meet the truck at a later stop. A customer the drone serves is
struck off the truck's route, so the truck genuinely drives less -- and since
the truck no longer stops there, it also skips that customer's doorstep
service time, which is where much of a real delivery route's time goes.

THE AGENT
---------
The agent is the *dispatcher*. At each truck stop it makes one decision:

    action 0              drive to the next required stop
    action 1 + k*R + r    launch a drone to the k-th nearest unserved customer,
                          to be recovered at the truck's (r+1)-th required stop

So the agent chooses not only *whom* to serve by air but *where to meet the
drone afterwards* (Review-1 commitment). Meeting further down the route lets a
drone take a longer sortie without the truck waiting for it; meeting sooner
frees the drone for the next launch. A sortie may therefore span several truck
legs, and two drones may be out at once with different meeting points.

WHAT THE WORLD DOES TO THE PLAN
-------------------------------
  * Real streets. Truck distances are directed shortest paths through the
    OpenStreetMap network, one-way streets included.
  * Doorstep service time. The truck spends ``truck_service_time_s`` at every
    customer it serves itself.
  * Traffic. Truck speed follows a time-of-day congestion profile with rush
    hours, a day-level factor, and unpredictable per-leg noise.
  * Wind. Each episode has a wind vector. A drone flies at constant airspeed,
    so a headwind stretches both its flight time and its energy; a tailwind
    shrinks them. Feasibility is checked against the wind it will actually
    meet.
  * Re-planning. Once drones have taken customers off the truck, the order of
    the remaining stops is re-optimised (asymmetric 2-opt + relocate), so the
    truck stops driving a tour shaped by stops it no longer makes.

The agent observes the wind and the current congestion level -- it can adapt
to both -- but not the per-leg traffic noise, which is the "unexpected" part.

OBJECTIVES (Review-1 commitment: time, energy, cost)
---------------------------------------------------
Each leg produces increments of three objectives: elapsed time, energy (truck
fuel incl. idling + drone electricity) and cost in rupees (driver, fuel,
electricity, drone wear). The reward scalarises them with a preference vector
``w`` on the simplex, each objective normalised by the truck-only baseline of
the same instance in the same world. With ``w = (1, 0, 0)`` the reward is
exactly the time-only reward of earlier versions: -2 per minute.

With ``preference_conditioned=True`` a new ``w`` is drawn each episode and
shown to the agent, so one policy learns the whole trade-off and an operator
picks the balance at run time instead of hand-tuning reward weights.

ACTION MASKING AND HONEST TERMINATION
-------------------------------------
``action_masks()`` and ``_dispatch_drone`` read the same launch plan, so the
mask is exact by construction: a launch is legal only if a drone is free, the
customer and rendezvous exist, and the sortie -- wind included -- leaves the
safety reserve intact. An episode ends at the depot; abandoned customers are
penalised and runs that abandon any are excluded from timing statistics.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .economics import DEFAULT_COSTS
from .physics import DronePhysics

_MDS_CACHE = {}

OBJECTIVES = ("time", "energy", "cost")
PREFERENCE_PRESETS = {
    "fastest":  (1.0, 0.0, 0.0),
    "balanced": (1 / 3, 1 / 3, 1 / 3),
    "cheapest": (0.0, 0.0, 1.0),
    "greenest": (0.0, 1.0, 0.0),
}


class TruckDroneEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        scenario,
        instance_ids=None,
        n_drones=2,
        drone_config=None,
        max_visible_customers=5,
        rendezvous_options=2,
        max_steps=None,
        time_limit_s=28800,
        payload_kg=1.0,
        recharge_pct_per_min=1.0,
        batteries_per_drone=2,
        battery_swap_time_s=60.0,
        truck_service_time_s=120.0,
        replan_tour=True,
        # --- stochastic world ---
        stochastic=True,
        wind_max_ms=8.0,
        wind_fixed=None,               # (speed m/s, heading rad or None)
        free_flow_kmh=30.0,
        rush_peak=1.7,
        day_sigma=0.10,
        traffic_sigma=0.15,
        start_hour_range=(8.0, 15.0),
        static_truck_kmh=25.0,         # used when stochastic=False
        # --- objectives ---
        preference=None,
        preference_conditioned=False,
        cost_model=DEFAULT_COSTS,
        # --- reward ---
        reward_delivery=10.0,
        reward_drone_delivery_bonus=2.0,
        reward_completion_bonus=100.0,
        penalty_per_minute=-2.0,
        penalty_per_step=-0.1,
        penalty_failed_sortie=-15.0,
        penalty_battery_death=-30.0,
        penalty_per_unserved=-30.0,
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
        self.R = max(1, int(rendezvous_options))
        self.time_limit_s = time_limit_s
        self.payload_kg = payload_kg
        self.recharge_pct_per_min = recharge_pct_per_min
        self.batteries_per_drone = max(1, int(batteries_per_drone))
        self.battery_swap_time_s = battery_swap_time_s
        self.service_s = truck_service_time_s
        self.replan_tour = replan_tour

        self.stochastic = stochastic
        self.wind_max_ms = wind_max_ms
        self.wind_fixed = wind_fixed
        self.free_flow_ms = free_flow_kmh / 3.6
        self.rush_peak = rush_peak
        self.day_sigma = day_sigma
        self.traffic_sigma = traffic_sigma
        self.start_hour_range = start_hour_range
        self.static_truck_ms = static_truck_kmh / 3.6

        self.preference_conditioned = preference_conditioned
        self._default_preference = np.asarray(preference or (1.0, 0.0, 0.0), float)
        self.costs = cost_model

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

        self.action_space = spaces.Discrete(1 + self.K * self.R)
        self.per_drone = 6
        self.per_candidate = 2 + 2 * self.R
        obs_size = (2 + self.per_drone * self.n_drones + 6 + len(OBJECTIVES)
                    + self.per_candidate * self.K)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        finite = self.road_matrix[np.isfinite(self.road_matrix)]
        self.max_dist = float(finite.max()) or 1.0
        self.node_xy = (np.asarray(scenario["node_xy"], dtype=np.float64)
                        if "node_xy" in scenario else self._embed_positions())
        self._max_steps_override = max_steps
        self._instance_cursor = 0

        self.reset()

    # ==================================================================
    # Setup
    # ==================================================================

    def _embed_positions(self):
        """
        Fallback for a scenario without a stored embedding. Normal scenarios
        carry ``node_xy`` (see ``scenario.embed_positions``), so this -- and
        the scikit-learn import it implies -- never runs in a training worker.
        """
        key = (self.n_nodes, float(np.nansum(self.road_matrix)))
        if key not in _MDS_CACHE:
            from .scenario import embed_positions
            _MDS_CACHE[key] = embed_positions(self.road_matrix)
        return _MDS_CACHE[key]

    def _load_instance(self, index):
        pos = index % len(self.instances)
        inst = self.instances[pos]
        self.instance_index = pos
        self.instance_global_id = self.instance_ids[pos]
        self.depot_index = int(inst["depot"])
        self.customer_indices = [int(c) for c in inst["customers"]]
        self.n_customers = len(self.customer_indices)
        self.base_route = [int(n) for n in inst["truck_route"]]
        self.truck_only_distance_m = float(inst["truck_route_distance_m"])
        self.max_steps = (self._max_steps_override
                          or len(self.base_route) * (self.n_drones + 3))

    # ==================================================================
    # The world: traffic and wind
    # ==================================================================

    def _draw_world(self, world_seed):
        rng = np.random.default_rng(world_seed)
        self.world_seed = int(world_seed)
        if self.stochastic:
            lo, hi = self.start_hour_range
            self.start_hour = float(rng.uniform(lo, hi))
            self.day_factor = float(rng.lognormal(0.0, self.day_sigma))
            speed = float(rng.uniform(0.0, self.wind_max_ms))
            heading = float(rng.uniform(0.0, 2 * np.pi))
            if self.wind_fixed is not None:
                # Pin the speed; a heading of None keeps the drawn direction,
                # so a wind-speed sweep still averages over directions.
                speed = float(self.wind_fixed[0])
                if self.wind_fixed[1] is not None:
                    heading = float(self.wind_fixed[1])
        else:
            self.start_hour, self.day_factor = 9.0, 1.0
            speed, heading = (self.wind_fixed or (0.0, 0.0))
            heading = 0.0 if heading is None else heading
        self.wind = np.array([speed * np.cos(heading), speed * np.sin(heading)])
        self.wind_speed = float(speed)

    def _congestion(self, t_s):
        """Multiplier on free-flow travel time at simulation time ``t_s``."""
        if not self.stochastic:
            return 1.0
        h = self.start_hour + t_s / 3600.0
        rush = max(np.exp(-0.5 * ((h - 9.5) / 1.2) ** 2),
                   np.exp(-0.5 * ((h - 18.5) / 1.4) ** 2))
        return (1.0 + (self.rush_peak - 1.0) * rush) * self.day_factor

    def _expected_drive_s(self, a, b, t_s):
        dist = float(self.road_matrix[a, b])
        if not self.stochastic:
            return dist / self.static_truck_ms
        return dist / self.free_flow_ms * self._congestion(t_s)

    def _drive_s(self, a, b, t_s):
        """
        Realised drive time. The per-leg noise uses common random numbers --
        seeded by the world, the leg and the 15-minute window -- so the
        truck-only baseline and every policy meet the *same* traffic on the
        same street at the same time, and comparisons stay paired.
        """
        base = self._expected_drive_s(a, b, t_s)
        if not self.stochastic or self.traffic_sigma <= 0 or a == b:
            return base
        seed = (self.world_seed * 1_000_003 + a * 7919 + b * 104_729
                + int(t_s // 900) * 31) % (2 ** 32)
        return base * float(np.random.default_rng(seed).lognormal(0.0, self.traffic_sigma))

    def _wind_equiv_m(self, a, b):
        """
        Still-air distance with the same flight time as flying a->b in this
        wind. The drone holds its airspeed, so time and energy both scale with
        time aloft; feeding the physics an equivalent distance keeps every
        battery and voltage-sag effect intact.
        """
        d = float(self.air_matrix[a, b])
        if d <= 0.0 or self.wind_speed <= 0.0:
            return d
        u = (self.coords[b] - self.coords[a]) / d
        w_par = float(self.wind @ u)
        w_perp = float(np.linalg.norm(self.wind - w_par * u))
        v = self.drone.cruise_speed_ms
        ground = np.sqrt(max(v * v - w_perp * w_perp, 0.0)) + w_par
        return d * v / max(ground, 0.25 * v)

    # ==================================================================
    # Gym API
    # ==================================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}

        if "instance" in options:
            index = int(options["instance"])
        elif seed is not None:
            index = int(seed)
        else:
            index = self._instance_cursor
            self._instance_cursor += 1
        self._load_instance(index)

        world_seed = options.get("world_seed")
        if world_seed is None:
            world_seed = int(self.np_random.integers(2 ** 31 - 1))
        self._draw_world(world_seed)

        if "preference" in options and options["preference"] is not None:
            pref = np.asarray(options["preference"], dtype=float)
        elif self.preference_conditioned:
            pref = self.np_random.dirichlet(np.ones(len(OBJECTIVES)))
        else:
            pref = self._default_preference
        self.preference = pref / max(pref.sum(), 1e-9)

        self.truck_route = list(self.base_route)
        self.current_step = 0
        self.truck_stop_idx = 0
        self.drone_packs = [[100.0] * self.batteries_per_drone
                            for _ in range(self.n_drones)]
        self.drone_installed = [0] * self.n_drones
        self.battery_swaps = 0
        self.min_landing_pct = 100.0
        self.served = np.zeros(self.n_customers, dtype=bool)
        self.total_time_s = 0.0
        self.drone_energy_wh = 0.0
        self.truck_energy_wh = 0.0
        self.truck_idle_wh = 0.0
        self.truck_distance_m = 0.0
        self.drone_distance_m = 0.0
        self.drone_deliveries = 0
        self.truck_deliveries = 0
        self.truck_wait_time_s = 0.0
        self.drone_wait_time_s = 0.0
        self.failed_sorties = 0
        self.battery_deaths = 0
        self.replans = 0
        self.late_rendezvous = 0
        self.current_dispatches = []
        self.route_complete = False
        self.trace = {"legs": [], "sorties": []}
        self._plans_cache = None
        self._baseline = self._simulate_truck_only()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.current_step += 1
        reward = self.P_step
        phi_before = self._potential()
        terminated = truncated = False

        if action == 0:
            reward += self._advance_truck()
        else:
            reward += self._dispatch_drone(int(action) - 1)
        self._plans_cache = None

        if any(soc <= 0 for packs in self.drone_packs for soc in packs):
            reward += self.P_battery_dead
            self.battery_deaths += 1
            self.drone_packs = [[max(0.0, s) for s in p] for p in self.drone_packs]

        all_served = bool(np.all(self.served))
        at_route_end = self.truck_stop_idx >= len(self.truck_route) - 1
        out_of_time = self.total_time_s >= self.time_limit_s

        if at_route_end or out_of_time:
            n_unserved = int(self.n_customers - np.sum(self.served))
            if all_served and at_route_end:
                self.route_complete = True
                reward += self.R_completion
            else:
                reward += self.P_unserved * n_unserved
            terminated = True
        elif self.current_step >= self.max_steps:
            reward += self.P_unserved * int(self.n_customers - np.sum(self.served))
            truncated = True

        if self.use_reward_shaping:
            phi_after = 0.0 if (terminated or truncated) else self._potential()
            reward += self.gamma * phi_after - phi_before

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    # ==================================================================
    # Route bookkeeping
    # ==================================================================

    def _truck_node(self):
        return self.truck_route[min(self.truck_stop_idx, len(self.truck_route) - 1)]

    def _pending_customer_at(self, node):
        """Local index of an unserved, undispatched customer at ``node``."""
        dispatched = {d["cust_local_idx"] for d in self.current_dispatches}
        for local_idx, cust_node in enumerate(self.customer_indices):
            if (cust_node == node and not self.served[local_idx]
                    and local_idx not in dispatched):
                return local_idx
        return None

    def _rendezvous_indices(self):
        return {d["recovery_idx"] for d in self.current_dispatches}

    def _required_after(self, from_idx, count, extra_skip=()):
        """
        The next ``count`` stops the truck must actually make after
        ``from_idx``: a stop is required if a customer there still needs the
        truck, if an airborne drone is due to meet it there, or if it is the
        final depot return. Everything else is driven past -- which is how a
        sortie removes truck mileage.
        """
        last = len(self.truck_route) - 1
        meet = self._rendezvous_indices()
        out = []
        for j in range(from_idx + 1, last + 1):
            if j == last or j in meet:
                out.append(j)
            else:
                local = self._pending_customer_at(self.truck_route[j])
                if local is not None and local not in extra_skip:
                    out.append(j)
            if len(out) == count:
                break
        return out

    def _next_required_stop(self, from_idx, extra_skip=()):
        return self._required_after(from_idx, 1, extra_skip)[0]

    # ==================================================================
    # Launch planning -- the single source of truth for mask and dispatch
    # ==================================================================

    def _get_nearest_unserved(self, truck_node):
        dispatched = {d["cust_local_idx"] for d in self.current_dispatches}
        unserved = [i for i in range(self.n_customers)
                    if not self.served[i] and i not in dispatched]
        unserved.sort(key=lambda i: self.air_matrix[truck_node, self.customer_indices[i]])
        return unserved[:self.K]

    def _best_pack(self, drone_idx):
        packs = self.drone_packs[drone_idx]
        idx = max(range(len(packs)), key=lambda p: packs[p])
        return idx, packs[idx]

    def _expected_arrival_s(self, stop_indices, t0, extra_skip=()):
        """Expected truck arrival time at the last of ``stop_indices``."""
        t = t0
        prev = self._truck_node()
        for n, j in enumerate(stop_indices):
            node = self.truck_route[j]
            t += self._expected_drive_s(prev, node, t)
            if n < len(stop_indices) - 1:
                local = self._pending_customer_at(node)
                if local is not None and local not in extra_skip:
                    t += self.service_s
            prev = node
        return t

    def _state_signature(self):
        return (self.truck_stop_idx, self.total_time_s, self.served.tobytes(),
                tuple(self.truck_route),
                tuple((d["drone_idx"], d["cust_local_idx"], d["recovery_idx"])
                      for d in self.current_dispatches),
                tuple(tuple(p) for p in self.drone_packs),
                tuple(self.drone_installed))

    def _plans(self):
        """
        Every (slot, rendezvous option) the agent could choose right now, with
        the features the observation needs and -- if legal -- the full sortie
        plan. The mask, the observation and the dispatch all read this one
        computation, so they cannot disagree.

        It is cached against a signature of the state rather than invalidated
        by hand: anything that changes the state -- a step, or a test restoring
        a snapshot -- changes the signature, so a stale plan can never be read.
        """
        sig = self._state_signature()
        if self._plans_cache is not None and self._plans_cache[0] == sig:
            return self._plans_cache[1]

        truck_node = self._truck_node()
        nearest = self._get_nearest_unserved(truck_node)
        busy = {d["drone_idx"] for d in self.current_dispatches}
        free = [i for i in range(self.n_drones) if i not in busy]
        drone_idx = (max(free, key=lambda i: self._best_pack(i)[1]) if free else None)
        t0 = self.total_time_s

        plans = {}
        for slot, cust_local in enumerate(nearest):
            cust_node = self.customer_indices[cust_local]
            opts = self._required_after(self.truck_stop_idx, self.R,
                                        extra_skip=(cust_local,))
            path = [truck_node] + [self.truck_route[j] for j in opts]
            for r in range(self.R):
                entry = {"slot": slot, "r": r, "cust_local_idx": cust_local,
                         "customer_node": cust_node, "plan": None,
                         "detour_m": 0.0, "margin_s": 0.0}
                plans[(slot, r)] = entry
                if r >= len(opts):
                    continue
                rdv_node = self.truck_route[opts[r]]
                # Road detour the truck avoids: the cheapest place to slot this
                # customer into the truck's path up to the meeting point.
                seg = path[:r + 2]
                entry["detour_m"] = max(0.0, min(
                    self.road_matrix[a, cust_node] + self.road_matrix[cust_node, b]
                    - self.road_matrix[a, b] for a, b in zip(seg[:-1], seg[1:])))
                out_eq = self._wind_equiv_m(truck_node, cust_node)
                back_eq = self._wind_equiv_m(cust_node, rdv_node)
                arrive = self._expected_arrival_s(opts[:r + 1], t0,
                                                  extra_skip=(cust_local,))
                if drone_idx is None:
                    entry["margin_s"] = (self.drone.sortie_time_s(out_eq, back_eq)
                                         - (arrive - t0))
                    continue
                pack_idx, soc = self._best_pack(drone_idx)
                swap_s = (self.battery_swap_time_s
                          if pack_idx != self.drone_installed[drone_idx] else 0.0)
                sortie_s = self.drone.sortie_time_s(out_eq, back_eq) + swap_s
                entry["margin_s"] = sortie_s - (arrive - t0)
                profile = self.drone.sortie_profile(out_eq, back_eq,
                                                    payload_kg=self.payload_kg,
                                                    soc_pct=soc)
                if profile["end_soc_pct"] < self.drone.safety_margin_pct:
                    continue
                entry["plan"] = {
                    "drone_idx": drone_idx, "cust_local_idx": cust_local,
                    "launch_node": truck_node, "launch_time": t0,
                    "recovery_idx": opts[r], "recovery_node": rdv_node,
                    "rendezvous_option": r, "pack_idx": pack_idx, "soc": soc,
                    "swap_time_s": swap_s, "sortie_time": sortie_s,
                    "battery_pct_cost": profile["battery_pct_cost"],
                    "energy_wh": profile["energy_wh"],
                    "distance_m": float(self.air_matrix[truck_node, cust_node]
                                        + self.air_matrix[cust_node, rdv_node]),
                    "detour_saved_m": entry["detour_m"],
                }
        self._plans_cache = (sig, plans)
        return plans

    def decode_action(self, action):
        """action >= 1 -> (slot, rendezvous option)."""
        idx = int(action) - 1
        return idx // self.R, idx % self.R

    def action_masks(self):
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True
        for (slot, r), entry in self._plans().items():
            if entry["plan"] is not None:
                mask[1 + slot * self.R + r] = True
        return mask

    def _dispatch_drone(self, idx):
        slot, r = idx // self.R, idx % self.R
        entry = self._plans().get((slot, r))
        if entry is None or entry["plan"] is None:
            self.failed_sorties += 1
            return self.P_failed
        plan = dict(entry["plan"])
        if plan["swap_time_s"] > 0:
            self.drone_installed[plan["drone_idx"]] = plan["pack_idx"]
            self.battery_swaps += 1
        self.current_dispatches.append(plan)
        return 0.0

    # ==================================================================
    # Truck movement
    # ==================================================================

    def _advance_truck(self):
        reward = 0.0
        from_idx = self.truck_stop_idx
        from_node = self.truck_route[from_idx]
        next_idx = self._next_required_stop(from_idx)
        to_node = self.truck_route[next_idx]
        t0 = self.total_time_s

        dist = float(self.road_matrix[from_node, to_node])
        drive_s = self._drive_s(from_node, to_node, t0)
        t_arrive = t0 + drive_s

        truck_local = self._pending_customer_at(to_node)
        service_s = self.service_s if truck_local is not None else 0.0
        truck_done = t_arrive + service_s

        # Drones due to meet the truck here. Each lands at its own time; the
        # leg is over when the truck has finished at this stop *and* every
        # drone due here is back aboard.
        landing = [d for d in self.current_dispatches if d["recovery_idx"] == next_idx]
        land_times = [d["launch_time"] + d["sortie_time"] for d in landing]
        t_end = max([truck_done] + land_times)
        idle_s = t_end - truck_done
        if idle_s > 0:
            self.truck_wait_time_s += idle_s
            self.late_rendezvous += 1
        for lt in land_times:
            self.drone_wait_time_s += max(0.0, t_arrive - lt)

        airborne = {d["drone_idx"]: d for d in self.current_dispatches}
        n_recovered = 0
        drone_wh = 0.0
        for d in landing:
            pack = d["pack_idx"]
            before = self.drone_packs[d["drone_idx"]][pack]
            self.drone_packs[d["drone_idx"]][pack] -= d["battery_pct_cost"]
            self.min_landing_pct = min(self.min_landing_pct,
                                       self.drone_packs[d["drone_idx"]][pack])
            self.drone_energy_wh += d["energy_wh"]
            drone_wh += d["energy_wh"]
            self.drone_distance_m += d["distance_m"]
            self.served[d["cust_local_idx"]] = True
            self.drone_deliveries += 1
            n_recovered += 1
            reward += self.R_delivery + self.R_drone_bonus
            if self.record_trace:
                self.trace["sorties"].append({
                    "drone_idx": d["drone_idx"],
                    "launch_node": d["launch_node"],
                    "customer_node": self.customer_indices[d["cust_local_idx"]],
                    "recovery_node": d["recovery_node"],
                    "rendezvous_option": d["rendezvous_option"],
                    "t_launch": d["launch_time"],
                    "t_land": d["launch_time"] + d["sortie_time"],
                    "t_recover": t_end,
                    "distance_m": d["distance_m"], "energy_wh": d["energy_wh"],
                    "battery_before": before,
                    "battery_after": before - d["battery_pct_cost"],
                    "swapped": d["swap_time_s"] > 0,
                    "detour_saved_m": d["detour_saved_m"],
                })
        self.current_dispatches = [d for d in self.current_dispatches
                                   if d["recovery_idx"] != next_idx]

        if truck_local is not None:
            self.served[truck_local] = True
            self.truck_deliveries += 1
            reward += self.R_delivery

        leg_s = t_end - t0
        truck_wh = self.drone.truck_energy_wh(dist)
        idle_wh = self.costs.idle_wh(idle_s)
        self.truck_energy_wh += truck_wh + idle_wh
        self.truck_idle_wh += idle_wh
        self.truck_distance_m += dist
        self.total_time_s = t_end
        self.truck_stop_idx = next_idx

        # Charging. A pack charges only while it is on the truck. A drone that
        # was airborne at any point in this leg was off the truck for all of it
        # -- launches happen at the start of a leg and recoveries at its end --
        # so its fitted pack gains nothing, while every spare charges for the
        # whole leg.
        for i in range(self.n_drones):
            for p in range(len(self.drone_packs[i])):
                if i in airborne and p == airborne[i]["pack_idx"]:
                    continue
                gain = leg_s / 60.0 * self.recharge_pct_per_min
                self.drone_packs[i][p] = min(100.0, self.drone_packs[i][p] + gain)

        # The objectives, scalarised by the preference and normalised by this
        # instance's truck-only baseline in this same world.
        d_time = leg_s
        d_energy = truck_wh + idle_wh + drone_wh
        d_cost = self.costs.cost_inr(leg_s, truck_wh + idle_wh, drone_wh, n_recovered)
        b = self._baseline
        scale = self.P_minute * b["time_s"] / 60.0
        reward += scale * float(self.preference @ np.array([
            d_time / b["time_s"], d_energy / b["energy_wh"], d_cost / b["cost_inr"]]))

        if self.record_trace:
            self.trace["legs"].append({
                "from_node": from_node, "to_node": to_node,
                "t_start": t0, "t_arrive": t_arrive, "t_end": t_end,
                "distance_m": dist, "service_s": service_s, "idle_s": idle_s,
                "truck_served_node": to_node if truck_local is not None else None,
                "packs_after": [[round(x, 1) for x in p] for p in self.drone_packs],
                "congestion": self._congestion(t0),
            })

        if self.replan_tour and (self.drone_deliveries or self.current_dispatches):
            self._replan()
        return reward

    # ==================================================================
    # Tour re-planning
    # ==================================================================

    def _replan(self):
        """
        Re-optimise the order of the truck's remaining stops.

        Only the tail *after* the last pending meeting point is reordered, so
        every airborne drone still finds the truck where it was promised. The
        distances are directed, so the moves are asymmetric-aware: 2-opt with
        prefix sums of the forward and reverse costs (reversing a segment
        changes what every street in it costs), plus single-stop relocation,
        which never reverses anything.
        """
        last = len(self.truck_route) - 1
        anchor = max([self.truck_stop_idx] + list(self._rendezvous_indices()))
        if anchor >= last - 1:
            return
        dispatched = {d["cust_local_idx"] for d in self.current_dispatches}
        tail = []
        for j in range(anchor + 1, last):
            node = self.truck_route[j]
            local = self._pending_customer_at(node)
            if local is not None and local not in dispatched:
                tail.append(node)
        start, end = self.truck_route[anchor], self.truck_route[last]
        before = [start] + tail + [end]
        after = _improve_open_path(before, self.road_matrix)
        if after != before or len(tail) != last - anchor - 1:
            self.truck_route = self.truck_route[:anchor + 1] + after[1:]
            if after != before:
                self.replans += 1

    # ==================================================================
    # Baseline in the same world
    # ==================================================================

    def _simulate_truck_only(self):
        """
        The OR-Tools truck-only tour, driven through *this episode's* world --
        same start hour, same congestion, same per-leg traffic draws (common
        random numbers), same doorstep service time. The comparison is paired
        down to the traffic jam.
        """
        t, dist = 0.0, 0.0
        route = self.base_route
        for a, b in zip(route[:-1], route[1:]):
            t += self._drive_s(a, b, t)
            dist += self.road_matrix[a, b]
            if b != self.depot_index:
                t += self.service_s
        wh = self.drone.truck_energy_wh(dist)
        return {"distance_m": float(dist), "time_s": float(t), "energy_wh": wh,
                "cost_inr": self.costs.cost_inr(t, wh, 0.0, 0),
                "co2_kg": self.costs.co2_kg(wh, 0.0)}

    def truck_only_metrics(self):
        return dict(self._baseline)

    # ==================================================================
    # Observation
    # ==================================================================

    def _potential(self):
        truck_node = self._truck_node()
        p = self.node_xy[truck_node]
        pending = [self.customer_indices[i] for i in range(self.n_customers)
                   if not self.served[i]]
        goal = (min(pending, key=lambda c: self.air_matrix[truck_node, c])
                if pending else self.depot_index)
        return self.omega / (1.0 + float(np.linalg.norm(self.node_xy[goal] - p)))

    def _get_obs(self):
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        truck_node = self._truck_node()
        tx, ty = self.node_xy[truck_node]
        obs[0], obs[1] = tx, ty

        airborne = {d["drone_idx"]: d for d in self.current_dispatches}
        base = 2
        for i in range(self.n_drones):
            packs = self.drone_packs[i]
            o = base + self.per_drone * i
            obs[o] = max(packs) / 100.0
            obs[o + 2] = sum(packs) / len(packs) / 100.0
            d = airborne.get(i)
            if d is not None:
                obs[o + 1] = 1.0
                land = d["launch_time"] + d["sortie_time"] - self.total_time_s
                obs[o + 3] = np.clip(land / 1800.0, -1, 1)
                rx, ry = self.node_xy[d["recovery_node"]]
                obs[o + 4] = np.clip((rx - tx) / 2.0, -1, 1)
                obs[o + 5] = np.clip((ry - ty) / 2.0, -1, 1)

        g = base + self.per_drone * self.n_drones
        obs[g] = 1.0 - np.sum(self.served) / max(1, self.n_customers)
        obs[g + 1] = min(1.0, self.total_time_s / self.time_limit_s)
        scale = max(self.wind_max_ms, 1.0)
        obs[g + 2] = np.clip(self.wind[0] / scale, -1, 1)
        obs[g + 3] = np.clip(self.wind[1] / scale, -1, 1)
        obs[g + 4] = np.clip((self._congestion(self.total_time_s) - 1.0) / 1.5, -1, 1)
        hour = self.start_hour + self.total_time_s / 3600.0
        obs[g + 5] = np.clip((hour - 13.0) / 6.0, -1, 1)
        obs[g + 6:g + 6 + len(OBJECTIVES)] = self.preference

        c0 = g + 6 + len(OBJECTIVES)
        plans = self._plans()
        nearest = self._get_nearest_unserved(truck_node)
        for k, local in enumerate(nearest):
            o = c0 + self.per_candidate * k
            node = self.customer_indices[local]
            obs[o] = min(1.0, self.air_matrix[truck_node, node] / self.max_dist)
            cx, cy = self.node_xy[node]
            obs[o + 1] = np.arctan2(cy - ty, cx - tx) / np.pi
            for r in range(self.R):
                e = plans.get((k, r))
                if e is None:
                    continue
                obs[o + 2 + 2 * r] = min(1.0, e["detour_m"] / self.max_dist)
                obs[o + 3 + 2 * r] = np.clip(e["margin_s"] / 1800.0, -1, 1)
        return obs

    # ==================================================================
    # Reporting
    # ==================================================================

    def _get_info(self):
        n_sorties = self.drone_deliveries
        total_wh = self.truck_energy_wh + self.drone_energy_wh
        return {
            "instance": self.instance_index,
            "instance_global_id": self.instance_global_id,
            "world_seed": self.world_seed,
            "preference": list(map(float, self.preference)),
            "wind_ms": self.wind_speed,
            "start_hour": self.start_hour,
            "total_time_s": self.total_time_s,
            "truck_distance_m": self.truck_distance_m,
            "drone_distance_m": self.drone_distance_m,
            "truck_energy_wh": self.truck_energy_wh,
            "truck_idle_wh": self.truck_idle_wh,
            "drone_energy_wh": self.drone_energy_wh,
            "total_energy_wh": total_wh,
            "cost_inr": self.costs.cost_inr(self.total_time_s, self.truck_energy_wh,
                                            self.drone_energy_wh, n_sorties),
            "co2_kg": self.costs.co2_kg(self.truck_energy_wh, self.drone_energy_wh),
            "drone_packs": [list(p) for p in self.drone_packs],
            "battery_swaps": self.battery_swaps,
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
            "replans": self.replans,
            "late_rendezvous": self.late_rendezvous,
            "truck_only": dict(self._baseline),
        }

    def render(self, mode="human"):
        status = ("Step {:3d} | stop {}/{} | packs {} | flying {} | served {}/{} | "
                  "{:6.1f} min | wind {:.1f} m/s".format(
                      self.current_step, self.truck_stop_idx,
                      len(self.truck_route) - 1,
                      [[round(x) for x in p] for p in self.drone_packs],
                      len(self.current_dispatches), int(np.sum(self.served)),
                      self.n_customers, self.total_time_s / 60.0, self.wind_speed))
        if mode == "human":
            print(status)
        return status

    def close(self):
        pass


# ======================================================================
# Local search for the re-planned tour
# ======================================================================

def _path_cost(path, D):
    return float(sum(D[a, b] for a, b in zip(path[:-1], path[1:])))


def _improve_open_path(path, D, max_rounds=50):
    """
    Improve an open path with fixed endpoints under a *directed* cost matrix.

    Two neighbourhoods, applied until neither helps:
      * 2-opt: reverse an interior segment. With asymmetric costs every street
        inside the segment flips direction, so its cost is recomputed from
        prefix sums of forward and reverse traversal -- O(1) per move.
      * relocate: lift one stop and reinsert it elsewhere. No reversal, so it
        finds improvements 2-opt cannot when one-way streets punish reversal.
    """
    p = list(path)
    m = len(p)
    if m <= 3:
        return p
    for _ in range(max_rounds):
        improved = False
        fwd = np.zeros(m)
        rev = np.zeros(m)
        for k in range(1, m):
            fwd[k] = fwd[k - 1] + D[p[k - 1], p[k]]
            rev[k] = rev[k - 1] + D[p[k], p[k - 1]]
        best, move = -1e-6, None
        for i in range(1, m - 2):
            for j in range(i + 1, m - 1):
                delta = (D[p[i - 1], p[j]] + D[p[i], p[j + 1]]
                         - D[p[i - 1], p[i]] - D[p[j], p[j + 1]]
                         + (rev[j] - rev[i]) - (fwd[j] - fwd[i]))
                if delta < best:
                    best, move = delta, ("2opt", i, j)
        for i in range(1, m - 1):
            a, x, b = p[i - 1], p[i], p[i + 1]
            gain = D[a, x] + D[x, b] - D[a, b]
            for j in range(0, m - 1):
                if j == i or j == i - 1:
                    continue
                u, v = p[j], p[j + 1]
                delta = D[u, x] + D[x, v] - D[u, v] - gain
                if delta < best:
                    best, move = delta, ("move", i, j)
        if move is None:
            break
        kind, i, j = move
        if kind == "2opt":
            p[i:j + 1] = p[i:j + 1][::-1]
        else:
            x = p.pop(i)
            p.insert(j + 1 if j < i else j, x)
    return p
