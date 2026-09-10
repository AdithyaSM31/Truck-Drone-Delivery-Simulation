"""
Tests for the properties the results actually depend on.

These are not coverage tests. Each one pins down a claim the write-up makes,
so that if someone changes the environment and a claim stops being true, the
suite says so instead of the number quietly moving.

    pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truckdrone import config                                  # noqa: E402
from truckdrone.baselines import POLICIES                      # noqa: E402
from truckdrone.env import TruckDroneEnv                       # noqa: E402
from truckdrone.evaluate import evaluate_policy, summarise     # noqa: E402
from truckdrone.physics import DronePhysics                    # noqa: E402
from truckdrone.scenario import (build_graph, load_scenario,   # noqa: E402
                                 split_instances)


@pytest.fixture(scope="module")
def scenario():
    return load_scenario()


@pytest.fixture(scope="module")
def eval_ids(scenario):
    return split_instances(scenario)[1]


@pytest.fixture(scope="module")
def env(scenario, eval_ids):
    return TruckDroneEnv(scenario, instance_ids=eval_ids, **config.env_kwargs())


# ======================================================================
# Scenario
# ======================================================================

def test_graph_reproduces_cached_road_matrix(scenario):
    """
    The animation redraws the road graph from coordinates alone. If that
    reconstruction drifted from the graph whose shortest paths produced the
    cached distance matrix, the dashboard would show the truck driving a route
    of a different length than the one being scored.
    """
    import networkx as nx

    G = build_graph(scenario["coords"])
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    n = scenario["n_nodes"]
    err = max(abs(lengths[i][j] - scenario["road_matrix"][i][j])
              for i in range(n) for j in range(n))
    assert err < 1e-6


def test_road_distance_never_shorter_than_air_distance(scenario):
    """
    The drone's advantage rests on flying straight while the truck follows
    roads. If any road path were shorter than the straight line between the
    same two nodes, the geometry would be broken.
    """
    road = np.asarray(scenario["road_matrix"])
    air = np.asarray(scenario["air_matrix"])
    assert np.all(road >= air - 1e-6)


def test_train_and_eval_instances_are_disjoint(scenario):
    train_ids, eval_ids = split_instances(scenario)
    assert set(train_ids).isdisjoint(eval_ids)
    assert len(eval_ids) > 0


# ======================================================================
# Action masking
# ======================================================================

def test_advance_is_always_legal(env):
    """
    Without this the agent could be trapped in a state with no legal action.
    Advancing the truck is what guarantees every episode can still progress.
    """
    for instance in range(5):
        env.reset(options={"instance": instance})
        for _ in range(env.max_steps):
            assert env.action_masks()[0]
            _, _, term, trunc, _ = env.step(0)
            if term or trunc:
                break


def test_mask_matches_what_the_env_will_actually_accept(env):
    """
    The mask is a promise: every action it marks legal must succeed, and every
    action it marks illegal must be refused. Check both directions by taking
    each action from a copy of the same state.
    """
    rng = np.random.default_rng(0)

    def snapshot():
        # Deep-copy the dispatches: accepting a launch re-prices every sortie
        # already in flight, mutating those dicts in place, so a shallow copy
        # would not restore the state between trial actions.
        return (env.truck_stop_idx, env.served.copy(),
                [list(packs) for packs in env.drone_packs],
                list(env.drone_installed),
                [dict(d) for d in env.current_dispatches])

    def restore(state):
        env.truck_stop_idx, env.served = state[0], state[1].copy()
        env.drone_packs = [list(packs) for packs in state[2]]
        env.drone_installed = list(state[3])
        env.current_dispatches = [dict(d) for d in state[4]]

    for instance in range(3):
        env.reset(options={"instance": instance})
        for _ in range(12):
            mask = env.action_masks().copy()
            state = snapshot()

            for action in range(env.action_space.n):
                restore(state)
                before = env.failed_sorties
                env.step(action)
                rejected = env.failed_sorties > before
                assert rejected != bool(mask[action]), \
                    "mask[{}]={} but rejected={}".format(
                        action, mask[action], rejected)

            restore(state)
            legal = np.flatnonzero(mask)
            _, _, term, trunc, _ = env.step(int(rng.choice(legal)))
            if term or trunc:
                break


def test_masked_rollouts_never_fail_a_sortie(scenario, eval_ids):
    """
    The headline claim of the masking change: illegal dispatches go to exactly
    zero, for every policy that respects the mask, on every held-out instance.
    """
    for name in ("random", "always_nearest", "greedy"):
        runs = evaluate_policy(POLICIES[name](), scenario, eval_ids)
        assert all(r["failed_sorties"] == 0 for r in runs), name


def test_no_sortie_is_dispatched_without_a_free_drone(env):
    env.reset(options={"instance": 0})
    for _ in range(env.max_steps):
        busy = len(env.current_dispatches)
        if busy >= env.n_drones:
            assert not env.action_masks()[1:].any()
        legal = np.flatnonzero(env.action_masks())
        _, _, term, trunc, _ = env.step(int(legal[-1]))
        if term or trunc:
            break


# ======================================================================
# Physics
# ======================================================================

def test_sortie_is_costed_against_the_rendezvous_the_truck_reaches(env):
    """
    Every drone is recovered where the truck next stops, and that point moves
    each time another drone is launched -- the truck can then skip one more
    customer. A sortie priced once at launch and never revisited ends up
    budgeted for a pickup the truck never makes: charged for a short return
    leg and flown on a long one, which understates energy and lets the
    feasibility check pass sorties that cannot be flown.

    So: at the moment of recovery, the distance each sortie was charged for
    must equal the distance actually flown to the truck's real position.
    """
    rng = np.random.default_rng(3)
    checked = 0

    for instance in range(len(env.instances)):
        env.reset(options={"instance": instance})
        for _ in range(env.max_steps):
            mask = env.action_masks()
            # Prefer launching, so multi-drone rendezvous shifts actually occur.
            sorties = np.flatnonzero(mask[1:])
            action = (int(rng.choice(sorties)) + 1
                      if len(sorties) and rng.random() < 0.75 else 0)

            if action == 0 and env.current_dispatches:
                to_node = env.truck_route[
                    env._next_required_stop(env.truck_stop_idx)]
                for d in env.current_dispatches:
                    cust = env.customer_indices[d["cust_local_idx"]]
                    flown = (env.air_matrix[d["launch_node"], cust]
                             + env.air_matrix[cust, to_node])
                    assert d["distance_m"] == pytest.approx(flown, rel=1e-9), (
                        "sortie charged for {:.0f} m but flies {:.0f} m"
                        .format(d["distance_m"], flown))
                    assert d["recovery_node"] == to_node
                    checked += 1

            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    assert checked > 20, "too few recoveries exercised ({})".format(checked)


def test_launch_that_would_strand_an_airborne_drone_is_illegal(env):
    """
    Because launching moves the shared rendezvous further away, a launch can
    put a drone already in the air out of range. The mask must forbid those,
    not just check the newcomer.
    """
    rng = np.random.default_rng(5)
    for instance in range(len(env.instances)):
        env.reset(options={"instance": instance})
        for _ in range(env.max_steps):
            mask = env.action_masks()
            for slot in np.flatnonzero(mask[1:]):
                cust_local_idx = env._get_nearest_unserved(
                    env.truck_route[env.truck_stop_idx])[slot]
                recovery = env._rendezvous_node(extra_skip=(cust_local_idx,))
                # Every drone already flying must still reach the rendezvous
                # this legal launch would create.
                assert env._all_feasible_at(env.current_dispatches, recovery)

            legal = np.flatnonzero(mask)
            _, _, term, trunc, _ = env.step(int(rng.choice(legal)))
            if term or trunc:
                break


def test_a_drone_does_not_recharge_while_it_is_flying(env):
    """
    Recharging happens on the truck, so a drone gets credit only for the part
    of the leg it was actually aboard. A drone airborne for the whole leg must
    come back with strictly less charge than it left with -- if it were
    credited for the full leg regardless, a long sortie could end with more
    battery than it started.
    """
    rng = np.random.default_rng(7)
    checked = 0

    for instance in range(len(env.instances)):
        env.reset(options={"instance": instance})
        for _ in range(env.max_steps):
            mask = env.action_masks()
            sorties = np.flatnonzero(mask[1:])
            action = (int(rng.choice(sorties)) + 1
                      if len(sorties) and rng.random() < 0.7 else 0)

            if action == 0 and env.current_dispatches:
                truck_s = env.drone.truck_travel_time_s(float(env.road_matrix[
                    env.truck_route[env.truck_stop_idx],
                    env.truck_route[env._next_required_stop(env.truck_stop_idx)]]))
                flying = {d["drone_idx"]: (d["sortie_time"], d["pack_idx"])
                          for d in env.current_dispatches}
                before = {i: env.drone_packs[i][pk]
                          for i, (_, pk) in flying.items()}
                env.step(0)
                longest = max(t for t, _ in flying.values())
                for idx, (sortie_s, pack) in flying.items():
                    # That pack was airborne the whole leg, so it spent no
                    # time on the charger and must come back with less.
                    if sortie_s >= max(truck_s, longest) - 1e-9:
                        assert env.drone_packs[idx][pack] < before[idx], (
                            "pack {} of drone {} gained charge in the air"
                            .format(pack, idx))
                        checked += 1
                continue

            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    assert checked > 10, "too few full-leg sorties exercised ({})".format(checked)


def test_the_spare_pack_charges_while_the_drone_is_away(env):
    """
    The whole case for carrying a spare: it sits on the truck charging for the
    entire leg, including all the time the drone is off flying. So a spare that
    is not full must gain charge across a leg during which its drone was
    airborne -- if spares only charged while the drone was home, carrying one
    would buy nothing.
    """
    if env.batteries_per_drone < 2:
        pytest.skip("configured without a spare")

    rng = np.random.default_rng(11)
    checked = 0

    for instance in range(len(env.instances)):
        env.reset(options={"instance": instance})
        # Drain every pack so there is headroom to observe charging.
        env.drone_packs = [[55.0] * env.batteries_per_drone
                           for _ in range(env.n_drones)]

        for _ in range(env.max_steps):
            mask = env.action_masks()
            sorties = np.flatnonzero(mask[1:])
            action = (int(rng.choice(sorties)) + 1
                      if len(sorties) and rng.random() < 0.7 else 0)

            if action == 0 and env.current_dispatches:
                flying = {d["drone_idx"]: d["pack_idx"]
                          for d in env.current_dispatches}
                before = [list(packs) for packs in env.drone_packs]
                env.step(0)
                for idx, flown_pack in flying.items():
                    for p in range(env.batteries_per_drone):
                        if p == flown_pack or before[idx][p] >= 100.0:
                            continue
                        assert env.drone_packs[idx][p] > before[idx][p], (
                            "spare {} of drone {} did not charge while the "
                            "drone was away".format(p, idx))
                        checked += 1
                continue

            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    assert checked > 10, "too few spares observed ({})".format(checked)


def test_a_swap_costs_ground_time(env):
    """
    Swapping is fast but not free. A launch that changes pack must carry the
    configured swap time on top of its flight profile, otherwise the model
    would make spares strictly better than they are.
    """
    if env.batteries_per_drone < 2:
        pytest.skip("configured without a spare")

    env.reset(options={"instance": 0})
    # Make pack 1 clearly the best, so the next launch must swap onto it.
    env.drone_packs = [[40.0, 100.0] for _ in range(env.n_drones)]
    env.drone_installed = [0] * env.n_drones

    legal = np.flatnonzero(env.action_masks()[1:])
    assert len(legal), "expected at least one legal sortie"

    swaps_before = env.battery_swaps
    env.step(int(legal[0]) + 1)
    d = env.current_dispatches[-1]

    assert d["swap_time_s"] == env.battery_swap_time_s
    assert env.battery_swaps == swaps_before + 1
    assert env.drone_installed[d["drone_idx"]] == 1
    assert d["soc"] == 100.0

    bare = env.drone.sortie_time_s(
        *env._sortie_legs(d, d["recovery_node"]))
    assert d["sortie_time"] == pytest.approx(bare + env.battery_swap_time_s)


def test_a_sortie_carries_exactly_one_parcel(env):
    """
    A drone has one parcel bay: it serves one customer, then must rejoin the
    truck before it can carry anything else. Two things have to hold for that
    to be true, and both are easy to break.

    First, a drone already in the air cannot be given a second customer -- so
    at any instant no drone appears twice among the outstanding dispatches.
    Second, each dispatch names exactly one customer, so the number of drone
    deliveries booked on a leg equals the number of sorties recovered on it.
    """
    rng = np.random.default_rng(13)
    sorties_seen = 0

    for instance in range(len(env.instances)):
        env.reset(options={"instance": instance})
        delivered_by_drone = 0

        for _ in range(env.max_steps):
            # No drone is ever assigned two customers at once.
            airborne = [d["drone_idx"] for d in env.current_dispatches]
            assert len(airborne) == len(set(airborne))
            assert len(airborne) <= env.n_drones
            assert all(d["cust_local_idx"] is not None
                       for d in env.current_dispatches)

            mask = env.action_masks()
            sorties = np.flatnonzero(mask[1:])
            action = (int(rng.choice(sorties)) + 1
                      if len(sorties) and rng.random() < 0.7 else 0)

            if action == 0:
                # One recovered sortie == one parcel delivered.
                expected = len(env.current_dispatches)
                before = env.drone_deliveries
                _, _, term, trunc, _ = env.step(0)
                assert env.drone_deliveries - before == expected
                delivered_by_drone += expected
                sorties_seen += expected
            else:
                _, _, term, trunc, _ = env.step(action)

            if term or trunc:
                break

        assert env.drone_deliveries == delivered_by_drone

    assert sorties_seen > 50, "too few sorties exercised ({})".format(sorties_seen)


def test_a_drone_ends_every_sortie_at_the_truck(env):
    """
    After delivering, the drone must come back to the truck -- it cannot park
    itself at the customer or wander to the next one. The recovery node of
    every sortie is therefore the node the truck is standing on when the
    sortie is booked in, never the customer it just served.
    """
    rng = np.random.default_rng(17)
    checked = 0

    for instance in range(len(env.instances)):
        env.reset(options={"instance": instance})
        for _ in range(env.max_steps):
            mask = env.action_masks()
            sorties = np.flatnonzero(mask[1:])
            action = (int(rng.choice(sorties)) + 1
                      if len(sorties) and rng.random() < 0.7 else 0)

            if action == 0 and env.current_dispatches:
                pending = list(env.current_dispatches)
                env.step(0)
                truck_node = env.truck_route[env.truck_stop_idx]
                for d in pending:
                    assert d["recovery_node"] == truck_node, (
                        "drone recovered at {} but truck is at {}"
                        .format(d["recovery_node"], truck_node))
                    checked += 1
                continue

            _, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    assert checked > 50, "too few sorties exercised ({})".format(checked)


def test_battery_never_goes_negative(scenario, eval_ids):
    for name in ("random", "always_nearest", "greedy"):
        runs = evaluate_policy(POLICIES[name](), scenario, eval_ids)
        assert all(r["min_battery_pct"] >= 0 for r in runs), name
        assert all(r["battery_deaths"] == 0 for r in runs), name


def test_infeasible_sortie_is_refused_when_the_battery_is_flat(env):
    env.reset(options={"instance": 0})
    env.drone_packs = [[1.0] * env.batteries_per_drone
                       for _ in range(env.n_drones)]
    assert not env.action_masks()[1:].any()


def test_sortie_energy_grows_with_distance():
    drone = DronePhysics()
    short = drone.sortie_profile(1000.0, 1000.0, payload_kg=1.0, soc_pct=100.0)
    long = drone.sortie_profile(3000.0, 3000.0, payload_kg=1.0, soc_pct=100.0)
    assert long["energy_wh"] > short["energy_wh"]
    assert long["time_s"] > short["time_s"]


def test_payload_costs_energy():
    drone = DronePhysics()
    light = drone.sortie_profile(2000.0, 2000.0, payload_kg=0.0, soc_pct=100.0)
    heavy = drone.sortie_profile(2000.0, 2000.0, payload_kg=2.0, soc_pct=100.0)
    assert heavy["energy_wh"] > light["energy_wh"]


# ======================================================================
# Bookkeeping the results depend on
# ======================================================================

def test_every_customer_is_served_exactly_once(env):
    """
    A customer counted twice would inflate the delivery total; one dropped
    silently would deflate it. Truck and drone deliveries must sum to the
    number served, with no overlap.
    """
    for instance in range(5):
        env.reset(options={"instance": instance})
        for _ in range(env.max_steps):
            legal = np.flatnonzero(env.action_masks())
            _, _, term, trunc, info = env.step(int(legal[-1]))
            if term or trunc:
                break
        assert info["drone_deliveries"] + info["truck_deliveries"] == info["served"]
        assert info["served"] <= info["total_customers"]


def test_truck_only_policy_matches_the_ortools_baseline(scenario, eval_ids):
    """
    Driving every stop with no sortie must reproduce the precomputed OR-Tools
    tour distance. If it does not, the baseline the improvements are quoted
    against is not the baseline being solved.
    """
    runs = evaluate_policy(POLICIES["truck_only"](), scenario, eval_ids)
    for r in runs:
        assert r["route_complete"]
        assert r["truck_km"] == pytest.approx(r["baseline_km"], rel=1e-6)


def test_drone_sorties_remove_truck_kilometres(scenario, eval_ids):
    """The mechanism the whole project rests on."""
    runs = evaluate_policy(POLICIES["always_nearest"](), scenario, eval_ids)
    flown = [r for r in runs if r["drone_deliveries"] > 0]
    assert flown, "expected at least one sortie"
    assert all(r["truck_km"] < r["baseline_km"] + 1e-6 for r in flown)


def test_incomplete_routes_are_excluded_from_timing_stats():
    """
    The correctness of the headline number depends on this. A run that
    abandoned most of its customers finishes fast; if it leaked into the
    average it would manufacture an improvement. Here a fast-but-incomplete
    run is mixed with slow complete ones and must not move the mean.
    """
    complete = [{"route_complete": True, "completion_pct": 100.0,
                 "failed_sorties": 0, "battery_deaths": 0, "battery_swaps": 3, "customers": 15,
                 "hybrid_time_min": 100.0, "baseline_time_min": 100.0,
                 "hybrid_energy_wh": 100.0, "baseline_energy_wh": 100.0,
                 "truck_km": 50.0, "baseline_km": 50.0, "drone_km": 5.0,
                 "drone_deliveries": 2, "truck_wait_min": 0.0,
                 "min_battery_pct": 80.0} for _ in range(6)]
    abandoned = dict(complete[0], route_complete=False, completion_pct=13.0,
                     hybrid_time_min=10.0)

    assert summarise(complete, "x")["time_change_pct"] == pytest.approx(0.0)

    mixed = summarise(complete + [abandoned], "x")
    assert mixed["time_change_pct"] == pytest.approx(0.0), \
        "an abandoned route leaked into the delivery-time average"
    assert mixed["completion_rate_pct"] == pytest.approx(600 / 7)
    assert mixed["n_completed"] == 6


def test_too_few_completed_routes_reports_no_statistic():
    """A policy that cannot finish gets no delivery number, not a flattering one."""
    runs = [{"route_complete": False, "completion_pct": 20.0,
             "failed_sorties": 40, "battery_deaths": 0, "battery_swaps": 3, "customers": 15,
             "hybrid_time_min": 10.0, "baseline_time_min": 150.0,
             "hybrid_energy_wh": 1.0, "baseline_energy_wh": 100.0,
             "truck_km": 5.0, "baseline_km": 60.0, "drone_km": 1.0,
             "drone_deliveries": 1, "truck_wait_min": 0.0,
             "min_battery_pct": 90.0} for _ in range(20)]
    s = summarise(runs, "broken")
    assert s["time_change_pct"] is None
    assert s["stats_basis"] == "insufficient"


# ======================================================================
# Gym API conformance
# ======================================================================

def test_observation_stays_inside_the_declared_space(env):
    obs, _ = env.reset(options={"instance": 0})
    assert env.observation_space.contains(obs)
    for _ in range(env.max_steps):
        legal = np.flatnonzero(env.action_masks())
        obs, reward, term, trunc, _ = env.step(int(legal[-1]))
        assert env.observation_space.contains(obs)
        assert np.isfinite(reward)
        if term or trunc:
            break


def test_reset_with_the_same_instance_is_deterministic(env):
    a, _ = env.reset(options={"instance": 3})
    b, _ = env.reset(options={"instance": 3})
    assert np.allclose(a, b)


def test_episode_always_terminates(env):
    """No policy, however perverse, may run past the step budget."""
    rng = np.random.default_rng(1)
    for instance in range(len(env.instances)):
        env.reset(options={"instance": instance})
        for step in range(env.max_steps + 1):
            legal = np.flatnonzero(env.action_masks())
            _, _, term, trunc, _ = env.step(int(rng.choice(legal)))
            if term or trunc:
                break
        else:
            pytest.fail("instance {} never ended".format(instance))
