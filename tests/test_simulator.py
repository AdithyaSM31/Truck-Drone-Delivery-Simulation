"""
Tests for the properties the results depend on.

Not coverage tests. Each one pins a claim the write-up makes, so if someone
changes the simulator and a claim stops being true, the suite says so instead
of a number quietly moving.

    pytest tests/ -v
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from truckdrone import config                                     # noqa: E402
from truckdrone.baselines import POLICIES, solve_tsp              # noqa: E402
from truckdrone.economics import DEFAULT_COSTS                    # noqa: E402
from truckdrone.env import (PREFERENCE_PRESETS, TruckDroneEnv,    # noqa: E402
                            _improve_open_path, _path_cost)
from truckdrone.evaluate import evaluate_policy, summarise        # noqa: E402
from truckdrone.physics import DronePhysics                       # noqa: E402
from truckdrone.scenario import (StreetPaths, load_scenario,      # noqa: E402
                                 split_instances, to_utm)

KW = config.env_kwargs()


@pytest.fixture(scope="module")
def scenario():
    return load_scenario("whitefield")


@pytest.fixture(scope="module")
def eval_ids(scenario):
    return split_instances(scenario)[1]


@pytest.fixture(scope="module")
def env(scenario, eval_ids):
    return TruckDroneEnv(scenario, instance_ids=eval_ids, **KW)


def _rollout(env, choose, instance, world_seed=11, preference=None, hook=None):
    """Run one episode, calling ``hook(env, action)`` before every step."""
    obs, _ = env.reset(options={"instance": instance, "world_seed": world_seed,
                                "preference": preference})
    info = None
    for _ in range(env.max_steps + 1):
        a = choose(env)
        if hook:
            hook(env, a)
        obs, _, te, tr, info = env.step(a)
        if te or tr:
            return info
    pytest.fail("episode did not terminate")


def _random_legal(seed, p_launch=0.7):
    rng = np.random.default_rng(seed)

    def choose(env):
        launches = np.flatnonzero(env.action_masks()[1:])
        if len(launches) and rng.random() < p_launch:
            return int(rng.choice(launches)) + 1
        return 0
    return choose


# ======================================================================
# Real street network
# ======================================================================

@pytest.mark.parametrize("city", ["whitefield", "chennai"])
def test_street_paths_reproduce_the_road_matrix(city):
    """
    The dashboard and the Folium maps draw the truck along stored street
    polylines; the simulator charges it the road matrix. If the two disagreed,
    the pictures would show a different route from the one being scored.
    """
    sc = load_scenario(city)
    paths = StreetPaths(city)
    rng = np.random.default_rng(0)
    for _ in range(60):
        a, b = rng.choice(sc["n_nodes"], size=2, replace=False)
        xy = paths.xy(a, b)
        drawn = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        assert drawn == pytest.approx(sc["road_matrix"][a, b], rel=0.02)


@pytest.mark.parametrize("city", ["whitefield", "chennai"])
def test_driving_is_never_shorter_than_flying(city):
    """The drone's advantage rests on this. Small slack for UTM scale error."""
    sc = load_scenario(city)
    road, air = np.asarray(sc["road_matrix"]), np.asarray(sc["air_matrix"])
    off = ~np.eye(len(road), dtype=bool)
    assert np.all(road[off] >= air[off] * 0.999)


def test_one_way_streets_make_distances_directed(scenario):
    road = np.asarray(scenario["road_matrix"])
    assert not np.allclose(road, road.T)
    assert np.allclose(np.diag(road), 0.0)


def test_train_and_eval_instances_are_disjoint(scenario):
    train_ids, eval_ids = split_instances(scenario)
    assert set(train_ids).isdisjoint(eval_ids) and eval_ids


def test_stored_truck_tours_are_reproducible(scenario):
    """OR-Tools on the stored directed matrix reproduces the cached tour length."""
    inst = scenario["instances"][25]
    nodes = [inst["depot"]] + inst["customers"]
    sub = np.asarray(scenario["road_matrix"])[np.ix_(nodes, nodes)]
    _, dist = solve_tsp(sub, depot=0, time_limit_s=2)
    assert dist == pytest.approx(inst["truck_route_distance_m"], rel=0.01)


# ======================================================================
# Action masking
# ======================================================================

def test_driving_on_is_always_legal(env):
    for inst in range(4):
        env.reset(options={"instance": inst, "world_seed": 1})
        for _ in range(env.max_steps):
            assert env.action_masks()[0]
            _, _, te, tr, _ = env.step(0)
            if te or tr:
                break


# Everything reset() initialises is episode state; a faithful snapshot copies
# all of it, so a trial action can be undone without counters drifting.
_STATE = ("truck_route", "current_step", "truck_stop_idx", "drone_packs",
          "drone_installed", "battery_swaps", "min_landing_pct", "served",
          "total_time_s", "drone_energy_wh", "truck_energy_wh", "truck_idle_wh",
          "truck_distance_m", "drone_distance_m", "drone_deliveries",
          "truck_deliveries", "truck_wait_time_s", "drone_wait_time_s",
          "failed_sorties", "battery_deaths", "replans", "late_rendezvous",
          "current_dispatches", "route_complete", "trace")


def _snapshot(env):
    import copy
    return {k: copy.deepcopy(getattr(env, k)) for k in _STATE}


def _restore(env, state):
    import copy
    for k, v in state.items():
        setattr(env, k, copy.deepcopy(v))


def test_mask_is_exactly_what_dispatch_accepts(env):
    """
    The mask is a promise: every action it allows succeeds, every action it
    forbids is refused. Checked by trying each action from a restored copy of
    the same state.
    """
    choose = _random_legal(0)
    for inst in range(3):
        env.reset(options={"instance": inst, "world_seed": 5})
        for _ in range(14):
            mask = env.action_masks().copy()
            state = _snapshot(env)
            for a in range(1, env.action_space.n):
                _restore(env, state)
                before = env.failed_sorties
                env.step(a)
                assert (env.failed_sorties > before) != bool(mask[a]), (
                    "mask[{}]={} disagrees with dispatch".format(a, mask[a]))
            _restore(env, state)
            _, _, te, tr, _ = env.step(choose(env))
            if te or tr:
                break


def test_masked_policies_never_attempt_an_infeasible_sortie(scenario, eval_ids):
    for name in ("random", "always_nearest", "greedy"):
        runs = evaluate_policy(POLICIES[name](), scenario, eval_ids[:8],
                               world_seeds=(3,))
        assert all(r["failed_sorties"] == 0 for r in runs), name


def test_no_launch_without_a_free_drone(env):
    def hook(e, a):
        if len(e.current_dispatches) >= e.n_drones:
            assert not e.action_masks()[1:].any()
    _rollout(env, _random_legal(1, p_launch=0.9), 0, hook=hook)


# ======================================================================
# Sorties, rendezvous and the parcel bay
# ======================================================================

def test_a_sortie_carries_one_parcel_and_ends_at_the_truck(env):
    """
    One parcel per sortie; a drone already airborne is never given a second
    customer; and every sortie ends on the node the truck is standing on.
    """
    recovered = 0

    def hook(e, a):
        nonlocal recovered
        airborne = [d["drone_idx"] for d in e.current_dispatches]
        assert len(airborne) == len(set(airborne)) <= e.n_drones
        if a == 0:
            nxt = e._next_required_stop(e.truck_stop_idx)
            for d in e.current_dispatches:
                if d["recovery_idx"] == nxt:
                    assert d["recovery_node"] == e.truck_route[nxt]
                    recovered += 1

    for inst in range(8):
        info = _rollout(env, _random_legal(inst), inst, hook=hook)
        assert info["drone_deliveries"] + info["truck_deliveries"] == info["served"]
    assert recovered > 20


def test_a_launch_never_moves_an_airborne_drones_meeting_point(env):
    """
    Meeting points are fixed when a drone launches. Launching a second drone
    can let the truck skip another stop, but never one where a drone is due.
    """
    checked = 0

    def hook(e, a):
        nonlocal checked
        if a == 0 or not e.current_dispatches:
            return
        due = {d["recovery_idx"]: d["recovery_node"] for d in e.current_dispatches}
        state = _snapshot(e)
        e.step(a)
        for d in e.current_dispatches:
            if d["recovery_idx"] in due:
                assert e.truck_route[d["recovery_idx"]] == due[d["recovery_idx"]]
                assert d["recovery_idx"] in set(
                    e._required_after(e.truck_stop_idx, len(e.truck_route)))
                checked += 1
        _restore(e, state)

    for inst in range(10):
        _rollout(env, _random_legal(100 + inst, p_launch=0.9), inst, hook=hook)
    assert checked > 5


def test_the_later_rendezvous_option_is_actually_used(env):
    seen = {0: 0, 1: 0}

    def hook(e, a):
        if a > 0:
            seen[e.decode_action(a)[1]] += 1
    for inst in range(10):
        _rollout(env, _random_legal(inst), inst, hook=hook)
    assert seen[0] > 0 and seen[1] > 0


def test_charged_distance_equals_flown_distance(env):
    """A sortie is charged for the straight-line legs to its actual meeting point."""
    checked = 0

    def hook(e, a):
        nonlocal checked
        for d in e.current_dispatches:
            c = e.customer_indices[d["cust_local_idx"]]
            flown = e.air_matrix[d["launch_node"], c] + e.air_matrix[c, d["recovery_node"]]
            assert d["distance_m"] == pytest.approx(flown)
            checked += 1
    for inst in range(4):
        _rollout(env, _random_legal(inst), inst, hook=hook)
    assert checked > 10


# ======================================================================
# Batteries
# ======================================================================

def test_a_flying_pack_does_not_charge_but_the_spare_does(env):
    """
    A drone off the truck for a leg cannot charge its fitted pack; the spare
    on the truck charges the whole leg. Packs are drained first so charging is
    observable.
    """
    checked_fly = checked_spare = 0

    def hook(e, a):
        nonlocal checked_fly, checked_spare
        if a != 0 or not e.current_dispatches:
            return
        flying = {d["drone_idx"]: d for d in e.current_dispatches}
        before = [list(p) for p in e.drone_packs]
        state = _snapshot(e)
        e.step(0)
        for i, d in flying.items():
            p = d["pack_idx"]
            landed = all(x["drone_idx"] != i for x in e.current_dispatches)
            expected = before[i][p] - (d["battery_pct_cost"] if landed else 0.0)
            assert e.drone_packs[i][p] == pytest.approx(expected)
            checked_fly += 1
            for q in range(len(before[i])):
                if q != p and before[i][q] < 100.0:
                    assert e.drone_packs[i][q] > before[i][q]
                    checked_spare += 1
        _restore(e, state)

    for inst in range(8):
        env.reset(options={"instance": inst, "world_seed": 2})
        env.drone_packs = [[70.0, 55.0] for _ in range(env.n_drones)]
        choose = _random_legal(inst, 0.8)
        for _ in range(env.max_steps):
            a = choose(env)
            hook(env, a)
            _, _, te, tr, _ = env.step(a)
            if te or tr:
                break
    assert checked_fly > 10 and checked_spare > 5


def test_a_pack_swap_costs_ground_time(env):
    env.reset(options={"instance": 0, "world_seed": 0})
    env.drone_packs = [[40.0, 100.0] for _ in range(env.n_drones)]
    env.drone_installed = [0] * env.n_drones
    legal = np.flatnonzero(env.action_masks()[1:])
    assert len(legal)
    swaps = env.battery_swaps
    env.step(int(legal[0]) + 1)
    d = env.current_dispatches[-1]
    assert d["swap_time_s"] == env.battery_swap_time_s
    assert env.battery_swaps == swaps + 1 and d["soc"] == 100.0


def test_batteries_never_die(scenario, eval_ids):
    for name in ("random", "always_nearest", "greedy"):
        runs = evaluate_policy(POLICIES[name](), scenario, eval_ids[:8], world_seeds=(9,))
        assert all(r["battery_deaths"] == 0 and r["min_battery_pct"] >= 0 for r in runs)


# ======================================================================
# Wind
# ======================================================================

def test_headwind_lengthens_and_tailwind_shortens_a_flight(scenario, eval_ids):
    calm = TruckDroneEnv(scenario, instance_ids=eval_ids, **config.env_kwargs(wind_fixed=(0.0, 0.0)))
    calm.reset(options={"instance": 0, "world_seed": 0})
    a, b = 3, 17
    u = calm.coords[b] - calm.coords[a]
    heading = float(np.arctan2(u[1], u[0]))
    tail = TruckDroneEnv(scenario, instance_ids=eval_ids, **config.env_kwargs(wind_fixed=(6.0, heading)))
    head = TruckDroneEnv(scenario, instance_ids=eval_ids, **config.env_kwargs(wind_fixed=(6.0, heading + np.pi)))
    for e in (tail, head):
        e.reset(options={"instance": 0, "world_seed": 0})
    d = calm._wind_equiv_m(a, b)
    assert d == pytest.approx(calm.air_matrix[a, b])
    assert tail._wind_equiv_m(a, b) < d < head._wind_equiv_m(a, b)


def test_strong_headwind_shrinks_what_the_mask_allows(scenario, eval_ids):
    """Feasibility is checked against the wind the drone will actually meet."""
    def legal_count(wind):
        e = TruckDroneEnv(scenario, instance_ids=eval_ids, **config.env_kwargs(wind_fixed=wind))
        total = 0
        for inst in range(10):
            e.reset(options={"instance": inst, "world_seed": 0})
            total += int(e.action_masks()[1:].sum())
        return total
    assert legal_count((12.0, 0.0)) < legal_count((0.0, 0.0))


# ======================================================================
# Traffic, service time and the baseline
# ======================================================================

def test_truck_only_policy_reproduces_the_simulated_baseline(env):
    """
    The baseline is the OR-Tools tour driven through the same world. Driving
    that tour with no drone must reproduce it exactly -- time, distance,
    energy, cost -- or every improvement is quoted against the wrong number.
    """
    for inst in range(6):
        for ws in (1, 2):
            info = _rollout(env, lambda e: 0, inst, world_seed=ws)
            b = info["truck_only"]
            assert info["route_complete"]
            assert info["total_time_s"] == pytest.approx(b["time_s"])
            assert info["truck_distance_m"] == pytest.approx(b["distance_m"])
            assert info["total_energy_wh"] == pytest.approx(b["energy_wh"])
            assert info["cost_inr"] == pytest.approx(b["cost_inr"])


def test_baseline_includes_doorstep_service_time(scenario, eval_ids):
    calm = TruckDroneEnv(scenario, instance_ids=eval_ids,
                         **config.env_kwargs(stochastic=False))
    calm.reset(options={"instance": 0, "world_seed": 0})
    route = calm.base_route
    drive = sum(calm.road_matrix[a, b] for a, b in zip(route[:-1], route[1:])) / calm.static_truck_ms
    assert calm.truck_only_metrics()["time_s"] == pytest.approx(
        drive + calm.n_customers * calm.service_s)


def test_same_world_same_episode(env):
    def run(ws):
        infos = _rollout(env, _random_legal(3), 4, world_seed=ws)
        return (infos["total_time_s"], infos["cost_inr"], infos["drone_deliveries"])
    assert run(42) == run(42)
    assert run(42) != run(43)


def test_rush_hour_slows_the_truck(env):
    env.reset(options={"instance": 0, "world_seed": 0})
    env.start_hour, env.day_factor = 6.0, 1.0
    early = env._expected_drive_s(3, 17, 0.0)
    env.start_hour = 9.5
    peak = env._expected_drive_s(3, 17, 0.0)
    assert peak > 1.4 * early


# ======================================================================
# Tour re-planning
# ======================================================================

def test_local_search_never_worsens_and_keeps_endpoints(scenario):
    D = np.asarray(scenario["road_matrix"])
    rng = np.random.default_rng(0)
    for _ in range(150):
        nodes = [int(x) for x in rng.choice(50, size=int(rng.integers(4, 15)), replace=False)]
        out = _improve_open_path(nodes, D)
        assert out[0] == nodes[0] and out[-1] == nodes[-1]
        assert sorted(out) == sorted(nodes)
        assert _path_cost(out, D) <= _path_cost(nodes, D) + 1e-6


def test_replanning_never_moves_a_promised_meeting_point(env):
    """Re-planning reorders only the tour after the last pending rendezvous."""
    def hook(e, a):
        if a != 0:
            return
        due = {d["recovery_idx"]: d["recovery_node"] for d in e.current_dispatches}
        state = _snapshot(e)
        e.step(0)
        for idx, node in due.items():
            if idx > e.truck_stop_idx:
                assert e.truck_route[idx] == node
        _restore(e, state)
    for inst in range(8):
        _rollout(env, _random_legal(50 + inst, 0.8), inst, hook=hook)


# ======================================================================
# Objectives
# ======================================================================

def test_time_preference_reproduces_minus_two_per_minute(scenario, eval_ids):
    """With w = (1,0,0) the objective term is exactly -2 per elapsed minute."""
    e = TruckDroneEnv(scenario, instance_ids=eval_ids,
                      **config.env_kwargs(use_reward_shaping=False))
    e.reset(options={"instance": 0, "world_seed": 3, "preference": (1, 0, 0)})
    for _ in range(3):
        t0, served0 = e.total_time_s, int(e.served.sum())
        _, r, te, tr, _ = e.step(0)
        delivered = int(e.served.sum()) - served0
        objective = r - e.P_step - e.R_delivery * delivered
        assert objective == pytest.approx(-2.0 * (e.total_time_s - t0) / 60.0)


def test_reported_cost_matches_the_cost_model(env):
    info = _rollout(env, _random_legal(7), 2)
    assert info["cost_inr"] == pytest.approx(DEFAULT_COSTS.cost_inr(
        info["total_time_s"], info["truck_energy_wh"], info["drone_energy_wh"],
        info["drone_deliveries"]))


def test_preference_is_observed_and_normalised(env):
    obs, _ = env.reset(options={"instance": 0, "world_seed": 0, "preference": (2, 1, 1)})
    assert env.preference.sum() == pytest.approx(1.0)
    g = 2 + env.per_drone * env.n_drones + 6
    assert np.allclose(obs[g:g + 3], [0.5, 0.25, 0.25])
    for name, w in PREFERENCE_PRESETS.items():
        assert sum(w) == pytest.approx(1.0), name


# ======================================================================
# Evaluation bookkeeping
# ======================================================================

def _fake_run(**kw):
    r = {"instance": 0, "world_seed": 0, "route_complete": True,
         "completion_pct": 100.0, "customers": 15, "failed_sorties": 0,
         "battery_deaths": 0, "battery_swaps": 3, "drone_deliveries": 4,
         "truck_wait_min": 0.0, "min_battery_pct": 50.0,
         "time_min": 100.0, "base_time_min": 100.0, "energy_kwh": 50.0,
         "base_energy_kwh": 50.0, "cost_inr": 500.0, "base_cost_inr": 500.0,
         "co2_kg": 10.0, "base_co2_kg": 10.0, "truck_km": 50.0, "base_truck_km": 50.0}
    r.update(kw)
    return r


def test_abandoned_routes_never_enter_timing_statistics():
    complete = [_fake_run(instance=i) for i in range(6)]
    abandoned = _fake_run(instance=9, route_complete=False, completion_pct=13.0,
                          time_min=10.0)
    s = summarise(complete + [abandoned], "x")
    assert s["time_change_pct"] == pytest.approx(0.0)
    assert s["n_completed"] == 6
    assert s["completion_rate_pct"] == pytest.approx(600 / 7)


def test_too_few_completed_routes_gets_no_statistic():
    runs = [_fake_run(instance=i, route_complete=False, time_min=10.0) for i in range(20)]
    s = summarise(runs, "broken")
    assert s["time_change_pct"] is None and s["stats_basis"] == "insufficient"


# ======================================================================
# Gym conformance and transfer
# ======================================================================

def test_observation_stays_in_its_space(env):
    def hook(e, a):
        assert e.observation_space.contains(e._get_obs())
    _rollout(env, _random_legal(2), 0, hook=hook)


def test_every_episode_terminates(env):
    for inst in range(len(env.instances)):
        _rollout(env, _random_legal(inst, 0.95), inst)


def test_an_agent_built_for_whitefield_runs_unchanged_in_chennai(scenario):
    """Transfer needs identical observation and action spaces across cities."""
    a = TruckDroneEnv(scenario, instance_ids=[20], **KW)
    b = TruckDroneEnv(load_scenario("chennai"), instance_ids=[20], **KW)
    assert a.observation_space.shape == b.observation_space.shape
    assert a.action_space.n == b.action_space.n
    _rollout(b, _random_legal(0), 0)


# ----------------------------------------------------------------------
# Analysis, ablation and presentation plumbing
# ----------------------------------------------------------------------

def test_pinned_wind_speed_keeps_a_random_direction(scenario, eval_ids):
    env = TruckDroneEnv(scenario, instance_ids=eval_ids,
                        **config.env_kwargs(wind_fixed=[12.0, None]))
    headings = set()
    for ws in (1, 2, 3):
        env.reset(options={"instance": 0, "world_seed": ws})
        assert env.wind_speed == pytest.approx(12.0)
        headings.add(round(float(np.arctan2(env.wind[1], env.wind[0])), 6))
    assert len(headings) == 3


def test_training_choice_ablations_are_scored_in_the_standard_world():
    from truckdrone.ablation import eval_env_kwargs
    assert eval_env_kwargs("abl_calm_training")["stochastic"] is True
    assert eval_env_kwargs("abl_specialist")["preference_conditioned"] is False
    assert eval_env_kwargs("abl_rendezvous1")["rendezvous_options"] == 1
    assert eval_env_kwargs("abl_reference") == config.env_kwargs()


def test_simplex_grid_covers_every_corner():
    from truckdrone.analysis import simplex_grid
    grid = simplex_grid(0.25)
    assert len(grid) == 15
    assert all(abs(sum(w) - 1) < 1e-9 and min(w) >= 0 for w in grid)
    for corner in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
        assert corner in grid


def test_animation_legs_chain_and_follow_real_streets(scenario):
    from truckdrone.rollout import build_trace
    t = build_trace(scenario, 0, 101, POLICIES["greedy"](), "greedy", "fastest")
    for part in (t["hybrid"], t["baseline"]):
        legs = part["legs"]
        assert legs[0]["from"] == t["depot"] and legs[-1]["to"] == t["depot"]
        for a, b in zip(legs, legs[1:]):
            assert a["to"] == b["from"]
            assert b["t_start"] >= a["t_start"]
        km = sum(leg["km"] for leg in legs)
        assert km == pytest.approx(part["stats"]["truck_km"], abs=0.05)


def test_real_street_map_is_written(scenario, tmp_path):
    pytest.importorskip("folium")
    from truckdrone.realmap import build_map
    from truckdrone.rollout import build_trace
    t = build_trace(scenario, 0, 101, POLICIES["greedy"](), "greedy", "fastest")
    out = build_map(scenario, t, str(tmp_path / "m.html"))
    html = open(out, encoding="utf-8").read()
    assert "leaflet" in html.lower() and "arcgisonline" in html
