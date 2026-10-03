"""
The experiment configuration, in one place.

Training, evaluation, rollout, the ablations and the dashboard all import from
here, so an agent is never scored under a world it did not train in. Getting
that wrong is an easy way to publish a number nobody can reproduce.
"""

# ----------------------------------------------------------------------
# Cities
# ----------------------------------------------------------------------
# Agents train in Whitefield. Chennai is a structurally different street grid
# (83% of site pairs one-way-asymmetric against Whitefield's 46%) used only to
# ask whether a policy transfers without retraining.
CITY = "whitefield"
TRANSFER_CITY = "chennai"

# ----------------------------------------------------------------------
# Fleet and dispatcher
# ----------------------------------------------------------------------
N_DRONES = 2
DRONE_COUNT_ABLATION = (1, 2, 3)
BATTERIES_PER_DRONE = 2               # one fitted + one hot-swappable spare
BATTERY_COUNT_ABLATION = (1, 2)
MAX_VISIBLE_CUSTOMERS = 5             # K nearest unserved customers
RENDEZVOUS_OPTIONS = 2                # meet at the next stop, or the one after

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
ENV_KWARGS = dict(
    n_drones=N_DRONES,
    max_visible_customers=MAX_VISIBLE_CUSTOMERS,
    rendezvous_options=RENDEZVOUS_OPTIONS,
    payload_kg=1.0,
    recharge_pct_per_min=1.0,
    batteries_per_drone=BATTERIES_PER_DRONE,
    battery_swap_time_s=60.0,
    truck_service_time_s=120.0,       # parking + walk to door + handover
    replan_tour=True,
    stochastic=True,                  # traffic + wind
    wind_max_ms=8.0,
    preference_conditioned=True,      # one policy for every time/energy/cost balance
    time_limit_s=28800,               # 8-hour driver shift
)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
TIMESTEPS = {"maskable_ppo": 2_000_000, "ppo": 1_000_000, "dqn": 1_000_000}
SEEDS = {"maskable_ppo": (0, 1, 2, 3, 4), "ppo": (0, 1, 2), "dqn": (0, 1, 2)}
N_ENVS = 4
EVAL_FREQ = 25_000                    # environment steps between evaluations
N_EVAL_EPISODES = 20

# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------
# Every policy meets the same worlds: 20 held-out instances x these world
# seeds, so every comparison is paired down to the traffic jam and the wind.
EVAL_WORLD_SEEDS = (101, 102, 103, 104, 105)
HEADLINE_PREFERENCE = "fastest"

# A fixed, deterministic set of worlds for choosing the best checkpoint during
# training -- disjoint from EVAL_WORLD_SEEDS, so checkpoint selection never
# peeks at the worlds the final numbers come from.
SELECTION_WORLD_SEED_BASE = 9000


def env_kwargs(**overrides):
    """ENV_KWARGS with overrides, for ablations."""
    kwargs = dict(ENV_KWARGS)
    kwargs.update(overrides)
    return kwargs
