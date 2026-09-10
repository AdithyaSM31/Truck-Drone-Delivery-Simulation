"""
The experiment configuration, in one place.

Training, evaluation, rollout and the dashboard all import from here, so the
agent is never scored under a fleet or reward setting different from the one
it trained on. Getting that wrong is an easy way to publish a number that
cannot be reproduced.
"""

# ----------------------------------------------------------------------
# Fleet
# ----------------------------------------------------------------------
# Two drones is the headline configuration. It is not arbitrary: a one-drone
# truck can only ever remove one customer per leg, which caps the achievable
# saving near 5% no matter how good the dispatcher is, so a single-drone study
# mostly measures the ceiling rather than the policy. With two drones the
# truck can skip two stops per leg and the interesting trade-off appears --
# whether to spend a drone on the customer that is closest, or hold it for the
# one whose road detour is expensive. The DRONE_COUNT_ABLATION runs 1, 2 and 3
# so the effect of the fleet size is reported rather than assumed.
N_DRONES = 2
DRONE_COUNT_ABLATION = (1, 2, 3)

# Customers visible to the dispatcher at any moment (the K nearest unserved).
# The action space is Discrete(K + 1).
MAX_VISIBLE_CUSTOMERS = 5

# Battery packs carried per drone: one fitted plus one spare. A spare charges
# on the truck the entire time the drone is away, so turnaround is set by how
# long a swap takes rather than how long a recharge takes -- which is how real
# drone delivery operations run. BATTERY_COUNT_ABLATION measures what the
# spare is actually worth rather than assuming it.
BATTERIES_PER_DRONE = 2
BATTERY_COUNT_ABLATION = (1, 2)

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------
ENV_KWARGS = dict(
    n_drones=N_DRONES,
    max_visible_customers=MAX_VISIBLE_CUSTOMERS,
    payload_kg=1.0,
    recharge_pct_per_min=1.0,
    batteries_per_drone=BATTERIES_PER_DRONE,
    battery_swap_time_s=60.0,
    time_limit_s=28800,          # 8-hour driver shift
)

# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------
TIMESTEPS = 300_000
SEED = 0
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 20


def env_kwargs(n_drones=None, **overrides):
    """ENV_KWARGS with optional overrides, for ablations."""
    kwargs = dict(ENV_KWARGS)
    if n_drones is not None:
        kwargs["n_drones"] = n_drones
    kwargs.update(overrides)
    return kwargs
