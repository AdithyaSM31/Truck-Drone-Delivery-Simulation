# Truck–Drone Collaborative Delivery — Deep Reinforcement Learning

A working simulator for last-mile delivery in which a truck acts as a **mobile
launch platform** for drones. The truck drives a road network; at each stop a
learned dispatcher decides whether to press on or to launch a drone to a
nearby customer. Any customer a drone serves is struck off the truck's route,
so sorties remove heavy-vehicle kilometres — that is the mechanism the whole
idea rests on.

The road network is real: 50 OpenStreetMap intersections across a 10 km × 10 km
slice of Whitefield, Bengaluru, projected to UTM 43N, with 87 weighted edges.
Truck movement follows road shortest paths; drone flight does not.

```bash
pip install -r requirements.txt
python run_demo.py
```

That trains the agents if they are missing, evaluates everything on held-out
instances, generates the figures, and opens the dashboard at
`http://127.0.0.1:5000`. Add `--ablation` to also run the fleet-size and
spare-battery studies (slower, and the source of figures 4 and 5), or
`--quick` for a 30k-step smoke test that finishes in a couple of minutes.

---

## What the agent actually decides

The agent is a **dispatcher**, not a pilot. At every truck stop it takes one
discrete action:

| action | meaning |
|---|---|
| `0` | advance the truck to its next required stop, recovering any drone in flight |
| `1 … K` | launch a drone to the *i*-th nearest unserved customer |

**Observation** (dimension `7 + 4K`, K = 5): truck position in a normalised
map frame; per drone its **fullest battery pack**, an in-flight flag and its
**mean charge in reserve**; fraction of customers still unserved; normalised
elapsed time; then four features for each of the K nearest unserved customers
— air distance, bearing, **the road detour the truck saves by skipping it**,
and **how the sortie duration compares with the truck's leg time**.

The truck's position is embedded by **multidimensional scaling of the road
distance matrix**, not raw map coordinates. Two intersections can be close as
the crow flies but far apart by road; MDS lays them out so that Euclidean
distance in the observation approximates driving distance.

**Reward.** The agent is charged for every minute the fleet spends and paid for
every package delivered, with a flat completion bonus and a real penalty for
abandoning customers. Charging elapsed time is what makes the reward agree with
the metric the results table reports, and it prices sorties with no special-case
term: a sortie that lets the truck skip a stop pays for itself in shorter legs,
while one that leaves the truck idling at the recovery point charges for every
minute it waits. A potential-based shaping term (Ng, Harada & Russell, 1999)
accelerates learning without changing the optimal policy.

## What one sortie is

A drone has a single parcel bay, so **one sortie carries exactly one package**:

```
truck stop  ──►  customer      ──►  rendezvous
   launch        drop the parcel     rejoin the truck at its NEXT stop
```

It cannot chain deliveries. A drone already in the air is not offered as a
choice to the dispatcher until it has been recovered, so no drone is ever
assigned two customers at once. Two tests pin this: one asserts that the
number of drone deliveries booked on a leg equals the number of sorties
recovered on it, the other that every sortie ends on the node the truck is
standing on.

The one subtlety is where "back to the truck" is. The drone does **not** return
to the point it launched from — that truck has driven on. It flies forward to
the truck's next required stop and meets it there, which is what makes a sortie
pay: the truck keeps moving the whole time.

That has a consequence worth knowing when you watch the animation. The
rendezvous is usually *another customer's location* — in a typical run, 7 of 8
of them are — because the truck's next stop is, by definition, a customer it
still has to serve. So the drone's path passes over one customer node, drops
nothing there, and lands. Early versions of the visualiser drew both legs in
one colour and it read like a single drone making two deliveries. Now the
drone carries a visible parcel that disappears at the doorstep, the outbound
leg is drone-orange and the return leg truck-blue, and a dashed ring marks the
rendezvous while a drone is inbound.

## Batteries: swap, don't wait

Each drone carries **two interchangeable packs** — one fitted, one spare. It
always takes up its fullest pack; if that is not the one fitted, the launch
costs 60 s of ground time while the crew changes it over.

The rule that matters is *when packs charge*. A pack charges only while it is
on the truck, so:

| pack | time on the charger during a leg |
|---|---|
| the one that flew | only what is left of the leg after it lands |
| the spare | the **whole leg**, including all the time the drone was away |

That asymmetry is the entire case for a spare, and it turns turnaround from a
recharge problem into a swap problem — which is why real drone-delivery
operations swap packs rather than charge them in place. What it is worth is
measured below rather than assumed.

## Five things this project fixes

This began from an earlier research codebase. Rebuilding it surfaced five
problems worth stating plainly, because each is easy to make and none is
visible in a summary statistic. The first four were found by refusing to
accept a number until a hand-written heuristic had been beaten with it; the
fifth was found by watching the animation and noticing a drone that never
made it back to the truck.

**1. The headline result was an artefact.** The old evaluation reported a 15%
delivery-time improvement. Two of its twenty runs finished in ~20 minutes
against a ~160-minute baseline — because the agent had abandoned 13 of 15
packages and gone home. Those runs dominated the average. `evaluate.py` now
reports completion rate separately and computes time, energy and distance
**over completed routes only**; a policy that cannot finish gets no delivery
statistic at all.

**2. Agents were burning their interaction budget on illegal actions.** The
old runs averaged 9.4 infeasible dispatch attempts per episode, and DQN could
get stuck looping on them until the step budget expired. The environment now
exposes `action_masks()`, and `MaskablePPO` never proposes an infeasible
sortie. Failed sorties go to exactly zero and route completion to 100%.

**3. The reward did not encode the objective.** With masking and honest
evaluation in place, the trained agent still scored −6.1% against a baseline
that a two-line heuristic beat at −12.9%. The cause was in the reward: it
charged `-0.1` per *decision*, but a decision can be two minutes of driving or
twenty, so nothing in it was proportional to delivery time. Worse, the
completion bonus was scaled by the fraction of customers the drone served, so
the agent could raise its return by flying even where flying made the route
slower — and it did. The reward now charges real elapsed minutes and pays a
flat completion bonus, which is both simpler and actually the thing being
measured. That took the agent from −6.1% to −11.6%.

**4. The observation did not contain the answer.** Even with a correct reward
the agent still trailed the heuristic. The reason was in the observation: it
described each candidate customer by *air* distance and bearing, but whether a
sortie is worth flying is a **road** question — how much driving the truck
avoids by skipping that stop. Two customers equally close as the crow flies
can differ several-fold in road detour, so no policy over that observation
could tell them apart. Adding the road detour and the sortie-versus-leg time
margin took the agent past every heuristic.

Measured as each change landed (all three under the pre-fix simulator, so
they are comparable with each other but not with the final table below, which
is a corrected and slightly harder world):

```
reward charges decisions,  air-distance observation   →   -6.1%
reward charges minutes,    air-distance observation   →  -11.6%
reward charges minutes,    road-detour observation    →  -14.0%
```

The general lesson is worth more than the numbers. An agent optimising a proxy
will find the gap between the proxy and the objective; an agent whose
observation omits the deciding quantity cannot close that gap however long it
trains. A competent heuristic is the cheapest instrument for detecting both,
because it fails in neither way — which is why `always_nearest` and `greedy`
are in the results table rather than only `random`.

**5. The simulator let drones cheat, in two ways.** Every airborne drone is
picked up wherever the truck next stops — and that pickup point *moves* each
time another drone launches, because the truck can then skip one more
customer. The first drone's energy, flight time and battery were locked in at
launch against the old, nearer pickup point, then it was flown to the new,
further one. Over the held-out set that under-charged **9% of sorties**, whose
return leg averaged 2 300 m as costed against 4 166 m as flown, worst case
4 877 m. It also let the feasibility check pass sorties that could not be
flown, and in the animation it looked exactly like what it was: a drone that
does not make it back to the truck. The rendezvous is now resolved for every
in-flight drone together whenever it moves, and a launch that would strand a
drone already airborne is refused.

Separately, **drones were recharging while in the air** — the recharge
credited every drone with the whole leg, including one that had just spent
that entire leg flying, which is where its battery went. Charging now accrues
only for the time a drone is actually aboard.

Both fixes make the world harder and honest, and both cost real percentage
points: with a single battery pack per drone, MaskablePPO moved from −14.4%
to −11.1%. Every policy moved down together and the ordering did not change,
which is the reassuring outcome — the earlier ranking was right, the earlier
magnitudes were flattered by a simulator that gave the drones free range and
free charge.

(The headline result is −20.5%, not −11.1%. The difference is the spare
battery described above, which was added after these fixes. −11.1% is what a
one-pack drone achieves in the corrected world — the `1 (no spare)` row of the
battery ablation, at the headline table's longer training budget.)

## Results

Held-out evaluation: 20 delivery instances the agents never trained on, 15
customers each, **2 drones each carrying a spare pack**, scored against the
OR-Tools truck-only tour for the same instance. Negative is better.

| policy | routes completed | delivery time | vs baseline | truck km | vs baseline | flown by drone |
|---|---|---|---|---|---|---|
| **maskable ppo** | **100%** | **123.6 min** | **−20.5%** | **50.5 km** | **−22.2%** | 51% |
| always nearest | 100% | 130.1 min | −16.2% | 54.1 km | −16.3% | 58% |
| greedy | 100% | 133.0 min | −14.5% | 51.9 km | −20.0% | 50% |
| random | 100% | 137.0 min | −11.6% | 54.8 km | −15.3% | 47% |
| dqn | 100% | 143.3 min | −7.8% | 59.0 km | −8.8% | 25% |
| ppo | 100% | 155.2 min | −0.1% | 64.7 km | −0.1% | 1% |
| truck only (baseline) | 100% | 155.5 min | — | 64.8 km | — | 0% |

The masked agent is the only learner to beat the hand-written heuristics, and
it does so while completing every route and never attempting a single
infeasible sortie.

**It wins by flying less, not more.** Note the last column: `always_nearest`
hands 58% of the customers to a drone and MaskablePPO only 51%, yet the agent
finishes four and a half points ahead. The operational numbers say why —

| policy | delivery time | pack swaps per route | truck idle at rendezvous |
|---|---|---|---|
| maskable ppo | −20.5% | 5.0 | 2.5 min |
| always nearest | −16.2% | 6.2 | 0.2 min |
| greedy | −14.5% | 5.1 | 8.4 min |

`always_nearest` flies constantly and burns more swaps to do it; `greedy`
chases expensive detours and leaves the truck standing for eight minutes a
route. The learned policy flies fewer, better-chosen sorties and accepts a
little idling where the detour is worth it. All three land with 13–17% charge
left, so every one of them is working the battery close to its 10% reserve —
the constraint is real and being used, not slack.

### Is the lead real?

Every policy runs the same 20 instances, so the comparison is paired and
instance difficulty cancels. Both a paired t-test and a Wilcoxon signed-rank
test are reported — Wilcoxon makes no normality assumption, and the two
agreeing means the answer does not rest on one.

| vs | mean difference | instances won | paired t | Wilcoxon |
|---|---|---|---|---|
| always nearest | −6.5 min | 17/20 | 3.0e-04 | 3.9e-04 |
| greedy | −9.4 min | 15/20 | 1.0e-02 | 7.3e-03 |
| random | −13.3 min | 16/20 | 1.8e-04 | 2.1e-04 |
| dqn | −19.6 min | 20/20 | 2.4e-07 | 1.9e-06 |
| ppo | −31.6 min | 20/20 | 2.5e-10 | 1.9e-06 |
| truck only | −31.8 min | 20/20 | 2.0e-10 | 1.9e-06 |

The margin over `always_nearest` — the toughest opponent — is 6.5 minutes a
route, won on 17 of 20 instances, at p ≈ 3 × 10⁻⁴. It is not noise. The win
count is quoted alongside the mean deliberately: a policy that is better on
average but loses half its instances would be a much weaker claim than one
that wins nearly all of them.

### The two failure modes, caught in one table

The rows above use each algorithm's **best** checkpoint, which is standard
practice. Scoring the **final** checkpoint instead is worth doing once,
because it exhibits both problems this project set out to fix:

| final checkpoint | routes completed | delivery time | vs baseline | illegal dispatches |
|---|---|---|---|---|
| ppo | **65%** | 126.5 min | −17.3% | **25.8** |
| dqn | 95% | 144.8 min | −6.6% | 4.2 |
| maskable ppo | 100% | 125.4 min | −19.2% | 0.0 |

Look at what unmasked PPO reports here: **−17.3%**, wildly better than its own
honest best-checkpoint score of −0.1%. It earns that by abandoning **35% of
its routes** — its deterministic policy gets stuck repeating an infeasible
dispatch until the step budget expires, and a run that quits early looks fast.
This is precisely the artefact that produced the original 15% claim,
reproduced here on demand, and the completion column is the only thing that
catches it.

### What the hardware buys

The agent is retrained at every setting, so each row is the achievable saving
with that hardware rather than one policy applied outside the world it learned
in. Ablation runs use a 200k-step budget against the headline table's 300k,
so they are comparable with each other but slightly conservative against it.

**A spare battery per drone** (2 drones throughout):

| packs per drone | random | greedy | always nearest | **maskable ppo** |
|---|---|---|---|---|
| 1 (no spare) | −6.6% | −8.1% | −9.7% | **−11.8%** |
| 2 (hot swap) | −11.6% | −14.5% | −16.2% | **−18.1%** |

**Fleet size** (every drone carrying a spare):

| drones | random | greedy | always nearest | **maskable ppo** |
|---|---|---|---|---|
| 1 | −8.5% | −4.4% | −8.7% | **−10.8%** |
| 2 | −11.6% | −14.5% | −16.2% | **−18.1%** |
| 3 | −17.6% | −19.1% | −20.4% | **−22.3%** |

The clearest finding in the project is the comparison *between* these two
tables. Adding a spare pack to a two-drone truck is worth **+6.3 points**
(−11.8% → −18.1%). Adding a whole third drone to that same truck is worth
**+4.2** (−18.1% → −22.3%). **The cheapest upgrade is a battery, not an
airframe** — and it is not close. Drone share rises from 37% to 55% when the
spare appears, which says plainly what was binding: not the drone's range, not
the number of airframes, but how long a single pack took to refill.

A **one-drone truck stays structurally limited** whatever else you give it: it
can remove at most one stop per leg, so no dispatcher gets past about 11%.

The RL margin over the best heuristic runs **+1.9 to +2.1 points** across the
ablation settings, and reaches +4.3 in the headline table where the agent gets
the full 300k-step budget. An earlier version of the fleet-size table appeared
to show the margin widening sharply with fleet size; that was an artefact of
the rendezvous bug described above, which rewarded exactly the multi-drone
launches whose cost was being understated.

## Repository layout

```
truckdrone/
  scenario.py    OSM road network, distance matrices, graph reconstruction
  physics.py     drone energy model — thrust, actuator disk, LiPo voltage sag
  env.py         the Gymnasium environment, action masking, honest termination
  baselines.py   truck-only, random, always-nearest, greedy; OR-Tools TSP
  train.py       MaskablePPO / PPO / DQN
  evaluate.py    held-out scoring, completion-gated statistics
  ablation.py    fleet-size and spare-battery studies
  rollout.py     turns a policy rollout into an animation trace
  report.py      figures and the results table
  config.py      one place for the experiment configuration
dashboard/
  server.py      Flask API — runs policies on demand
  static/        the live visualiser
data/
  whitefield_scenario.npz   distance matrices + 40 solved delivery instances
  whitefield_drive.graphml  the raw OpenStreetMap extract it was built from
tests/           property tests for the claims the results depend on
```

## Running the pieces

```bash
python -m truckdrone.scenario                          # verify the network
python -m truckdrone.train --algo all --timesteps 300000
python -m truckdrone.evaluate                          # results table
python -m truckdrone.evaluate --which final            # the failure-mode table
python -m truckdrone.ablation                          # fleet size + spare battery
python -m truckdrone.report                            # figures
python dashboard/server.py                             # dashboard only
python -m truckdrone.export_static                     # freeze to static site/
tensorboard --logdir models                            # training curves
```

## The dashboard

- **Live map** — truck crawling along real road paths, drones flying straight
  lines to customers, hovering to drop, then cutting ahead to the recovery
  point. Customers fill in as they are served.
- **Race view** — the same instance run side by side against the truck-only
  OR-Tools baseline on one clock. You watch one finish first.
- **Live state** — elapsed delivery clock, packages delivered, truck and drone
  distance, energy, and per-drone battery showing the pack in the air against
  the spare charging on the truck.
- **Sortie log** — every launch: the single parcel it carries, where it meets
  the truck afterwards, the road detour it saved, what it cost the battery and
  whether a pack was swapped.
- **Held-out evaluation table** — filled in once `truckdrone.evaluate` has run.

Any policy can be selected, learned or hand-written, on any of the 20 held-out
instances.

## Deploying the dashboard

The live dashboard runs policies on demand, which means shipping PyTorch — 544 MB
of it, before OR-Tools and SciPy. That is more than twice Vercel's 250 MB
serverless budget, and on a container host it buys a cold start long enough to
ruin a demo.

None of it is necessary. Every rollout the dashboard can display is
**deterministic**: learned policies are evaluated with `deterministic=True`, the
heuristics have no randomness, and `RandomPolicy` is seeded. So the set of things
a visitor can ask for is finite and known in advance — 7 policies × 20 held-out
instances — and each trace is about 9 KB of JSON.

```bash
python -m truckdrone.export_static      # 140 traces, ~9 s, 1.2 MB total
```

That writes `site/`: a self-contained static build with no Python at all. It
loads instantly, costs nothing to host, and cannot fall over mid-presentation.

`dashboard/static/index.html` serves **both** modes from one file — the exporter
prepends `window.__STATIC__`, which swaps the `/api/...` calls for files under
`data/`. There is no second copy of the dashboard to keep in sync, and because
the rollouts are deterministic a precomputed trace is byte-identical to what the
server would have returned.

| | live Flask | static build |
|---|---|---|
| dependencies | ~800 MB | none |
| cold start | seconds (torch import + MDS) | none |
| new policy on demand | yes | no — the 140 precomputed combinations are all the UI exposes |

Re-run the exporter after retraining, or the site will serve the old agents.

## Methodology notes

- **Held-out instances.** 40 delivery problems share one road network; the
  agent trains on 20 and every reported number comes from the other 20.
- **Paired comparison.** Each hybrid run is compared against the OR-Tools
  truck-only tour *for that same instance*, so instance difficulty cancels.
- **Same code path for every policy.** Heuristics and learned agents are scored
  through one function, so no policy gets an accidental advantage.
- **Identical hyperparameters** for PPO and MaskablePPO, so the masking
  ablation isolates masking.
- **The baseline is strong.** OR-Tools runs guided local search on the truck
  tour. The comparison is against a well-solved TSP, not a naive one.
- **Differences are tested, not eyeballed.** Paired t and Wilcoxon signed-rank
  over the 20 shared instances, reported with the win count.
- **Single training seed.** Every learned result comes from `seed=0`
  (`config.SEED`). The evaluation is paired across 20 instances, which controls
  for instance difficulty but not for training variance — a second seed could
  shift the learned rows by a point or so. Re-run with
  `--seed` to check; the heuristics are deterministic and unaffected.

## Limitations

Stated plainly, because a result is only as good as what it leaves out. The
first two both make the reported savings **conservative** — a fuller model
would favour the drone more, not less.

- **The truck pays no service time at a stop.** Only driving time is charged,
  so removing a stop saves the detour distance but not the minutes a driver
  would spend parking and walking to the door. Real last-mile service time is
  a substantial share of a route, and it is time a drone sortie removes
  entirely. In the sortie log you can occasionally see the agent fly to a
  customer that sits directly on the truck's path, saving 0 km of detour —
  under this model that sortie is wasted; with service time modelled it would
  not be.
- **The truck route is fixed.** Each instance uses one OR-Tools tour computed
  over *all* customers, and drone-served stops are skipped from it. The tour
  is never re-optimised for the customers that actually remain, so the truck
  drives a route shaped by stops it no longer makes. Joint re-optimisation is
  the obvious next step and would raise every hybrid number here.
- **One recovery point per sortie.** The drone must meet the truck at its next
  required stop. Allowing recovery further down the tour would permit longer,
  more valuable sorties.
- **One parcel per sortie.** A drone carries a single package and must rejoin
  the truck before it can take another. Multi-parcel drones exist and would
  change the economics substantially, but they are not modelled here.
- **Deterministic.** No traffic, wind, failed handovers, or stochastic service
  times. Battery and flight physics are modelled in detail; the world around
  them is not.
- **One road network.** The 40 instances vary depot and customers but share a
  single 50-node graph, so these results speak to one city region rather than
  to road networks in general.
- **Swap time and charge rate are assumptions, not measurements.** A pack
  changes over in 60 s and charges at 1% per minute on the truck. The spare
  battery result is worth roughly what those two numbers are worth: a slower
  swap or a faster charger would narrow the gap between one pack and two. The
  ablation is the place to re-run if you want to test that sensitivity.

## Credits

Built on research by Adithya SM and Nitin. The road network, the drone energy
model and the environment's core dynamics come from that work; the action
masking, the corrected evaluation, the baseline suite and the simulator are
this project.
