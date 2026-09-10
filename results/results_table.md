Held-out evaluation: 20 delivery instances, 15 customers each. Time and distance are averaged over completed routes only.
| policy | routes completed | delivery time | vs baseline | truck km | vs baseline | delivered by drone | illegal dispatches |
|---|---|---|---|---|---|---|---|
| maskable ppo | 100% | 123.6 min | -20.5% | 50.5 km | -22.2% | 51% | 0.0 |
| always nearest | 100% | 130.1 min | -16.2% | 54.1 km | -16.3% | 58% | 0.0 |
| greedy | 100% | 133.0 min | -14.5% | 51.9 km | -20.0% | 50% | 0.0 |
| random | 100% | 137.0 min | -11.6% | 54.8 km | -15.3% | 47% | 0.0 |
| dqn | 100% | 143.3 min | -7.8% | 59.0 km | -8.8% | 25% | 0.0 |
| ppo | 100% | 155.2 min | -0.1% | 64.7 km | -0.1% | 1% | 0.0 |
| truck only | 100% | 155.5 min | -0.0% | 64.8 km | +0.0% | 0% | 0.0 |

Paired comparison for **maskable ppo** on the same 20 instances. Negative favours it.

| vs | mean difference | instances won | paired t | Wilcoxon |
|---|---|---|---|---|
| always nearest | -6.5 min | 17/20 | 3.0e-04 | 3.9e-04 |
| greedy | -9.4 min | 15/20 | 1.0e-02 | 7.3e-03 |
| random | -13.3 min | 16/20 | 1.8e-04 | 2.1e-04 |
| dqn | -19.6 min | 20/20 | 2.4e-07 | 1.9e-06 |
| ppo | -31.6 min | 20/20 | 2.5e-10 | 1.9e-06 |
| truck only | -31.8 min | 20/20 | 2.0e-10 | 1.9e-06 |
