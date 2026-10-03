Held-out evaluation, whitefield: 20 instances × 5 worlds, 15 customers each, preference 'fastest'. Paired % change vs the OR-Tools truck-only tour in the same world, completed routes only; ± is the 95% CI across training seeds.

| policy | seeds | routes completed | time | cost | energy | CO₂ | truck km | by drone | illegal dispatches |
|---|---|---|---|---|---|---|---|---|---|
| maskable ppo | 5 | 100% | -27.1% ± 0.6 | -19.9% ± 0.9 | -24.2% ± 0.8 | -23.2% ± 0.8 | -25.5% ± 0.8 | 52% | 0.00 |
| always nearest | — | 100% | -23.9% | -13.0% | -16.4% | -15.3% | -17.3% | 67% | 0.00 |
| greedy | — | 100% | -22.6% | -14.3% | -17.4% | -16.3% | -18.6% | 54% | 0.00 |
| dqn | 3 | 88% | -20.7% ± 4.9 | -14.6% ± 4.7 | -17.4% ± 4.9 | -16.7% ± 4.9 | -18.3% ± 4.8 | 41% | 9.52 |
| random | — | 100% | -20.7% | -12.8% | -15.8% | -14.7% | -16.8% | 51% | 0.00 |
| ppo | 3 | 99% | -0.7% ± 2.9 | -0.5% ± 2.1 | -0.6% ± 2.1 | -0.6% ± 2.0 | -0.6% ± 2.2 | 1% | 0.56 |
| truck only | — | 100% | +0.0% | -0.0% | -0.0% | -0.0% | +0.0% | 0% | 0.00 |

Paired tests on **time_min** for maskable ppo (n = 100 instance-world pairs; negative favours it):

| vs | mean difference | won | paired t p | Wilcoxon p |
|---|---|---|---|---|
| always nearest | -5.97 min | 67/100 | 1.4e-06 | 1.1e-05 |
| greedy | -8.12 min | 76/100 | 3.3e-11 | 1.8e-10 |
| dqn | -12.18 min | 69/72 | 1.1e-17 | 6.0e-13 |
| random | -11.41 min | 81/100 | 1.9e-15 | 2.1e-12 |
| ppo | -47.95 min | 98/98 | 1.1e-60 | 8.3e-18 |
| truck only | -49.35 min | 100/100 | 3.7e-63 | 3.9e-18 |


Paired tests on **cost_inr** for maskable ppo (n = 100 instance-world pairs; negative favours it):

| vs | mean difference | won | paired t p | Wilcoxon p |
|---|---|---|---|---|
| always nearest | -111.76 ₹ | 94/100 | 3.9e-20 | 2.5e-16 |
| greedy | -91.36 ₹ | 88/100 | 9.2e-19 | 4.2e-15 |
| dqn | -91.29 ₹ | 69/72 | 5.2e-17 | 5.8e-13 |
| random | -114.19 ₹ | 90/100 | 5.8e-22 | 2.0e-15 |
| ppo | -316.32 ₹ | 98/98 | 2.0e-56 | 8.3e-18 |
| truck only | -324.88 ₹ | 100/100 | 2.0e-59 | 3.9e-18 |


Paired tests on **energy_kwh** for maskable ppo (n = 100 instance-world pairs; negative favours it):

| vs | mean difference | won | paired t p | Wilcoxon p |
|---|---|---|---|---|
| always nearest | -5.99 kWh | 95/100 | 9.7e-23 | 5.8e-17 |
| greedy | -5.20 kWh | 91/100 | 1.2e-22 | 1.7e-15 |
| dqn | -5.40 kWh | 70/72 | 1.8e-20 | 2.0e-13 |
| random | -6.51 kWh | 95/100 | 8.9e-28 | 2.3e-17 |
| ppo | -18.19 kWh | 98/98 | 1.9e-65 | 8.3e-18 |
| truck only | -18.60 kWh | 100/100 | 1.8e-69 | 3.9e-18 |

**Robustness** (time % vs truck-only):

| setting | maskable ppo | greedy | always nearest |
|---|---|---|---|
| wind 0 m/s | -28.3% ± 0.8 | -23.2% | -24.8% |
| wind 4 m/s | -27.1% ± 0.6 | -22.1% | -23.4% |
| wind 8 m/s | -22.7% ± 0.8 | -20.6% | -21.3% |
| wind 12 m/s | -16.4% ± 0.6 | -17.4% | -16.6% |
| calm traffic | -27.2% ± 0.8 | -22.1% | -24.5% |
| nominal traffic | -27.1% ± 0.6 | -22.6% | -23.9% |
| heavy traffic | -27.8% ± 1.0 | -23.7% | -24.0% |


**Generalisation** (time % vs truck-only):

| test set | maskable ppo | greedy | always nearest |
|---|---|---|---|
| 10 customers | -23.7% ± 1.3 | -23.6% | -20.1% |
| 20 customers | -28.0% ± 0.8 | -22.5% | -26.7% |
| Chennai (zero-shot) | -24.7% ± 1.0 | -20.7% | -22.9% |


**Ablations** (MaskablePPO, seed 0, 1M steps each):

| variant | agent time | greedy, same world | agent vs reference |
|---|---|---|---|
| Full configuration, same 1M-step budget | -27.5% | -22.6% | — |
| Drone must meet the truck at the next stop (no rendezvous choice) | -24.9% | -21.5% | +2.6 pp |
| Truck tour never re-optimised after drone deliveries | -27.9% | -22.6% | -0.4 pp |
| One drone instead of two | -18.3% | -13.9% | +9.2 pp |
| Three drones instead of two | -30.2% | -27.4% | -2.7 pp |
| No spare battery: one pack per drone | -18.1% | -14.1% | +9.4 pp |
| Trained for time only (no preference conditioning) | -26.9% | -22.6% | +0.6 pp |
| Trained without traffic or wind, tested with them | -26.3% | -22.6% | +1.2 pp |
