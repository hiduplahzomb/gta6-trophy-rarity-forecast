# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hiduplahzomb/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hiduplahzomb/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every GTA VI PlayStation trophy's earn rate on Exophase at 2026-11-26T00:00:00Z (one week after launch), as five quantiles [0.05, 0.25, 0.5, 0.75, 0.95] per trophy, scored by pinball loss on the logit scale.

## Method

Count decomposition plus category priors — not a direct rate extrapolation:

1. **Dedupe the history.** Exophase refreshes a game's counts on its own schedule (hours apart), so most reads repeat the same state. Only the distinct states carry information. The current state's age (how long it has been observed) is measured on the full history, since deduping destroys it.
2. **Players tracked** follow a power law in time since release, fit on the distinct reads, capped at 8x the last read and 1,000,000 absolute (launch sales are front-loaded, so a power law fit on the launch rush overstates later growth). A flat history (no refresh inside the window) falls back to a refresh-jump expectation — a refresh is due, and the median jump is ~3x, scaled by how stale the state is.
3. **Earned counts** per trophy follow a power law in time, blended in log space with a time-dependent category prior anchored on Red Dead Redemption 2 / GTA V trophy-rarity analogues (RDR2 PSNProfiles mature rates: chapter 1 80%, final mission 42%, 100% completion 4.2%, platinum 2.9%; GTA V PSN: final mission 16.8%, platinum 0.1%). Counts are capped by the category ceiling and floored at the last observed count — counts never fall.
4. **Saturation.** A trophy (nearly) all tracked players hold stays saturated: its count scales with players, not with time. (The intro trophy held 100.00% through a 6x refresh jump.) The count blends toward the sticky-rate count as the last observed rate approaches 100%.
5. **Rate = earned / players.** This captures the dilution-vs-progression dynamic: a refresh jump dilutes every rate while play pushes earned counts up. The rate series is jumpy; the count series is smooth and monotone.
6. **Quantiles** are logit-normal around the median with a spread that grows with the horizon, widens for grind/platinum trophies, and adds a refresh component for short windows (a refresh landing inside the window moves every rate at once).

Standard library only, deterministic, no network.

## Validation

Practice cases scored against the reconstructed answers (the Exophase page fetched 2026-11-19 10:54 UTC still shows the 04:30Z refresh state — 3,868 players — which is both practice answers): **pinball loss 0.333 (p00) / 0.305 (p01) vs the house baselines' 1.51**, typical miss 0.14 / 0.13 pp, 90% coverage 0.98. Two runs byte-identical. A synthetic final-like case (2.5 days of history with refresh jumps, 4.5-day horizon) gives sane week-one medians: intro 99.99%, chapter 1 60%, chapter 2 43%, final mission 14%, platinum 0.01%, 100% completion 0.03%.

Caveat: the practice is a 90-minute horizon dominated by one refresh jump; the final is a 4.5-day horizon dominated by progression and players growth. The practice score does not guarantee a final score — players growth is the biggest uncertainty, and the category priors are analogues, not measurements.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=2wgaefze&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
