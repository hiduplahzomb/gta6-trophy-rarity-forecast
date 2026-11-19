# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hiduplahzomb/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/hiduplahzomb/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for each of the 55 GTA VI PlayStation 5 trophies, the share of
players Exophase tracks who have earned it at 2026-11-26T00:00:00Z — one
week after the game's 19 November 2026 launch — as five quantiles
(0.05 / 0.25 / 0.5 / 0.75 / 0.95). The program is standard-library-only
Python, deterministic, makes no network calls, and runs each case in
well under a second.

## Method

A trophy's earn rate is a ratio, earned / players, and the two halves
move for different reasons, so the model forecasts them separately and
divides.

**Players.** Exophase's players-tracked count jumps in steps when the
page refreshes. The distinct reads in a case's history are fit to a
power law in time since release, read at the target, and capped at eight
times the last observed count and at 1,000,000 absolute. When the
history shows only one distinct state — the page has not refreshed yet —
the forecast uses the refresh-jump expectation instead: the next refresh
multiplies players by roughly three, scaled up the longer the current
state has been stale. Launch night's first refresh was about 6x, from
a tiny pre-launch baseline of 638 players.

**Earned counts.** Each trophy's earned count (rate x players / 100)
is fit to the same power-law shape in time since release and read at the
target, then blended in log space with a category prior: a ceiling on
the share of players who will eventually earn a trophy of that kind,
approached with a characteristic time. The blend weight on the observed
trend fades with horizon — about 98% at 90 minutes, about 40% at 4.5
days. The blended count is floored at the last observed count (counts
never fall) and capped at the category ceiling. A trophy whose last rate
is near 100% saturates: its count is pulled toward the sticky-rate
count, last rate x forecast players, so the opening trophy stays at
~100%.

**Categories.** Each trophy is assigned one of twelve category priors
from its tier, name and description: intro, chapter 1–4, final mission,
story, side strands, grind, 100% completion, platinum, misc. The
ceilings and times are anchored on the mature trophy rates of GTA V and
Red Dead Redemption 2 on PlayStation (see the data notes below).

**Quantiles.** The median rate is the forecast count over the forecast
players. The five quantiles come from a logit-normal spread around it:
the spread grows with the square root of the horizon in days, widens
for grind and platinum trophies, and carries an extra refresh component
on short horizons, because a refresh inside the window moves every rate
at once.

## Data

- **Exophase GTA VI PS5 trophy page** (https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/):
  the 55-trophy list went public on 18 Nov 2026 (Push Square, citing
  Exophase). Launch night the page showed 638 players tracked and
  every trophy's rarity; by the 04:30Z refresh on 19 Nov it showed
  3,868 players. The kit's practice cases are cut from that recording.
- **GTA V and Red Dead Redemption 2 trophy rates** on PlayStation
  (Exophase, read 2026-11-19): story trophies settle at 60–90% of
  players for early chapters and 10–20% for the finale; grind trophies
  (250 stunts, all gold medals, 100 locations) sit below 2%; 100%
  completion and the platinum sit below 1%. These anchor the category
  ceilings.
- **Launch sales**: Sensor Tower's third-party tracker estimates about
  17.1 million units (~$1.65 billion) sold by noon ET on 18 Nov 2026,
  pre-orders included (reported 19 Nov 2026); Take-Two / Rockstar have
  not published an official unit figure, and no 39–40M first-day figure
  has been confirmed. That scale is why the players cap is set at
  1,000,000 Exophase-tracked players — Exophase tracks a small
  fraction of the total player base.

## Validation

`gta6-trophy-validate.py` (kept outside this repo) runs the entry on
the kit's two practice cases and a synthetic final-like case
(2.5 days of history, 4.5-day horizon). Results, 19 November 2026:

| case | pinball loss | typical miss | 90% coverage |
|------|-------------|--------------|--------------|
| p00  | 0.194 | 0.044 pp | 0.96 |
| p01  | 0.200 | 0.048 pp | 0.96 |

House baseline practice loss: 1.51. Both runs are byte-identical
(deterministic), and every output is 55 rows, 5 quantiles, non-decreasing,
inside (0, 100], and under 65,536 bytes.

Synthetic final-like medians are sane: intro 99.99%, chapter 1 60%
(ceiling), chapter 2 45% (ceiling), final mission 15.8%, 100%
completion 0.03%, platinum 0.01%, misc trophies ~10.3%.

## Running it

```sh
MAKEFUN_INPUT_DIR=/path/to/cases MAKEFUN_OUTPUT_DIR=/path/to/out python3 main.py
```

Reads every `*.in` in the input directory and writes `<id>.out` for
each. Each output is validated against its case (ids match, quantiles
non-decreasing, inside (0, 100], at most 65,536 bytes) before it is
written; a malformed case fails fast.
