"""Forecast Exophase GTA VI (PS5) trophy earn rates at a future cut.

Method: count decomposition plus category priors.

A trophy's earn rate is a ratio, earned / players, and the two halves move
for different reasons. Players jump in steps whenever Exophase refreshes
its page; earned counts grow steadily as players progress through the
game. So the model forecasts the numerator and the denominator
separately, then divides.

Players.  The distinct reads (see distinct_reads) are fit to a power law
in time since release, players = exp(a + b * log1p(hours)), and the fit
is read at the target time, capped at eight times the last observed
players and at 1,000,000 absolute. When the history shows only one
distinct state the page has not refreshed yet, so the fit is impossible
and the forecast uses the refresh-jump expectation instead: the next
refresh multiplies players by roughly three, scaled up the longer the
current state has been stale.

Earned counts.  Each trophy's earned count is rate * players / 100 at
every distinct read. Counts are fit to the same power-law shape in time
since release and read at the target, then blended in log space with a
category prior: the prior says what share of players will eventually
earn a trophy of that kind, and how fast that share approaches its
ceiling. The blend weight on the observed trend fades with horizon (the
recent data pins down a 90-minute forecast; over four days the prior
carries more of the weight). The blended count is floored at the last
observed count (counts never fall) and capped at the category ceiling
times the forecast players. A trophy whose last rate is near 100%
saturates: its count is pulled toward the sticky-rate count, last rate
times forecast players, so the introduction trophy stays at ~100%.

Quantiles.  The median rate is the forecast count over forecast
players. The five quantiles come from a logit-normal spread around it:
the spread grows with the square root of the horizon in days, widens
for grind and platinum trophies (slow, uncertain), and carries an extra
refresh component on short horizons, because a refresh inside the
window moves every rate at once.

Only the standard library is used; the forecast is deterministic.
"""

import math
from datetime import datetime, timezone

# Category priors: (eventual ceiling in percent of players,
# characteristic time in hours for the rate to approach it).
# Anchored on the mature trophy rates of GTA V and Red Dead Redemption 2
# on PlayStation, where story trophies settle near their ceilings within
# the first week and grind trophies stay in the low single digits.
PRIORS = {
    "intro": (100.0, 1.0),
    "chapter1": (60.0, 36.0),
    "chapter2": (45.0, 60.0),
    "chapter3": (35.0, 84.0),
    "chapter4": (25.0, 108.0),
    "final": (18.0, 132.0),
    "story": (20.0, 96.0),
    "side": (10.0, 120.0),
    "grind": (2.0, 720.0),
    "completion": (0.5, 2000.0),
    "platinum": (0.1, 2000.0),
    "misc": (12.0, 72.0),
}

# z-scores for the five output levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95.
Z_SCORES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)

MAX_PLAYERS = 1_000_000
PLAYERS_GROWTH_CAP = 8.0
# How fast the blend weight on the observed trend fades with horizon:
# weight = 1 / (1 + horizon_days / BLEND_HALF_LIFE_DAYS).
BLEND_HALF_LIFE_DAYS = 3.0
# Refresh-jump expectation for a page that has not refreshed yet:
# base multiplier plus one per hour the current state has been stale.
JUMP_BASE = 2.5
JUMP_PER_STALE_HOUR = 1.0
# Logit-normal spread: base scale per sqrt(day), extra for slow
# categories, and the short-window refresh component.
SPREAD_PER_SQRT_DAY = 0.5
SPREAD_SLOW_CATEGORY = 0.5
SPREAD_REFRESH = 0.6
SPREAD_REFRESH_DECAY_DAYS = 1.0


def categorize(trophy):
    """Assign one of the category priors to a trophy.

    The trophy list for this game is fixed, so the rules read the
    trophy's tier, name and description. Story beats are recognised by
    their description ("Complete Chapter N.", "Complete the final
    mission.", the opening mission "Gone Fishing"); side strands by
    "the final mission for <name>"; the platinum by tier; 100%
    completion by name. What is left is split by effort: long,
    repetitive or all-of-a-kind achievements are grind, everything else
    is misc.
    """
    tier = (trophy.get("tier") or "").lower()
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").strip().lower()

    if tier == "platinum":
        return "platinum"
    if "100% completion" in desc:
        return "completion"
    if desc.startswith("complete chapter "):
        chapter = desc[len("complete chapter "):].split(".")[0].strip()
        if chapter in ("1", "2", "3", "4"):
            return "chapter" + chapter
    if desc == "complete the final mission.":
        return "final"
    if "gone fishing" in desc or name == "just like paradise":
        return "intro"
    if "final mission for" in desc or "mission strands" in desc:
        return "side"
    if desc.startswith("complete ") and desc.endswith("."):
        return "story"
    grind_markers = (
        "250 stunts",
        "gold medal in all",
        "every location",
        "20 different races",
        "3 collectible card sets",
        "100 locations",
        "50 challenges",
        "5 bank vaults",
        "every spot",
    )
    if any(marker in desc for marker in grind_markers):
        return "grind"
    return "misc"


def parse_time(text):
    """Parse an ISO 8601 timestamp (trailing Z allowed) to aware UTC."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since(release, moment):
    """Hours from the release instant to a moment, floored at zero."""
    return max((moment - release).total_seconds() / 3600.0, 0.0)


def distinct_reads(history, reference):
    """Collapse the history to one entry per distinct Exophase state.

    The house recorder reads the page every few minutes, so most reads
    repeat the previous state exactly. A state is the players-tracked
    count together with the whole rates vector; reads that match an
    earlier state carry no new information. Returns the distinct states
    in first-seen order, each as (time, players, rates), plus the age in
    hours of the current (last) state at the reference time, measured
    on the full history -- how long the page has shown this state.
    """
    seen = {}
    order = []
    for read in history:
        rates = tuple(read.get("rates") or [])
        key = (read.get("players_tracked"), rates)
        when = parse_time(read.get("at") or read.get("slot"))
        if key not in seen:
            seen[key] = when
            order.append((when, read.get("players_tracked"), rates))
    if not order:
        return [], 0.0
    last_time, last_players, last_rates = order[-1]
    first_seen = seen[(last_players, last_rates)]
    age = (reference - first_seen).total_seconds() / 3600.0
    return order, max(age, 0.0)


def fit_log_power(times, values):
    """Least-squares fit of log(value) = a + b * log1p(time).

    Returns (a, b), or None when there are fewer than two usable points
    or the values carry no variation. Times are hours since release,
    already floored at zero, so log1p is always defined.
    """
    points = [(math.log1p(t), math.log(v))
              for t, v in zip(times, values) if v > 0]
    if len(points) < 2:
        return None
    n = float(len(points))
    mean_x = sum(p[0] for p in points) / n
    mean_y = sum(p[1] for p in points) / n
    sxx = sum((p[0] - mean_x) ** 2 for p in points)
    if sxx <= 0:
        return None
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in points)
    slope = sxy / sxx
    return (mean_y - slope * mean_x, slope)


def eval_log_power(fit, t):
    """Evaluate a log-power fit at t hours since release."""
    a, b = fit
    return math.exp(a + b * math.log1p(max(t, 0.0)))


def prior_rate(category, t_hours):
    """Category-prior rate (percent) at t hours since release."""
    ceiling, tau = PRIORS[category]
    return ceiling * (1.0 - math.exp(-max(t_hours, 0.0) / tau))


def forecast_players(release, states, state_age, last_players, target_hours):
    """Forecast the players-tracked count at the target time."""
    if len(states) >= 2:
        times = [hours_since(release, when) for when, _, _ in states]
        players = [p for _, p, _ in states]
        fit = fit_log_power(times, players)
        if fit is not None:
            guess = eval_log_power(fit, target_hours)
            return min(guess, PLAYERS_GROWTH_CAP * last_players,
                       MAX_PLAYERS)
    # Flat history: the page has not refreshed yet. The next refresh
    # multiplies players by roughly three, scaled up the longer the
    # current state has been stale.
    jump = min(JUMP_BASE + JUMP_PER_STALE_HOUR * state_age,
               PLAYERS_GROWTH_CAP)
    return min(last_players * jump, MAX_PLAYERS)


def blend_log(trend, prior, weight):
    """Blend two positive estimates in log space; weight is the trend's."""
    return math.exp(weight * math.log(trend) + (1.0 - weight) * math.log(prior))


def forecast_trophy(category, counts, times, last_rate, last_count,
                    players_target, target_hours, trend_weight):
    """Forecast one trophy's earn rate (percent) at the target time."""
    ceiling, _tau = PRIORS[category]
    # The trend: a power law in time through the observed counts.
    fit = fit_log_power(times, counts) if len(counts) >= 2 else None
    if fit is not None:
        trend = eval_log_power(fit, target_hours)
    else:
        trend = last_count
    # The prior, expressed as a count at the forecast players.
    prior_count = prior_rate(category, target_hours) / 100.0 * players_target
    count = blend_log(max(trend, 1e-9), max(prior_count, 1e-9), trend_weight)
    # Counts never fall, and never pass the category ceiling.
    count = max(count, last_count)
    count = min(count, ceiling / 100.0 * players_target)
    # Saturation: a trophy almost everyone already has keeps its rate,
    # so its count grows with players rather than with time.
    if last_rate >= 90.0:
        stick = min((last_rate - 90.0) / 10.0, 1.0)
        sticky = last_rate / 100.0 * players_target
        count = (1.0 - stick) * count + stick * sticky
    return count / players_target * 100.0


def spread_for(category, horizon_days):
    """Logit-normal spread (in logit units) for a forecast."""
    spread = SPREAD_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))
    if category in ("grind", "platinum"):
        spread += SPREAD_SLOW_CATEGORY
    if horizon_days < SPREAD_REFRESH_DECAY_DAYS * 3:
        spread += SPREAD_REFRESH * math.exp(
            -horizon_days / SPREAD_REFRESH_DECAY_DAYS)
    return spread


def logit(p):
    p = min(max(p, 0.01), 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    return 100.0 / (1.0 + math.exp(-x))


def quantiles(median_rate, spread):
    """Five quantiles of a logit-normal centred on the median rate."""
    centre = logit(median_rate)
    out = []
    for z in Z_SCORES:
        q = inv_logit(centre + z * spread)
        out.append(min(max(q, 0.01), 99.99))
    return out


def forecast(case):
    """Forecast every trophy's earn rate at the case's target.

    Returns a list of {"id": n, "q": [q05, q25, q50, q75, q95]} rows
    in trophy-id order.
    """
    trophies = case["trophies"]
    history = case.get("history") or []
    target = parse_time(case["target"])
    at = parse_time(case["at"])
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if "release_date" in case["game"] \
        else datetime(2026, 11, 19, tzinfo=timezone.utc)
    target_hours = hours_since(release, target)
    horizon_days = max(case.get("horizon_minutes", 0), 0) / 1440.0
    trend_weight = 1.0 / (1.0 + horizon_days / BLEND_HALF_LIFE_DAYS)

    states, state_age = distinct_reads(history, at)
    if not states:
        raise ValueError("case has no history reads")
    last_time, last_players, last_rates = states[-1]
    players_target = forecast_players(release, states, state_age,
                                      last_players, target_hours)

    # Hours since release at each distinct read, for the count fits.
    state_times = [hours_since(release, when) for when, _, _ in states]

    rows = []
    for index, trophy in enumerate(trophies):
        category = categorize(trophy)
        counts = []
        times = []
        for (when, players, rates), t in zip(states, state_times):
            if index >= len(rates) or rates[index] is None:
                continue
            counts.append(rates[index] / 100.0 * players)
            times.append(t)
        last_rate = last_rates[index] if index < len(last_rates) \
            and last_rates[index] is not None else 0.0
        last_count = counts[-1] if counts else 0.0
        median = forecast_trophy(category, counts, times, last_rate,
                                 last_count, players_target, target_hours,
                                 trend_weight)
        rows.append({"id": trophy["id"],
                     "q": quantiles(median, spread_for(category,
                                                       horizon_days))})
    rows.sort(key=lambda r: r["id"])
    return rows
