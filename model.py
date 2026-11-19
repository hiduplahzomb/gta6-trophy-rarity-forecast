"""Forecast GTA VI PS5 trophy earn rates on Exophase at a target time.

Model: count decomposition + category priors.

  1. Parse the case history and dedupe consecutive identical reads.
     Exophase refreshes a game's counts on its own schedule (hours apart),
     so most reads repeat the same state; only the distinct states carry
     information.
  2. Players tracked: a power law in time since release, fit on the
     distinct reads. A flat history (no refresh inside the window) falls
     back to a refresh-jump expectation: a refresh is due, and the median
     jump is ~3x (the first refresh was 6x, from a tiny pre-launch
     baseline; later ones are smaller).
  3. Earned counts per trophy: a power law in time, blended in log space
     with a time-dependent category prior anchored on Red Dead Redemption
     2 / GTA V trophy-rarity analogues, then capped by the category
     ceiling and floored at the last observed count (counts never fall).
  4. Rate = earned / players. Quantiles are logit-normal around the
     median with a horizon-, category- and refresh-aware spread.

Why counts, not rates: a refresh jump changes the denominator but not
the numerator, so the rate series is jumpy while the earned-count series
is smooth and monotone. Fitting the counts and dividing by a separately
forecast players count captures the dilution-vs-progression dynamic that
a direct rate extrapolation misses.

Case schema (makefun/gta6-trophies-case/v1): game.release_date is a date
("2026-11-19"); history is a list of {slot, at, players_tracked,
trophies_earned, rates} where rates is a list in trophies order;
trophies carry {id, name, description, tier, points, secret}.

Standard library only. Deterministic: no randomness, no wall clock, no
network. Every input is read from the case file.
"""

import math
from datetime import datetime, timezone

# ---------------------------------------------------------------- constants

MAX_PLAYERS = 1_000_000   # sanity cap on tracked players at the target
MAX_PLAYERS_GROWTH = 8.0  # cap on players growth vs the last read: launch
                          # sales are front-loaded, so a power law fit on
                          # the launch rush overstates later growth
FLAT_REFRESH_MAX = 3.0    # median players jump when a flat history hides a
                          # refresh that is overdue (first refresh was 6x)
SIGMA_PER_SQRT_DAY = 0.5  # horizon spread, logit units per sqrt(day)
REFRESH_SIGMA = 0.8       # extra spread for a refresh landing in-window
REFRESH_DECAY_HOURS = 12.0  # refresh uncertainty decays with horizon
BLEND_HALF_LIFE_DAYS = 3.0  # trend weight halves every 3 days of horizon
MIN_SIGMA = 0.3
MAX_SIGMA = 2.0

# category: (ceiling %, tau hours, sigma_cat)
# Ceiling = the long-run share of tracked players who will ever earn a
# trophy of this kind. Anchored on RDR2 PSNProfiles mature rates
# (chapter 1 80%, final mission 42%, 100% completion 4.2%, platinum
# 2.9%) and GTA V PSN rates (final mission 16.8%, platinum 0.1%),
# scaled to a tracker population. tau = characteristic time to approach
# the ceiling; the prior rate at time t is ceiling * t / (t + tau).
CATEGORIES = {
    "intro":      (100.0, 1.0,    0.30),
    "chapter1":   (60.0,  36.0,   0.30),
    "chapter2":   (45.0,  60.0,   0.30),
    "chapter3":   (35.0,  84.0,   0.30),
    "chapter4":   (25.0,  108.0,  0.35),
    "final":      (18.0,  132.0,  0.40),
    "story":      (20.0,  96.0,   0.40),
    "side":       (10.0,  120.0,  0.50),
    "grind":      (2.0,   720.0,  0.60),
    "completion": (0.5,   2000.0, 0.80),
    "platinum":   (0.1,   2000.0, 0.80),
    "misc":       (12.0,  72.0,   0.50),
}

_STORY_MISSIONS = ("breakfast run", "cartel heist", "child's play",
                   "headliners", "vcia takedown", "operation savage glory")
_GRIND_WORDS = ("250 stunts", "gold medal", "collectible card", "waink",
                "50 challenges", "meditate", "explore all",
                "purchase all", "20 different races", "drone race")


def categorize(trophy):
    """Assign a trophy to a category by its tier, name and description."""
    text = (trophy.get("name", "") + " " +
            trophy.get("description", "")).lower()
    tier = str(trophy.get("tier", "")).lower()
    if tier == "platinum":
        return "platinum"
    if "gone fishing" in text:
        return "intro"
    if "attain 100% completion" in text:
        return "completion"
    if "complete chapter 1" in text:
        return "chapter1"
    if "complete chapter 2" in text:
        return "chapter2"
    if "complete chapter 3" in text:
        return "chapter3"
    if "complete chapter 4" in text:
        return "chapter4"
    # Side strands ("final mission for X") before the story final.
    if "final mission for" in text or "mission strands for strangers" in text:
        return "side"
    if "complete the final mission" in text:
        return "final"
    if any(m in text for m in _STORY_MISSIONS):
        return "story"
    if any(w in text for w in _GRIND_WORDS):
        return "grind"
    return "misc"


# ------------------------------------------------------------------ helpers

def parse_time(s):
    """Parse an ISO timestamp ending in Z into an aware datetime."""
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


def parse_release(game):
    """Release is a date ("2026-11-19") = midnight UTC that day."""
    raw = str(game.get("release_date", "2026-11-19"))
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        dt = datetime(2026, 11, 19)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def logit(p):
    """Logit of a percentage in (0, 100)."""
    p = min(max(p, 1e-9), 100.0 - 1e-9) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    """Inverse logit, returned as a percentage in (0, 100)."""
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def norm_ppf(p):
    """Inverse standard-normal CDF (Acklam's approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
                + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= 1.0 - p_low:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r
                + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r
                + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
             + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _ols(xs, ys):
    """Least-squares slope/intercept of y on x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return (my - b * mx, b)


def _power_law_fit(times, values):
    """Fit log(value + 1) = a + b * log(1 + t). Returns (a, b) or None.

    times are hours since release; values are counts (players or earns).
    Pre-release reads (t < 0) are clamped to t = 0: the pre-launch
    baseline is the t = 0 state, not part of the growth curve.
    """
    xs = [math.log1p(max(t, 0.0)) for t in times]
    ys = [math.log(v + 1.0) for v in values]
    return _ols(xs, ys)


def _power_law_eval(fit, t):
    """Evaluate a fitted power law at time t (hours); returns a count."""
    a, b = fit
    return math.expm1(a + b * math.log1p(t))


# ------------------------------------------------------------------- model

def _read_time(read):
    """A history read's time: its slot (nominal), else its `at`."""
    return parse_time(read.get("slot") or read["at"])


def _distinct_reads(case):
    """History reads, deduplicated to one per distinct state.

    Consecutive reads with identical (players_tracked, rates) are the
    same Exophase refresh seen repeatedly; keep the FIRST read of each
    run (the earliest time that state was visible).
    """
    out = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates")
        players = read.get("players_tracked")
        if players is None or not rates:
            continue
        key = (players, tuple(rates))
        if key != prev_key:
            out.append(read)
            prev_key = key
    return out


def _current_state_age_hours(case):
    """How long (hours) the latest history state has been observed.

    Walks the full history backwards from the end while the state
    (players_tracked, rates) matches the last state; the age is the
    span from the first read of that run to the last read. Must be
    computed on the FULL history: deduping collapses the run to one
    read and destroys the staleness information.
    """
    hist = [r for r in case.get("history", [])
            if r.get("players_tracked") is not None and r.get("rates")]
    if not hist:
        return 0.0
    last = hist[-1]
    key = (last["players_tracked"], tuple(last["rates"]))
    first = last
    for read in reversed(hist[:-1]):
        if (read["players_tracked"], tuple(read["rates"])) == key:
            first = read
        else:
            break
    return max(0.0,
               (_read_time(last) - _read_time(first)).total_seconds()
               / 3600.0)


def _forecast_players(distinct, release, target, state_age_hours):
    """Forecast players_tracked at the target time."""
    t_target = (target - release).total_seconds() / 3600.0
    players = [r["players_tracked"] for r in distinct]
    last = players[-1]
    if len(set(players)) < 2:
        # Flat history: no refresh inside the window, so a refresh is due.
        # Scale the expected jump by how stale the current state is.
        factor = 1.0 + (FLAT_REFRESH_MAX - 1.0) * min(
            1.0, state_age_hours / 3.0)
        return max(float(last), min(float(MAX_PLAYERS), last * factor))
    times = [(_read_time(r) - release).total_seconds() / 3600.0
             for r in distinct]
    fit = _power_law_fit(times, players)
    if fit is None:
        return float(last)
    a, b = fit
    # Players only grow; clamp the slope to a sane band.
    b = min(max(b, 0.0), 3.0)
    est = _power_law_eval((a, b), t_target)
    cap = min(float(MAX_PLAYERS), float(last) * MAX_PLAYERS_GROWTH)
    return max(float(last), min(cap, est))


def forecast(case):
    """Forecast every trophy's quantiles at the case target.

    Returns a list of {"id": int, "q": [...]} sorted by trophy id, one
    quantile per level in case["levels"], rates as percentages rounded
    to 4 decimals.
    """
    release = parse_release(case.get("game", {}))
    target = parse_time(case["target"])
    levels = list(case.get("levels") or (0.05, 0.25, 0.5, 0.75, 0.95))
    trophies = sorted(case["trophies"], key=lambda t: t["id"])
    # rates lists follow the trophies order in the case file.
    pos = {t["id"]: i for i, t in enumerate(case["trophies"])}
    distinct = _distinct_reads(case)
    t_target = (target - release).total_seconds() / 3600.0

    horizon_minutes = case.get("horizon_minutes")
    if horizon_minutes is None:
        at = parse_time(case["at"])
        horizon_minutes = (target - at).total_seconds() / 60.0
    horizon_days = max(0.0, float(horizon_minutes) / 1440.0)
    horizon_hours = horizon_days * 24.0

    state_age = _current_state_age_hours(case)
    players_target = _forecast_players(distinct, release, target,
                                       state_age) \
        if distinct else 10000.0

    # Trend weight: near 1 for short horizons (trust the data), fading
    # toward the category prior as the horizon grows.
    w_trend = 1.0 / (1.0 + horizon_days / BLEND_HALF_LIFE_DAYS)

    # Refresh uncertainty: a refresh landing inside a short window moves
    # every rate at once; it washes out over days.
    refresh_sigma = REFRESH_SIGMA * math.exp(-horizon_hours /
                                             REFRESH_DECAY_HOURS)

    results = []
    for trophy in trophies:
        tid = trophy["id"]
        cat = categorize(trophy)
        ceiling, tau, sigma_cat = CATEGORIES[cat]
        # Time since release cannot be negative: a target before release
        # (or a release-date typo) must not drive the prior negative.
        prior_rate = (ceiling * max(t_target, 0.0)
                      / (max(t_target, 0.0) + tau))
        prior_count = max(0.05, players_target * prior_rate / 100.0)

        # Observed earned counts from the distinct reads.
        counts = []
        times = []
        idx = pos[tid]
        for read in distinct:
            rates = read.get("rates") or []
            if idx >= len(rates) or rates[idx] is None:
                continue
            counts.append(rates[idx] * read["players_tracked"] / 100.0)
            times.append((_read_time(read) - release).total_seconds()
                         / 3600.0)

        if counts:
            last_count = counts[-1]
            if len(set(round(c, 6) for c in counts)) < 2:
                trend_count = last_count
            else:
                fit = _power_law_fit(times, counts)
                trend_count = _power_law_eval(fit, t_target) \
                    if fit is not None else last_count
        else:
            # No history at all (e.g. a trophy added after the reads):
            # the prior is the whole forecast.
            last_count = 0.0
            trend_count = prior_count

        # Blend trend and prior in log space.
        log_c = (w_trend * math.log(max(trend_count, 0.05)) +
                 (1.0 - w_trend) * math.log(prior_count))

        # Saturation: a trophy (nearly) all tracked players hold stays
        # saturated — its count scales with players, not with time.
        # (The intro trophy held 100.00% through a 6x refresh jump.)
        # Blend toward the sticky-rate count as the last observed rate
        # approaches 100%; no effect below a 50% last rate.
        if counts:
            last_rate = (last_count / distinct[-1]["players_tracked"]
                         * 100.0)
            w_sticky = max(0.0, (last_rate - 50.0) / 50.0)
            if w_sticky > 0.0:
                sticky_count = last_rate / 100.0 * players_target
                log_c = (w_sticky * math.log(max(sticky_count, 0.05)) +
                         (1.0 - w_sticky) * log_c)

        # Cap by the ceiling and floor at the last observed count
        # (counts never fall).
        count = math.exp(log_c)
        count = min(count, players_target * ceiling / 100.0)
        count = min(count, float(players_target))
        count = max(count, last_count)

        median = 100.0 * count / players_target
        median = min(max(median, 0.01), 99.99)

        sigma = math.sqrt((SIGMA_PER_SQRT_DAY * math.sqrt(horizon_days)) ** 2
                          + sigma_cat ** 2 + refresh_sigma ** 2)
        sigma = min(max(sigma, MIN_SIGMA), MAX_SIGMA)

        centre = logit(median)
        qs = [round(inv_logit(centre + norm_ppf(lv) * sigma), 4)
              for lv in levels]
        # Force non-decreasing after rounding, keep inside (0, 100).
        for i in range(1, len(qs)):
            qs[i] = max(qs[i], qs[i - 1])
        qs = [min(max(q, 0.01), 99.99) for q in qs]
        results.append({"id": tid, "q": qs})
    return results
