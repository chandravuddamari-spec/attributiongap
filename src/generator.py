"""User journeys with a KNOWN causal structure.

Some channels genuinely cause conversions. Others merely appear on the path of
people who were going to convert anyway - paid search on a branded term being
the canonical example, where the user had already decided and the click is a
toll booth rather than a cause.

Attribution cannot tell these apart, because both look identical in a
touchpoint log. The generator therefore records which is which, so the gap
between attributed credit and real incrementality becomes measurable rather
than arguable.
"""

from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

CHANNELS = (
    "brand_search", "generic_search", "paid_social", "display",
    "affiliate", "email", "organic",
)

#: TRUE incremental effect: the probability the channel ADDS to conversion
#: when it appears. Zero means it is on the path but causes nothing.
TRUE_INCREMENTALITY: dict[str, float] = {
    "brand_search": 0.02,     # almost pure toll booth
    "generic_search": 0.11,   # genuinely finds new demand
    "paid_social": 0.09,
    "display": 0.01,          # nearly worthless, and heavily attributed
    "affiliate": 0.03,        # mostly last-click capture of existing intent
    "email": 0.07,
    "organic": 0.05,
}

#: How likely the channel is to APPEAR on the path of a converting user,
#: independent of whether it caused anything. High values here with low
#: incrementality above is exactly the toll-booth pattern.
APPEARANCE_PROPENSITY: dict[str, float] = {
    "brand_search": 0.62,
    "generic_search": 0.34,
    "paid_social": 0.30,
    "display": 0.48,
    "affiliate": 0.41,
    "email": 0.36,
    "organic": 0.44,
}

#: Where in the path the channel tends to sit (0 = early, 1 = late). Last-click
#: attribution rewards lateness, which is why brand search and affiliate
#: dominate under it.
POSITION_BIAS: dict[str, float] = {
    "brand_search": 0.92,
    "generic_search": 0.45,
    "paid_social": 0.22,
    "display": 0.15,
    "affiliate": 0.88,
    "email": 0.55,
    "organic": 0.60,
}


@dataclass(frozen=True)
class Journey:
    path: tuple[str, ...]
    converted: bool
    #: Would this user have converted with NO marketing at all? The quantity
    #: attribution can never observe, and the whole reason it is misleading.
    would_convert_anyway: bool


def simulate_journeys(
    users: int = 200_000, seed: int = 20260911,
    base_conversion: float = 0.06,
) -> list[Journey]:
    rnd = random.Random(seed)
    journeys: list[Journey] = []

    for _ in range(users):
        # Intent, drawn before any marketing happens.
        intent = rnd.random()
        would_anyway = intent > (1 - base_conversion)

        touched: list[tuple[str, float]] = []
        for channel in CHANNELS:
            # High-intent users are MORE likely to be touched by late-funnel
            # channels. This is the confounding that makes attribution wrong:
            # the channel does not cause the conversion, it follows it.
            propensity = APPEARANCE_PROPENSITY[channel]
            if POSITION_BIAS[channel] > 0.7:
                propensity *= 0.4 + 1.4 * intent
            if rnd.random() < propensity:
                touched.append((channel, POSITION_BIAS[channel]))

        touched.sort(key=lambda t: t[1] + rnd.gauss(0, 0.12))
        path = tuple(c for c, _ in touched)

        converted = would_anyway
        if not converted:
            for channel in path:
                if rnd.random() < TRUE_INCREMENTALITY[channel]:
                    converted = True
                    break

        journeys.append(Journey(path, converted, would_anyway))
    return journeys


def true_incremental_conversions(
    journeys: list[Journey], channel: str,
) -> float:
    """Conversions that would NOT have happened without this channel.

    Computed from the generator's own knowledge, so it is the ground truth the
    experiment below is designed to approximate.
    """
    total = 0.0
    for j in journeys:
        if not j.converted or j.would_convert_anyway:
            continue
        if channel not in j.path:
            continue
        # Share of the credit proportional to this channel's incrementality
        # among those on the path - the generator's own causal accounting.
        weights = sum(TRUE_INCREMENTALITY[c] for c in j.path)
        if weights > 0:
            total += TRUE_INCREMENTALITY[channel] / weights
    return total


# ---------------------------------------------------------------------------
# Holdout experiments
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Holdout:
    channel: str
    #: Conversions per user in the treated group and the held-out group.
    treated_rate: float
    holdout_rate: float
    n_treated: int
    n_holdout: int
    #: When the experiment ran, in days before "today".
    age_days: int

    @property
    def incremental_rate(self) -> float:
        return self.treated_rate - self.holdout_rate

    def incremental_conversions(self, population: int) -> float:
        return self.incremental_rate * population


def run_holdout(
    channel: str, users: int = 60_000, seed: int = 7, age_days: int = 0,
) -> Holdout:
    """Split users, suppress the channel for one half, compare.

    This is the ground truth attribution is scored against. It is expensive
    and slow, which is precisely why it is run rarely - and why the bias
    factors below exist to carry its findings into the daily numbers.
    """
    rnd = random.Random(zlib.crc32(channel.encode()) ^ seed)
    treated = simulate_journeys(users // 2, seed=rnd.randrange(1 << 30))
    holdout_source = simulate_journeys(users // 2, seed=rnd.randrange(1 << 30))

    # In the holdout the channel is suppressed: it never appears, and its
    # incremental contribution is therefore absent.
    holdout: list[Journey] = []
    for j in holdout_source:
        if channel not in j.path:
            holdout.append(j)
            continue
        path = tuple(c for c in j.path if c != channel)
        converted = j.would_convert_anyway
        if not converted:
            for c in path:
                if rnd.random() < TRUE_INCREMENTALITY[c]:
                    converted = True
                    break
        holdout.append(Journey(path, converted, j.would_convert_anyway))

    treated_rate = sum(1 for j in treated if j.converted) / len(treated)
    holdout_rate = sum(1 for j in holdout if j.converted) / len(holdout)
    return Holdout(channel, treated_rate, holdout_rate,
                   len(treated), len(holdout), age_days)
