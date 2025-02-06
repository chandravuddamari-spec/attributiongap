"""Attribution models, and the bias factors that reconcile them with truth.

Why this module exists.

The attribution tool says paid search drove 4,200 conversions. The holdout
experiment says its incremental contribution was 1,100. Both numbers appear in
the same deck, quoted by different teams, and the organisation never resolves
which to budget against - so it budgets against the larger one.

A Markov removal-effect model is a better attribution model than last-click,
and it is still an attribution model: it distributes observed credit among
observed touchpoints, which is a different quantity from incrementality and no
amount of modelling closes that gap.

So this does not stop at building the model. It SCORES the model against
experimental truth and publishes a per-channel BIAS FACTOR, which is then used
to correct the fast-moving daily numbers between infrequent experiments - with
the factor's age carried alongside it, because a bias factor goes stale and a
stale correction applied silently is its own problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .generator import CHANNELS, Holdout, Journey


# ---------------------------------------------------------------------------
# Attribution models
# ---------------------------------------------------------------------------

def last_click(journeys: list[Journey]) -> dict[str, float]:
    credit = {c: 0.0 for c in CHANNELS}
    for j in journeys:
        if j.converted and j.path:
            credit[j.path[-1]] += 1.0
    return credit


def linear(journeys: list[Journey]) -> dict[str, float]:
    credit = {c: 0.0 for c in CHANNELS}
    for j in journeys:
        if not j.converted or not j.path:
            continue
        share = 1.0 / len(j.path)
        for c in j.path:
            credit[c] += share
    return credit


def _transition_matrix(
    journeys: list[Journey], exclude: str | None = None,
) -> tuple[dict[str, dict[str, float]], set[str]]:
    """Build transition counts over START -> channels -> CONVERT/NULL.

    When `exclude` is given the channel becomes an ABSORBING NULL state:
    every transition into it goes to NULL instead, so any path that passed
    through it cannot reach CONVERT.

    This is the definition, and getting it wrong is easy. Simply deleting the
    channel from each path leaves the journey converting exactly as before, so
    every removal effect comes out at zero and the model silently reports that
    no channel does anything. The arithmetic never complains.
    """
    counts: dict[str, dict[str, float]] = {}
    states: set[str] = {"START", "CONVERT", "NULL"}

    def bump(a: str, b: str) -> None:
        counts.setdefault(a, {})
        counts[a][b] = counts[a].get(b, 0.0) + 1.0
        states.add(a)
        states.add(b)

    for j in journeys:
        path = list(j.path)
        if not path:
            bump("START", "CONVERT" if j.converted else "NULL")
            continue

        if exclude is not None and exclude in path:
            # Absorb into NULL at the point the excluded channel appears.
            cut = path.index(exclude)
            prefix = path[:cut]
            if not prefix:
                bump("START", "NULL")
            else:
                bump("START", prefix[0])
                for a, b in zip(prefix, prefix[1:]):
                    bump(a, b)
                bump(prefix[-1], "NULL")
            continue

        bump("START", path[0])
        for a, b in zip(path, path[1:]):
            bump(a, b)
        bump(path[-1], "CONVERT" if j.converted else "NULL")

    probabilities: dict[str, dict[str, float]] = {}
    for state, row in counts.items():
        total = sum(row.values())
        probabilities[state] = {k: v / total for k, v in row.items()}
    return probabilities, states


def _conversion_probability(
    matrix: dict[str, dict[str, float]], states: set[str],
    iterations: int = 200,
) -> float:
    """Probability of reaching CONVERT from START, by value iteration."""
    value = {s: 0.0 for s in states}
    value["CONVERT"] = 1.0
    value["NULL"] = 0.0

    for _ in range(iterations):
        updated = dict(value)
        for state in states:
            if state in ("CONVERT", "NULL"):
                continue
            row = matrix.get(state)
            if not row:
                updated[state] = 0.0
                continue
            updated[state] = sum(p * value[nxt] for nxt, p in row.items())
        if all(abs(updated[s] - value[s]) < 1e-12 for s in states):
            value = updated
            break
        value = updated
    return value["START"]


def markov_removal_effect(journeys: list[Journey]) -> dict[str, float]:
    """Credit proportional to the drop in conversion when a channel is removed.

    A genuine improvement on last-click: it accounts for position and for
    paths that would still convert without the channel. It is still measuring
    observed credit, not causation - and tests/test_attribution.py shows exactly
    how far that falls short.
    """
    base_matrix, base_states = _transition_matrix(journeys)
    base_probability = _conversion_probability(base_matrix, base_states)
    conversions = sum(1 for j in journeys if j.converted)

    removal: dict[str, float] = {}
    for channel in CHANNELS:
        matrix, states = _transition_matrix(journeys, exclude=channel)
        probability = _conversion_probability(matrix, states)
        removal[channel] = max(0.0, base_probability - probability)

    total = sum(removal.values())
    if total <= 0:
        return {c: 0.0 for c in CHANNELS}
    return {c: conversions * v / total for c, v in removal.items()}


# ---------------------------------------------------------------------------
# Bias factors
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BiasFactor:
    channel: str
    attributed: float
    incremental: float
    age_days: int
    #: Half-life in days for how fast a bias factor loses relevance.
    half_life_days: int = 90

    @property
    def factor(self) -> float:
        """attributed / incremental. Above 1 means over-credited."""
        if self.incremental <= 0:
            return math.inf
        return self.attributed / self.incremental

    @property
    def confidence(self) -> float:
        """Decays with age. A factor measured a year ago is not evidence."""
        return 0.5 ** (self.age_days / self.half_life_days)

    @property
    def stale(self) -> bool:
        return self.confidence < 0.25

    def corrected(self, attributed_today: float) -> float:
        """Apply the correction, shrunk toward no-correction as it ages.

        Applying a year-old factor at full strength is its own failure mode,
        and silently continuing to apply it is worse than not having it.
        """
        if self.incremental <= 0:
            return 0.0
        raw = attributed_today / self.factor
        weight = self.confidence
        return weight * raw + (1 - weight) * attributed_today


def compute_bias_factors(
    attributed: dict[str, float], holdouts: dict[str, Holdout],
    population: int,
) -> dict[str, BiasFactor]:
    out: dict[str, BiasFactor] = {}
    for channel, holdout in holdouts.items():
        out[channel] = BiasFactor(
            channel=channel,
            attributed=attributed.get(channel, 0.0),
            incremental=max(0.0, holdout.incremental_conversions(population)),
            age_days=holdout.age_days,
        )
    return out


def trustworthiness(factors: dict[str, BiasFactor]) -> list[tuple[str, float]]:
    """Where the next experiment should run.

    Ranked by how far the attribution is from truth, weighted by how stale the
    evidence is. A channel with a large, old bias factor is the one costing
    the most money per week of not knowing.
    """
    scored: list[tuple[str, float]] = []
    for channel, f in factors.items():
        if not math.isfinite(f.factor):
            scored.append((channel, math.inf))
            continue
        distance = abs(math.log(max(f.factor, 1e-6)))
        scored.append((channel, distance * (1 - f.confidence + 0.2)))
    return sorted(scored, key=lambda kv: -kv[1])
