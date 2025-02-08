"""The demo: two numbers, and the factor that reconciles them.

Run: python -m src.demo
"""

from __future__ import annotations

from .attribution import (
    compute_bias_factors, last_click, linear, markov_removal_effect,
    trustworthiness,
)
from .generator import (
    CHANNELS, TRUE_INCREMENTALITY, run_holdout, simulate_journeys,
    true_incremental_conversions,
)

USERS = 60_000


def main() -> None:
    journeys = simulate_journeys(users=USERS, seed=20260911)
    conversions = sum(1 for j in journeys if j.converted)
    organic = sum(1 for j in journeys if j.would_convert_anyway)

    markov = markov_removal_effect(journeys)
    last = last_click(journeys)
    even = linear(journeys)
    truth = {c: true_incremental_conversions(journeys, c) for c in CHANNELS}

    print("\n  attributiongap: two numbers in the same deck")
    print("  " + "=" * 78)
    print(f"  {USERS:,} users, {conversions:,} conversions, of which "
          f"{organic:,} ({organic / conversions:.0%}) would have")
    print("  happened with no marketing at all - the quantity no attribution")
    print("  model can observe.\n")

    def share(model: dict[str, float], channel: str) -> float:
        total = sum(model.values()) or 1.0
        return model[channel] / total

    print("  SHARE OF CREDIT, BY MODEL")
    print("  " + "-" * 78)
    print(f"  {'channel':<17}{'last-click':>12}{'linear':>10}{'markov':>10}"
          f"{'TRUTH':>10}{'markov bias':>14}")
    for channel in sorted(CHANNELS, key=lambda c: -share(markov, c)):
        m = share(markov, channel)
        t = share(truth, channel)
        bias = m / t if t > 0 else float("inf")
        print(f"  {channel:<17}{share(last, channel):>11.1%}"
              f"{share(even, channel):>10.1%}{m:>10.1%}{t:>10.1%}"
              f"{bias:>13.2f}x")

    print("\n  Two biases, pulling opposite ways:")
    print(f"    brand_search  late toll booth   last-click "
          f"{share(last, 'brand_search'):.0%} vs truth "
          f"{share(truth, 'brand_search'):.0%}")
    print(f"    display       early, ubiquitous markov     "
          f"{share(markov, 'display'):.0%} vs truth "
          f"{share(truth, 'display'):.0%}")
    print("  Markov fixes last-click's late bias and introduces an early,")
    print("  high-frequency one. Neither is measuring incrementality, and")
    print("  neither is a step on the road to it.\n")

    print("  THE HOLDOUT EXPERIMENTS")
    print("  " + "-" * 78)
    tested = ["brand_search", "generic_search", "display", "affiliate"]
    ages = {"brand_search": 400, "generic_search": 12,
            "display": 95, "affiliate": 210}
    holdouts = {
        c: run_holdout(c, users=40_000, seed=3, age_days=ages[c])
        for c in tested
    }
    for channel, h in holdouts.items():
        print(f"    {channel:<17}treated {h.treated_rate:.3%}  "
              f"holdout {h.holdout_rate:.3%}  "
              f"incremental {h.incremental_rate:+.3%}  "
              f"({h.age_days}d old)")

    factors = compute_bias_factors(markov, holdouts, USERS)
    print("\n  BIAS FACTORS (attribution / incrementality)")
    print("  " + "-" * 78)
    print(f"    {'channel':<17}{'attributed':>12}{'incremental':>14}"
          f"{'factor':>9}{'age':>7}{'confidence':>12}{'':>8}")
    for channel, f in factors.items():
        flag = "STALE" if f.stale else ""
        factor = "inf" if f.factor == float("inf") else f"{f.factor:.2f}x"
        print(f"    {channel:<17}{f.attributed:>12,.0f}{f.incremental:>14,.0f}"
              f"{factor:>9}{f.age_days:>6}d{f.confidence:>11.0%}{flag:>8}")

    print("\n  CORRECTED DAILY REPORTING")
    print("  " + "-" * 78)
    print(f"    {'channel':<17}{'raw':>12}{'corrected':>12}{'why':>34}")
    for channel, f in factors.items():
        raw = markov[channel]
        corrected = f.corrected(raw)
        why = ("full correction" if f.confidence > 0.8
               else f"shrunk: evidence is {f.age_days}d old")
        print(f"    {channel:<17}{raw:>12,.0f}{corrected:>12,.0f}{why:>34}")
    print("\n    A stale factor is shrunk toward no-correction rather than")
    print("    applied at full strength. Silently correcting today's numbers")
    print("    with a year-old experiment is its own failure mode.\n")

    print("  WHERE THE NEXT EXPERIMENT SHOULD RUN")
    print("  " + "-" * 78)
    for channel, score in trustworthiness(factors)[:3]:
        f = factors[channel]
        print(f"    {channel:<17}score {score:>6.2f}   "
              f"factor {f.factor:.2f}x, evidence {f.age_days}d old")
    print("    Ranked by how far attribution is from truth, weighted by how")
    print("    stale the evidence is - which is the cost per week of not")
    print("    knowing.\n")


if __name__ == "__main__":
    main()
