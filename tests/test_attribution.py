"""The gap between attributed credit and incrementality, measured."""

from __future__ import annotations

import math
import unittest

from src.attribution import (
    BiasFactor, compute_bias_factors, last_click, linear,
    markov_removal_effect, trustworthiness,
)
from src.generator import (
    CHANNELS, TRUE_INCREMENTALITY, run_holdout, simulate_journeys,
    true_incremental_conversions,
)

USERS = 40_000


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.journeys = simulate_journeys(users=USERS, seed=20260911)
        cls.markov = markov_removal_effect(cls.journeys)
        cls.last = last_click(cls.journeys)
        cls.truth = {
            c: true_incremental_conversions(cls.journeys, c) for c in CHANNELS
        }


class TestModels(Base):
    def test_last_click_credits_only_the_final_touch(self):
        total = sum(self.last.values())
        conversions = sum(1 for j in self.journeys if j.converted and j.path)
        self.assertAlmostEqual(total, conversions, places=6)

    def test_linear_spreads_credit_across_the_path(self):
        spread = linear(self.journeys)
        self.assertGreater(spread["display"], self.last["display"],
                           "an early-funnel channel gets more under linear")

    def test_markov_distributes_the_same_total(self):
        total = sum(self.markov.values())
        conversions = sum(1 for j in self.journeys if j.converted)
        self.assertAlmostEqual(total / conversions, 1.0, delta=0.02)

    def test_MARKOV_TRADES_ONE_BIAS_FOR_ANOTHER(self):
        """The finding that justifies the whole reconciliation step.

        Markov is the standard "better than last-click" answer, and it does
        fix last-click's bias toward the final touch. What is rarely said is
        that it introduces its own: absorbing a channel into NULL kills every
        path that reached it, so a channel appearing EARLY and OFTEN scores
        highly whether or not it causes anything.

        In this fixture `display` is early, ubiquitous and nearly worthless
        (true incrementality 0.01). Last-click barely credits it; Markov
        credits it heavily. Neither model is measuring incrementality, and
        neither is a step on the road to measuring it.
        """
        last_share = self.last["display"] / sum(self.last.values())
        markov_share = self.markov["display"] / sum(self.markov.values())
        true_share = self.truth["display"] / sum(self.truth.values())

        self.assertGreater(
            markov_share, last_share,
            "Markov should credit an early, frequent channel more than "
            "last-click does",
        )
        self.assertGreater(
            markov_share, true_share,
            "and that credit exceeds what the channel actually causes",
        )

    def test_the_two_models_disagree_about_late_touch_channels_too(self):
        # In the other direction: last-click over-credits the final touch.
        last_share = self.last["brand_search"] / sum(self.last.values())
        markov_share = self.markov["brand_search"] / sum(self.markov.values())
        self.assertGreater(
            last_share, markov_share,
            "last-click should over-credit the late toll-booth channel "
            "relative to Markov",
        )

    def test_NEITHER_MODEL_RANKS_THE_CHANNELS_CORRECTLY(self):
        def rank_error(model: dict[str, float]) -> int:
            order = [c for c, _ in sorted(model.items(), key=lambda kv: -kv[1])]
            truth_order = [
                c for c, _ in sorted(self.truth.items(), key=lambda kv: -kv[1])]
            return sum(
                abs(order.index(c) - truth_order.index(c)) for c in CHANNELS)

        # Both are materially wrong. Choosing between them is choosing which
        # bias you prefer, which is why the experimental reconciliation is not
        # optional.
        self.assertGreater(rank_error(self.last), 0)
        self.assertGreater(rank_error(self.markov), 0)


class TestTheGap(Base):
    def test_THE_HEADLINE_attribution_over_credits_the_toll_booths(self):
        """brand_search appears on most converting paths and causes almost
        nothing. Attribution cannot see the difference."""
        attributed_share = (
            self.markov["brand_search"] / sum(self.markov.values()))
        true_share = self.truth["brand_search"] / sum(self.truth.values())
        self.assertGreater(
            attributed_share, true_share,
            f"brand_search attributed {attributed_share:.1%} of credit "
            f"against a true {true_share:.1%}",
        )

    def test_a_genuinely_incremental_channel_is_UNDER_credited(self):
        attributed_share = (
            self.markov["generic_search"] / sum(self.markov.values()))
        true_share = self.truth["generic_search"] / sum(self.truth.values())
        self.assertLess(attributed_share, true_share)

    def test_the_attribution_ranking_differs_from_the_truth_ranking(self):
        attributed = [c for c, _ in
                      sorted(self.markov.items(), key=lambda kv: -kv[1])]
        actual = [c for c, _ in
                  sorted(self.truth.items(), key=lambda kv: -kv[1])]
        self.assertNotEqual(
            attributed[:3], actual[:3],
            "if the top three agreed, there would be nothing to reconcile",
        )

    def test_no_attribution_model_closes_the_gap(self):
        # The point that is worth making out loud: this is not a modelling
        # deficiency to be fixed with a better model.
        for model in (self.last, linear(self.journeys), self.markov):
            share = model["brand_search"] / sum(model.values())
            true_share = self.truth["brand_search"] / sum(self.truth.values())
            self.assertGreater(share, true_share)


class TestHoldouts(unittest.TestCase):
    def test_a_holdout_recovers_a_channel_with_real_incrementality(self):
        holdout = run_holdout("generic_search", users=40_000, seed=3)
        self.assertGreater(
            holdout.incremental_rate, 0.0,
            "suppressing a genuinely incremental channel must reduce "
            "conversions",
        )

    def test_a_holdout_finds_little_for_a_toll_booth_channel(self):
        real = run_holdout("generic_search", users=40_000, seed=3)
        toll = run_holdout("brand_search", users=40_000, seed=3)
        self.assertLess(
            toll.incremental_rate, real.incremental_rate,
            "brand search should show far less incrementality than generic",
        )

    def test_the_experiment_reports_its_sample_sizes(self):
        holdout = run_holdout("display", users=20_000, seed=5)
        self.assertGreater(holdout.n_treated, 1_000)
        self.assertGreater(holdout.n_holdout, 1_000)


class TestBiasFactors(unittest.TestCase):
    def test_an_over_credited_channel_has_a_factor_above_one(self):
        f = BiasFactor("brand_search", attributed=4_200, incremental=1_100,
                       age_days=0)
        self.assertAlmostEqual(f.factor, 4_200 / 1_100, places=6)
        self.assertGreater(f.factor, 1.0)

    def test_the_correction_moves_the_daily_number_toward_truth(self):
        f = BiasFactor("brand_search", attributed=4_200, incremental=1_100,
                       age_days=0)
        corrected = f.corrected(4_200)
        self.assertLess(corrected, 4_200)
        self.assertAlmostEqual(corrected, 1_100, delta=1.0)

    def test_A_STALE_FACTOR_IS_SHRUNK_not_applied_at_full_strength(self):
        """Applying a year-old correction silently is its own failure."""
        fresh = BiasFactor("x", attributed=4_000, incremental=1_000, age_days=0)
        old = BiasFactor("x", attributed=4_000, incremental=1_000, age_days=365)

        self.assertAlmostEqual(fresh.corrected(4_000), 1_000, delta=1.0)
        self.assertGreater(
            old.corrected(4_000), fresh.corrected(4_000),
            "an old factor should be trusted less, not the same",
        )
        self.assertLess(old.corrected(4_000), 4_000,
                        "but it should still carry some weight")

    def test_staleness_is_flagged(self):
        self.assertFalse(
            BiasFactor("x", 100, 50, age_days=30).stale)
        self.assertTrue(
            BiasFactor("x", 100, 50, age_days=300).stale)

    def test_confidence_halves_every_half_life(self):
        f = BiasFactor("x", 100, 50, age_days=90, half_life_days=90)
        self.assertAlmostEqual(f.confidence, 0.5, places=6)

    def test_a_channel_with_zero_incrementality_is_handled(self):
        f = BiasFactor("x", attributed=500, incremental=0.0, age_days=0)
        self.assertEqual(f.factor, math.inf)
        self.assertEqual(f.corrected(500), 0.0)


class TestExperimentPrioritisation(Base):
    def test_it_names_where_the_next_experiment_should_run(self):
        holdouts = {
            "brand_search": run_holdout("brand_search", 20_000, 3, age_days=400),
            "generic_search": run_holdout("generic_search", 20_000, 3, age_days=10),
        }
        factors = compute_bias_factors(self.markov, holdouts, USERS)
        ranked = trustworthiness(factors)
        self.assertEqual(len(ranked), 2)
        # The stale, badly-biased channel should outrank the fresh one.
        self.assertEqual(ranked[0][0], "brand_search")

    def test_every_channel_with_an_experiment_gets_a_factor(self):
        holdouts = {c: run_holdout(c, 10_000, 3) for c in list(CHANNELS)[:3]}
        factors = compute_bias_factors(self.markov, holdouts, USERS)
        self.assertEqual(set(factors), set(holdouts))


if __name__ == "__main__":
    unittest.main()
