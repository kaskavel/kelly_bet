#!/usr/bin/env python3
"""
Tests for volatility-scaled barrier placement.

Fixed +5%/-3% barriers were the wrong shape for an 860-asset universe. Measured on
the cached history, a 15-day time barrier left 46.8% of bets reaching neither level,
and one forex position stayed open 4.6 months because it could not plausibly travel
3%. Scaling the barriers to each asset's own volatility makes the bets comparable
events that resolve on a similar clock.

The tests below pin the three properties that matter:

  * the payoff ratio -- and therefore the barrier geometry the model trains against
    -- is identical on every asset;
  * barrier width tracks the asset's volatility;
  * fees do NOT scale, so break-even rises on quiet assets, and those assets are
    refused rather than silently traded at a 60% break-even.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.trading.barriers import BarrierPolicy


def price_frame(daily_vol: float, n: int = 300, seed: int = 4) -> pd.DataFrame:
    """A synthetic OHLCV series with a known daily volatility."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, daily_vol, n)))
    return pd.DataFrame({
        'Open': close,
        'High': close * (1 + abs(rng.normal(0, daily_vol / 3, n))),
        'Low': close * (1 - abs(rng.normal(0, daily_vol / 3, n))),
        'Close': close,
        'Volume': np.full(n, 1_000_000.0),
    })


def config(**barrier_overrides) -> dict:
    barrier = {
        'mode': 'volatility',
        'win_sigma': 1.5,
        'loss_sigma': 1.0,
        'horizon_days': 10,
        'vol_lookback': 20,
        'max_barrier_pct': 40.0,
        'max_break_even_pct': 47.0,
    }
    barrier.update(barrier_overrides)
    return {'trading': {
        'win_threshold': 5.0,
        'loss_threshold': 3.0,
        'trading_fee_percentage': 0.25,
        'barrier': barrier,
    }}


class GeometryTestCase(unittest.TestCase):

    def setUp(self):
        self.policy = BarrierPolicy(config())

    def test_geometry_probability_is_independent_of_asset(self):
        """
        The driftless win probability is m/(k+m) on every asset.

        This is what makes one pooled model legitimate: the event has the same shape
        everywhere, only the distance scales.
        """
        self.assertAlmostEqual(self.policy.geometry_probability, 1.0 / 2.5, places=9)

        for daily_vol in (0.003, 0.010, 0.018, 0.045, 0.090):
            spec = self.policy.for_series(price_frame(daily_vol))
            self.assertIsNotNone(spec)
            self.assertAlmostEqual(spec.win_pct / spec.loss_pct,
                                   self.policy.payoff_ratio, places=6)

    def test_geometry_is_not_fifty_percent(self):
        """Guards against the recurring assumption that 50% is neutral."""
        self.assertNotAlmostEqual(self.policy.geometry_probability, 0.5, places=2)
        self.assertAlmostEqual(self.policy.geometry_probability, 0.40, places=6)

    def test_barrier_width_tracks_volatility(self):
        """A more volatile asset gets proportionally wider barriers."""
        widths = []
        for daily_vol in (0.005, 0.010, 0.020, 0.040):
            spec = self.policy.for_series(price_frame(daily_vol))
            widths.append(spec.win_pct)

        for narrower, wider in zip(widths, widths[1:]):
            self.assertLess(narrower, wider)

    def test_sigma_scales_with_square_root_of_horizon(self):
        """sigma_h = sigma_daily * sqrt(h): quadrupling h doubles the barrier."""
        frame = price_frame(0.018)
        ten = BarrierPolicy(config(horizon_days=10)).for_series(frame)
        forty = BarrierPolicy(config(horizon_days=40)).for_series(frame)

        self.assertAlmostEqual(forty.win_pct / ten.win_pct, 2.0, delta=0.02)

    def test_extreme_volatility_is_capped_without_distorting_the_ratio(self):
        """The cap must preserve the payoff ratio, or the geometry shifts."""
        policy = BarrierPolicy(config(max_barrier_pct=10.0))
        spec = policy.for_series(price_frame(0.090))

        self.assertLessEqual(spec.win_pct, 10.0 + 1e-9)
        self.assertAlmostEqual(spec.win_pct / spec.loss_pct, policy.payoff_ratio, places=6)


class EconomicFilterTestCase(unittest.TestCase):

    def setUp(self):
        self.policy = BarrierPolicy(config())

    def test_break_even_rises_as_volatility_falls(self):
        """
        Fees are fixed while barriers scale, so quiet assets need a higher win rate.

        This is the arithmetic that should have excluded the forex book.
        """
        break_evens = []
        for daily_vol in (0.040, 0.018, 0.010, 0.003):
            spec = self.policy.for_series(price_frame(daily_vol))
            break_evens.append(self.policy.break_even(spec.win_pct, spec.loss_pct))

        for lower, higher in zip(break_evens, break_evens[1:]):
            self.assertLess(lower, higher)

    def test_quiet_assets_are_refused(self):
        """A currency-pair-like series is uneconomic and must be rejected."""
        spec = self.policy.for_series(price_frame(0.003))
        self.assertFalse(self.policy.is_economic(spec))
        self.assertIn("break-even", self.policy.rejection_reason(spec))

    def test_normal_and_high_volatility_assets_are_accepted(self):
        for daily_vol in (0.015, 0.018, 0.035, 0.045):
            spec = self.policy.for_series(price_frame(daily_vol))
            self.assertTrue(self.policy.is_economic(spec),
                            f"daily vol {daily_vol:.1%} was rejected")
            self.assertIsNone(self.policy.rejection_reason(spec))

    def test_no_floor_inflates_a_quiet_asset_into_acceptance(self):
        """
        There must be no minimum barrier width.

        An earlier version widened quiet assets up to a 1% floor, which rescued
        exactly the assets the fee arithmetic is meant to exclude and meant the
        barrier was no longer the requested multiple of sigma.
        """
        spec = self.policy.for_series(price_frame(0.0025))
        sigma = spec.sigma_pct

        self.assertAlmostEqual(spec.win_pct, 1.5 * sigma, places=6)
        self.assertAlmostEqual(spec.loss_pct, 1.0 * sigma, places=6)
        self.assertFalse(self.policy.is_economic(spec))

    def test_win_barrier_below_the_fee_is_refused(self):
        """A profit target smaller than the round trip can never be favourable."""
        policy = BarrierPolicy(config(win_sigma=0.05, loss_sigma=0.05))
        spec = policy.for_series(price_frame(0.010))
        self.assertFalse(policy.is_economic(spec))


class BarrierSeriesTestCase(unittest.TestCase):

    def setUp(self):
        self.policy = BarrierPolicy(config())

    def test_series_matches_the_pointwise_spec(self):
        """barrier_series() and for_series() must agree on the final bar."""
        frame = price_frame(0.018)
        win, loss = self.policy.barrier_series(frame)
        spec = self.policy.for_series(frame)

        self.assertAlmostEqual(win.iloc[-1], spec.win_pct, places=9)
        self.assertAlmostEqual(loss.iloc[-1], spec.loss_pct, places=9)

    def test_warmup_bars_are_nan(self):
        """
        No barrier before there is enough history to estimate volatility.

        Warm-up is exactly `vol_lookback` bars. The robust estimator centres the MAD
        on zero rather than on a rolling median precisely so it stays one window deep
        -- stacking two windows would double the warm-up.
        """
        frame = price_frame(0.018)
        win, loss = self.policy.barrier_series(frame)

        lookback = self.policy.vol_lookback
        self.assertTrue(win.iloc[:lookback].isna().all())
        self.assertTrue(win.iloc[lookback:].notna().any())

    def test_post_warmup_nans_are_only_uneconomic_bars(self):
        """
        After warm-up, a NaN barrier means the bar was refused as uneconomic.

        Volatility drifts, so a normally tradeable asset can pass through quiet
        stretches where its barriers no longer clear the fee. Those bars must be
        dropped -- but for that reason and no other.
        """
        frame = price_frame(0.018)
        win, loss = self.policy.barrier_series(frame)
        sigma = self.policy.realised_sigma_series(frame['Close'])

        lookback = self.policy.vol_lookback
        for index in win.index[lookback:]:
            if win.loc[index] == win.loc[index]:      # not NaN
                continue
            # Every remaining NaN must be explained by the economic ceiling.
            bar_sigma = sigma.loc[index]
            self.assertEqual(bar_sigma, bar_sigma, "sigma unexpectedly NaN")
            break_even = self.policy.break_even(
                self.policy.win_sigma * bar_sigma,
                self.policy.loss_sigma * bar_sigma) * 100.0
            self.assertGreater(break_even, self.policy.max_break_even_pct,
                               f"bar {index} dropped but was economic")

    def test_robust_sigma_ignores_a_corrupted_bar(self):
        """
        One fake gap must not blow up the barrier.

        The price cache mixes adjusted with unadjusted prices, producing false jumps
        such as AVB's -61% bar. Under a standard-deviation estimate that single
        outlier pushed AVB's sigma to 67.7% and its barrier to the 40% cap, which
        would have sized a real bet against a nonsense target.
        """
        import numpy as np

        frame = price_frame(0.018, n=400, seed=12)
        clean = BarrierPolicy(config()).for_series(frame)

        # The fake gap has to fall INSIDE the trailing lookback window, or neither
        # estimator sees it: rescaling a whole tail leaves the returns within that
        # tail unchanged. Corrupting the last 5 bars puts one -61% return inside the
        # final 20-bar window.
        corrupted = frame.copy()
        for column in ('Open', 'High', 'Low', 'Close'):
            corrupted.loc[corrupted.index[-5:], column] *= 0.39   # a false -61% gap

        robust = BarrierPolicy(config(robust_sigma=True)).for_series(corrupted)
        naive = BarrierPolicy(config(robust_sigma=False)).for_series(corrupted)

        # The robust estimate barely moves; the SD estimate explodes to the cap.
        self.assertLess(abs(robust.sigma_pct - clean.sigma_pct), 2.0)
        self.assertGreater(naive.sigma_pct, robust.sigma_pct * 3)
        self.assertAlmostEqual(naive.win_pct, self.policy.max_barrier_pct, places=6)

    def test_robust_sigma_still_reads_volatile_assets_as_volatile(self):
        """Robustness must not flatten genuinely high-volatility assets."""
        quiet = BarrierPolicy(config()).for_series(price_frame(0.008, seed=21))
        wild = BarrierPolicy(config()).for_series(price_frame(0.060, seed=21))
        self.assertGreater(wild.sigma_pct, quiet.sigma_pct * 4)

    def test_robust_sigma_agrees_with_sd_on_clean_data(self):
        """On clean normal returns the MAD estimator matches the SD within ~10%."""
        frame = price_frame(0.02, n=500, seed=33)
        robust = BarrierPolicy(config(robust_sigma=True)).realised_sigma_series(frame['Close'])
        naive = BarrierPolicy(config(robust_sigma=False)).realised_sigma_series(frame['Close'])
        ratio = robust.dropna().mean() / naive.dropna().mean()
        self.assertAlmostEqual(ratio, 1.0, delta=0.12)

    def test_uneconomic_bars_are_masked_out(self):
        """
        Bars whose barriers cannot cover fees are not labelled.

        A bet that would never be placed is not evidence about the strategy, so it
        must not enter the training set or the backtest.
        """
        win, loss = self.policy.barrier_series(price_frame(0.002))
        self.assertTrue(win.isna().all())

    def test_volatility_estimate_uses_only_past_bars(self):
        """
        The barrier at bar i must not depend on anything after bar i.

        Truncating the series after bar i must leave that bar's barrier unchanged.
        """
        frame = price_frame(0.018, n=200)
        full_win, _ = self.policy.barrier_series(frame)

        for i in (60, 120, 180):
            truncated_win, _ = self.policy.barrier_series(frame.iloc[:i + 1])
            self.assertAlmostEqual(truncated_win.iloc[-1], full_win.iloc[i], places=9,
                                   msg=f"barrier at bar {i} changed when future bars were removed")


class FixedModeTestCase(unittest.TestCase):

    def test_fixed_mode_still_works(self):
        """Fixed barriers remain available and unchanged."""
        policy = BarrierPolicy({'trading': {
            'win_threshold': 5.0, 'loss_threshold': 3.0,
            'trading_fee_percentage': 0.25,
            'barrier': {'mode': 'fixed'},
        }})

        self.assertFalse(policy.is_volatility_scaled)
        spec = policy.for_series(price_frame(0.018))
        self.assertEqual(spec.win_pct, 5.0)
        self.assertEqual(spec.loss_pct, 3.0)
        self.assertAlmostEqual(policy.geometry_probability, 0.375, places=6)
        self.assertAlmostEqual(policy.break_even(5.0, 3.0), 0.4375, places=6)

    def test_fixed_mode_needs_no_price_history(self):
        """Fixed barriers do not depend on a volatility estimate."""
        policy = BarrierPolicy({'trading': {
            'win_threshold': 5.0, 'loss_threshold': 3.0,
            'barrier': {'mode': 'fixed'},
        }})
        spec = policy.for_series(pd.DataFrame({'Close': [100.0, 101.0]}))
        self.assertIsNotNone(spec)


class LabellingIntegrationTestCase(unittest.TestCase):
    """The labeller must accept per-bar barriers, not just scalars."""

    def setUp(self):
        from src.prediction.algorithms.base_algorithm import BasePredictionAlgorithm

        class Probe(BasePredictionAlgorithm):
            async def predict(self, data):
                return None

            async def train(self, data, target_data=None):
                return None

        self.probe = Probe('probe', {})
        self.policy = BarrierPolicy(config())

    def test_per_bar_barriers_produce_labels(self):
        frame = price_frame(0.018, n=300)
        win, loss = self.policy.barrier_series(frame)

        outcomes = self.probe.barrier_outcomes(frame, win, loss, max_hold=30)
        labelled = outcomes.dropna(subset=['label'])

        self.assertGreater(len(labelled), 50)
        self.assertTrue(set(labelled['label'].unique()).issubset({-1.0, 0.0, 1.0}))
        # Each labelled bar records the barriers it actually used.
        self.assertTrue(labelled['win_pct'].notna().all())

    def test_win_rate_approaches_the_geometry_on_a_driftless_walk(self):
        """
        On a driftless walk the win rate sits near the barrier geometry, ~40%.

        Averaged over INDEPENDENT paths. Barrier outcomes within a single path are
        strongly correlated because their forward windows overlap, so one path
        estimates the rate to only about +/-5 percentage points -- wide enough to be
        mistaken for a real edge, which is why the reference is measured rather than
        read off a closed form.
        """
        rates = []
        for seed in range(12):
            frame = price_frame(0.02, n=1500, seed=2000 + seed)
            win, loss = self.policy.barrier_series(frame)
            labels = self.probe._barrier_labels(frame, win, loss, max_hold=150).dropna()
            if len(labels):
                rates.append(labels.mean())

        self.assertGreaterEqual(len(rates), 10)
        mean_rate = float(np.mean(rates))

        # Far from 50%, which is the misreading the whole design has to avoid.
        self.assertLess(mean_rate, 0.47)
        # And in the neighbourhood of the geometry approximation.
        self.assertAlmostEqual(mean_rate, self.policy.geometry_probability, delta=0.06)

    def test_geometry_estimate_is_invariant_to_asset_volatility(self):
        """
        The measured baseline must not depend on the asset's volatility.

        That invariance is the whole reason for scaling barriers to sigma: it is what
        makes one pooled model legitimate across the universe.
        """
        estimates = [
            self.policy.estimate_geometry_probability(daily_vol=vol, paths=10, bars=1200)
            for vol in (0.010, 0.020, 0.040)
        ]

        means = [e['mean'] for e in estimates]
        self.assertLess(max(means) - min(means), 0.06,
                        f"baseline moved with volatility: {means}")

    def test_short_time_barrier_truncates_wins_preferentially(self):
        """
        A tight time barrier cuts off wins more than losses.

        The win barrier sits 1.5 sigma out and the stop 1.0 sigma, so expected
        time-to-touch is ~2.25x longer for a win. A short limit therefore depresses
        the win rate below the barrier geometry -- which is a property of the clock,
        not of the model, and was worth 8 points of win rate on real data.
        """
        frame = price_frame(0.02, n=2500, seed=31)
        win, loss = self.policy.barrier_series(frame)

        rates = {}
        timeouts = {}
        for max_hold in (20, 60, 150):
            outcomes = self.probe.barrier_outcomes(
                frame, win, loss, max_hold).dropna(subset=['label'])
            rates[max_hold] = (outcomes['label'] == 1).mean()
            timeouts[max_hold] = (outcomes['label'] == -1).mean()

        # Timeouts fall as the limit lengthens...
        self.assertGreater(timeouts[20], timeouts[60])
        self.assertGreater(timeouts[60], timeouts[150])
        # ...and the win rate rises towards the geometry as truncation disappears.
        self.assertLess(rates[20], rates[150])

    def test_measured_baseline_is_near_the_approximation(self):
        """The closed form is documented as approximate; check it is not misleading."""
        estimate = self.policy.estimate_geometry_probability(paths=12, bars=1500)
        self.assertAlmostEqual(estimate['mean'], self.policy.geometry_probability,
                               delta=0.05)


if __name__ == '__main__':
    unittest.main()


class MinorUnitCurrencyTestCase(unittest.TestCase):
    """
    The London Stock Exchange quotes in PENCE, not pounds.

    CRH.L trades at 8,418 meaning GBp 8,418 = GBP 84.18. Treating that as pounds
    reported the price as $11,402 instead of $114.02 -- a 100x error across all 73
    `.L` assets in this universe.
    """

    def setUp(self):
        import pandas as pd
        from src.utils.currency_converter import CurrencyConverter
        self.converter = CurrencyConverter()
        self.converter.update_rates({
            'GBPUSD=X': pd.DataFrame({'Close': [1.3545]}),
            'USDJPY=X': pd.DataFrame({'Close': [153.5]}),
        })

    def test_lse_symbols_are_priced_in_pence(self):
        self.assertEqual(self.converter.currency_for_symbol('CRH.L'), 'GBX')
        self.assertEqual(self.converter.currency_for_symbol('LLOY.L'), 'GBX')

    def test_gbx_is_one_hundredth_of_gbp(self):
        self.assertAlmostEqual(self.converter.get_rate('GBX'),
                               self.converter.get_rate('GBP') / 100.0, places=12)

    def test_real_lse_prices_convert_plausibly(self):
        """Actual cached closes must land in a believable share-price range."""
        for raw, symbol in ((8418.00, 'CRH.L'), (11650.00, 'AZN.L'),
                            (557.10, 'BP.L'), (151.25, 'CNA.L')):
            usd = self.converter.convert_to_usd(
                raw, self.converter.currency_for_symbol(symbol))
            self.assertLess(usd, 500.0, f"{symbol} converted to an implausible ${usd:,.2f}")
            self.assertGreater(usd, 0.5)

    def test_crh_converts_to_the_expected_figure(self):
        usd = self.converter.convert_to_usd(8418.00, 'GBX')
        self.assertAlmostEqual(usd, 114.02, delta=0.05)

    def test_non_lse_currencies_are_unaffected(self):
        self.assertEqual(self.converter.currency_for_symbol('8053.T'), 'JPY')
        self.assertAlmostEqual(self.converter.convert_to_usd(1840.0, 'JPY'),
                               1840.0 / 153.5, places=6)
        self.assertEqual(self.converter.currency_for_symbol('AAPL'), 'USD')
        self.assertEqual(self.converter.convert_to_usd(315.34, 'USD'), 315.34)

    def test_asset_selector_labels_uk_stocks_in_pence(self):
        """The metadata source must agree with the feed's units."""
        import asyncio
        import yaml
        from src.utils.asset_selector import AssetSelector

        config = yaml.safe_load(open('config/config.yaml'))
        assets = asyncio.run(AssetSelector(config).get_all_assets())
        uk = [a for a in assets if a['symbol'].endswith('.L')]
        self.assertGreater(len(uk), 20)
        self.assertTrue(all(a['currency'] == 'GBX' for a in uk),
                        "UK listings must be labelled GBX, not GBP")
