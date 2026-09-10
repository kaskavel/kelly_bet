"""
Barrier policy: how far away the win and loss levels sit for a given asset.

Fixed percentage barriers were the wrong shape for a multi-asset universe. A +5%/-3%
bet means something completely different on RIVN than on EURGBP: measured on the
cached history, a 15-day time barrier left 46.8% of bets reaching neither level, and
one forex position sat open for 4.6 months because it could not plausibly travel 3%.
Sizing, labelling and the model were all being applied to events that were not
comparable to one another.

Volatility-scaled barriers fix that. With

    win  = +k * sigma_h
    loss = -m * sigma_h

where sigma_h is the asset's own realised volatility over the intended holding
horizon, the expected time-to-touch is comparable across the universe, so:

  * one pooled model is predicting the same shape of event on every asset;
  * the driftless barrier probability stays put regardless of asset (~40% for
    the configured 1.5/1.0, verified invariant to volatility by simulation);
  * bets resolve on a similar clock, which is what makes a time barrier meaningful.

The one identity that governs this whole design
------------------------------------------------
For a driftless asset the barrier geometry probability is l/(w+l) -- and the PRE-FEE
break-even probability is also l/(w+l). They are the same number. So:

    **No choice of barrier shape creates expected profit.**

Fixed or volatility-scaled, any k/m ratio, any width: on a driftless asset the
probability you get is exactly the probability you need. Changing the geometry moves
both sides together. Everything beyond that has to come from genuine predictive edge.

What barrier width DOES change is the fee drag, which is the only structural term:

    required edge = break_even - geometry = 2c / (w + l)

That is a real, measurable improvement and it is why scaling to sigma is worth doing:

        barriers                geometry   break-even   edge needed
        fixed 5% / 3%             37.50%      43.75%       6.25 pts
        volatility (mean 9.2/6.1) 40.01%      43.28%       3.26 pts
        volatility, high-vol      40.00%      42.00%       2.00 pts
        volatility, forex         40.00%      60.83%      20.83 pts

Widening the barriers roughly halves the edge the model must supply, and it makes
explicit that a quiet currency pair would need 20+ points -- which is why
`is_economic()` refuses it rather than trading at a 60% break-even.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ArrayLike = Union[float, pd.Series, np.ndarray]


def _simulate_barrier_rate(close: np.ndarray, win_pct: float, loss_pct: float,
                           max_hold: int) -> Optional[float]:
    """
    Fraction of bars whose bet touches the win barrier first, on one price path.

    Close-only touch detection, which is what the barrier labeller falls back to when
    no intraday High/Low is available. Bars without a full forward window are skipped.
    """
    n = len(close)
    upper_mult = 1.0 + win_pct / 100.0
    lower_mult = 1.0 - loss_pct / 100.0

    wins = 0
    total = 0

    for i in range(n - max_hold - 1):
        entry = close[i]
        upper = entry * upper_mult
        lower = entry * lower_mult
        window = close[i + 1:i + 1 + max_hold]

        up_hit = np.flatnonzero(window >= upper)
        down_hit = np.flatnonzero(window <= lower)

        first_up = up_hit[0] if up_hit.size else np.inf
        first_down = down_hit[0] if down_hit.size else np.inf

        total += 1
        if first_up < first_down:
            wins += 1

    return wins / total if total else None


@dataclass
class BarrierSpec:
    """Concrete barrier levels for one bet, as positive percentages."""
    win_pct: float
    loss_pct: float
    mode: str
    sigma_pct: Optional[float] = None   # sigma_h used, in percent
    horizon_days: Optional[int] = None

    def __str__(self) -> str:
        if self.sigma_pct is not None:
            return (f"+{self.win_pct:.2f}%/-{self.loss_pct:.2f}% "
                    f"(sigma_{self.horizon_days}d = {self.sigma_pct:.2f}%)")
        return f"+{self.win_pct:.2f}%/-{self.loss_pct:.2f}% (fixed)"


class BarrierPolicy:
    """
    Single source of truth for barrier placement.

    Every consumer -- the training labels, the Kelly calculator, bet placement and
    the monitoring loop -- must derive barriers from here, or they will silently
    disagree about what bet is being made.
    """

    def __init__(self, config: Dict):
        trading = config.get('trading', {})
        barrier = trading.get('barrier', {}) or {}

        self.mode = barrier.get('mode', 'fixed')

        # Fixed mode
        self.fixed_win_pct = float(trading.get('win_threshold', 5.0))
        self.fixed_loss_pct = float(trading.get('loss_threshold', 3.0))

        # Volatility mode
        self.win_sigma = float(barrier.get('win_sigma', 1.5))
        self.loss_sigma = float(barrier.get('loss_sigma', 1.0))
        self.horizon_days = int(barrier.get('horizon_days', 10))
        self.vol_lookback = int(barrier.get('vol_lookback', 20))
        # Estimate sigma from the median absolute deviation instead of the standard
        # deviation, so a single corrupted bar cannot blow up the barrier. See
        # realised_sigma_series() for why this is not optional on this dataset.
        self.robust_sigma = bool(barrier.get('robust_sigma', True))

        # Ceiling for extreme volatility. There is deliberately NO floor that widens
        # a quiet asset's barriers: inflating them up to a minimum would rescue
        # exactly the assets the fee arithmetic is supposed to exclude, and the
        # barrier would no longer be the requested multiple of sigma.
        self.max_barrier_pct = float(barrier.get('max_barrier_pct', 40.0))

        # Fee per side, as a decimal. Needed to judge whether a barrier is economic.
        self.fee_rate = float(trading.get('trading_fee_percentage', 0.25)) / 100.0

        # Reject a bet whose fee-adjusted break-even exceeds this. The driftless
        # geometry probability is m/(k+m), so the gap between that and this ceiling
        # is how much genuine edge the model would have to supply. At 40% geometry
        # and a 47% ceiling, that is at most 7 points.
        self.max_break_even_pct = float(barrier.get('max_break_even_pct', 47.0))

    # ------------------------------------------------------------------ helpers

    @property
    def is_volatility_scaled(self) -> bool:
        return self.mode == 'volatility'

    @property
    def payoff_ratio(self) -> float:
        """win/loss ratio, constant across assets in volatility mode."""
        if self.is_volatility_scaled:
            return self.win_sigma / self.loss_sigma
        return self.fixed_win_pct / self.fixed_loss_pct

    @property
    def geometry_probability(self) -> float:
        """
        APPROXIMATE P(win barrier first) for a driftless asset: m/(k+m).

        The point this number makes is the important one: the neutral reference is
        around 40%, not 50%, so a "60% probability" was never the edge it looked
        like. But treat the exact value as an approximation, not a constant.

        The clean l/(w+l) result is for a continuous driftless ARITHMETIC walk that
        lands exactly on its barriers. Real bars are discrete and prices compound, so
        the true value is a couple of points away and depends on the step size
        relative to the barrier width. Measured over 24 independent simulated paths:

            barriers     l/(w+l)   log-space   measured
            5% / 3%       0.375      0.384      0.400 +/- 0.007
            6% / 4%       0.400      0.412      0.414 +/- 0.008
            3% / 2%       0.400      0.406      0.434 +/- 0.005
            20% / 10%     0.333      0.366      0.344 +/- 0.017

        Neither closed form wins, so where the reference actually matters use
        `estimate_geometry_probability()`, which measures it, or read the base rate
        the backtest reports from real data.
        """
        if self.is_volatility_scaled:
            return self.loss_sigma / (self.win_sigma + self.loss_sigma)
        return self.fixed_loss_pct / (self.fixed_win_pct + self.fixed_loss_pct)

    def estimate_geometry_probability(self, win_pct: Optional[float] = None,
                                      loss_pct: Optional[float] = None,
                                      daily_vol: float = 0.02,
                                      max_hold: int = 250,
                                      paths: int = 24,
                                      bars: int = 2500,
                                      seed: int = 9000) -> Dict[str, float]:
        """
        Measure the driftless win rate for these barriers by simulation.

        Returns {'mean', 'stderr', 'paths'}. Averaged over independent paths because
        barrier outcomes on overlapping windows within one path are strongly
        correlated: a single 2,500-bar path estimates the rate to only about +/-5
        percentage points, which is wide enough to mistake for a real edge.
        """
        if win_pct is None or loss_pct is None:
            if self.is_volatility_scaled:
                sigma_pct = daily_vol * np.sqrt(self.horizon_days) * 100.0
                spec = self._spec_from_sigma(sigma_pct)
                win_pct, loss_pct = spec.win_pct, spec.loss_pct
            else:
                win_pct, loss_pct = self.fixed_win_pct, self.fixed_loss_pct

        rates = []
        for offset in range(paths):
            rng = np.random.default_rng(seed + offset)
            close = 100 * np.exp(np.cumsum(rng.normal(0, daily_vol, bars)))
            rate = _simulate_barrier_rate(close, win_pct, loss_pct, max_hold)
            if rate is not None:
                rates.append(rate)

        if not rates:
            return {'mean': float('nan'), 'stderr': float('nan'), 'paths': 0}

        array = np.asarray(rates)
        stderr = array.std(ddof=1) / np.sqrt(len(array)) if len(array) > 1 else float('nan')
        return {'mean': float(array.mean()), 'stderr': float(stderr), 'paths': len(array)}

    def required_edge(self, win_pct: float, loss_pct: float) -> float:
        """
        Percentage points of edge over the geometry that this bet needs to break even.

        Equals 2c/(w+l): the entire structural cost of the strategy. Wider barriers
        shrink it, which is the measurable benefit of scaling to volatility. A bet
        needing 20 points is not a bet.
        """
        round_trip = 2 * self.fee_rate
        total = (win_pct + loss_pct) / 100.0
        if total <= 0:
            return 100.0
        return (round_trip / total) * 100.0

    def break_even(self, win_pct: float, loss_pct: float) -> float:
        """
        Fee-adjusted break-even probability for a specific barrier pair.

        In volatility mode this varies per asset, because the fee is fixed while the
        barriers scale. That variation is the point: it is what makes a low-volatility
        asset correctly uneconomic.
        """
        round_trip = 2 * self.fee_rate
        win_net = win_pct / 100.0 - round_trip
        loss_net = loss_pct / 100.0 + round_trip
        if win_net <= 0:
            return 1.0
        return loss_net / (win_net + loss_net)

    def is_economic(self, spec: BarrierSpec) -> bool:
        """
        Whether this barrier pair is worth trading at all.

        Volatility scaling will happily place a +1.4% win barrier on a quiet currency
        pair, where a 0.50% round trip consumes a third of the profit and break-even
        climbs past 60%. Such a bet needs 20+ points of edge over the geometry to be
        worth taking, which no signal here supplies -- so it is refused.
        """
        if spec.win_pct <= 2 * self.fee_rate * 100.0:
            return False
        return self.break_even(spec.win_pct, spec.loss_pct) * 100.0 <= self.max_break_even_pct

    def rejection_reason(self, spec: BarrierSpec) -> Optional[str]:
        """Why a barrier pair was refused, for logging."""
        if self.is_economic(spec):
            return None
        break_even_pct = self.break_even(spec.win_pct, spec.loss_pct) * 100.0
        return (f"break-even {break_even_pct:.1f}% exceeds the "
                f"{self.max_break_even_pct:.1f}% ceiling "
                f"({spec} vs {2 * self.fee_rate * 100.0:.2f}% round-trip fee)")

    # -------------------------------------------------------------- computation

    def realised_sigma_pct(self, close: pd.Series) -> Optional[float]:
        """
        Trailing horizon volatility as a percentage, from the last vol_lookback bars.

        Uses only data up to and including the final bar, so it is safe to call at
        decision time.
        """
        series = self.realised_sigma_series(close)
        if series is None or series.empty:
            return None
        value = series.iloc[-1]
        return None if not np.isfinite(value) else float(value)

    def realised_sigma_series(self, close: pd.Series) -> Optional[pd.Series]:
        """
        Per-bar trailing horizon volatility, in percent. ROBUST to bad bars.

        Log returns, rolling volatility over vol_lookback bars, scaled to the holding
        horizon by sqrt(time). The window is strictly backward-looking and the result
        is NOT shifted, so the value at bar i uses returns up to and including bar i --
        exactly what is known when a bet is opened at bar i's close.

        Volatility is estimated from the median absolute deviation rather than the
        standard deviation, because **the price cache contains corrupted bars**. It
        mixes split/dividend-adjusted with unadjusted prices, which manufactures fake
        gaps: AVB jumps from $168.50 (2026-04-08) to $65.08 (2026-08-11), a false
        -61% "return", and there are 126 such bars across 75 assets.

        A single outlier like that inflates a 20-bar standard deviation enough to push
        the barrier to its 40% cap, which would place a real bet on a nonsense target.
        MAD ignores a few outliers while still reading genuinely volatile assets as
        volatile, so crypto is not quietly flattened. Scaled by 1.4826 so it equals
        the SD for normally distributed returns.
        """
        try:
            close = pd.Series(close).astype(float)
            if len(close) < self.vol_lookback + 2:
                return None

            log_returns = np.log(close / close.shift(1))

            if self.robust_sigma:
                # MAD about ZERO, not about a rolling median. Daily returns are
                # near-zero-mean over 20 bars, and centring on zero needs a single
                # rolling window -- stacking two windows would double the warm-up to
                # 40 bars, which this sparse cache cannot spare.
                # For zero-mean normal returns median(|r|) = 0.6745*sigma, so the
                # 1/0.6745 = 1.4826 factor makes this agree with the SD on clean data.
                mad = log_returns.abs().rolling(
                    self.vol_lookback, min_periods=self.vol_lookback).median()
                daily_sigma = 1.4826 * mad
            else:
                daily_sigma = log_returns.rolling(
                    self.vol_lookback, min_periods=self.vol_lookback).std()

            horizon_sigma = daily_sigma * np.sqrt(self.horizon_days)
            return horizon_sigma * 100.0

        except Exception as e:
            logger.error(f"Could not compute realised volatility: {e}")
            return None

    def for_series(self, data: pd.DataFrame) -> Optional[BarrierSpec]:
        """
        Barriers for a bet opened at the last bar of `data`.

        Returns None when volatility cannot be estimated, so the caller skips the
        asset rather than falling back to an arbitrary fixed barrier.
        """
        if not self.is_volatility_scaled:
            return BarrierSpec(
                win_pct=self.fixed_win_pct,
                loss_pct=self.fixed_loss_pct,
                mode='fixed',
            )

        if 'Close' not in data:
            return None

        sigma_pct = self.realised_sigma_pct(data['Close'])
        if sigma_pct is None or sigma_pct <= 0:
            return None

        return self._spec_from_sigma(sigma_pct)

    def _spec_from_sigma(self, sigma_pct: float) -> BarrierSpec:
        """Scale the barriers to sigma, capping only the extreme end."""
        win_pct = self.win_sigma * sigma_pct
        loss_pct = self.loss_sigma * sigma_pct

        # Cap the wider leg and rescale the other, so the payoff ratio -- and
        # therefore the barrier geometry the model is trained against -- is preserved.
        widest = max(win_pct, loss_pct)
        if widest > self.max_barrier_pct:
            scale = self.max_barrier_pct / widest
            win_pct *= scale
            loss_pct *= scale

        return BarrierSpec(
            win_pct=win_pct,
            loss_pct=loss_pct,
            mode='volatility',
            sigma_pct=sigma_pct,
            horizon_days=self.horizon_days,
        )

    def barrier_series(self, data: pd.DataFrame):
        """
        Per-bar barrier percentages for labelling a whole history.

        Returns (win_pct, loss_pct) as Series aligned to `data`, or scalars in fixed
        mode. Bars where volatility is unavailable carry NaN and will not be labelled.
        """
        if not self.is_volatility_scaled:
            return self.fixed_win_pct, self.fixed_loss_pct

        sigma = self.realised_sigma_series(data['Close'])
        if sigma is None:
            nan = pd.Series(np.nan, index=data.index)
            return nan, nan

        win = self.win_sigma * sigma
        loss = self.loss_sigma * sigma

        # Vectorised equivalent of the cap in _spec_from_sigma.
        widest = pd.concat([win, loss], axis=1).max(axis=1)
        over = widest > self.max_barrier_pct
        if over.any():
            scale = (self.max_barrier_pct / widest).where(over, 1.0)
            win, loss = win * scale, loss * scale

        # Bars whose barriers are uneconomic are not labelled: a bet that would
        # never be taken is not evidence about the strategy.
        round_trip_pct = 2 * self.fee_rate * 100.0
        win_net = win - round_trip_pct
        loss_net = loss + round_trip_pct
        break_even = (loss_net / (win_net + loss_net)) * 100.0
        uneconomic = (win_net <= 0) | (break_even > self.max_break_even_pct)
        win = win.mask(uneconomic)
        loss = loss.mask(uneconomic)

        return win, loss

    def describe(self) -> str:
        """One-line summary for logs."""
        if self.is_volatility_scaled:
            return (f"volatility barriers: +{self.win_sigma}sigma / -{self.loss_sigma}sigma "
                    f"over {self.horizon_days}d (vol lookback {self.vol_lookback} bars), "
                    f"capped at {self.max_barrier_pct}%, "
                    f"break-even ceiling {self.max_break_even_pct}%; "
                    f"geometry probability ~{self.geometry_probability:.1%} (approx)")
        return (f"fixed barriers: +{self.fixed_win_pct}% / -{self.fixed_loss_pct}%; "
                f"geometry probability ~{self.geometry_probability:.1%} (approx)")
