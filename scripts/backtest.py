#!/usr/bin/env python3
"""
Walk-forward backtester for the Kelly barrier strategy.

CLAUDE.md documented `scripts/backtest.py` but it never existed, which is why six
months of live paper trading produced 84 observations instead of the tens of
thousands the same code can generate offline. Every threshold in config.yaml was set
by intuition because there was no way to measure one.

What it does
------------
Walks forward through cached daily bars in slices. For each slice it:

  1. Trains the supervised models on data STRICTLY BEFORE the slice, with a purge
     embargo so no training label's forward window reaches into the slice.
  2. Scores every asset on every bar in the slice, out of sample.
  3. Resolves each score against the real first-touch barrier outcome, using
     intraday High/Low, a time barrier, and both legs of the fee.

It emits two things:

  * a calibration dataset  (raw score, realised outcome)  ->  data/backtest_scores.csv
  * a simulated trade log and equity curve                ->  data/backtest_trades.csv

Both are written in chronological order, which is what the calibrator's time-ordered
holdout assumes.

Usage
-----
    python scripts/backtest.py                          # default: 120 assets, 4 folds
    python scripts/backtest.py --max-assets 300 --folds 6
    python scripts/backtest.py --threshold 60 --no-trade-log
"""

import argparse
import asyncio
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.kelly.calculator import KellyCalculator
from src.prediction.predictor import PredictionEngine

logger = logging.getLogger('backtest')


# --------------------------------------------------------------------------- data

def load_history(db_path: Path, max_assets: int, min_bars: int) -> Dict[str, pd.DataFrame]:
    """
    Load cached daily OHLCV per symbol.

    Prices are in each asset's native currency, which is correct here: the barriers
    are percentage moves, so a bet's outcome is currency-invariant. Only position
    SIZING needs USD, and that is handled in the simulation by working in fractions.
    """
    conn = sqlite3.connect(db_path)
    try:
        counts = pd.read_sql_query(
            """
            SELECT a.symbol, a.asset_type, COUNT(*) AS bars
            FROM price_data p JOIN assets a ON a.asset_id = p.asset_id
            GROUP BY a.symbol, a.asset_type
            HAVING bars >= ?
            ORDER BY bars DESC, a.symbol
            """,
            conn, params=(min_bars,),
        )
        if counts.empty:
            return {}

        symbols = counts['symbol'].head(max_assets).tolist()
        placeholders = ','.join('?' * len(symbols))
        frame = pd.read_sql_query(
            f"""
            SELECT a.symbol, p.timestamp, p.open, p.high, p.low, p.close, p.volume
            FROM price_data p JOIN assets a ON a.asset_id = p.asset_id
            WHERE a.symbol IN ({placeholders})
            ORDER BY a.symbol, p.timestamp
            """,
            conn, params=symbols,
        )
    finally:
        conn.close()

    # Normalise to the CALENDAR DAY. Bars are stored with the exchange's own UTC
    # offset (-04:00 for US, +01:00 for London, shifting again with DST), so the same
    # trading day appears as several distinct timestamps: the database holds 1,899
    # distinct timestamps for just 331 calendar dates. Left as-is, a US bar and a
    # London bar from the same session are treated as different dates, which both
    # inflates any date-based index by ~5.7x and breaks cross-asset alignment.
    frame['timestamp'] = pd.to_datetime(
        frame['timestamp'], utc=True, errors='coerce', format='mixed'
    ).dt.tz_localize(None).dt.normalize()
    frame = frame.dropna(subset=['timestamp'])

    history: Dict[str, pd.DataFrame] = {}
    for symbol, group in frame.groupby('symbol', sort=False):
        df = group.drop(columns=['symbol']).copy()
        df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low',
                                'close': 'Close', 'volume': 'Volume'})
        df = df.set_index('timestamp').sort_index()
        df = df[~df.index.duplicated(keep='last')]
        df = df[(df[['Open', 'High', 'Low', 'Close']] > 0).all(axis=1)]
        if len(df) >= min_bars:
            history[symbol] = df

    return history


def fold_boundaries(history: Dict[str, pd.DataFrame], folds: int,
                    warmup_frac: float = 0.5, purge_days: int = 0,
                    min_train_bars: int = 120,
                    min_trainable_assets: int = 40) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Split the date range into `folds` consecutive test windows.

    Placement has to satisfy two competing constraints:

      * the supervised models need `min_train_bars` of history BEFORE the purge gap,
        and the purge widens with the time barrier;
      * every bar kept needs `max_hold` bars of forward data to resolve, so the tail
        of the history is unusable.

    On this cache those constraints nearly collide: the median asset holds 209 bars
    across a 320-trading-day span, so 120 training bars consume most of the series.
    Placing the start too early leaves the models untrained (they then score with
    heuristics only, silently, which makes the output useless for judging the score);
    too late and there is no test window left.

    The resolution is that an asset need not be TRAINED on to be SCORED -- the
    features are scale-free and the model is pooled. So this picks the earliest start
    where `min_trainable_assets` can train, and scores the full universe there.
    """
    all_dates = sorted({ts for df in history.values() for ts in df.index})
    if len(all_dates) < 200:
        return []

    # Calendar days per union-date, used to convert the purge into an index offset.
    span_days = max((all_dates[-1] - all_dates[0]).days, 1)
    dates_per_day = len(all_dates) / span_days
    purge_index = int(np.ceil(purge_days * dates_per_day))

    # Pick the EARLIEST test start at which enough assets can train.
    #
    # The features are scale-free and the model is pooled, so an asset does not need
    # to appear in training to be scored -- `score_fold` trains on whatever has
    # enough history and scores everything. Requiring *every* asset to be trainable
    # (an earlier version of this function did) pushes the test window so late that
    # almost nothing is left: on this cache it collapsed the usable sample from
    # ~12,500 observations to 82, because the median asset holds only 209 bars over a
    # 320-trading-day span and a 30-BAR forward window therefore reaches ~45 calendar
    # days back.
    #
    # So: scan forward for the first start where `min_trainable_assets` have
    # `min_train_bars` before the purge gap, and stop there.
    start_index = None
    for candidate in range(0, len(all_dates) - 20):
        if candidate < purge_index:
            continue
        train_end = all_dates[candidate] - pd.Timedelta(days=purge_days)
        trainable = sum(1 for df in history.values()
                        if len(df[df.index < train_end]) >= min_train_bars)
        if trainable >= min_trainable_assets:
            start_index = candidate
            break

    if start_index is None:
        logger.warning(
            f"No test start leaves {min_trainable_assets} assets with "
            f"{min_train_bars}+ training bars before a {purge_days}-day purge. "
            f"Falling back to a {warmup_frac:.0%} warm-up; the supervised models may "
            f"not train. Reduce --max-hold, or extend the price history."
        )
        start_index = int(len(all_dates) * warmup_frac)

    test_dates = all_dates[start_index:]
    if len(test_dates) < folds * 10:
        folds = max(1, len(test_dates) // 20)

    edges = np.linspace(0, len(test_dates), folds + 1).astype(int)
    return [
        (test_dates[edges[i]], test_dates[edges[i + 1] - 1])
        for i in range(folds)
        if edges[i + 1] - 1 > edges[i]
    ]


# ---------------------------------------------------------------------- scoring

async def score_fold(engine: PredictionEngine, history: Dict[str, pd.DataFrame],
                     train_end: pd.Timestamp, test_start: pd.Timestamp,
                     test_end: pd.Timestamp, win_pct: float, loss_pct: float,
                     max_hold: int, stride: int) -> List[Dict]:
    """
    Train on data before `train_end`, then score every bar in the test window.

    Returns one record per (symbol, bar) with the raw ensemble score and the realised
    barrier outcome.
    """
    # --- train ---------------------------------------------------------------
    train_slices = {}
    thin = 0
    for symbol, df in history.items():
        past = df[df.index < train_end]
        if len(past) >= 120:
            train_slices[symbol] = past
        elif not past.empty:
            thin += 1

    for algorithm in engine.algorithms.values():
        algorithm.is_trained = False

    if train_slices:
        logger.info(f"  training on {len(train_slices)} assets, bars < {train_end.date()}")
        await engine._auto_train_models(train_slices)
    else:
        # Do NOT abandon the fold. SMA and RSI need no training, so the fold is
        # still scoreable -- just without the supervised models. Returning early
        # here made a long time barrier look like "no data" when the real cause was
        # that the purge gap had eaten the training window.
        logger.warning(
            f"  no asset has 120+ bars before {train_end.date()} "
            f"({thin} asset(s) had some history but too little) - "
            f"scoring this fold with untrained algorithms only. "
            f"A longer --max-hold pushes the purge gap further back; extend history "
            f"or lower --folds to give the models something to learn from."
        )

    trained = [name for name, algo in engine.algorithms.items() if algo.is_trained]
    logger.info(f"  trained: {trained or 'none (heuristics only)'}")

    # --- score ---------------------------------------------------------------
    records: List[Dict] = []
    probe = next(iter(engine.algorithms.values()))
    policy = engine.barrier_policy

    for symbol, df in history.items():
        # Outcomes need the full series so forward windows resolve properly.
        # In volatility mode each bar carries its own barriers, derived from the
        # trailing volatility known at that bar; uneconomic bars come back NaN and
        # are therefore never labelled or scored.
        bar_win, bar_loss = policy.barrier_series(df)
        outcomes = probe.barrier_outcomes(df, bar_win, bar_loss, max_hold)

        window = df[(df.index >= test_start) & (df.index <= test_end)]
        if window.empty:
            continue

        positions = [df.index.get_loc(ts) for ts in window.index]
        for offset, (ts, row_index) in enumerate(zip(window.index, positions)):
            if offset % stride:
                continue

            row = outcomes.iloc[row_index]
            if not np.isfinite(row['label']):
                continue  # forward window not complete: unresolved

            # Only bars up to and including this one are visible to the model.
            visible = df.iloc[:row_index + 1]
            if len(visible) < 90:
                continue

            algo_predictions = await engine._predict_asset(symbol, visible)
            if not algo_predictions:
                continue

            raw_score = engine._calculate_ensemble_score(algo_predictions)
            label = int(row['label'])

            records.append({
                'timestamp': ts,
                'symbol': symbol,
                'raw_score': raw_score,
                # Binary training/calibration target: did the WIN barrier come first?
                'outcome': 1 if label == 1 else 0,
                # Which barrier resolved it: win / loss / timeout.
                'exit_type': {1: 'win_barrier', 0: 'loss_barrier', -1: 'time_barrier'}[label],
                # Actual gross return realised, used by the simulation.
                'exit_return_pct': float(row['exit_return']) * 100.0,
                'bars_held': int(row['bars_held']),
                # The barriers this bet actually used, so the simulation can price
                # each bet on its own geometry rather than a single global pair.
                'win_pct': float(row['win_pct']),
                'loss_pct': float(row['loss_pct']),
                'entry_price': float(visible['Close'].iloc[-1]),
                'n_algorithms': len(algo_predictions),
                **{f"algo_{p['algorithm']}": p['probability'] for p in algo_predictions},
            })

    logger.info(f"  scored {len(records)} (symbol, bar) observations")
    return records


# ------------------------------------------------------------------- simulation

def simulate(records: pd.DataFrame, config: Dict, threshold: float,
             max_concurrent: int) -> Tuple[pd.DataFrame, Dict]:
    """
    Simulate trading the scored bars, with fees, barriers and Kelly sizing.

    Deliberately simple about concurrency: a bet occupies a slot for `max_hold` days,
    and new bets are only opened while a slot is free. That understates capital
    efficiency slightly but never overstates it.
    """
    trading = config['trading']
    win_pct = float(trading['win_threshold'])
    loss_pct = float(trading['loss_threshold'])
    fee = float(trading['trading_fee_percentage']) / 100.0
    max_hold = int(trading['max_hold_days'])

    kelly = KellyCalculator(config)

    capital = float(trading['initial_capital'])
    peak = capital
    max_drawdown = 0.0

    open_until: Dict[str, pd.Timestamp] = {}
    trades: List[Dict] = []

    for record in records.sort_values('timestamp').itertuples():
        now = record.timestamp

        # Free slots whose holding period has elapsed.
        for symbol in [s for s, until in open_until.items() if until <= now]:
            del open_until[symbol]

        if record.raw_score < threshold:
            continue
        if record.symbol in open_until:
            continue
        if len(open_until) >= max_concurrent:
            continue

        # Size on this bet's own barriers, matching what place_bet() does live.
        recommendation = kelly.calculate_bet_size(
            probability=record.raw_score,
            current_price=record.entry_price,
            available_capital=capital,
            win_threshold=getattr(record, 'win_pct', None) or win_pct,
            loss_threshold=getattr(record, 'loss_pct', None) or loss_pct,
        )
        if not recommendation.is_favorable or recommendation.recommended_amount <= 0:
            continue

        stake = recommendation.recommended_amount

        # Realised P&L on the stake, using the ACTUAL exit return (which for a
        # time-barrier exit is the terminal move, not a full stop-out), both fee
        # legs included.
        gross = record.exit_return_pct / 100.0
        net = gross - 2 * fee
        pnl = stake * net

        capital += pnl
        peak = max(peak, capital)
        max_drawdown = max(max_drawdown, (peak - capital) / peak * 100.0 if peak > 0 else 0.0)

        held_days = max(1, int(record.bars_held))
        open_until[record.symbol] = now + pd.Timedelta(days=held_days)

        trades.append({
            'timestamp': now,
            'symbol': record.symbol,
            'raw_score': record.raw_score,
            'outcome': record.outcome,
            'exit_type': record.exit_type,
            'stake': stake,
            'fraction': recommendation.fraction_of_capital,
            'gross_return_pct': record.exit_return_pct,
            'net_return_pct': net * 100.0,
            'bars_held': held_days,
            'pnl': pnl,
            'capital_after': capital,
        })

        if capital < float(config.get('risk', {}).get('min_capital', 1000.0)):
            logger.warning(f"  simulation halted: capital fell below minimum at {now.date()}")
            break

    trade_frame = pd.DataFrame(trades)

    if trade_frame.empty:
        return trade_frame, {
            'trades': 0,
            'note': f'no bar scored above the {threshold:.1f} threshold',
        }

    wins = int((trade_frame['outcome'] == 1).sum())
    initial = float(trading['initial_capital'])

    stats = {
        'trades': int(len(trade_frame)),
        'win_rate_pct': 100.0 * wins / len(trade_frame),
        'break_even_pct': kelly.break_even_probability * 100.0,
        'total_pnl': float(trade_frame['pnl'].sum()),
        'final_capital': float(capital),
        'total_return_pct': (capital / initial - 1) * 100.0,
        'turnover': float(trade_frame['stake'].sum()),
        'return_on_turnover_pct': 100.0 * trade_frame['pnl'].sum() / trade_frame['stake'].sum(),
        'edge_per_trade_pct': float(trade_frame['net_return_pct'].mean()),
        'max_drawdown_pct': max_drawdown,
        'mean_stake_fraction_pct': float(trade_frame['fraction'].mean()) * 100.0,
    }
    return trade_frame, stats


# ------------------------------------------------------------------------- main

async def run(args):
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    db_path = Path(config['database']['sqlite']['path'])

    # Barrier-geometry overrides, so the parameters can be swept without editing
    # config. They are written back into the config dict so every consumer -- the
    # algorithms' training labels and the Kelly calculator -- sees the same bet.
    trading = config['trading']
    if args.win is not None:
        trading['win_threshold'] = args.win
    if args.loss is not None:
        trading['loss_threshold'] = args.loss
    if args.max_hold is not None:
        trading['max_hold_days'] = args.max_hold

    barrier = trading.setdefault('barrier', {})
    if args.barrier_mode is not None:
        barrier['mode'] = args.barrier_mode
    if args.win_sigma is not None:
        barrier['win_sigma'] = args.win_sigma
    if args.loss_sigma is not None:
        barrier['loss_sigma'] = args.loss_sigma
    if args.horizon is not None:
        barrier['horizon_days'] = args.horizon
    if args.max_break_even is not None:
        barrier['max_break_even_pct'] = args.max_break_even

    win_pct = float(trading['win_threshold'])
    loss_pct = float(trading['loss_threshold'])
    max_hold = int(trading['max_hold_days'])

    logger.info(f"Loading history from {db_path}")
    history = load_history(db_path, args.max_assets, args.min_bars)
    if not history:
        logger.error("No usable price history found. Run the system once to populate "
                     "price_data, or lower --min-bars.")
        return 1

    total_bars = sum(len(df) for df in history.values())
    logger.info(f"Loaded {len(history)} assets, {total_bars} bars")

    folds = fold_boundaries(history, args.folds, purge_days=max_hold)
    if not folds:
        logger.error("Not enough history to build walk-forward folds")
        return 1

    logger.info(f"Barrier: +{win_pct}% / -{loss_pct}%, time barrier {max_hold} bars")
    logger.info(f"Random-walk baseline win rate: {loss_pct / (win_pct + loss_pct):.1%}")
    logger.info(f"{len(folds)} walk-forward folds")

    engine = PredictionEngine(config)
    await engine.initialize()

    all_records: List[Dict] = []
    for i, (test_start, test_end) in enumerate(folds, 1):
        logger.info(f"Fold {i}/{len(folds)}: {test_start.date()} .. {test_end.date()}")
        # Purge: training stops max_hold bars before the test window opens.
        train_end = test_start - pd.Timedelta(days=max_hold)
        records = await score_fold(engine, history, train_end, test_start, test_end,
                                   win_pct, loss_pct, max_hold, args.stride)
        all_records.extend(records)

    if not all_records:
        logger.error("No observations scored; nothing to report")
        return 1

    frame = pd.DataFrame(all_records).sort_values('timestamp').reset_index(drop=True)

    scores_path = Path(args.out_scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(scores_path, index=False)
    logger.info(f"Wrote {len(frame)} scored observations to {scores_path}")

    report(frame, config, args)

    if not args.no_trade_log:
        threshold = args.threshold if args.threshold is not None else float(
            trading.get('min_probability', 66.0))
        trades, stats = simulate(frame, config, threshold,
                                 int(trading.get('max_concurrent_bets', 30)))
        print()
        print(f"SIMULATION at score threshold {threshold:.1f}")
        print("-" * 64)
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key:28s} {value:12.3f}")
            else:
                print(f"  {key:28s} {value:>12}")

        if not trades.empty:
            trades_path = Path(args.out_trades)
            trades.to_csv(trades_path, index=False)
            logger.info(f"Wrote {len(trades)} simulated trades to {trades_path}")

    return 0


def report(frame: pd.DataFrame, config: Dict, args):
    """Print the reliability table and the threshold sweep."""
    from src.trading.barriers import BarrierPolicy

    trading = config['trading']
    policy = BarrierPolicy(config)
    fee = float(trading['trading_fee_percentage']) / 100.0

    # In volatility mode each bet has its own barriers, so break-even varies per bet.
    if 'win_pct' in frame.columns and frame['win_pct'].notna().any():
        per_bet_break_even = frame.apply(
            lambda r: policy.break_even(r['win_pct'], r['loss_pct']), axis=1) * 100.0
        break_even = per_bet_break_even.mean()
        net_return = frame['exit_return_pct'] - 2 * fee * 100.0
    else:
        break_even = KellyCalculator(config).break_even_probability * 100.0
        per_bet_break_even = None
        net_return = frame['exit_return_pct'] - 2 * fee * 100.0

    frame = frame.assign(net_return_pct=net_return)

    print()
    print("=" * 72)
    print("OUT-OF-SAMPLE RELIABILITY: raw ensemble score vs realised win rate")
    print("=" * 72)
    print(f"  {policy.describe()}")
    print(f"  observations: {len(frame)}   overall win rate: {frame['outcome'].mean():.2%}")
    print(f"  random-walk geometry: {policy.geometry_probability:.2%}")
    if per_bet_break_even is not None:
        print(f"  fee-adjusted break-even: mean {break_even:.2f}% "
              f"(range {per_bet_break_even.min():.2f}%-{per_bet_break_even.max():.2f}%)")
        print(f"  barriers: win {frame['win_pct'].mean():.2f}% mean "
              f"({frame['win_pct'].min():.2f}%-{frame['win_pct'].max():.2f}%)")
        print(f"  mean net return per bet: {frame['net_return_pct'].mean():+.3f}%")
        required = frame.apply(
            lambda r: policy.required_edge(r['win_pct'], r['loss_pct']), axis=1)
        print(f"  edge the model must supply: {required.mean():.2f} pts mean "
              f"(= 2c/(w+l); this is the ONLY structural cost -- for a driftless "
              f"asset the geometry probability equals the pre-fee break-even, so no "
              f"barrier shape creates profit)")
    else:
        print(f"  fee-adjusted break-even: {break_even:.2f}%")
    # How bets actually resolve. If most reach neither barrier, the time barrier -
    # not the model - is deciding the outcome.
    if 'exit_type' in frame.columns:
        print()
        print("  HOW BETS RESOLVE")
        counts = frame['exit_type'].value_counts()
        for exit_type, count in counts.items():
            share = count / len(frame)
            mean_return = frame.loc[frame['exit_type'] == exit_type, 'exit_return_pct'].mean()
            print(f"    {exit_type:<14} {count:>7} ({share:>6.1%})  "
                  f"mean return {mean_return:+7.2f}%")

    print()
    print(f"  {'raw score band':<18}{'n':>8}{'win-barrier rate':>19}{'mean net return':>18}")

    edges = [0, 40, 45, 50, 55, 60, 65, 70, 101]
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = frame[(frame['raw_score'] >= lo) & (frame['raw_score'] < hi)]
        if len(band) < 20:
            continue
        print(f"  {f'{lo}-{hi}%':<18}{len(band):>8}{band['outcome'].mean():>18.2%}"
              f"{band['net_return_pct'].mean():>17.3f}%")

    print()
    print("  THRESHOLD SWEEP (mean net return per trade, actual exits, fees included)")
    print(f"  {'threshold':<12}{'trades':>9}{'win rate':>11}{'net/trade':>12}{'total':>11}")
    for threshold in (40, 45, 50, 55, 60, 65, 70, 75):
        selected = frame[frame['raw_score'] >= threshold]
        if len(selected) < 20:
            continue
        net = selected['net_return_pct'].mean()
        print(f"  {threshold:<12}{len(selected):>9}{selected['outcome'].mean():>10.2%}"
              f"{net:>11.3f}%{net * len(selected):>10.1f}%")

    # Does the strategy work better on volatile assets? With volatility-scaled
    # barriers, fees are a smaller share of the profit target on high-vol names, so
    # break-even is lower there.
    if 'win_pct' in frame.columns and frame['win_pct'].notna().any():
        print()
        print("  BY BARRIER WIDTH (a proxy for the asset's own volatility)")
        print(f"  {'win barrier':<16}{'bets':>8}{'win rate':>11}{'break-even':>13}{'net/trade':>12}")
        quantiles = frame['win_pct'].quantile([0, .25, .5, .75, 1.0]).tolist()
        for lo, hi in zip(quantiles[:-1], quantiles[1:]):
            band = frame[(frame['win_pct'] >= lo) & (frame['win_pct'] <= hi)]
            if len(band) < 20:
                continue
            band_break_even = band.apply(
                lambda r: policy.break_even(r['win_pct'], r['loss_pct']), axis=1).mean() * 100
            print(f"  {f'{lo:.1f}-{hi:.1f}%':<16}{len(band):>8}"
                  f"{band['outcome'].mean():>10.2%}{band_break_even:>12.2f}%"
                  f"{band['net_return_pct'].mean():>11.3f}%")

    print()
    print("  Next step: python scripts/fit_calibration.py")
    print("  That turns these raw scores into probabilities, so the threshold can be")
    print("  set against break-even instead of guessed.")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(description='Walk-forward backtest of the barrier strategy')
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--max-assets', type=int, default=120,
                        help='Assets to include, longest history first (default: 120)')
    parser.add_argument('--min-bars', type=int, default=150,
                        help='Minimum bars an asset needs to be included (default: 150)')
    parser.add_argument('--folds', type=int, default=4,
                        help='Walk-forward test windows (default: 4)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Score every Nth bar; raise to speed up (default: 1)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Score threshold for the simulation (default: config min_probability)')
    parser.add_argument('--win', type=float, default=None,
                        help='Override win_threshold %% (fixed-barrier mode only)')
    parser.add_argument('--loss', type=float, default=None,
                        help='Override loss_threshold %% (fixed-barrier mode only)')
    parser.add_argument('--max-hold', type=int, default=None,
                        help='Override max_hold_days. This is the dominant parameter: '
                             'too short and most bets reach neither barrier.')
    parser.add_argument('--barrier-mode', choices=['volatility', 'fixed'], default=None,
                        help='Override barrier.mode')
    parser.add_argument('--win-sigma', type=float, default=None,
                        help='Win barrier in sigma units (volatility mode)')
    parser.add_argument('--loss-sigma', type=float, default=None,
                        help='Loss barrier in sigma units (volatility mode)')
    parser.add_argument('--horizon', type=int, default=None,
                        help='Days used to scale sigma (volatility mode)')
    parser.add_argument('--max-break-even', type=float, default=None,
                        help='Reject bets whose fee-adjusted break-even exceeds this %%')
    parser.add_argument('--out-scores', default='data/backtest_scores.csv')
    parser.add_argument('--out-trades', default='data/backtest_trades.csv')
    parser.add_argument('--no-trade-log', action='store_true')
    parser.add_argument('--log-level', default='INFO')
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )

    return asyncio.run(run(args))


if __name__ == '__main__':
    sys.exit(main())
