#!/usr/bin/env python3
"""
Edge analysis with honest error bars.

Barrier outcomes are NOT independent observations, and treating them as if they were
is what made earlier readings of this data wrong twice over. Two sources of
dependence:

  * **Overlap within an asset.** A bet opened at bar i and one opened at bar i+1
    share almost their whole forward window. With a 60-bar time barrier, 60
    consecutive "observations" are close to one observation.
  * **Correlation across assets.** Every bet is a long position, so a market-wide
    move resolves many of them the same way on the same day.

Naive SE = sqrt(p(1-p)/n) therefore understates uncertainty badly: on simulated data
a single 2,500-bar path estimates a barrier win rate to only about +/-5 percentage
points, where the naive formula would claim +/-1.

This script reports, for every slice:

  * the naive SE, for reference;
  * an ASSET-CLUSTERED SE (each asset counts as one observation);
  * a DATE-BLOCK-CLUSTERED SE (each calendar block counts as one), which is what
    catches market-wide correlation;
  * the effective sample size implied by the widest of those.

It then tests the specific hypotheses we care about:

  1. Does the ensemble score predict the barrier outcome at all?
  2. Is the apparent edge in the mid-width barrier band real, or one asset?
  3. Does any slice with positive edge replicate across independent time folds?

Usage:
    python scripts/analyze_edge.py data/scores_vol60.csv
    python scripts/analyze_edge.py data/scores_vol60.csv --fee 0.25 --blocks 12
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ------------------------------------------------------------------ statistics

def clustered_stats(frame: pd.DataFrame, value_col: str,
                    cluster_col: str) -> Dict[str, float]:
    """
    Mean and standard error treating each cluster as one independent observation.

    The cluster mean is the unit of analysis, so within-cluster correlation -- however
    strong -- cannot inflate the sample size.
    """
    groups = frame.groupby(cluster_col)[value_col].mean().dropna()
    n = len(groups)
    if n < 2:
        return {'mean': float(groups.mean()) if n else float('nan'),
                'stderr': float('nan'), 'clusters': n}
    return {
        'mean': float(groups.mean()),
        'stderr': float(groups.std(ddof=1) / np.sqrt(n)),
        'clusters': n,
    }


def date_blocks(frame: pd.DataFrame, blocks: int) -> pd.Series:
    """Assign each row to one of `blocks` contiguous calendar blocks."""
    if 'timestamp' not in frame.columns:
        return pd.Series('all', index=frame.index)
    ranks = frame['timestamp'].rank(method='dense')
    return pd.cut(ranks, bins=blocks, labels=False).astype('Int64').astype(str)


def summarise(frame: pd.DataFrame, value_col: str, blocks: int) -> Dict[str, float]:
    """Naive, asset-clustered and date-clustered views of the same mean."""
    values = frame[value_col].dropna()
    n = len(values)
    naive_se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float('nan')

    by_asset = clustered_stats(frame, value_col, 'symbol') if 'symbol' in frame else {}
    working = frame.assign(_block=date_blocks(frame, blocks))
    by_date = clustered_stats(working, value_col, '_block')

    candidates = [se for se in (by_asset.get('stderr'), by_date.get('stderr'))
                  if se is not None and np.isfinite(se)]
    worst_se = max(candidates) if candidates else naive_se

    effective_n = (values.std(ddof=1) / worst_se) ** 2 if worst_se and worst_se > 0 else float('nan')

    return {
        'n': n,
        'mean': float(values.mean()),
        'naive_se': naive_se,
        'asset_se': by_asset.get('stderr', float('nan')),
        'assets': by_asset.get('clusters', 0),
        'date_se': by_date.get('stderr', float('nan')),
        'blocks': by_date.get('clusters', 0),
        'worst_se': worst_se,
        'effective_n': effective_n,
    }


def verdict(mean: float, stderr: float, null: float = 0.0) -> str:
    """Plain-language read on whether a mean is distinguishable from `null`."""
    if not np.isfinite(stderr) or stderr == 0:
        return "no SE"
    z = (mean - null) / stderr
    if abs(z) < 1.0:
        return f"z={z:+.2f} noise"
    if abs(z) < 2.0:
        return f"z={z:+.2f} weak"
    if abs(z) < 3.0:
        return f"z={z:+.2f} SUGGESTIVE"
    return f"z={z:+.2f} STRONG"


# --------------------------------------------------------------------- loading

def load(path: Path, fee_pct: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if 'timestamp' in frame.columns:
        frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True, errors='coerce')
        frame = frame.sort_values('timestamp')

    round_trip = 2 * fee_pct
    if 'exit_return_pct' in frame.columns:
        frame['net_return_pct'] = frame['exit_return_pct'] - round_trip
    else:
        raise SystemExit(f"{path} has no exit_return_pct column; re-run the backtest")

    if 'win_pct' not in frame.columns:
        frame['win_pct'] = np.nan
    if 'loss_pct' not in frame.columns:
        frame['loss_pct'] = np.nan

    # Break-even and required edge per bet, from that bet's own barriers.
    total = (frame['win_pct'] + frame['loss_pct']) / 100.0
    frame['break_even_pct'] = ((frame['loss_pct'] / 100.0 + round_trip / 100.0)
                               / total.where(total > 0)) * 100.0
    frame['required_edge_pct'] = (round_trip / 100.0 / total.where(total > 0)) * 100.0
    return frame


def non_overlapping(frame: pd.DataFrame, min_gap_bars: int) -> pd.DataFrame:
    """
    Keep, per asset, only bets whose forward windows do not overlap.

    Crude but decisive: if an apparent edge survives here, it is not an artefact of
    counting the same market move dozens of times.
    """
    if 'bars_held' not in frame.columns or 'symbol' not in frame.columns:
        return frame

    keep = []
    for _, group in frame.groupby('symbol', sort=False):
        group = group.sort_values('timestamp') if 'timestamp' in group else group
        next_free = None
        for row in group.itertuples():
            ts = getattr(row, 'timestamp', None)
            if next_free is None or ts is None or ts >= next_free:
                keep.append(row.Index)
                if ts is not None:
                    next_free = ts + pd.Timedelta(days=min_gap_bars * 1.45)
    return frame.loc[keep]


# ----------------------------------------------------------------------- report

def header(title: str):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_slice(label: str, stats: Dict[str, float], null: float = 0.0,
                width: int = 22):
    print(f"  {label:<{width}}{stats['n']:>7}"
          f"{stats['mean']:>+10.3f}"
          f"{stats['naive_se']:>9.3f}"
          f"{stats['worst_se']:>10.3f}"
          f"{stats['effective_n']:>10.0f}"
          f"   {verdict(stats['mean'], stats['worst_se'], null)}")


def slice_table(frame: pd.DataFrame, title: str, blocks: int):
    print()
    print(f"  {title}")
    print(f"  {'slice':<22}{'n':>7}{'net%':>10}{'naiveSE':>9}{'clustSE':>10}"
          f"{'eff.n':>10}   verdict")


def main():
    parser = argparse.ArgumentParser(description='Edge analysis with clustered errors')
    parser.add_argument('scores', nargs='?', default='data/backtest_scores.csv')
    parser.add_argument('--fee', type=float, default=0.25,
                        help='Fee per side, percent (default 0.25)')
    parser.add_argument('--blocks', type=int, default=10,
                        help='Calendar blocks for date clustering (default 10)')
    parser.add_argument('--max-hold', type=int, default=60,
                        help='Time barrier, used for the non-overlapping subsample')
    args = parser.parse_args()

    path = Path(args.scores)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/backtest.py")

    frame = load(path, args.fee)

    header(f"EDGE ANALYSIS - {path.name}")
    print(f"  observations           {len(frame)}")
    if 'symbol' in frame:
        print(f"  distinct assets        {frame['symbol'].nunique()}")
    if 'timestamp' in frame:
        print(f"  date range             {frame['timestamp'].min().date()} .. "
              f"{frame['timestamp'].max().date()}")
    print(f"  win-barrier rate       {frame['outcome'].mean():.2%}")
    if frame['break_even_pct'].notna().any():
        print(f"  break-even (mean)      {frame['break_even_pct'].mean():.2f}%")
        print(f"  required edge (mean)   {frame['required_edge_pct'].mean():.2f} pts")
    print(f"  mean net return/bet    {frame['net_return_pct'].mean():+.3f}%")

    if 'exit_type' in frame:
        print()
        print("  resolution mix:")
        for exit_type, count in frame['exit_type'].value_counts().items():
            print(f"    {exit_type:<14}{count:>7} ({count / len(frame):>6.1%})")

    # ---------------------------------------------------------------- overall
    header("1. IS THERE ANY EDGE OVERALL?")
    overall = summarise(frame, 'net_return_pct', args.blocks)
    slice_table(frame, "net return per bet, vs zero", args.blocks)
    print_slice("all bets", overall)
    print()
    print(f"  Naive SE would claim +/-{overall['naive_se']:.3f}%; clustering says "
          f"+/-{overall['worst_se']:.3f}%.")
    print(f"  Effective sample is ~{overall['effective_n']:.0f}, not {overall['n']} "
          f"-- a factor of {overall['n'] / max(overall['effective_n'], 1):.0f} fewer.")

    # ------------------------------------------------------------ score signal
    header("2. DOES THE ENSEMBLE SCORE PREDICT ANYTHING?")
    print(f"  {'score band':<22}{'n':>7}{'net%':>10}{'naiveSE':>9}{'clustSE':>10}"
          f"{'eff.n':>10}   verdict")
    edges = [0, 35, 40, 45, 50, 55, 101]
    for lo, hi in zip(edges[:-1], edges[1:]):
        band = frame[(frame['raw_score'] >= lo) & (frame['raw_score'] < hi)]
        if len(band) < 40:
            continue
        print_slice(f"{lo}-{hi}%", summarise(band, 'net_return_pct', args.blocks))

    # Spearman with an asset-clustered error bar.
    print()
    per_asset_corr = []
    for symbol, group in frame.groupby('symbol'):
        if len(group) >= 30 and group['raw_score'].std() > 0:
            per_asset_corr.append(group['raw_score'].corr(group['outcome'],
                                                          method='spearman'))
    if len(per_asset_corr) >= 3:
        arr = np.array([c for c in per_asset_corr if np.isfinite(c)])
        se = arr.std(ddof=1) / np.sqrt(len(arr))
        print(f"  Spearman(score, outcome) within-asset: mean {arr.mean():+.4f} "
              f"+/- {se:.4f} across {len(arr)} assets   "
              f"{verdict(arr.mean(), se)}")
        print(f"  (computed per asset then averaged, so cross-asset differences in "
              f"base rate cannot masquerade as signal)")

    # -------------------------------------------------- within-asset timing test
    header("2b. IS THE SCORE A TIMING SIGNAL, IN EITHER DIRECTION?")
    print("  Score ranked WITHIN each asset, so a persistently high-scoring asset")
    print("  cannot be confused with a well-timed entry. If the score is reliably")
    print("  anti-predictive, inverting it is itself an edge -- worth knowing.")
    print()

    if 'symbol' in frame.columns:
        ranked = frame.copy()
        ranked['score_pct_within_asset'] = ranked.groupby('symbol')['raw_score'].rank(pct=True)
        usable = ranked[ranked.groupby('symbol')['raw_score'].transform('count') >= 20]

        if len(usable) >= 100:
            print(f"  {'within-asset rank':<22}{'n':>7}{'net%':>10}{'naiveSE':>9}"
                  f"{'clustSE':>10}{'eff.n':>10}   verdict")
            for lo, hi, label in ((0.0, 0.2, 'bottom 20%'), (0.2, 0.4, '20-40%'),
                                  (0.4, 0.6, 'middle 20%'), (0.6, 0.8, '60-80%'),
                                  (0.8, 1.01, 'top 20%')):
                band = usable[(usable['score_pct_within_asset'] >= lo)
                              & (usable['score_pct_within_asset'] < hi)]
                if len(band) >= 30:
                    print_slice(label, summarise(band, 'net_return_pct', args.blocks))

            top = usable[usable['score_pct_within_asset'] >= 0.8]
            bottom = usable[usable['score_pct_within_asset'] < 0.2]
            if len(top) >= 30 and len(bottom) >= 30:
                spread = top['net_return_pct'].mean() - bottom['net_return_pct'].mean()
                # SE of the difference, using the clustered SE of each leg.
                top_se = summarise(top, 'net_return_pct', args.blocks)['worst_se']
                bot_se = summarise(bottom, 'net_return_pct', args.blocks)['worst_se']
                diff_se = float(np.sqrt(top_se ** 2 + bot_se ** 2))
                print()
                print(f"  top-minus-bottom spread: {spread:+.3f}% +/- {diff_se:.3f}   "
                      f"{verdict(spread, diff_se)}")
                print(f"  (a large POSITIVE spread means the score times entries; a large")
                print(f"   NEGATIVE one means inverting it does)")
        else:
            print("  Too few assets with enough observations to rank within asset.")

    # ------------------------------------------------------- barrier width band
    if frame['win_pct'].notna().any() and frame['win_pct'].nunique() > 4:
        header("3. IS THE MID-WIDTH BARRIER BAND REAL, OR ONE ASSET?")
        frame = frame.assign(width_q=pd.qcut(frame['win_pct'], 4, labels=False,
                                             duplicates='drop'))
        print(f"  {'width quartile':<22}{'n':>7}{'net%':>10}{'naiveSE':>9}"
              f"{'clustSE':>10}{'eff.n':>10}   verdict")
        for q in sorted(frame['width_q'].dropna().unique()):
            band = frame[frame['width_q'] == q]
            lo, hi = band['win_pct'].min(), band['win_pct'].max()
            print_slice(f"Q{int(q) + 1} {lo:.1f}-{hi:.1f}%",
                        summarise(band, 'net_return_pct', args.blocks))

        # Concentration check on the best quartile.
        best_q = max(frame['width_q'].dropna().unique(),
                     key=lambda q: frame[frame['width_q'] == q]['net_return_pct'].mean())
        band = frame[frame['width_q'] == best_q]
        print()
        print(f"  Best quartile is Q{int(best_q) + 1}. Concentration check:")
        by_symbol = band.groupby('symbol')['net_return_pct'].agg(['count', 'mean'])
        by_symbol = by_symbol.sort_values('mean', ascending=False)
        total = band['net_return_pct'].sum()
        top3 = by_symbol.head(3)
        top3_contribution = (frame[frame['symbol'].isin(top3.index)
                                   & (frame['width_q'] == best_q)]['net_return_pct'].sum())
        print(f"    {len(by_symbol)} assets contribute; "
              f"top 3 supply {top3_contribution / total:.0%} of the total return"
              if total else "    (zero total return)")
        print(f"    assets with positive mean: "
              f"{(by_symbol['mean'] > 0).sum()}/{len(by_symbol)}")
        for symbol, row in top3.iterrows():
            print(f"      {symbol:<12} n={row['count']:>4.0f}  mean {row['mean']:+.3f}%")

    # --------------------------------------------------------- time replication
    if 'timestamp' in frame:
        header("4. DOES ANY EDGE REPLICATE ACROSS TIME?")
        halves = frame.assign(half=pd.qcut(frame['timestamp'].rank(method='dense'), 2,
                                           labels=['first half', 'second half']))
        print(f"  {'period':<22}{'n':>7}{'net%':>10}{'naiveSE':>9}{'clustSE':>10}"
              f"{'eff.n':>10}   verdict")
        for period, group in halves.groupby('half', observed=True):
            print_slice(str(period), summarise(group, 'net_return_pct', args.blocks))

        if 'width_q' in frame.columns:
            print()
            print("  Best-quartile edge, split by period (does it hold up?):")
            print(f"  {'period':<22}{'n':>7}{'net%':>10}{'naiveSE':>9}{'clustSE':>10}"
                  f"{'eff.n':>10}   verdict")
            best_q = max(frame['width_q'].dropna().unique(),
                         key=lambda q: frame[frame['width_q'] == q]['net_return_pct'].mean())
            for period, group in halves[halves['width_q'] == best_q].groupby(
                    'half', observed=True):
                if len(group) >= 30:
                    print_slice(str(period), summarise(group, 'net_return_pct', args.blocks))

    # ------------------------------------------------ non-overlapping subsample
    header("5. THE SAME QUESTION ON NON-OVERLAPPING BETS")
    independent = non_overlapping(frame, args.max_hold)
    print(f"  {len(independent)} of {len(frame)} bets survive the no-overlap filter")
    if len(independent) >= 40:
        print()
        print(f"  {'slice':<22}{'n':>7}{'net%':>10}{'naiveSE':>9}{'clustSE':>10}"
              f"{'eff.n':>10}   verdict")
        print_slice("all (independent)", summarise(independent, 'net_return_pct', args.blocks))
        if 'width_q' in independent.columns:
            for q in sorted(independent['width_q'].dropna().unique()):
                band = independent[independent['width_q'] == q]
                if len(band) >= 30:
                    print_slice(f"Q{int(q) + 1} (independent)",
                                summarise(band, 'net_return_pct', args.blocks))
    else:
        print("  Too few independent bets to say anything; widen the sample.")

    # ------------------------------------------------------------------- power
    header("6. IS THIS SAMPLE EVEN BIG ENOUGH TO DETECT AN EDGE?")
    values = frame['net_return_pct'].dropna()
    sd = float(values.std(ddof=1))
    naive_se = overall['naive_se']
    worst_se = overall['worst_se']
    inflation = (worst_se / naive_se) ** 2 if naive_se > 0 else float('nan')

    span = float((frame['win_pct'] + frame['loss_pct']).mean())
    if not np.isfinite(span) or span <= 0:
        span = 8.0  # fallback: a 5%/3% bet

    print(f"  sd(net return per bet)   {sd:.2f}%")
    print(f"  variance inflation       ~{inflation:.0f}x from overlap and co-movement")
    print(f"  observations held        {len(values)}  (effective ~{overall['effective_n']:.0f})")
    print()
    print(f"  {'edge to detect':<20}{'net%/bet':>11}{'indep. bets':>14}{'raw obs needed':>17}{'have?':>8}")
    for edge_pts in (2, 4, 6, 8):
        per_bet = edge_pts / 100.0 * span
        need_se = per_bet / 2.0            # aim for z = 2
        need_effective = (sd / need_se) ** 2
        need_raw = need_effective * inflation if np.isfinite(inflation) else float('nan')
        have = "YES" if len(values) >= need_raw else "no"
        print(f"  {f'{edge_pts} pts of win rate':<20}{per_bet:>10.2f}%"
              f"{need_effective:>14.0f}{need_raw:>17,.0f}{have:>8}")
    print()
    print("  An underpowered run cannot distinguish 'no edge' from 'edge we cannot")
    print("  see'. If the answer above is 'no', scale the run (more assets, stride 1,")
    print("  more folds) before concluding anything from a null result.")

    header("READING THIS")
    print("  'noise' means the slice is indistinguishable from zero edge once the")
    print("  overlap between bets is accounted for. Only STRONG, replicated across")
    print("  both time periods and not concentrated in a couple of assets, is worth")
    print("  trading. Anything else is a hypothesis for the next run.")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
