"""
Reliability and execution-quality panels for the dashboard.

These exist because the dashboard previously rendered an uncalibrated ensemble score
in a column labelled "Probability", with nothing on screen to compare it against. It
was therefore possible to run the system for six months, see "62%" on every top pick,
and never notice that the realised win rate was 44% against a break-even of 43.7%.

Three things are surfaced here:

  * the fee-adjusted break-even probability, which is the real neutral point for a
    barrier bet (37.5% before fees for a 5%/3% bet, not 50%);
  * predicted band vs delivered win rate, computed live from the bets table;
  * exit slippage, so an overshooting stop is visible the day it happens.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st


@st.cache_data(ttl=60)
def load_closed_bets(db_path: str) -> pd.DataFrame:
    """Closed bets with the fields needed for reliability and slippage."""
    conn = sqlite3.connect(db_path)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(bets)")}
        optional = [c for c in ('exit_reason', 'exit_slippage_pct') if c in columns]
        select = ", ".join([
            'bet_id', 'symbol', 'asset_type', 'status', 'amount', 'entry_price',
            'exit_price', 'entry_time', 'exit_time', 'realized_pnl',
            'probability_when_placed',
        ] + optional)

        return pd.read_sql_query(
            f"SELECT {select} FROM bets WHERE status IN ('won','lost') ORDER BY exit_time",
            conn,
        )
    finally:
        conn.close()


def render_decision_panel(db_path: str, break_even_pct: float, geometry_pct: float,
                          required_edge_pct: float, is_calibrated: bool,
                          expected_days: Optional[float] = None,
                          open_positions: int = 0,
                          correlation_haircut: Optional[float] = None):
    """
    What the user needs in order to decide, rather than a status report on our models.

    The previous version of this banner explained our internal state ("no calibration
    has been fitted, run these two scripts") which tells a trader nothing about
    whether to place a bet.

    The useful insight is that **the economics of a bet are exactly knowable even when
    its probability is not**. Barrier distances, break-even, the skill the payoff
    demands and the fee drag all follow from the asset's own volatility and the fee.
    Those are shown here as hard numbers. The model's score is presented for what it
    can actually do -- rank assets against each other -- and the delivered track record
    is put next to the bar it has to clear.
    """
    bets = load_closed_bets(db_path)
    delivered = None
    sample = 0
    if not bets.empty:
        sample = len(bets)
        delivered = (bets['status'] == 'won').mean() * 100

    col1, col2, col3, col4 = st.columns(4)

    # Stated as a hit rate out of 100, because "43.28%" invites reading it as a
    # probability of something rather than as a bar to clear.
    col1.metric(
        "Must win", f"{break_even_pct:.0f} in 100",
        help=f"Of every 100 such bets, this many must win just to break even after "
             f"fees ({break_even_pct:.2f}% exactly). Below that line a bet loses "
             f"money on average no matter how confident it looks.")

    col2.metric(
        "Fees take", f"{required_edge_pct:.1f}% of pot",
        help=f"The broker's cut as a share of everything at stake. The barrier shape "
             f"alone already wins about {geometry_pct:.0f} in 100, so pure luck gets "
             f"you most of the way — this is the extra you must supply, and it is "
             f"entirely the fee. Like a casino rake: more volatile assets have wider "
             f"barriers, so the same fee is a smaller slice of a bigger pot.")

    if delivered is not None:
        col3.metric(
            "Actually won", f"{delivered:.0f} in 100",
            delta=f"{delivered - break_even_pct:+.1f} vs the bar",
            delta_color="normal" if delivered >= break_even_pct else "inverse",
            help=f"Real track record: {sample} closed bets, {delivered:.1f}% won, "
                 f"against a {break_even_pct:.2f}% break-even.")
    else:
        col3.metric("Actually won", "no data",
                    help="No closed bets yet.")

    if is_calibrated:
        col4.metric("Score means", "Probability",
                    help="Calibrated against realised outcomes, so a displayed 55% "
                         "has been observed to win about 55% of the time.")
    else:
        col4.metric("Score means", "Rank only",
                    help="The score orders assets but its level is not a probability. "
                         "A displayed 62 is not a 62% chance.")

    # Facts that are the same for every bet belong here, once, not repeated down a
    # column. Under volatility-scaled barriers the expected time-to-target is
    # identical across assets by construction -- that is the point of scaling to
    # sigma -- so a per-row "days" column would carry no information.
    context = []
    if expected_days:
        context.append(f"Every bet targets roughly **{expected_days:.0f} trading "
                       f"days** to reach its profit target — barriers are scaled to "
                       f"each asset's volatility so the clock is comparable.")
    if open_positions and correlation_haircut is not None and correlation_haircut < 0.999:
        context.append(
            f"**{open_positions} positions open**, so new bets are sized at "
            f"**{correlation_haircut:.0%}** of standalone Kelly — a book of "
            f"correlated longs is closer to one leveraged bet than to "
            f"{open_positions} independent ones.")
    if context:
        st.caption(" ".join(context))

    if is_calibrated:
        st.caption(
            f"**How to read this screen.** Scores are calibrated, so compare them "
            f"directly against the {break_even_pct:.2f}% line: above it the bet has "
            f"positive expected value, below it it does not. Position size already "
            f"scales with the margin."
        )
    else:
        st.caption(
            f"**How to read this screen.** The score is a ranking, not a price — use "
            f"it to choose *between* the assets below, not to decide *whether* a bet "
            f"is worth taking. That second question is answered by the columns that do "
            f"not depend on the model: **break-even**, **skill required** and "
            f"**expected days to resolve**, which are exact for each asset. "
            f"On the record so far the system has delivered "
            f"{f'{delivered:.1f}%' if delivered is not None else 'no measured'} "
            f"against a {break_even_pct:.2f}% bar, so treat any position as research "
            f"sizing until the Reliability tab shows predicted matching delivered."
        )


def render_reliability(db_path: str, break_even_pct: float):
    """Predicted band vs delivered win rate, from the live bets table."""
    st.subheader("Reliability: predicted vs delivered")

    bets = load_closed_bets(db_path)
    if bets.empty:
        st.info("No closed bets yet - nothing to measure.")
        return

    bets = bets.dropna(subset=['probability_when_placed'])
    if bets.empty:
        st.info("Closed bets carry no recorded probability.")
        return

    wins = (bets['status'] == 'won')
    overall_win_rate = wins.mean() * 100
    mean_predicted = bets['probability_when_placed'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Mean predicted", f"{mean_predicted:.1f}%")
    col2.metric("Delivered win rate", f"{overall_win_rate:.1f}%",
                delta=f"{overall_win_rate - mean_predicted:+.1f} pts vs predicted",
                delta_color="inverse" if overall_win_rate < mean_predicted else "normal")
    col3.metric("Break-even needed", f"{break_even_pct:.2f}%",
                delta=f"{overall_win_rate - break_even_pct:+.2f} pts of edge",
                delta_color="normal" if overall_win_rate >= break_even_pct else "inverse")

    # Realised payoff ratio: what execution actually delivers, versus the design.
    won = bets[wins]
    lost = bets[~wins]
    if not won.empty and not lost.empty:
        avg_win = (won['realized_pnl'] / won['amount'] * 100).mean()
        avg_loss = -(lost['realized_pnl'] / lost['amount'] * 100).mean()
        realised_break_even = avg_loss / (avg_win + avg_loss) * 100 if (avg_win + avg_loss) else 0

        st.caption(
            f"Realised payoff: **+{avg_win:.2f}%** on wins, **-{avg_loss:.2f}%** on losses "
            f"(ratio {avg_win / avg_loss:.3f}). At those payoffs the break-even win rate "
            f"is **{realised_break_even:.2f}%**, not {break_even_pct:.2f}% - the gap is "
            f"execution slippage past the barriers."
        )

    bands = [(0, 50), (50, 58), (58, 62), (62, 66), (66, 72), (72, 101)]
    rows: List[Dict] = []
    for low, high in bands:
        band = bets[(bets['probability_when_placed'] >= low) &
                    (bets['probability_when_placed'] < high)]
        if band.empty:
            continue
        band_wins = (band['status'] == 'won')
        rows.append({
            'Predicted band': f"{low}-{high}%",
            'Bets': len(band),
            'Mean predicted': band['probability_when_placed'].mean(),
            'Delivered': band_wins.mean() * 100,
            'Gap (pts)': band_wins.mean() * 100 - band['probability_when_placed'].mean(),
            'P&L': band['realized_pnl'].sum(),
            'Return on stake': band['realized_pnl'].sum() / band['amount'].sum() * 100,
        })

    if not rows:
        return

    frame = pd.DataFrame(rows)
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Mean predicted': st.column_config.NumberColumn(format="%.1f%%"),
            'Delivered': st.column_config.NumberColumn(format="%.1f%%"),
            'Gap (pts)': st.column_config.NumberColumn(format="%+.1f"),
            'P&L': st.column_config.NumberColumn(format="$%.2f"),
            'Return on stake': st.column_config.NumberColumn(format="%+.2f%%"),
        },
    )
    st.caption(
        "A well-calibrated system has 'Gap' near zero in every band. Persistent "
        "negative gaps mean the probabilities are overstated and every downstream "
        "threshold is set against the wrong scale."
    )


def render_execution_quality(db_path: str):
    """
    Exit slippage: how far fills land past the barrier they were meant to hit.

    Barrier overshoot is the largest measurable cost in this strategy's history -- a
    -3% stop realised -6.97% on average -- and it was completely invisible.
    """
    st.subheader("Execution quality: exit slippage")

    bets = load_closed_bets(db_path)
    if bets.empty:
        st.info("No closed bets yet.")
        return

    if 'exit_slippage_pct' not in bets.columns or bets['exit_slippage_pct'].isna().all():
        # Fall back to deriving overshoot from prices for bets closed before the
        # slippage column existed.
        st.caption("No recorded slippage yet (column added recently). "
                   "Showing overshoot derived from entry/exit prices instead.")
        derived = bets.dropna(subset=['entry_price', 'exit_price']).copy()
        if derived.empty:
            return
        derived['Realised move %'] = (derived['exit_price'] / derived['entry_price'] - 1) * 100
        summary = derived.groupby('status')['Realised move %'].agg(['count', 'mean', 'min', 'max'])
        st.dataframe(summary, use_container_width=True)
        st.caption(
            "Compare these against the configured barriers. A loss column averaging "
            "well beyond the stop distance means exits are not happening at the barrier."
        )
        return

    slippage = bets.dropna(subset=['exit_slippage_pct'])
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean slippage", f"{slippage['exit_slippage_pct'].mean():+.2f}%")
    col2.metric("Worst slippage", f"{slippage['exit_slippage_pct'].min():+.2f}%")
    col3.metric("Exits measured", f"{len(slippage)}")

    if 'exit_reason' in slippage.columns:
        by_reason = slippage.groupby('exit_reason')['exit_slippage_pct'].agg(
            ['count', 'mean', 'min'])
        by_reason.columns = ['Exits', 'Mean slippage %', 'Worst %']
        st.dataframe(by_reason, use_container_width=True)

    st.caption(
        "Negative slippage on a stop means the fill was worse than the stop price. "
        "Persistent overshoot is a monitoring-latency problem: barriers enforced by "
        "broker bracket orders do not drift."
    )
