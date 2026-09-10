"""
When a bet settles. One implementation, used by every caller.

There were three settlement paths and they disagreed:

  * `trading_system._monitor_existing_bets` -- price barriers AND the time barrier,
    but it only runs under `main.py --mode manual|automated`;
  * `bet_monitor._monitor_and_settle_positions` -- price barriers ONLY, no time
    barrier at all;
  * the dashboard, which delegates to the second one.

So anyone driving the system from the dashboard had no time barrier, and a position
that reached neither price level stayed open indefinitely. One forex bet sat open for
293 days against a 60-day limit. Divergent copies of a rule mean the rule does not
exist; this module is the single copy.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SettlementDecision:
    """Whether a position closes now, and why."""
    should_close: bool
    exit_type: str = ""          # win_barrier | loss_barrier | time_barrier
    reason: str = ""
    exit_price: Optional[float] = None

    @property
    def is_time_barrier(self) -> bool:
        return self.exit_type == 'time_barrier'


def decide(entry_price: float,
           win_price: float,
           loss_price: float,
           entry_time: datetime,
           current_price: Optional[float],
           max_hold_days: int,
           now: Optional[datetime] = None,
           bet_type: str = 'long') -> SettlementDecision:
    """
    Decide whether one position should close.

    Price barriers are checked first, then the time barrier, because a position that
    reaches a barrier ON the day it expires should be recorded as a barrier hit --
    that is what the calibration data needs to distinguish.

    `current_price` may be None (the feed failed, or the asset is no longer priced).
    A price barrier cannot then be evaluated, but **the time barrier still can**: a
    stale position must not become immortal just because its quote went missing. It
    closes at the last known price, and the reason says so.
    """
    now = now or datetime.now()
    held_days = (now - entry_time).days
    expired = bool(max_hold_days) and held_days >= max_hold_days

    if current_price is None:
        if expired:
            return SettlementDecision(
                should_close=True, exit_type='time_barrier',
                reason=(f"TIME BARRIER: held {held_days}d >= {max_hold_days}d limit, "
                        f"no current price available - closing at last known price"),
                exit_price=entry_price)
        return SettlementDecision(should_close=False,
                                  reason="no current price available")

    if bet_type == 'long':
        hit_win = current_price >= win_price
        hit_loss = current_price <= loss_price
        return_pct = ((current_price - entry_price) / entry_price) * 100
    else:
        hit_win = current_price <= win_price
        hit_loss = current_price >= loss_price
        return_pct = ((entry_price - current_price) / entry_price) * 100

    if hit_win:
        target_pct = (win_price / entry_price - 1) * 100
        return SettlementDecision(
            should_close=True, exit_type='win_barrier',
            reason=f"WIN THRESHOLD HIT: {return_pct:+.2f}% (target: {target_pct:+.2f}%)",
            exit_price=current_price)

    if hit_loss:
        stop_pct = (loss_price / entry_price - 1) * 100
        return SettlementDecision(
            should_close=True, exit_type='loss_barrier',
            reason=f"LOSS THRESHOLD HIT: {return_pct:+.2f}% (stop: {stop_pct:+.2f}%)",
            exit_price=current_price)

    if expired:
        return SettlementDecision(
            should_close=True, exit_type='time_barrier',
            reason=(f"TIME BARRIER: held {held_days}d >= {max_hold_days}d limit, "
                    f"closing at market at {return_pct:+.2f}%"),
            exit_price=current_price)

    return SettlementDecision(should_close=False,
                              reason=f"open, {held_days}d held, {return_pct:+.2f}%")


async def settle_positions(portfolio, prices: Dict[str, float],
                           max_hold_days: int,
                           now: Optional[datetime] = None) -> List[Dict]:
    """
    Apply `decide` to every open position and close the ones that qualify.

    Args:
        portfolio: a PortfolioManager
        prices: symbol -> current price IN USD. Symbols absent from this map are
            still evaluated for the time barrier.
        max_hold_days: the time barrier
        now: injectable for testing

    Returns a record per closed position, for reporting.
    """
    alive = await portfolio.get_alive_bets()
    if not alive:
        return []

    settled = []
    for bet in alive:
        decision = decide(
            entry_price=bet.entry_price,
            win_price=bet.win_price,
            loss_price=bet.loss_price,
            entry_time=bet.entry_time,
            current_price=prices.get(bet.symbol),
            max_hold_days=max_hold_days,
            now=now,
            bet_type=getattr(bet, 'bet_type', 'long'),
        )

        if not decision.should_close:
            logger.debug(f"{bet.symbol}: {decision.reason}")
            continue

        logger.info(f"CLOSING {bet.symbol}: {decision.reason}")
        try:
            await portfolio.close_bet(bet.bet_id, decision.exit_price, decision.reason)
            settled.append({
                'bet_id': bet.bet_id,
                'symbol': bet.symbol,
                'exit_type': decision.exit_type,
                'exit_price': decision.exit_price,
                'entry_price': bet.entry_price,
                'reason': decision.reason,
            })
        except Exception as e:
            logger.error(f"Error closing {bet.symbol} ({bet.bet_id}): {e}")

    if settled:
        by_type: Dict[str, int] = {}
        for record in settled:
            by_type[record['exit_type']] = by_type.get(record['exit_type'], 0) + 1
        logger.info(f"Settled {len(settled)} position(s): {by_type}")

    return settled
