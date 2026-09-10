"""
Which opportunities are worth proposing, and in what order.

The obvious default -- rank by ensemble score, take the top 10 -- is the one thing the
evidence rules out. Measured on 11,942 out-of-sample observations across 399 assets,
the score is *anti*-predictive within asset:

    Spearman(score, outcome) = -0.0452 +/- 0.0145 across 321 assets,  z = -3.11

That is the only statistically strong result this project has produced, and it points
the wrong way. Sorting by score descending would sort by the thing measured to precede
worse outcomes. (Sorting ascending is not the answer either: the anti-signal is real
but economically small, and the best band still loses 0.212% per bet.)

So the default ranking uses what IS known exactly and does not depend on any model:

    required edge = 2c / (w + l)

the fee's share of everything at stake -- the rake. It varies from about 0.9% to over
20% across this universe purely because barriers scale with volatility while the fee
does not. A low-rake bet needs less than one extra correct call per hundred to break
even; a high-rake bet needs five or more, which is beyond what professional systematic
funds achieve. Ranking by rake therefore orders candidates by *how little skill they
demand*, which is a real and honest basis for a shortlist.

The score is still displayed, and still gates nothing away silently -- but it does not
drive the order until a calibration exists that shows it earning its keep.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# Rake bands, in percentage points of the pot. See BarrierPolicy.required_edge().
HURDLE_BANDS = (
    (1.5, "Low"),
    (3.0, "Moderate"),
    (5.0, "High"),
    (float('inf'), "Very high"),
)


@dataclass
class Shortlist:
    """The proposals, plus why everything else was left out."""
    proposals: List[Dict] = field(default_factory=list)
    considered: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)
    basis: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.proposals

    def summary(self) -> str:
        if self.is_empty:
            reasons = ", ".join(f"{count} {reason}"
                                for reason, count in sorted(self.rejected.items(),
                                                            key=lambda kv: -kv[1]))
            return (f"Nothing worth proposing out of {self.considered} assets"
                    + (f" ({reasons})." if reasons else "."))
        return (f"{len(self.proposals)} of {self.considered} assets clear the bar, "
                f"ranked by {self.basis}.")


def hurdle_label(required_edge_pct: Optional[float]) -> Optional[str]:
    """
    Plain reading of how much skill a bet demands.

    Calibration: professional systematic funds typically run on a couple of points of
    edge, so anything in the top band is not a realistic bet for anyone.
    """
    if required_edge_pct is None:
        return None
    for ceiling, label in HURDLE_BANDS:
        if required_edge_pct < ceiling:
            return label
    return "Very high"


def select_proposals(opportunities: List[Dict], limit: int = 10,
                     max_required_edge_pct: float = 3.0,
                     held_symbols: Optional[set] = None) -> Shortlist:
    """
    Pick the shortlist worth putting in front of a person.

    A candidate must clear every one of these, and each rejection is counted so the
    UI can explain an empty shortlist rather than just showing nothing:

      * **economically viable** -- its barriers cover the round-trip fee with enough
        margin that break-even sits under the configured ceiling;
      * **sizeable** -- fractional Kelly, after the correlation haircut for the open
        book, puts it above the minimum bet;
      * **a cheap enough table** -- the rake is at or under `max_required_edge_pct`.
        The default of 3.0 admits the "Low" and "Moderate" bands and refuses anything
        demanding more skill than a good fund delivers;
      * **not already held** -- doubling a position is a separate decision.

    Ties on rake are broken by the score, ascending, since that is the direction the
    evidence weakly supports. It is a tie-break only; it does not drive the order.
    """
    held = held_symbols or set()
    shortlist = Shortlist(considered=len(opportunities),
                         basis="lowest skill required (the fee's share of the pot)")

    def reject(reason: str):
        shortlist.rejected[reason] = shortlist.rejected.get(reason, 0) + 1

    candidates = []
    for opp in opportunities:
        symbol = opp.get('symbol')

        if symbol in held:
            reject("already held")
            continue

        required_edge = opp.get('required_edge_pct')
        if required_edge is None:
            reject("no barrier data")
            continue

        if opp.get('tradeable') is False:
            reject("fees exceed the target")
            continue

        if not opp.get('is_favorable'):
            # Distinguish "too small to bet" from "negative expected value", because
            # they call for completely different responses from the user.
            warning = (opp.get('risk_warning') or '').lower()
            reject("below minimum size" if 'below minimum' in warning
                   else "negative expected value")
            continue

        if required_edge > max_required_edge_pct:
            reject(f"skill required above {max_required_edge_pct:.1f} pts")
            continue

        candidates.append(opp)

    candidates.sort(key=lambda o: (o['required_edge_pct'],
                                   o.get('raw_score', o.get('final_probability', 0.0))))

    shortlist.proposals = candidates[:limit]
    for rank, opp in enumerate(shortlist.proposals, 1):
        opp['proposal_rank'] = rank
        opp['hurdle'] = hurdle_label(opp['required_edge_pct'])

    logger.info(shortlist.summary())
    return shortlist
