#!/usr/bin/env python3
"""
Tests for the proposal shortlist.

The behaviour that matters most is the refusal: the shortlist must be able to come
back empty, with a reason, rather than always producing ten rows because the screen
has ten slots.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.trading.selection import hurdle_label, select_proposals


def candidate(symbol, required_edge=2.0, score=55.0, favorable=True,
              tradeable=True, warning=''):
    return {
        'symbol': symbol,
        'required_edge_pct': required_edge,
        'raw_score': score,
        'final_probability': score,
        'is_favorable': favorable,
        'tradeable': tradeable,
        'risk_warning': warning,
    }


class HurdleLabelTestCase(unittest.TestCase):

    def test_bands(self):
        self.assertEqual(hurdle_label(0.9), "Low")
        self.assertEqual(hurdle_label(1.49), "Low")
        self.assertEqual(hurdle_label(1.5), "Moderate")
        self.assertEqual(hurdle_label(2.99), "Moderate")
        self.assertEqual(hurdle_label(3.0), "High")
        self.assertEqual(hurdle_label(4.99), "High")
        self.assertEqual(hurdle_label(5.0), "Very high")
        self.assertEqual(hurdle_label(20.0), "Very high")

    def test_missing_is_none(self):
        self.assertIsNone(hurdle_label(None))


class SelectProposalsTestCase(unittest.TestCase):

    def test_ranks_by_lowest_skill_required_not_by_score(self):
        """
        The score is anti-predictive (z = -3.11 on 11,942 observations), so ranking by
        it descending would sort by the thing measured to precede worse outcomes.
        Order comes from the rake instead.
        """
        opportunities = [
            candidate('HIGH_SCORE_EXPENSIVE', required_edge=2.9, score=90.0),
            candidate('LOW_SCORE_CHEAP', required_edge=0.9, score=45.0),
            candidate('MID', required_edge=1.8, score=70.0),
        ]

        shortlist = select_proposals(opportunities)
        self.assertEqual([o['symbol'] for o in shortlist.proposals],
                         ['LOW_SCORE_CHEAP', 'MID', 'HIGH_SCORE_EXPENSIVE'])

    def test_score_only_breaks_ties_and_prefers_lower(self):
        """Ties on rake break toward the lower score, the direction evidence supports."""
        opportunities = [
            candidate('A', required_edge=1.0, score=80.0),
            candidate('B', required_edge=1.0, score=40.0),
        ]
        shortlist = select_proposals(opportunities)
        self.assertEqual([o['symbol'] for o in shortlist.proposals], ['B', 'A'])

    def test_limit_is_respected(self):
        opportunities = [candidate(f'S{i}', required_edge=1.0 + i * 0.1)
                         for i in range(30)]
        self.assertEqual(len(select_proposals(opportunities, limit=10).proposals), 10)
        self.assertEqual(len(select_proposals(opportunities, limit=3).proposals), 3)

    # --- the refusals ------------------------------------------------------

    def test_empty_when_nothing_qualifies(self):
        """
        The shortlist must be able to say "nothing", rather than filling ten slots
        because ten slots exist.
        """
        opportunities = [candidate(f'S{i}', required_edge=8.0) for i in range(20)]
        shortlist = select_proposals(opportunities)

        self.assertTrue(shortlist.is_empty)
        self.assertIn("Nothing worth proposing", shortlist.summary())
        self.assertEqual(shortlist.considered, 20)

    def test_expensive_tables_are_refused(self):
        opportunities = [
            candidate('CHEAP', required_edge=1.0),
            candidate('DEAR', required_edge=5.4),
        ]
        shortlist = select_proposals(opportunities, max_required_edge_pct=3.0)

        self.assertEqual([o['symbol'] for o in shortlist.proposals], ['CHEAP'])
        self.assertIn('skill required above 3.0 pts', shortlist.rejected)

    def test_uneconomic_barriers_are_refused_with_their_own_reason(self):
        opportunities = [candidate('FX', required_edge=20.0, tradeable=False,
                                   favorable=False)]
        shortlist = select_proposals(opportunities)

        self.assertTrue(shortlist.is_empty)
        self.assertEqual(shortlist.rejected.get('fees exceed the target'), 1)

    def test_too_small_is_distinguished_from_negative_ev(self):
        """
        These need different responses from the user -- add capital versus do not bet
        -- so they must not be collapsed into one reason.
        """
        opportunities = [
            candidate('SMALL', favorable=False,
                      warning='Kelly size $31.75 below minimum $100.00 - no bet'),
            candidate('NEGATIVE', favorable=False,
                      warning='Negative expected value - no bet recommended'),
        ]
        shortlist = select_proposals(opportunities)

        self.assertTrue(shortlist.is_empty)
        self.assertEqual(shortlist.rejected.get('below minimum size'), 1)
        self.assertEqual(shortlist.rejected.get('negative expected value'), 1)

    def test_held_symbols_are_excluded(self):
        opportunities = [candidate('OWNED', required_edge=1.0),
                         candidate('FREE', required_edge=2.0)]
        shortlist = select_proposals(opportunities, held_symbols={'OWNED'})

        self.assertEqual([o['symbol'] for o in shortlist.proposals], ['FREE'])
        self.assertEqual(shortlist.rejected.get('already held'), 1)

    def test_missing_barrier_data_is_refused(self):
        shortlist = select_proposals([candidate('NODATA', required_edge=None)])
        self.assertTrue(shortlist.is_empty)
        self.assertEqual(shortlist.rejected.get('no barrier data'), 1)

    def test_empty_input(self):
        shortlist = select_proposals([])
        self.assertTrue(shortlist.is_empty)
        self.assertEqual(shortlist.considered, 0)

    # --- annotations -------------------------------------------------------

    def test_proposals_are_ranked_and_labelled(self):
        opportunities = [candidate('A', required_edge=0.9),
                         candidate('B', required_edge=2.0)]
        shortlist = select_proposals(opportunities)

        self.assertEqual(shortlist.proposals[0]['proposal_rank'], 1)
        self.assertEqual(shortlist.proposals[0]['hurdle'], 'Low')
        self.assertEqual(shortlist.proposals[1]['proposal_rank'], 2)
        self.assertEqual(shortlist.proposals[1]['hurdle'], 'Moderate')

    def test_summary_counts_rejections(self):
        opportunities = [
            candidate('OK', required_edge=1.0),
            candidate('DEAR', required_edge=9.0),
            candidate('OWNED', required_edge=1.0),
        ]
        shortlist = select_proposals(opportunities, held_symbols={'OWNED'})

        self.assertEqual(len(shortlist.proposals), 1)
        self.assertIn("1 of 3 assets clear the bar", shortlist.summary())


if __name__ == '__main__':
    unittest.main()
