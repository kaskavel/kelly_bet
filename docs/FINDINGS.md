# Kelly Bet — Findings

Running record of what has been measured, what was fixed, and where an edge could
plausibly come from. Written for whoever picks this up next, including a future me.

**The question this project has to answer:** can we find a signal that predicts the
first-touch barrier event well enough to clear its break-even? Everything else is
plumbing in service of measuring that honestly.

Last updated: 2026-09-10.

## In flight

The powered run identified in §6 item 0 is **executing now**:

```bash
python scripts/backtest.py --max-assets 400 --folds 1 --stride 2 --max-hold 30 \
    --out-scores data/scores_powered.csv --no-trade-log
```

Expect ~12,000 observations — the first sample large enough to see a 4-point edge
(§3 says ~10,100 are needed). Its analysis is chained and will appear at
`docs/powered_run_analysis.txt`. If that file is missing, re-run:

```bash
python scripts/analyze_edge.py data/scores_powered.csv --max-hold 30
```

**Read §3 before interpreting it.** Look for a STRONG verdict that replicates across
both time halves and is not concentrated in a handful of assets. Anything less is a
hypothesis, and this project has already produced two of those that evaporated.

---

## 0. The one identity that governs everything

For a driftless asset, the barrier geometry probability is `l/(w+l)` — and the
**pre-fee break-even probability is also `l/(w+l)`**. They are the same number.

```
required edge = break_even − geometry = 2c / (w + l)
```

Consequences, and they are not negotiable:

- **No barrier shape creates profit.** Fixed or σ-scaled, any k/m ratio, any width:
  on a driftless asset the probability you get is exactly the probability you need.
  Retuning the geometry moves both sides together.
- **Fees are the only structural cost**, and they shrink as barriers widen.
- **Everything else must come from genuine predictive edge.** There is no
  configuration that substitutes for it.

Verified numerically (`src/trading/barriers.py::required_edge`, asserted in tests):

| barriers | geometry | break-even | edge needed |
|---|---|---|---|
| fixed 5% / 3% | 37.50% | 43.75% | **6.25 pts** |
| σ-scaled (mean 9.2/6.1) | 40.01% | 43.28% | **3.26 pts** |
| σ-scaled, high-vol asset | 40.00% | 42.00% | 2.00 pts |
| σ-scaled, forex | 40.00% | 60.83% | 20.83 pts |

This is why the forex book was hopeless: it needed 20+ points of edge. It is now
refused automatically (`BarrierPolicy.is_economic`).

### The neutral point is ~40%, not 50%

The original system compared its "probability" against 50%, and treated 60% as a
10-point edge. It was never that. Note also that `l/(w+l)` is an **approximation** —
real bars are discrete and prices compound, so the truth is a couple of points off
and depends on step size relative to barrier width:

| barriers | `l/(w+l)` | log-space | measured (24 paths) |
|---|---|---|---|
| 5% / 3% | 0.375 | 0.384 | 0.400 ± 0.007 |
| 6% / 4% | 0.400 | 0.412 | 0.414 ± 0.008 |
| 3% / 2% | 0.400 | 0.406 | 0.434 ± 0.005 |
| 20% / 10% | 0.333 | 0.366 | 0.344 ± 0.017 |

Neither closed form wins. Where the reference matters, call
`BarrierPolicy.estimate_geometry_probability()` (which measures it) or read the base
rate the backtest reports from real data.

---

## 1. Where the original system stood

Six months of live paper trading, 87 bets, Oct 2025 → Apr 2026.

| metric | value |
|---|---|
| Realized P&L | +$216.92 on $41,943 turnover (**+0.52%**) |
| Reported probability at placement | 63.7% |
| Delivered win rate | **44.05%** (z = −3.75 vs the claim) |
| Break-even at realized payoffs | 43.73% |
| Edge per bet | **+0.05% of stake** — statistically zero |
| Reported equity | $11,001.32 (+10.0%) |
| **Actual equity** | **$10,112.83 (+1.1%)** |

The system was carefully built and not carefully designed: excellent plumbing around
a number that was not a probability.

---

## 2. What was broken, and is now fixed

Each of these is covered by a regression test.

### Correctness / safety

| defect | consequence | status |
|---|---|---|
| `can_continue_trading()` called with no argument | 608 lines of risk control never executed; `risk_events` had 0 rows after 6 months | fixed; 7 tests force each breach |
| `close_bet()` never removed the position from `active_bets` | equity double-counted every settled position — read +10.0% on a +1.1% account | fixed; equity now derived from the DB |
| Positions never marked to market | `current_price == entry_price` for a position's whole life; unrealized P&L permanently 0 | `mark_to_market()` added |
| Currency conversion lived only in the UI | trading core compared ¥/HK$ quotes to USD barriers; one bet recorded entry $31.67 vs current ¥4,899 | moved into `MarketDataManager.get_latest_data()`; missing rates now raise instead of passing through |
| HKD rate inverted | HK$100 converted to **$780** instead of $12.82 | fixed |
| **LSE prices treated as pounds, but the feed quotes PENCE** | CRH.L trades at 8,418 meaning GBP 84.18; labelling it GBP reported **$11,402 instead of $114.02**. All **73 `.L` assets** were 100x out. Dollar P&L happened to survive (shares and price scale inversely) but every displayed price, share count and absolute comparison was wrong | GBX added as a first-class currency worth 1/100 of GBP, derived automatically from the GBP rate; `.L` maps to GBX in both the converter and AssetSelector. 6 regression tests on real cached closes |
| No time barrier | one forex position open 4.6 months | `max_hold_days`, force-close at market |
| **THREE divergent settlement paths** | `trading_system` checked price + time barriers, but `bet_monitor` (which the dashboard delegates to) checked **price barriers only**. Anyone driving the system from the UI had no time limit, and the EURGBP=X position reached **293 days against a 60-day setting**. Divergent copies of a rule mean the rule does not exist | one implementation in `src/trading/settlement.py`; all three callers delegate. 16 tests, including the 293-day case and the unpriceable-and-expired case |
| Settlement was not checked on every entry into the app | the dashboard's page-load path never checked at all, so a position could stay overdue indefinitely while inflating equity, distorting the win rate and tightening the correlation haircut on new bets | settlement now runs on **every** entry and refresh. The page-load check is **offline** — stored marks only, zero API calls — which still catches every time-barrier exit, since the time barrier needs no quote. A structural test asserts all six entry paths call it |
| A position with no available quote could never expire | the realistic path to 293 days: the time barrier was only evaluated for symbols that returned a price, so a delisted or unsupported symbol became immortal | the time barrier is now evaluated even without a price, closing at the last known mark and saying so in the reason |
| `asset_type` hardcoded `'stock'` | all 87 bets mislabelled, including SLV and EURGBP=X | derived from metadata |
| 82,429 prediction rows stored as float32 BLOBs | `AVG(probability)` returned 6035.52 | all repaired; insert path casts with `float()` |
| Exchange timezone offsets not normalised | **1,899 distinct timestamps for 331 calendar dates** — cross-asset date alignment broken, any date index inflated ~5.7× | normalised to calendar day |
| **Cache mixes adjusted and unadjusted prices** | fake gaps: AVB shows $168.50 (2026-04-08) then $65.08 (2026-08-11), a false −61% bar. **126 such bars across 75 assets.** Under an SD volatility estimate this pushed AVB's σ to 67.7% and its barrier to the 40% cap — the UI would have offered a real bet against a nonsense target | volatility now estimated from the **median absolute deviation**, which ignores isolated outliers (AVB σ 67.7% → 4.5%, barrier 40% → 6.8%) while still reading genuine volatility (MANA 26.6% → 23.8%, AAPL 4.11% → 3.90%). **The underlying data still needs re-fetching with a consistent adjustment setting.** |
| Dashboard auto-refreshed on a timer | re-fetched the whole universe unattended and exhausted the market-data rate limit; every Streamlit rerun (tab switch, sort, row click) risked another fetch | auto-refresh removed; the page renders from the database and only an explicit "Refresh prices" button calls an API |
| Kelly called without per-asset barriers in BOTH dashboard paths | under volatility-scaled barriers every asset was sized against the config defaults — pricing a different bet than the one on screen | both paths now pass the asset's own barriers and the open-position count |
| Cash edits, resets and backups were loose hand-run scripts | no audit trail, and a portfolio could reach a state nobody could reconstruct | `src/portfolio/admin.py`: cash only moves through the ledger, destructive actions snapshot first, snapshots are never collateral damage. 23 tests |
| Snapshot filenames used second-resolution timestamps | restore takes an automatic safety snapshot, so two restores in the same second could **overwrite the snapshot being restored** — destroying it and making the restore a silent no-op | filenames uniquified, plus an explicit guard refusing a safety snapshot that collides with the restore source |
| Dashboard `place_bet()` did not forward the per-asset barriers | under volatility-scaled barriers **every placement path raised** — correctly, since the alternative is silently sizing a different bet than the one displayed and agreed to | the full opportunity row is now passed through; 4 regression tests pin the barriers into the stored `win_price`/`loss_price` |
| `PortfolioAdmin.summary()` built its counts after closing the connection | the resulting `ProgrammingError` was swallowed as a zero: a 121 MB database with 263k price bars reported "0 bars, 0 bets" | counts taken while the connection is open; regression test asserts they are non-zero |

### Statistical soundness

| defect | consequence | status |
|---|---|---|
| Label was `Close.shift(-5)/Close − 1 > 0.03` | path-independent, ignored the stop, 5-bar horizon against an 18.8-day hold — predicted an event nobody bets on | replaced with triple-barrier first-touch labels; validated against theory (random walk labels at 36.95% vs 37.5% geometry) |
| `train_test_split(random_state=42)` — shuffles by default | severe leakage: adjacent bars share indicator windows *and* overlapping label windows | purged, embargoed time-ordered splits |
| `class_weight='balanced'` | `predict_proba` shifted off the true base rate, then fed straight to Kelly | removed; all four models wrapped in calibration |
| Raw price levels as features | one scaler fitted on US mega-caps, applied to ¥4,899 and crypto quotes | scale-free feature set, **verified invariant to 1e-8 across a 24-million-fold price range** |
| LSTM/regression squashed point forecasts (`sigmoid(10r)`, `norm.cdf(r/0.02)`) | arbitrary mappings; 17.1% of LSTM calls pinned at the hardcoded 90% clamp | both now output the barrier probability directly |
| Kelly used `f=(bp−q)/b` | that is the all-or-nothing-stake formula; this bet risks only the stop distance | corrected to `f* = p/l − q/w` on fee-adjusted legs |
| Fees absent from EV | break-even understated by 6.25 pts | fees in both legs |
| `min_bet_amount` was a floor | raised small Kelly bets up to the minimum, breaching the position cap exactly when capital was lowest | now a skip threshold |
| `update_algorithm_performance()` was a stub | weights frozen at 0.2 since 2025-09-04; "the system learns which algorithms work best" was fiction | implemented (Brier-scored); first real weights produced |
| Hardcoded `if best_prob < 50.0` | wrong neutral point for a barrier bet | now each bet's own break-even |
| No backtester | every threshold set by intuition; 6 months produced 84 observations | `scripts/backtest.py`, walk-forward with purge |
| No calibration | 63.7% claimed vs 44.1% delivered, with nothing on screen to compare against | isotonic layer + a Reliability tab showing the break-even line |

Test count: **12 → 85**.

---

## 3. The measurement trap that made me wrong twice

**Barrier outcomes are not independent observations.** Two sources of dependence:

- **Overlap within an asset.** A bet at bar *i* and one at bar *i+1* share almost
  their entire forward window. With a 60-bar time barrier, 60 consecutive
  "observations" are close to one.
- **Correlation across assets.** Every bet is long, so a market-wide move resolves
  many of them the same way on the same day.

Measured: **a single 2,500-bar path estimates a barrier win rate to only about ±5
percentage points**, where the naive `sqrt(p(1-p)/n)` would claim ±1.

This burned me twice in this project:

1. I reported per-algorithm correlations as "within 1.4 SE of zero" using a naive SE.
   The conclusion held, but the confidence was overstated.
2. A 14-asset spot check said fixed barriers beat σ-scaled (−0.447% vs −0.814% per
   bet). With 5–9× the data the ranking **reversed** (−0.391% vs −0.126%). The spot
   check was noise and I should not have reported its direction.

`scripts/analyze_edge.py` now reports naive, asset-clustered and date-block-clustered
standard errors side by side, plus the implied effective sample size. **Use it before
believing any slice.**

### Measured effective sample sizes

| dataset | raw observations | effective | inflation |
|---|---|---|---|
| `scores_vol60.csv` | 1,787 | **127** | 14× |
| `scores_vol30.csv` | 512 | **14** | 36× |

The no-overlap filter is even blunter: of 1,787 bets in `scores_vol60`, only **134**
have non-overlapping forward windows.

### And therefore: every null result so far is under-powered

`analyze_edge.py` §6 computes what it would actually take. For `scores_vol60`
(sd of net return 7.43%, 14× inflation):

| edge to detect | net %/bet | independent bets needed | raw observations needed | had |
|---|---|---|---|---|
| 2 pts of win rate | 0.28% | 2,877 | **40,548** | 1,787 |
| 4 pts | 0.55% | 719 | **10,137** | 1,787 |
| 6 pts | 0.83% | 320 | **4,505** | 1,787 |
| 8 pts | 1.11% | 180 | 2,534 | 1,787 |

**This reframes everything below.** We have not shown there is no edge. We have shown
that a 4-point edge — roughly what the strategy needs — is invisible at the sample
sizes run so far. Distinguishing "no edge" from "edge we cannot see" requires roughly
an order of magnitude more observations, which is a matter of scaling the run (more
assets, stride 1, more folds), not of finding new data.

---

## 4. What we have learned about the edge hunt

### 4a. The signal, so far, is not there

On 1,035 out-of-sample observations, correlation with the actual barrier outcome:

```
regression  −0.028      rsi  −0.042      ENSEMBLE  −0.035
rf          +0.015      sma  −0.023      svm       −0.009
```

All indistinguishable from zero. Worse, **selecting on the score makes returns
worse**: the σ-scaled run's full sample was −0.126%/bet, but score ≥ 40 gave −0.464%
and score ≥ 45 gave −2.347%.

The monotone score bands I first found in the 84 live bets (27.3% → 41.7% → 45.0% →
56.7%) **did not reproduce** out of sample. That was noise at n=84.

### 4b. σ-scaled barriers are structurally better

Not because they create edge — they cannot — but because they halve the fee drag.

Matched A/B, 40 assets, 2 folds, `max_hold=30`:

| | fixed 5%/3% | σ-scaled 1.5σ/1.0σ |
|---|---|---|
| observations | 1,921 | 1,064 |
| win rate | 29.36% | 36.18% |
| geometry | 37.50% | 40.00% |
| gap to geometry | −8.14 pt | **−3.82 pt** |
| timeouts | 20.2% | **13.8%** |
| required edge | 6.25 pt | **3.96 pt** |
| **mean net per bet** | **−0.391%** | **−0.126%** |

### 4c. The time barrier truncates wins, and it matters

The win barrier sits 1.5σ out and the stop 1.0σ, so expected time-to-touch is ~2.25×
longer for a win. Too tight a limit therefore cuts off wins preferentially — a
property of the clock, not the model:

| `max_hold` | timeouts | win rate (geometry ≈ 40%) |
|---|---|---|
| 30d | 20.5% | 28.2% |
| 60d | 5.4% | 33.0% |
| 90d | 1.5% | 35.8% |

Set to 60. Worth 8 points of win rate over 30.

### 4d. One open lead: the mid-width barrier band

From the σ-scaled run's by-width breakdown:

```
win barrier      bets   win rate   break-even   net/trade
 4.3- 6.2%        266     34.59%      45.80%      -0.562%
 6.2- 7.9%        266     34.59%      44.28%      -0.739%
 7.9-10.0%        266     47.37%      43.41%      +1.441%   <- 4 pts of edge
10.0-27.4%        266     28.20%      42.36%      -0.642%
```

**Tested, and it did not survive. Consider this lead closed.**

Three independent checks, all negative:

1. **It moves.** In `scores_vol60` the best quartile was Q2 (5.6–7.2%), not Q3. The
   "mid-width band" is not a stable location.
2. **It is a handful of assets.** In `scores_vol30` the best quartile's top 3 assets
   supplied **541% of the total return** on n = 4, 7 and 6 bets (CSCO +15.6%,
   AMD +14.8%, ELV +12.1%). Only 11 of 25 assets had a positive mean. Remove three
   lucky names and the cell is negative.
3. **It does not replicate in time.** First half −1.955% (z = −1.89), second half
   +1.036% (noise). And on the 134 non-overlapping bets, nothing reaches even "weak".

Every quartile in both datasets reads "noise" or "weak" once clustered. This is what
a lead looks like when it was always noise — and it is exactly why §3's tooling
exists.

---

## 4d-bis. Translating the numbers into human terms

The quantities that matter are exact but their natural units are meaningless to a
person deciding whether to place a bet. The dashboard therefore states each one twice
— once precisely, once plainly:

| quant term | what it is | plain reading |
|---|---|---|
| break-even 43.28% | win rate needed for zero EV after fees | **"must win 43 in 100"** |
| required edge 3.26 pts | `2c/(w+l)`, the fee's share of everything at stake | **"fees take 3.3% of the pot"** — a rake |
| geometry ~40% | win rate the barrier shape gives a coin flip | **"pure luck already wins 40 in 100"** |
| barriers +9.28% / −6.19% | σ-scaled target and stop | **"risk $6.19 to make $9.28 per $100"** |
| — | how hard that is in practice | **Low / Moderate / High / Very high** |

The rake framing is the one that carries the insight. A 0.50% round trip on a bet
that only swings 10% consumes 5% of the pot; on one that swings 57% it consumes 0.9%.
Same fee, wildly different table. That is the entire structural argument for σ-scaled
barriers and for preferring volatile assets, expressed in a way that needs no maths.

Calibration for the "hurdle" bands: professional systematic funds typically run on a
couple of points of edge, so **anything reading "Very high" (5%+) is not a realistic
bet for anyone**, and the label says so.

---

## 4f. What the dashboard proposes, and on what basis

The default view is a shortlist of up to 10 proposals
(`src/trading/selection.py`). The ranking basis is worth stating plainly, because the
obvious choice is the one the evidence rules out.

**Not by ensemble score.** §4a measured the score as mildly *anti*-predictive within
asset (z = −3.11). Sorting by it descending would rank by the quantity that precedes
worse outcomes. Sorting ascending is not the answer either: the anti-signal is real
but economically small, and even the best band loses 0.212% per bet. The score is
displayed, and breaks ties in the direction the evidence weakly favours, but it does
not set the order.

**Ranked instead by the rake** — `2c/(w+l)`, the fee's share of everything at stake.
It is exact, model-free, and ranges from ~0.9% to over 20% across this universe purely
because barriers scale with volatility while the fee does not. Ordering by it ranks
candidates by *how little skill they demand*, which is defensible today.

A candidate must also be economically viable, sizeable after the correlation haircut,
under the rake ceiling (default 3.0 pts), and not already held. **Each rejection is
counted so an empty shortlist explains itself** — a screen that always finds ten
opportunities because it has ten slots conveys nothing. Live example:

```
10 of 60 assets clear the bar, ranked by lowest skill required.
 # Symbol Need right Fees take   Hurdle     Risk/Reward  Score Size
 1    NEM  42 in 100      1.8% Moderate -$10.86/+$16.28   60.2 $105
 2 5802.T  42 in 100      1.9% Moderate -$10.30/+$15.45   63.3 $130
 3   PALL  42 in 100      2.0% Moderate -$10.10/+$15.15   65.1 $143
rejected: {'skill required above 3.0 pts': 40, 'below minimum size': 4}
```

The screen says these are **the cheapest tables, not predicted winners**, because
that is what is true.

---

## 4e. The tooling built to answer these questions

The measurement apparatus is the durable output of this work. Without it, none of the
findings above were reachable.

| tool | what it answers |
|---|---|
| `scripts/backtest.py` | Walk-forward, purged, per-bar barriers, real first-touch outcomes with intraday High/Low, both fee legs. Emits a calibration dataset and a trade log. Sweeps barrier geometry via `--win-sigma/--loss-sigma/--horizon/--max-hold/--barrier-mode`. |
| `scripts/analyze_edge.py` | **Is a slice real?** Naive vs asset-clustered vs date-clustered SEs, effective sample size, score-band and within-asset timing tests, asset-concentration check, time replication, non-overlapping subsample, and a power table. |
| `scripts/fit_calibration.py` | Fits the isotonic score→probability map; **refuses to save** a calibration that cannot beat the base rate. |
| `src/trading/barriers.py` | Single source of truth for barrier placement; `required_edge()`, `break_even()`, `is_economic()`, `estimate_geometry_probability()`. |
| `src/prediction/calibration.py` | The isotonic layer, with a reliability report. |
| `src/ui/reliability_panel.py` | Dashboard tab: break-even line, predicted-vs-delivered by band, exit slippage. |
| `scripts/repair_prediction_blobs.py` | One-off repair of BLOB-encoded predictions. |

Typical loop:

```bash
python scripts/backtest.py --max-assets 400 --folds 3 --stride 1 --max-hold 30 \
    --out-scores data/scores_powered.csv --no-trade-log
python scripts/analyze_edge.py data/scores_powered.csv --max-hold 30
python scripts/fit_calibration.py --scores data/scores_powered.csv
```

**Rule of thumb established here:** never read a backtest number without running
`analyze_edge.py` on it first. Two of my own conclusions in this project were wrong
because I read naive standard errors.

---

## 5. Data constraints worth knowing

- **History**: 2025-06-02 → 2026-09-09, **320 trading dates**, 859 assets with ≥150
  bars.
- **The cache is SPARSE, and this is the binding constraint.** The median asset holds
  only **209 bars across a 320-trading-day span — ~35% of days are missing**. Two
  consequences, both nasty:
  - A `max_hold` of 30 *bars* reaches ~45 calendar days back, so the unusable tail of
    each series is much longer than it looks.
  - Indicator windows span more real time than intended: a "20-bar" volatility
    estimate covers ~30 calendar days.

  This is a data-collection gap, not a modelling choice, and **backfilling it is the
  single highest-leverage fix available**. It would roughly halve the effective time
  barrier and materially increase the usable sample.
- **Fold placement is delicate as a result.** Requiring every asset to be trainable
  collapsed the usable sample from ~24,400 observations to 82. Since features are
  scale-free and the model is pooled, an asset need not be trained on to be scored;
  `fold_boundaries` now picks the earliest start where ~40 assets can train and
  scores the whole universe from there.
- **Bars are daily**, despite `price_data.interval` having been labelled `'30min'`
  and the docs promising 30-minute data. Intraday High/Low is used for touch
  detection, but a barrier crossed and reversed within one day is invisible.
- **A purged walk-forward at `max_hold=60` cannot train the models on the earliest
  fold** — 120 training bars plus a 60-day purge exceeds what the front of the
  history provides. Model-based tests therefore run at `max_hold=30`; barrier-economics
  tests can use 60. **More history is the single cheapest unlock here.**
- Crypto universe is **1 asset** against a spec of 50.

---

## 6. Where an edge could still come from

Ranked by expected value of the experiment, not by ease.

0. ~~Scale the run~~ — **done, and it settled the wrong question.** Breadth added
   rows but not evidence (333× inflation). Do not re-run this expecting a different
   answer; the binding constraint is time, not assets.

1. **Get more history. This is now the critical path.** Detecting a 4-point edge needs
   ~208,000 effectively-independent-ish observations, and only distinct time periods
   supply independence. Eight months of test window cannot answer the question no
   matter how many assets are scored. Concretely: backfill the sparse cache (§5 — the
   median asset is missing ~35% of trading days) and extend the history back several
   years. Everything else on this list is premature until this is done.

1b. **Invert or replace the current features.** The one STRONG result is that the
   score is anti-predictive (z = −3.11). Momentum crossovers and RSI extremes are
   selecting bad entries for this bet. That is a concrete direction: test mean-
   reversion framings, and test the existing signals with the sign flipped — not
   because inverting is profitable as-is (it isn't, −0.212%/bet) but because a
   reliable negative is a much better starting point than a zero.
2. **Intraday bars.** Daily bars cannot see a barrier touched and reversed intra-day,
   and the original system's −18% realized stops were a monitoring-latency artefact.
   Finer bars fix both measurement and execution.
3. **Cross-sectional features.** Every current feature is per-asset and
   self-referential. Relative strength versus the universe, sector, or an index is
   the obvious missing family, and the date-normalisation fix above is what makes it
   computable.
4. **Asymmetric barriers with a directional signal.** Since geometry always equals
   pre-fee break-even, the only way a barrier choice helps is if it is *conditioned*
   on a signal. Worth testing k/m as a function of predicted direction.
5. **Regime conditioning.** Test whether any signal works in one volatility regime
   and not another, rather than pooling everything together.
6. ~~The mid-width barrier band~~ — **closed, see §4d.** It was three lucky assets.

**What is NOT worth more effort**: retuning k, m, horizon, or thresholds on the
current feature set. The identity in §0 says geometry cannot produce profit, and §4a
says these features supply no edge. Those knobs are exhausted.

---

## 6b. Is this worth continuing? A straight answer

**The specific system — daily technical indicators, first-touch barrier bets, liquid
large caps, 0.25% per side — is not salvageable.** Three independent reasons, none of
which is about implementation quality:

### It spends the whole prize on commissions

| configuration | fee drag | share of the equity risk premium |
|---|---|---|
| current: 22-day holds, 0.25%/side | **−5.60% / yr** | **86%** |
| same bet, 0.02%/side broker | −0.45% / yr | 7% |
| 6-month holds, 0.25%/side | −1.05% / yr | 16% |
| 6-month holds, 0.02%/side | −0.08% / yr | 1% |

Simply owning a diversified equity portfolio returns roughly **+6.5%/yr with zero
forecasting skill**. The current configuration burns 86% of that in commissions
*before* any model is consulted. The strategy is not competing against zero; it is
competing against a free 6.5%, and starting 5.6 points behind.

### It cannot be validated in a working lifetime

Detecting 3 points of edge at z = 2, given the measured sd and correlation:

| concurrent slots | correlation | years of live trading needed |
|---|---|---|
| 5 | 0.3 | **51** |
| 30 | 0.3 | **38** |
| 30 | 0 (fantasy) | 3.9 |

You cannot know whether it works, let alone profit from it. That is a property of the
turnover and the cross-asset correlation, not of the code.

### It is hunting in the most-mined dataset in existence

Daily OHLCV technicals on liquid large caps is what every retail system has tried
since the 1980s. RSI-14 and moving-average crossovers are in every textbook. §4a
found the ensemble mildly *anti*-predictive, which is what you would expect from
features this well known.

### But the dream is achievable — just not by prediction

"A statistically positive flow" does not require beating anyone at forecasting. The
leverage, ranked:

| change | required edge | improvement |
|---|---|---|
| nothing (fixed 5%/3%, 0.25% fee) | 6.25 pts | — |
| wider barriers (done) | 3.12 pts | 2× |
| **cheaper broker (0.25% → 0.02%/side)** | **0.25 pts** | **25×** |

**The single highest-leverage action in this entire project is changing broker, not
model.** A 25× reduction in the hurdle, available today, versus 2-3 points of edge
that has never materialised.

Three honest paths:

**A. Take the free money.** Diversified equity, periodic rebalance, near-zero
turnover. ~6.5%/yr expected, statistically positive, verifiable, and it uses the
portfolio ledger, risk controls and reconciliation already built here. Unexciting and
almost certainly the right answer.

**B. Keep the machine, change the bet.** Same infrastructure, but: a low-cost broker
or futures; holds of months rather than 22 days; and target documented risk premia
(12-month cross-sectional momentum, value, low-volatility) instead of 5-day
technicals. Those have decades of out-of-sample evidence behind them, unlike anything
currently in `prediction/`. Fee drag falls to ~0.08%/yr and validation moves from
decades to years.

**C. Keep hunting predictive edge, but move where the data is not mined.** Intraday
bars, small caps, crypto alts, or event-driven structure (index additions, earnings
drift). Plausible for a careful solo operator, but a multi-year project with a low
base rate, and it needs the data problems in §5 fixed first.

**What is definitively closed:** tuning `k`, `m`, `horizon`, thresholds or model
hyperparameters on this dataset. §0 shows barrier geometry cannot create profit and
§4a shows these features supply none. That search is over.

### The part of this work that keeps its value

`scripts/analyze_edge.py` — clustered standard errors, effective sample size,
concentration and time-replication checks, and a power table. It is strategy-agnostic
and it is the thing that stops you fooling yourself. It caught two of my own wrong
conclusions and one textbook overfit (§3). Whatever path is taken, run every candidate
strategy through it before believing anything.

---

## 7. Honest status

The system is now **correct**: it measures itself, refuses to present uncalibrated
scores as probabilities, sizes on the right formula, enforces its risk limits, keeps
one set of books, and can be backtested. That was the achievable part.

It is **not profitable**, and nothing measured so far suggests it is close.

But be precise about what has and has not been shown:

- **Shown:** the original "63.7% probability" was not a probability; the reported
  equity was wrong; the risk controls never ran; the label predicted the wrong event;
  fixed barriers are structurally worse than σ-scaled ones; the mid-width barrier
  lead was three lucky assets.
- **NOT shown:** that no edge exists. Every null result so far is under-powered by
  roughly an order of magnitude (§3). "We measured nothing" and "there is nothing"
  are different claims, and only the first is supported.

The honest position: **we have built an instrument capable of detecting an edge, and
have not yet run it at a sample size where an edge would be visible.** That run is
queued as §6 item 0.

Do not deploy capital on the basis of anything in §4 until a slice survives
`analyze_edge.py` with a STRONG verdict, replicated across both time halves and not
concentrated in a handful of assets.
