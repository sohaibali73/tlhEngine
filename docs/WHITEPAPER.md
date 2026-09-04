# TLH Engine with YANG: the algorithms, the rigor, and why tax-loss harvesting should be AI-assisted

*Technical and positioning write-up. Everything described here is implemented in this repository and covered by the
test suite (249 tests, wash-sale and holding-period rules hand-verified against IRS Publication 550 examples).*

---

## 1. Executive summary

Tax-loss harvesting is usually sold as a simple idea: sell losers, buy something similar, bank the loss. In practice
it is a constrained, multi-period optimisation under a hostile rule set. Doing it well requires four things that
rarely live in one place: lot-level tax accounting that is right to the day, a factor risk model that knows how
different two securities really are, an optimizer that trades tax alpha against tracking error without ever
tripping a wash sale, and a multi-year view of embedded gains, brackets and concentration.

TLH Engine puts all four in a single desktop application and adds **YANG**, an embedded co-pilot built on Claude that
operates the engine through 63 typed tools. YANG can run and tune harvests, build model portfolios with a dozen
construction strategies, design and evaluate trade plans, fit and validate risk-model variants, plan multi-year
diversification of concentrated positions, and even propose changes to the engine's own code, all inside guardrails:
it cannot trade, it cannot bypass the wash-sale engine, and every code change runs in a sandbox with the compliance
test suite before a human approves it.

The consequence for an advisory business is simple. The mathematics is done, consistently, on every account, every
day, with an audit trail. The advisor's job becomes the client: understanding their tax picture, explaining the
plan, and winning and keeping relationships. Section 10 makes that argument in detail.

---

## 2. Architecture in one page

Five decoupled layers, all persistent, all reproducible.

| Layer | What it holds | Key design decision |
|---|---|---|
| Data | Norgate EOD prices (total-return and unadjusted), GICS classifications, fundamentals, shares outstanding, index membership through time, macro series | Every pull is an immutable **snapshot** (Parquet + DuckDB). Every run and model fit records the snapshot id it consumed, so "what did the model see on date X" is answerable offline. |
| Portfolio & tax | Entities (households), accounts (taxable, IRA, Roth, 401k), lots, transactions, closures, scheduled DRIPs, tax profiles, carryforwards | SQLite state of record in WAL mode. Persist by asset id, display by symbol. |
| Risk & optimisation | Two factor risk models, harvest optimizer, basket builder, twelve strategies, backtester, glide-path optimizer, risk analytics | Everything is a plain Python module the co-pilot can read and propose diffs to. |
| AI orchestration | YANG (Claude via the Anthropic Messages API), tool executor, local sandbox, change registry, scheduled agent | Manual tool loop, streamed; code changes are draft → sandbox → diff → human approval → versioned promotion. |
| Presentation | PySide6 desktop app, Plotly charts, Excel export | Twelve screens, dense sortable tables, an interactive how-to. Nothing places orders; trade tickets are the terminal output. |

---

## 3. The tax engine

### 3.1 Lots and holding periods
Every purchase creates a lot with acquisition date, quantity, cost per share (fees included), a holding-period start
that can differ from the acquisition date after wash-sale tacking, and a basis adjustment that carries disallowed
losses. Lot selection supports FIFO, LIFO, HIFO, specific identification and a harvest-oriented MAX_LOSS method that
prefers short-term losses (worth more) and then the highest basis.

Long-term status follows the statutory convention exactly: the holding period begins the day after acquisition, so a
sale on the anniversary is still short-term and the first long-term day is anniversary + 1. Leap-day acquisitions roll
to March 1. This is tested against dates on both sides of every boundary.

### 3.2 The wash-sale engine (IRC §1091)
The engine is bidirectional and cross-account:

- **Window**: 30 calendar days before and after the loss sale, 61 days inclusive.
- **Scope**: every acquisition in every account of the same tax entity, including IRAs and Roths (Rev. Rul. 2008-5:
  a replacement inside an IRA disallows the loss permanently, with no basis step-up).
- **Substantially identical**: three tiers in an editable mapping. Identical (same security, share classes of one
  issuer), presumed identical (different-issuer ETFs on the same index; on by default because the IRS has not ruled),
  and substitute (highly correlated, different index: the wash-safe replacement set).
- **Matching**: replacement shares absorb loss shares in chronological order of acquisition; each replacement share
  can be used once across all sales; the shares being sold are never their own replacement; partial matches disallow
  a proportional part of the loss.
- **Consequences**: the disallowed loss is added to the replacement lot's basis and the sold lot's holding period
  tacks onto it.
- **Both directions in time**: recording a loss sale looks back for replacements and forward at scheduled purchases
  and DRIPs; recording a purchase looks back 30 days and retroactively washes earlier loss sales.
- **Explainability**: every determination produces a sentence a client can read: which shares, in which account, on
  which date, how many days from the sale, how much loss is disallowed, when the name may be repurchased.

The optimizer never treats wash safety as a soft penalty. Candidate sells that fail the screen never enter the
problem; candidate buys are screened against realised and proposed loss sales; the final trade list is re-screened and
the run raises if anything slipped through.

### 3.3 Netting and carryforward
Schedule D netting is implemented as written: net within character (carryovers included), net across characters
with the surviving loss keeping its character, apply the ordinary-income offset short-term first, carry the remainder
forward retaining character. Filing status sets the offset.

### 3.4 Brackets
A bracket schedule (federal ordinary, federal long-term, NIIT threshold, state rate; 2026 approximate defaults per
filing status, editable) lets every after-tax number stack the gain on the client's other taxable income instead of
assuming a flat rate. The same schedule is expressed as a sum of hinge functions,

    tax(g) = Σ_k (r_k − r_{k−1}) · max(0, g + I − T_k) + r_NIIT · max(0, g − max(0, N − I)) + r_state · g,

which is convex and therefore usable directly inside the optimizers. Tests confirm the hinge form equals the stacked
bracket computation to the cent across incomes and gain sizes.

---

## 4. Risk models

Two estimators share one output object (exposures, factor covariance, specific variance, factor returns, diagnostics),
so every downstream component works with either.

### 4.1 barra_lite
Market, six styles (value, momentum, quality, size, low-vol, growth), GICS sectors, optional macro block. Cap-weighted
z-score standardisation with winsorisation, daily cross-sectional weighted least squares with the constraint that
cap-weighted sector returns sum to zero (so the market factor is the market), EWMA factor covariance with diagonal
shrinkage, EWMA specific variance shrunk toward the cross-sectional median. Fits 640 names in about three seconds.

### 4.2 The Equity Risk Model (ERM)
A production-grade multi-descriptor model in the Barra USE4 tradition.

- **Descriptors → styles**: size (log cap), non-linear size (cube of size, orthogonalised), beta (EWMA regression on
  the cap-weighted market), momentum (12-1 relative strength with 126-day half-life plus 6-1), residual volatility
  (EWMA daily std, cumulative range, idiosyncratic sigma), value (B/P, E/P, S/P, FCF/P), quality (ROE, net and gross
  margin, low leverage), growth (EPS and revenue growth), liquidity (1/3/12-month share turnover), leverage.
  Fixed composite weights, standardised, re-standardised, orthogonalised where Barra does (residual vol on beta and
  size, liquidity on size). Descriptors with no data drop out of their composite; styles with no usable descriptors
  drop out of the model, and the fitted spec records what was actually used.
- **Industries**: GICS industry groups by default (sector or industry selectable), constrained so the market absorbs
  the cap-weighted industry mean.
- **Estimation**: daily WLS with square-root-cap weights capped at the 95th percentile so mega-caps do not dominate,
  optional Huber M-estimation, factor t-statistics and R² recorded per day.
- **Factor covariance**: EWMA volatility (half-life 84) times EWMA correlation (half-life 504), Newey-West correction
  for serial correlation, eigenfactor risk adjustment (Monte Carlo, scale 1.2, inflating variances of eigenfactors
  that sample covariance systematically understates), and a volatility regime adjustment from the EWMA of the
  cross-sectional bias statistic.
- **Specific risk**: EWMA residual variance with its own regime adjustment, Bayesian shrinkage toward the cap-decile
  mean, and a structural (characteristic) regression for names with too little history.
- **Assets without characteristics** (ETFs, funds) receive exposures by ridge time-series regression on the factor
  returns, so benchmark ETFs and replacement ETFs live in the same covariance as single stocks.

Live on the current universe: 36 factors, 25 industry groups, average cross-sectional R² of 0.39, market t-statistic
above 5, eigenfactor gammas in [0.97, 1.13], regime multiplier 1.12.

### 4.3 Validation, not just fitting
The Risk lab runs out-of-sample **bias tests**: refit the model at the start of each period, predict the volatility of
test portfolios (cap-weighted market, equal weight, the client's holdings, long-short style portfolios), compare with
realised returns. The bias statistic std(realised / predicted) should sit inside 1 ± √(2/n). A model that
under-forecasts risk is flagged before it is used to size a harvest.

---

## 5. The harvest optimizer

### 5.1 Formulation
Variables: s_l ∈ [0, 1], the fraction of candidate loss lot l to sell (taxable accounts only), and b_j ≥ 0, dollars to
buy of replacement candidate j. Post-trade holdings h' = h − S s + B b, weights w = h'/V, active weights a = w − w_b.

Maximise

    w_tax · Σ_l benefit_l s_l / V
    − w_te · TE(a)² / te_budget²
    − w_fac · ‖X_style' (w − w_0)‖² / drift_budget²
    − cost_bps · turnover

subject to cash neutrality within a tolerance, no shorts, per-name weight caps, sector drift bounds, a turnover budget,
an optional hard tracking-error cap and an optional target loss. TE² is expressed through the factor structure,
Σ = X F X' + D, in Cholesky form so the problem stays a convex QP (CLARABEL, with OSQP and SCS fallbacks).

### 5.2 What makes it more than a QP
- **Constraint hierarchy as a first-class control**: the user orders tax alpha, tracking error and factor neutrality;
  the order maps to geometrically decaying weights, and a six-way comparison shows how the trade list changes under
  every ordering. A **frontier** sweeps hard TE budgets and plots tax alpha against TE, so the trade-off is a curve,
  not a point.
- **Tax alpha, properly defined**: benefit today at the marginal rate of the loss's character minus the present value
  of the higher tax on eventual liquidation (deferral haircut over a configurable horizon; infinite horizon models a
  step-up at death).
- **Replacements are found, not assumed**: curated substitutes first, then GICS peers ranked by trailing correlation,
  every candidate wash-screened, every candidate's correlation shown.
- **Minimum trade sizes** via a two-pass heuristic (solve, zero sub-threshold trades, fix them, re-solve), documented
  as a heuristic rather than hidden.
- **Two modes**: opportunistic (replacements may absorb at most 125% of the proceeds of the names they replace, so a
  small harvest cannot quietly rebalance the whole book) and full-rebalance (buys may include benchmark or model-
  portfolio constituents, which is how the book migrates toward a target while harvesting).
- **Persisted runs**: every run stores trades, blocked lots with reasons, replacement candidates, before/after
  exposures, TE decomposition and sector weights, so any past recommendation can be reopened without recomputation.

---

## 6. Model portfolios and construction strategies

The basket builder solves a min-TE QP with a name cap enforced by iterative pruning, weight caps, a sector band that
relaxes gracefully when a tight name cap makes it infeasible, and style tilts. A basket can become the **benchmark**,
turning "harvest toward a model portfolio" into the same optimizer with a different target.

Twelve construction strategies share one contract and one covariance input (factor model or Ledoit-Wolf sample):

| Strategy | Formulation |
|---|---|
| Minimum variance | argmin w'Σw |
| Maximum diversification | min y'Σy s.t. y'σ = 1 (Choueifaty-Coignard), then capped |
| Risk parity | convex log-barrier: min ½ w'Σw − (1/n) Σ log w_i, then normalised; risk contributions equalise to within 0.1% |
| Hierarchical risk parity | single-linkage clustering on correlation distance, quasi-diagonalisation, recursive bisection with inverse-variance allocation |
| Mean-variance | Grinold alphas α = IC · σ · z from style signals; benchmark-relative risk by default |
| Black-Litterman | equilibrium prior π = δΣw_b with δ from a market Sharpe, views with confidence-scaled Ω, posterior mean into MVO |
| Minimum CVaR | Rockafellar-Uryasev LP on trailing daily scenarios |
| Stratified indexing | sector × size strata sampled in proportion to benchmark weight, then min-TE reweighting |
| Factor tilt | min TE with target active style exposures |
| Tax-aware transition | explicit sell/buy variables so net realised gain (gain-fraction · sell) is linear; losses offset gains inside the budget; a name may not be sold and re-bought in the same rebalance |
| Equal weight, cap weight | baselines |

A walk-forward **backtester** rebalances monthly, quarterly or weekly with trailing shrunk covariance and price-based
signals recomputed at each date, drifting weights between rebalances, one-way turnover costs, and point-in-time index
membership when the snapshot has it. It reports CAGR, volatility, Sharpe, drawdown, tracking error, information ratio,
turnover and hit rate, and states its own caveats (survivorship where membership is missing, look-ahead in
fundamental signals because Norgate fundamentals are current-only).

---

## 7. The TLH model builder

A drag-and-drop pipeline of typed blocks: Universe → Filter → Rank → Benchmark → Construction → Tax-aware transition
→ Harvest → Save/export. Left-to-right order is execution order, parameters are edited in a schema-driven panel,
validation runs live, and the run produces the same artifacts as the manual screens (a basket, a harvest run, an Excel
workbook). Pipelines are JSON, so YANG can author them from a sentence.

---

## 8. Embedded gains and concentration

This is where most TLH products stop and most advisors improvise. The engine treats it as an optimisation problem.

### 8.1 Diagnostics
Per position: weight, embedded gain split by term, tax if liquidated (bracket-stacked, NIIT, state; taxable accounts
only), share of total or active risk, specific volatility, beta, idiosyncratic share of own risk, and a **lock-in
ratio**: tax cost of liquidating the name per 1% of tracking error it removes. Portfolio level: HHI, effective N,
gain-weighted concentration, locked-in percentage.

### 8.2 The glide path
Choose dollars d_t to sell in each of T periods to minimise

    Σ_t disc_t [ tax_t(d_t) + cost · d_t ] + (1 − p_stepup) · terminal_tax(R_T)
    + (λ/2) Σ_t disc_t σ_ε² R_t² / W − Σ_t disc_t α R_t

where R_t is the concentrated value still held, tax_t is the convex bracket function on the taxable gain, loss offsets
enter as variables u_t with a cumulative availability constraint (today's harvestable losses, expected future losses,
carryforwards), short-term lots pay ordinary rates until they turn long-term, and α is the client's alpha view on the
stock. Optional constraints: an annual gain budget and minimum diversification by a given year. The optimum is compared
against sell-now, equal instalments and hold on the same objective, with infeasible policies flagged. Live example:
a 5-year schedule that lands exactly on a 0.5% gain budget while cutting tracking error to target from 10.1% to 4.7%.

### 8.3 Monte Carlo
Stock paths are beta × market plus idiosyncratic noise from the risk model; each policy is applied annually with
bracket taxes on each sale and terminal long-term tax unless step-up. Output: fan charts, terminal after-tax wealth
percentiles, CVaR and the probability that a policy beats selling now.

### 8.4 Hedging and alternatives
Black-Scholes collars with a zero-cost call solver, payoff tables, and flags that matter in practice: constructive-sale
exposure when the collar band is narrower than the 15% practitioner threshold (§1259), straddle rules when the lot is
short-term (§1092), in-the-money covered calls that can terminate the holding period. Alternatives are compared on
one page: donate appreciated shares versus sell-then-donate (with the 30%-of-AGI flag), gifting to a lower bracket
(carryover basis, kiddie tax noted), Section 721 exchange fund breakeven (fees versus the present value of deferral),
and the option value of a basis step-up.

### 8.5 From analysis to trades
A **completion portfolio** holds the locked names at their current weights and optimises the rest of the book to
minimise tracking error (live: 5.6% → 2.5% with a 40-name sleeve around three locked positions). A **gain-offset plan**
sells a chosen amount of the concentrated name from the highest-basis lots and pairs it with wash-safe harvestable
losses so the net realised gain is minimised; the result is a normal reviewable run with every leg screened.

---

## 9. Risk analytics

Total or active risk decomposed into market, style, industry and specific contributions, per factor and per holding
(marginal contribution and share of risk). Factor stress tests in sigma or raw units with conditional propagation to
correlated factors (Δf_others = Σ_oS Σ_SS⁻¹ Δf_S), presets from market -2σ to liquidity shocks, historical replay of the
model's own factor returns through today's exposures, and parametric VaR and expected shortfall.

---

## 10. Why TLH engines should be AI-assisted with tools

### 10.1 The three failure modes of today's offerings
1. **Black-box robo harvesters** run a fixed daily rule. They are fine on ETF portfolios and nearly useless on the
   cases that matter: concentrated stock, embedded gains, multi-account households, a client who wants to keep
   their employer's shares, a client whose bracket changes next year.
2. **Spreadsheet quants** can do all of it, for one client at a time, with no audit trail and no consistency. Every
   advisor's book becomes hostage to one person's availability.
3. **Chatbots without tools** talk about tax-loss harvesting fluently and cannot compute a single wash-sale window
   correctly.

### 10.2 What "AI with tools" changes
YANG is a reasoning layer over a deterministic engine. It never computes a wash-sale window itself; it calls the
engine that does and that is tested against IRS examples. It never estimates risk from memory; it fits the model. It
never invents a trade list; it runs the optimizer or hands a plan to the evaluator that screens every leg. The value
of the model is in *choosing what to ask*, sequencing tools, reading the results, and explaining them in English.

Concretely, the division of labour is:

| The engine (deterministic, tested) | YANG (reasoning, tools) | The advisor (relationship) |
|---|---|---|
| Lot accounting, holding periods, wash-sale determinations, netting | Runs harvests with the right configuration for this client, explains every blocked lot | Knows the client's income, plans, risk tolerance, sentiment about the stock |
| Risk model fitting and validation | Fits variants, compares them, flags a stale model | Decides how much tracking error the client will tolerate |
| Optimisation, frontier, strategies, backtests | Builds and ranks model portfolios, writes the summary | Chooses the model portfolio narrative that fits the client |
| Glide paths, Monte Carlo, hedging maths | Turns "he wants out of the stock over five years without a tax shock" into a plan and a trade ticket | Has the conversation, wins the account, keeps it |
| Audit log, run history, exports | Files unattended reports, queues code changes for approval | Reviews, approves, signs |

### 10.3 Guardrails are what make delegation safe
- **No trading**: the engine emits tickets; nothing routes orders.
- **No bypass**: wash-sale screening is a pre-filter and a post-check in every path, including plans YANG designs.
- **Code changes are governed**: YANG may propose changes to the model, optimizer or substitute map, but the change
  runs in an isolated sandbox with the module's tests plus the wash-sale suite as a canary; a human reviews the diff
  and the test output; promotion creates a new code version and the prior version stays available for rollback.
- **Everything is logged**: every tool call, run, change, approval and cost.
- **Every number carries its assumptions**: bracket schedule, other income, snapshot id, model version.

### 10.4 What this does to an advisory practice
- **Consistency**: every household gets the same quality of tax logic and the same explanations, whether the book is
  ten accounts or a thousand.
- **Time**: the quant work that took a director days per complex client now takes minutes, unattended if scheduled
  (daily harvest scan, wash-window watch, weekly model health, month-end transition check).
- **Explainability sells**: a client shown why a lot is blocked, what a collar costs, or why year three of the glide
  path is heavier than year two trusts the plan. Every screen and every YANG answer is built to produce that sentence.
- **Compliance posture**: an audit trail of determinations, runs and approvals is stronger than any spreadsheet.
- **Focus**: advisors spend their hours on prospecting, discovery and retention. The engine and its co-pilot do the
  mathematics, and the advisor supervises rather than computes.

### 10.5 Why not just let the AI do everything?
Because the failure cost is asymmetric. A wash sale, a constructive sale, a mis-bracketed liquidation are compliance
and client-trust events. Deterministic, tested code should own those decisions; the model should own the judgment
about which analyses to run and how to explain them; the human should own approval. That is exactly the split this
system enforces.

---

## 11. Engineering rigor

- 222 automated tests; the tax rules are tested with hand-computed expected values, including Publication 550's
  partial-wash example (100 shares sold, 75 replacement shares, $750 of $1,000 disallowed), IRA replacements, share-class
  groups, same-day buy-and-sell preservation of total loss, retroactive washes, and every day of the 61-day window.
- Convex formulations verified against independent computations (hinge-function tax equals bracket tax; tracking
  error from the factor structure equals the dense covariance; risk decomposition sums to total variance).
- Live verification against Norgate data at every step: model fits, harvests, baskets, strategies, glide paths,
  hedges and the co-pilot's tool loop.
- Reproducibility: immutable snapshots, versioned models, persisted runs, versioned code.
- Portable: one-click PyInstaller build; `.env` and state travel with the executable.

---

## 12. Limitations, stated plainly

- Norgate fundamentals are current values, so fundamental style factors carry look-ahead in the historical fit and in
  backtests; today's exposures and covariance are unaffected. A point-in-time vendor would remove this.
- Bracket figures are approximate 2026 defaults and must be verified against the published tables; state treatment is
  a flat rate.
- Hedging tax flags follow practitioner heuristics, not rulings; the engine flags, a tax adviser clears.
- The eigenfactor adjustment assumes Gaussian factor returns; the macro block is available in barra_lite only.
- Nothing here is tax or investment advice. The engine quantifies conventions and shows its assumptions; people decide.

---

## 13. Roadmap that follows naturally

Read-only broker connectivity for live holdings (still no order routing), point-in-time fundamentals, household-level
asset location across taxable and deferred accounts, and multi-client batch runs with YANG-written client letters.

---

## 14. September 2026 additions: the flagship release

### 14.1 A model library instead of a model
The engine now ships thirteen named risk models, each a `RiskModelSpec` preset with a one-paragraph rationale. Three
families:

- **Fundamental.** ERM standard, short-horizon (fast half-lives for a one-month decision), long-horizon (four-year
  window for strategic budgets), robust (Huber cross-sections), plus barra_lite with and without the macro block.
- **Statistical.** The *Potomac calibrated covariance* from the 2026 calibration study (126-day window, equal weights,
  Ledoit-Wolf constant-correlation shrinkage; 189-day exponential for the one-month horizon), the tight-pair sample
  covariance for near-identical substitutes, and an asymptotic PCA model whose factor count comes from the
  Ahn-Horenstein eigenvalue-ratio test. Statistical covariances travel through the same `FittedRiskModel` object as
  eigen-factor models (`stat:k` factors), so optimizers and analytics need no special case.
- **Hybrid and dynamic.** ERM plus principal components of its own residuals (co-movement the descriptors miss), and two
  dynamic factor covariances for any fundamental model: GARCH(1,1) variance forecasts per factor over the decision
  horizon with EWMA correlations, and a calm/stress mixture weighted by the probability of stress given today's market
  volatility.

### 14.2 Calibration as a feature
The calibration study that chose the statistical model is now a tool, not a document. The Risk lab re-runs the
walk-forward grid (lookback x weighting x estimator x horizon) on the live snapshot with non-overlapping forward
windows anchored to the longest lookback, twenty seeded five-name baskets against the equal-weight universe, sampled
pairs and the seven metrics of the original paper (volatility bias ratio and Spearman, correlation bias, RMSE and
Spearman, tracking-error bias ratio and Spearman) ranked within horizon into a composite score. Two extensions: a PCA
arm in the estimator grid and the client's own holdings as an additional test basket. A substitute-pair study shows
the one place the recommendation reverses: a 0.997 correlation shrunk toward the universe average triples the forecast
tracking error of an IVV-versus-SPY pair, so pairs use the unshrunk sample matrix.

### 14.3 Construction beyond long-only
Five strategies join the twelve: an **integrated multi-factor** portfolio (value, momentum, quality and low volatility
combined into one sector-neutral score, with the "mixed sleeves" alternative for comparison), **defensive equity** with
a covariance-implied beta cap, **quality-momentum**, a **130/30-style long/short extension** and a **market-neutral
overlay** around existing holdings. The long/short solver splits names into disjoint tranches by composite score
before optimising (the bottom tranche is short-eligible, the rest long-eligible), so a name is never held long and
short at once, the extension is sector- and style-neutral, and the net book carries the target beta. The point of the
extension is continuous loss generation: `optim/longshort.py` simulates net capital losses by year for long-only and
long/short books from first principles (basis dispersion, volatility, wash lockouts), prices the financing, and builds
the tax-neutral "Exchange" glide that divests a concentrated position at exactly the pace the extension's losses pay
for. Published industry profiles are carried as reference data for comparison only. `optim/overlay.py` sizes
index-futures beta overlays (micro and E-mini contracts, SPAN-style margin, carry) and states the Section 1256 and
straddle questions that tax counsel must clear. A seventeen-recipe **sample model-portfolio library** builds all of it
in one click.

### 14.4 Every state's tax rules
`tax/state_rates.yaml` encodes how each of the fifty states and DC taxes capital gains: ordinary treatment, no tax,
percentage exclusions (Arkansas, New Mexico, North Dakota, South Carolina, Vermont, Wisconsin), flat capital-gains
rates (Hawaii, Massachusetts, Montana), Washington's excise, and the investment-income surtaxes of Maryland,
Massachusetts and Minnesota, with simplified brackets and local-tax notes. Combined with the federal schedules and
NIIT this yields the marginal short- and long-term rates a harvested loss is worth for a client in any state at any
income. The figures are approximate planning values and are labelled as such wherever they appear.

### 14.5 Built for the person who is not a quant
A **Start here** screen reduces the product to three clicks: import a broker export (column names are mapped from a
dictionary of broker aliases; missing dates and costs get documented defaults and a flag), pick the client's state,
filing status and income, and press one button that refreshes data, fits a model if none exists, runs the wash-safe
harvest and explains the result in sentences generated from the numbers. A simple mode hides the quant workbenches; an
expert mode shows them. YANG gained tools for the same flow (`import_holdings`, `set_tax_setup`, `state_tax_rates`,
`one_click_harvest`) so the conversation can start from "here is the client's Schwab file".

### 14.6 Leverage without futures, and a tactical overlay for Potomac's strategies
The custodians that hold advisory accounts do not offer futures and do not permit direct shorting, but they do allow
leveraged and inverse ETFs on margin. The `levered_beta` model therefore builds a 1.5-beta book from S&P 500 stocks plus
2x/3x S&P funds and an optional Reg-T loan, minimising tracking variance against 1.5 times the index plus the real
costs of leverage (fund expense ratios, the volatility drag of daily rebalancing, margin interest) under initial-margin,
house-maintenance-by-leverage and equity-buffer constraints. Tracking error comes first: by default every index name is
held at 1.5 times its weight (full replication, no name cap, no minimum position), so the stock sleeve tracks the index
exactly and the only tracking error is what the leverage layer adds. Leveraged funds are modelled as k times the index
basket itself plus their measured tracking noise, because a fitted time-series beta under-states a 2x fund and would
under-size the position, and because a fitted fund row carries an idiosyncratic term that would make the fund look
riskier than it is. Costs are compared on one footing: a fund pays its expense ratio, the financing embedded in its swaps
and the volatility drag of daily rebalancing, a loan pays margin interest. On live data the default book (index at 1.5x
on a 50% loan) has a model tracking error of about 0.03% and a realised leverage-layer tracking error of about 0.3% a year
at monthly rebalancing; the cash-only book, where 2x and 3x funds carry the leverage, is about 0.5%. The model is always
built against the index benchmark, never against a saved basket. The margin report states the uniform market drop that
would trigger a call. The tactical overlay is the plug-in for Potomac's own strategies: each strategy holds the same five
tactical funds at 80/5/5/5/5 target allocations, and its exposure is read daily from the funds' NAVs (a fund in cash prints a
flat NAV; a risk-on day is worth the fund's slow beta), generated at the prior close and traded at the next close. That signal,
a strategy export, a manual beta or a blend sets a target beta between 0 and 1.5, and the engine sizes the single
leveraged or inverse fund that moves the whole book to that beta without selling a share of the tax-sensitive core,
reporting the margin consumed, the annual carry and the capital-gains tax the alternative of selling stock would have
realised. A daily simulator replays any signal with fund compounding, fees, interest and costs.

### 14.7 The research laboratory: defending the parameters
Before an advisor commits to a trigger, a basket size or a minimum account, the rule has to survive due diligence. The
research laboratory backtests the harvesting rules on the point-in-time S&P 500 (every member since 1999, delisted names
sold at their last close) over rolling five- and ten-year windows that start on the first trading day of every calendar
year from 2000, reviewed monthly, with whole shares and the two wash-sale rules that bind a research account. It reports
the *distribution* of outcomes across windows (median and interquartile range) for losses harvested per year, harvest
life and half-life before the account ossifies, realised and forecast tracking error, turnover and names held, as a
function of account size ($10k to $1m), basket size (50 to 300 names), harvest trigger (0.01% to 1% of account value),
harvesting approach (pairs on the fly within sector or index, SARD-paired twin baskets, or the tracking-error optimizer
with sector and factor-alignment bands) and a concentrated starting position (size times embedded gain), which is unwound
only as far as realised losses cover its gain. The risk model throughout is the calibrated 126-day Ledoit-Wolf covariance.
Runs execute across every core, resume if interrupted, and export a markdown and Excel write-up with the method,
findings, tables and caveats a reviewer expects.

### 14.8 Fast
The window appears in about 1.5 seconds instead of six: the solvers are imported lazily and warmed up in the
background, the settings loader lost its pydantic dependency, screens are constructed on first visit, and charts update
through `Plotly.react` instead of reloading a four-megabyte script per figure. The test suite grew to 249 tests.
