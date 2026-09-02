# Norgate Data — Python API Reference (cleaned)

Condensed from Norgate's official `norgatedata` package documentation. Trimmed: version history, install/upgrade boilerplate, and the third-party backtesting-framework integration snippets (Zipline/Backtrader/pybacktest/Tensorflow/Keras/the general package list) — none of that is relevant to building the TLH engine. Everything functional is kept and de-duplicated. Pair this with `tlh-engine-claude-code-brief.md`.

---

## Requirements & setup

- Python 3.5+, **Windows only**.
- Dependencies (`pandas`, `numpy`, `requests`, `logbook`) auto-install via pip.
- Requires an active Norgate Data subscription **and** the Norgate Data Updater (NDU) app installed and running at all times — NDU is a Windows-only background service; without it the package cannot fetch data.
- Needs a writable local folder `C:\Users\<username>\.norgatedata` (override with env var `NORGATEDATA_ROOT`).
- Install: `pip install norgatedata` · Upgrade: `pip install norgatedata --upgrade`
- `import norgatedata`

**Design implication for this project:** NDU is Windows-only and must be a running background process, so the data layer needs a startup check (`norgatedata.status()`) and a clear error surface in the GUI if NDU isn't running — don't let the app silently fail on data calls.

---

## Price & volume time series

`norgatedata.price_timeseries(symbol_or_assetid, **kwargs)` is the core call. Output format, date range, adjustment, and padding are all controlled by keyword arguments:

```python
import norgatedata

pricedata = norgatedata.price_timeseries(
    'AAPL',                                                   # symbol (str) or assetid (int)
    stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
    padding_setting=norgatedata.PaddingType.NONE,
    start_date='1990-01-01',                                  # or None = full history
    end_date='2000-01-01',                                    # or None = today
    limit=500,                                                # optional: last N records instead of a date range
    interval='D',                                              # 'D' daily, 'W' weekly, 'M' monthly
    timeseriesformat='pandas-dataframe',                       # or 'numpy-recarray' (default), 'numpy-ndarray'
    datetimeformat='datetime64ns',                             # see date formats below
)
```

**Date inputs** accept string (`'YYYY-MM-DD'` or `'YYYYMMDD'`), `datetime`, `pandas.Timestamp`, or `numpy.datetime64` interchangeably.

**Price adjustment types** (`StockPriceAdjustmentType`):
| Value | Meaning |
|---|---|
| `NONE` | Raw prices |
| `CAPITAL` | Adjusted for splits/capital reconstructions only |
| `CAPITALSPECIAL` | Adjusted for capital events + special dividends |
| `TOTALRETURN` | Adjusted for everything, incl. ordinary dividends — **default** |

**Padding types** (`PaddingType`) — repeats the prior close on days with no record:
| Value | Meaning |
|---|---|
| `NONE` | No padding — **default** |
| `ALLMARKETDAYS` | Pad every market day |
| `ALLWEEKDAYS` | Pad every weekday |
| `ALLCALENDARDAYS` | Pad every calendar day (useful for clean end-of-month/quarter snapshots) |

**Columns returned:** `Date, Open, High, Low, Close`, plus (where applicable) `Volume`, `Turnover`, `Unadjusted Close`, `Dividend` (stocks), `Open Interest` (futures/ASX options), `Delivery Month` (continuous futures).

**Datetime output formats** (`datetimeformat=`): `'datetime'` (Python `datetime.datetime`), `'date'` (Python `datetime.date`), `'datetime64ns'`, `'datetime64ms'`, `'m8d'` (`'<M8[D]'`, the NumPy default). A `timezone=` kwarg (e.g. `'UTC'`, `'US/Eastern'`) can localize the date column; time-of-day stays `00:00:00`.

**Feeding an existing array back in:** `price_timeseries` (and `index_constituent_timeseries`, below) can accept a previously-returned `numpy_recarray=` or `pandas_dataframe=` and append a new column to it in place, rather than re-fetching.

**Dividend column semantics** — what shows up depends on the adjustment method requested:
- Capital-only adjustment → sum of special + ordinary dividends for that day.
- Capital + special adjustment → sum of ordinary dividends only.
- Fully adjusted (total return) → dividend column is empty (already priced in).
- A dividend is attributed to the day *before* the ex-date (i.e., you're entitled to it if you held at that close) — relevant for cost-basis and reinvestment logic.

---

## Time series indicators

All follow the same call shape as `price_timeseries` (symbol/assetid, optional `start_date`/`end_date`/`padding_setting`, `timeseriesformat`, optional pass-through array).

| Function | Returns |
|---|---|
| `index_constituent_timeseries(symbol, indexname, ...)` | Whether the stock was a constituent of `indexname` (e.g. `'S&P 500'`, `'Russell 3000'`, or an index symbol like `$SPX`) on each date. **Requires Platinum/Diamond Stocks subscription.** |
| `major_exchange_listed_timeseries(symbol, ...)` | 1 = listed on a major exchange (NYSE, Nasdaq, NYSE American/Arca, Cboe BZX, IEX), 0 = OTC/Pink Sheet. US equities/ETPs, data from 2000+. **Platinum/Diamond.** |
| `blank_check_company_timeseries(symbol, ...)` | Periods flagged as a blank-check (SPAC-type) company. **Platinum/Diamond.** |
| `capital_event_timeseries(symbol, ...)` | 1 on the day a capital event (split, reverse split, bonus issue, stock dividend, reorg) takes effect (day before ex-date), else 0. |
| `dividend_yield_timeseries(symbol, ...)` | Trailing-12-month split-adjusted ordinary-dividend yield vs. close, recalculated as of each entitlement date. Excludes specials/distributions/spin-offs. |
| `padding_status_timeseries(symbol, ...)` | Flags which records were synthetically padded, per the padding setting used. |
| `unadjusted_close_timeseries(symbol, ...)` | Raw close (also already available as a column in `price_timeseries`) — mainly a helper for other libraries. |

`index_constituent_timeseries` is the one to lean on for benchmark membership and for validating that a factor model's universe matches a real index over time.

---

## Single-value metadata

### Security identity & info
```python
assetid = norgatedata.assetid('AMZN')          # symbol -> unchanging numeric ID
symbol  = norgatedata.symbol(129769)            # assetid -> current symbol

norgatedata.domicile(symbol)                    # country code
norgatedata.currency(symbol)                    # trading currency
norgatedata.exchange_name(symbol)               # short name, e.g. 'NYSE'
norgatedata.exchange_name_full(symbol)          # full name, e.g. 'New York Stock Exchange'
norgatedata.security_name(symbol)               # e.g. 'General Electric Co Common'
norgatedata.base_type(symbol)                   # Stock Market / Futures Market / Forex / Economic / etc.
norgatedata.subtype1(symbol)                    # Equity / Derivative / Debt / ETP / etc.
norgatedata.subtype2(symbol)                    # Operating Company / ETF / ETN / Preferred / Warrant / etc.
norgatedata.subtype3(symbol)                    # MLP / Closed End Fund / SPAC / etc.
norgatedata.financial_summary(symbol)           # prose financial summary
norgatedata.business_summary(symbol)            # prose business description
norgatedata.first_quoted_date(symbol, datetimeformat='iso')
norgatedata.last_quoted_date(symbol, datetimeformat='iso')     # None if still trading
norgatedata.second_last_quoted_date(symbol, datetimeformat='iso')
```
`subtype2`/`subtype3` are the fields most useful for building the "substantially identical security" mapping the wash-sale engine needs (e.g. distinguishing ETF share classes, ETNs, preferreds).

### Classifications (sector/industry — feeds the risk model)
```python
norgatedata.classification(symbol, schemename, classificationresulttype)
# schemename: 'GICS' | 'TRBC' | 'NorgateDataFuturesClassification'
# classificationresulttype: 'ClassificationId' | 'Name'

norgatedata.classification_at_level(symbol, schemename, classificationresulttype, level)
# level: e.g. 1 (broad sector) through 4 (sub-industry) for GICS/TRBC

norgatedata.corresponding_industry_index(symbol, indexfamilycode, level, indexreturntype)
# indexfamilycode: e.g. '$SPX', '$SP1500'; indexreturntype: 'PR' (price) | 'TR' (total return)
```
Returns `None` where no classification/index applies. GICS or TRBC at your chosen level is the direct input for the risk model's sector-factor block.

### Shares outstanding / float
```python
sharesout, asof   = norgatedata.sharesoutstanding(symbol, datetimeformat='iso')
sharesfloat, asof = norgatedata.sharesfloat(symbol, datetimeformat='iso')
```
`datetimeformat` also accepts `'pandas-timestamp'`, `'numpy-datetime64'`, `'datetime'`.

### Fundamental data
```python
value, asof = norgatedata.fundamental(symbol, fieldname, datetimeformat='iso')
# example fieldnames: 'mktcap', 'ttmepsxlcx', 'peexclxor', 'projepsq',
#                      'sharesoutstanding', 'sharesfloat' (many more fields exist)
```
Returns `(None, None)` if the field isn't available for that security. `asof` is the date the value applies to (typically quarter-end, or last-change date for current-ratio-type fields) — use it, not the pull date, when timestamping factor inputs.

### Futures metadata
```python
norgatedata.tick_size(symbol)                       # value per tick
norgatedata.point_value(symbol)                      # value per full point move
norgatedata.margin(symbol)                           # current margin requirement
norgatedata.first_notice_date(symbol, datetimeformat='iso')   # None if not deliverable pre-expiry
norgatedata.lowest_ever_tick_size(symbol)
norgatedata.session_type(symbol)                     # e.g. 'Electronic', 'Electronic (Last)'
norgatedata.futures_market_name(market_symbol)       # e.g. 'CL' -> 'Crude Oil'
norgatedata.futures_market_session_name(session_symbol)
norgatedata.futures_market_session_symbol(symbol)    # contract -> session symbol
norgatedata.futures_market_symbol(symbol)            # contract -> market symbol
norgatedata.futures_market_session_contracts(session_symbol)  # list, all contracts past+present
norgatedata.futures_market_symbols()                 # list of all market symbols
norgatedata.futures_market_session_symbols()          # list of all session symbols
```
`symbol` here can be an individual contract, continuous-contract symbol, session symbol, or market symbol — except date-related calls, which require an individual contract.

---

## Security lists

```python
# Watchlists (as defined in NDU's Watchlist Library)
symbols   = norgatedata.watchlist_symbols('S&P 500')
symbols   = norgatedata.watchlist_symbols('Russell 3000 Current & Past')
contents  = norgatedata.watchlist('Russell 3000 Current & Past')   # symbol + assetid + name
all_names = norgatedata.watchlists()

# Databases (as seen under NDU's "Database" tab)
symbols   = norgatedata.database_symbols('US Equities')
symbols   = norgatedata.database_symbols('World Indices')
contents  = norgatedata.database('AU Equities')                    # symbol + assetid + name
all_names = norgatedata.databases()
```
A Platinum-level US Stocks subscription typically exposes: `US Equities`, `US Equities Delisted`, `US Indices`, `World Indices`, `Economic`, `Forex Spot`, `Futures Continuous`. **`US Equities Delisted` is the one that matters for accurate historical cost-basis/lot reconstruction** — don't build the universe from `US Equities` alone or acquired/delisted names will silently vanish from history.

---

## Other informational functions

```python
norgatedata.last_database_update_time(database)   # local-PC datetime DB was last updated
                                                    # database shortnames: au, aueto, auwarrant, auindex,
                                                    # ca, caindex, us, usindex, cashcommodity, economic,
                                                    # future, forex, contfuture, worldindex
norgatedata.last_price_update_time(symbol)        # local-PC datetime symbol was last updated
norgatedata.status()                               # True if NDU is running, else False
```
Note: both "last update" calls report *when the local database was refreshed*, not the latest price date contained in it — don't conflate the two when building a staleness check.

---

## Operational notes

- **Assetid vs. symbol:** every function taking `symbol` also accepts the numeric `assetid` — prefer assetid for anything persisted (positions, lots, saved model inputs), since a symbol can change but the assetid never does. (`MSFT` = 134016, `AMZN` = 129769, as examples.)
- **Error handling:** invalid symbol/parameters → `ValueError`, with a human-readable message also logged via `logbook`. Missing data (e.g. a fundamental field the company doesn't report) → `None`, or `(None, None)` for calls that return a value+date pair.
- **Concurrency:** the package is safe to use under multithreading/multiprocessing — worth exploiting for bulk universe pulls (factor model refits over hundreds of names) rather than looping serially.

---

## Direct relevance to the TLH engine

- `price_timeseries` (total-return adjusted) → the core price series for both the risk model and the tax-lot P&L math.
- `US Equities Delisted` database + `last_quoted_date` → required for correct historical cost-basis reconstruction on names that no longer trade.
- `classification_at_level('GICS', ...)` → sector-factor block for the risk model.
- `index_constituent_timeseries` → benchmark membership for tracking-error and factor-exposure comparisons.
- `subtype2`/`subtype3` + `corresponding_industry_index` → starting point for the "substantially identical security" mapping the wash-sale engine and the optimizer's replacement-security logic both depend on.
- `fundamental()` (mktcap, EPS, P/E fields) → style-factor inputs (value, quality) alongside price-derived factors (momentum, low-vol, size via market cap).
- `sharesoutstanding`/`sharesfloat` → free-float weighting for factor exposure and portfolio-vs-benchmark comparisons.
- `norgatedata.status()` → the startup health check the data layer should run before anything else.
