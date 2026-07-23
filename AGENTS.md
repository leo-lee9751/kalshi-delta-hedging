# AGENTS.md

## Cursor Cloud specific instructions

This repo is a Python (3.12) algorithmic-trading project for Kalshi's `KXBTC15M`
15-minute BTC prediction markets. It has two halves that share the root modules:

- Research/backtesting scripts at the repo root (`analyze*.py`, `simulate*.py`,
  `run.py`, `hyperopt.py`, `collect_15s_data.py`).
- The production live trader in `live/` (`live/trader.py`), the documented product.

Dependencies are plain `pip` from `requirements.txt` (root) and `live/requirements.txt`
(installed by the startup update script). There is no database. There is no
lint/test/build tooling in the repo (no pytest/ruff/CI); `python -m py_compile *.py live/*.py`
is a useful quick sanity check.

### Running things

- Backtest / build empirical fair-price table:
  `python analyze_coin.py --series KXBTC15M --symbol BTC-USD --days N`
  (writes `data/logs/minute_analysis_2d_kxbtc15m.csv`). `run.py` is the older T+5 backtest.
- Live trader (paper mode, no real orders): `cd live && python3 trader.py`.
  Config comes from `live/.env` (see `live/.env.example`; `.env` is gitignored).

### Non-obvious gotchas (important)

- Paper mode requires `KALSHI_PRIVATE_KEY` to be EMPTY/unset in `live/.env`. The
  placeholder in `.env.example` (`your-private-key-here`) is not valid PEM and
  crashes `trader.py` at import (`load_private_key` raises). For paper testing,
  copy `.env.example` then blank out `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY`.
  Real RSA credentials are only needed to place live orders / check portfolio.
- `live/trader.py` hard-requires `data/logs/minute_analysis_2d_15s_kxbtc15m.csv`
  at startup or it exits. This file is gitignored/regeneratable. The intended
  generator (`simulate_15s.py`) needs 15-second Kalshi snapshots collected live by
  `collect_15s_data.py` (not usually present). As a working substitute, generate
  the minute-resolution table with `analyze_coin.py` and re-key it to the
  `offset_secs` schema the trader loads (`offset_secs = minute*60`, columns
  `offset_secs,bucket,n,win_rate,avg_fill`). The trader is designed to fall back to
  nearest-minute / hardcoded fair prices for intermediate 15s marks.
- The Kalshi market-list fetch in `kalshi_client.py` and `analyze_coin.py` sends the
  `min_close_ts` query param, which Kalshi's API now rejects with HTTP 503. Plain
  cursor pagination (no `min_close_ts`) works. To run backtests, prime the market
  cache first: fetch settled `KXBTC15M` markets via pagination and write them to
  `data/cache/markets_KXBTC15M_<days>d.json` (analyze_coin) and
  `data/cache/markets_<days>d.json` (kalshi_client). Those functions read the cache
  before hitting the API, so the broken request is skipped.
- Kalshi has maintenance/closed windows where `GET /trade-api/v2/exchange/status`
  returns `{"exchange_active":false,...}` and market/candlestick queries return HTTP
  503 (`service_unavailable`, service `query-exchange`). During these windows there
  are NO open `KXBTC15M` markets, so the live trader correctly logs
  "No open KXBTC15M market found. Skipping window." and waits for the next window;
  it auto-trades (paper) once a window opens. Data endpoints are also intermittently
  503 under load — retry with backoff.
- Coinbase REST blocks the default `urllib` User-Agent (HTTP 403), but `requests`
  works. The live BTC price feed uses the Coinbase WebSocket (`btc_feed.py`), which
  is unaffected.

### Network

Outbound network is required: Coinbase (BTC price via REST candles + WebSocket) and
Kalshi (`api.elections.kalshi.com`, markets/candlesticks/WebSocket). No inbound
ports/services are started.
