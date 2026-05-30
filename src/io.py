"""I/O utilities: EODHD client (cache-backed) and persistence helpers.

Shared across NB01 (fundamentals, EOD, earnings persistence), NB02 (artifact
persistence), and NB03 (news/earnings — added in Step 3).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm

from src.config import EODHD_API_KEY, EODHD_BASE_URL, OFFLINE_MODE, PATHS


# ---------------------------------------------------------------------------
# EODHDClient
# ---------------------------------------------------------------------------

_DEFAULT_TTL: dict[str, int] = {
    "fundamentals":      30 * 86400,
    "eod":                1 * 86400,
    "calendar/earnings":  7 * 86400,
    "news":               1 * 86400,
}


class EODHDClient:
    """Thin EODHD All-In-One client with on-disk JSON cache.

    Each public method returns the raw JSON payload as-is (no coercion).
    Cache freshness is per-endpoint via `ttl`; stale entries are refetched
    transparently. Pass `force=True` on any call to bypass the cache and
    overwrite the cached entry.

    When `offline=True` the client serves any existing cache entry regardless
    of TTL; a cache miss raises RuntimeError rather than hitting the network.
    """

    def __init__(
        self,
        cache_dir: Path = PATHS.CACHE,
        timeout: int = 30,
        api_token: str = EODHD_API_KEY,
        offline: bool = OFFLINE_MODE,
        base_url: str = EODHD_BASE_URL,
        ttl: dict[str, int] | None = None,
    ):
        self._token    = api_token            # leading underscore: do not log/print
        self.base_url  = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl       = dict(ttl if ttl is not None else _DEFAULT_TTL)
        self.timeout   = timeout
        self.offline   = offline
        self.stats     = {"hits": 0, "misses": 0, "writes": 0, "forced": 0}

    # ---- cache plumbing -----------------------------------------------------

    @staticmethod
    def _canonical(endpoint: str, params: dict) -> str:
        """endpoint + sorted-query-string, with api_token stripped."""
        clean = {k: v for k, v in params.items() if k != "api_token"}
        qs = "&".join(f"{k}={clean[k]}" for k in sorted(clean))
        return f"{endpoint}?{qs}"

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        h = hashlib.sha256(self._canonical(endpoint, params).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.json"

    def _ttl_for(self, endpoint: str) -> int:
        """Longest-prefix match: 'calendar/earnings' wins over 'calendar'."""
        for k in sorted(self.ttl, key=len, reverse=True):
            if endpoint.startswith(k):
                return self.ttl[k]
        return 86400  # safe default: 1 day

    def _is_fresh(self, path: Path, endpoint: str) -> bool:
        return path.exists() and (time.time() - path.stat().st_mtime) < self._ttl_for(endpoint)

    # ---- core GET -----------------------------------------------------------

    def _get(self, endpoint: str, params: dict | None = None,
             *, force: bool = False) -> Any:
        params = dict(params or {})
        params.setdefault("fmt", "json")
        path = self._cache_path(endpoint, params)

        # Offline mode: serve any existing cache entry, ignore TTL/force
        if self.offline:
            if path.exists():
                self.stats["hits"] += 1
                return json.loads(path.read_text())
            raise RuntimeError(
                f"OFFLINE_MODE: no cache for {endpoint}. Run once online to populate cache first."
            )

        if not force and self._is_fresh(path, endpoint):
            self.stats["hits"] += 1
            return json.loads(path.read_text())

        self.stats["forced" if force else "misses"] += 1

        if not self._token:
            raise RuntimeError("api_token is empty and no valid cache entry exists.")

        url = f"{self.base_url}/{endpoint}"
        r = requests.get(url, params={**params, "api_token": self._token},
                         timeout=self.timeout)
        if not r.ok:
            # Redact api_token from URL before raising so the token never appears in tracebacks
            redacted_url = re.sub(r"api_token=[^&]*", "api_token=<redacted>", r.url)
            raise requests.HTTPError(
                f"{r.status_code} {r.reason} for url: {redacted_url}",
                response=r,
            )
        payload = r.json()

        # Atomic write
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")))
        tmp.replace(path)
        self.stats["writes"] += 1
        return payload

    # ---- public surface (v2 pull contract) ----------------------------------

    def fundamentals(self, code: str, *, force: bool = False) -> dict:
        return self._get(f"fundamentals/{code}", force=force)

    def eod(self, code: str, from_date: str | None = None,
            to_date: str | None = None, *, force: bool = False) -> list[dict]:
        params: dict = {}
        if from_date: params["from"] = from_date
        if to_date:   params["to"]   = to_date
        return self._get(f"eod/{code}", params, force=force)

    def earnings_calendar(self, from_date: str, to_date: str | None = None,
                          symbols: str | None = None,
                          *, force: bool = False) -> dict:
        params: dict = {"from": from_date}
        if to_date: params["to"]      = to_date
        if symbols: params["symbols"] = symbols
        return self._get("calendar/earnings", params, force=force)

    def news(self, code: str, from_date: str, to_date: str | None = None,
             limit: int = 1000, *, force: bool = False) -> list[dict]:
        params: dict = {"s": code, "from": from_date, "limit": limit}
        if to_date: params["to"] = to_date
        return self._get("news", params, force=force)


# ---------------------------------------------------------------------------
# flatten_fundamentals
# ---------------------------------------------------------------------------

def flatten_fundamentals(ticker: str, payload: dict, country_of_origin: str
                         ) -> tuple[dict, list[str]]:
    """Flatten EODHD fundamentals JSON into one ordered row + quality flags."""
    flags: list[str] = []
    row: dict = {"Ticker": ticker}

    # --- General slim ---
    g = payload.get("General") or {}
    for k in ("Code", "Name", "Exchange", "Sector", "Industry",
              "GicSector", "GicGroup", "GicIndustry", "GicSubIndustry",
              "IsDelisted", "IPODate", "HomeCategory"):
        v = g.get(k)
        row[k] = v
        if v is None:
            flags.append(f"General.{k}=None")

    # --- HQ country (EODHD) ---
    ad = (g.get("AddressData") or {})
    row["HQCountryEODHD"] = ad.get("Country")
    if row["HQCountryEODHD"] is None:
        flags.append("General.AddressData.Country=None")

    # --- CountryOfOrigin (yaml authority) ---
    row["CountryOfOrigin"] = country_of_origin

    # --- Highlights (full v1 column set) ---
    h = payload.get("Highlights") or {}
    if not h:
        flags.append("Highlights missing/empty")
    for k in ("MarketCapitalization", "MarketCapitalizationMln", "EBITDA",
              "PERatio", "PEGRatio", "WallStreetTargetPrice", "BookValue",
              "DividendShare", "DividendYield", "EarningsShare",
              "EPSEstimateCurrentYear", "EPSEstimateNextYear",
              "EPSEstimateCurrentQuarter", "EPSEstimateNextQuarter",
              "MostRecentQuarter", "ProfitMargin", "OperatingMarginTTM",
              "ReturnOnAssetsTTM", "ReturnOnEquityTTM",
              "RevenueTTM", "RevenuePerShareTTM",
              "QuarterlyRevenueGrowthYOY", "GrossProfitTTM",
              "DilutedEpsTTM", "QuarterlyEarningsGrowthYOY"):
        row[k] = h.get(k)
        if h and h.get(k) is None:
            flags.append(f"Highlights.{k}=None")

    # --- Valuation ---
    v = payload.get("Valuation") or {}
    if not v:
        flags.append("Valuation missing/empty")
    for k in ("TrailingPE", "ForwardPE", "PriceSalesTTM", "PriceBookMRQ",
              "EnterpriseValue", "EnterpriseValueRevenue", "EnterpriseValueEbitda"):
        row[k] = v.get(k)
        if v and v.get(k) is None:
            flags.append(f"Valuation.{k}=None")

    # --- SharesStats subset + derived FreeFloatPercent ---
    ss = payload.get("SharesStats") or {}
    if not ss:
        flags.append("SharesStats missing/empty")
    sf, so = ss.get("SharesFloat"), ss.get("SharesOutstanding")
    row["SharesFloat"]         = sf
    row["SharesOutstanding"]   = so
    row["PercentInsiders"]     = ss.get("PercentInsiders")
    row["PercentInstitutions"] = ss.get("PercentInstitutions")
    row["ShortPercentFloat"]   = ss.get("ShortPercentFloat")
    if sf and so:
        if sf > so:
            # Float exceeding outstanding is physically impossible — almost always an
            # ADR-vs-local-share unit mismatch in the vendor payload (e.g., SUPV reports
            # SharesFloat in local Class B units but SharesOutstanding in ADR units).
            row["FreeFloatPercent"] = None
            flags.append(f"FreeFloatPercent suspect (float={sf:,.0f} > outstanding={so:,.0f}; "
                         f"likely ADR ratio mismatch — do not use downstream)")
        else:
            row["FreeFloatPercent"] = sf / so * 100.0
    else:
        row["FreeFloatPercent"] = None
        flags.append("FreeFloatPercent unavailable (SharesFloat or SharesOutstanding missing)")

    return row, flags


# ---------------------------------------------------------------------------
# Manifest helpers (internal)
# ---------------------------------------------------------------------------

def _manifest_entry(
    path: Path,
    obj: pd.DataFrame | pd.Series,
    snapshot: str | None,
    *,
    timeseries: bool = False,
) -> dict:
    entry: dict = {
        "sha256":       hashlib.sha256(path.read_bytes()).hexdigest(),
        "row_count":    int(len(obj)),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    if isinstance(obj, pd.DataFrame) and obj.ndim == 2:
        entry["column_count"] = int(obj.shape[1])
    if snapshot:
        entry["snapshot_date"] = snapshot
    if timeseries and isinstance(obj, pd.DataFrame) and not obj.empty:
        entry["date_min"] = str(obj.index.min().date())
        entry["date_max"] = str(obj.index.max().date())
    return entry


def _write_manifest(manifest_path: Path, root: Path, entries: dict[Path, dict]) -> None:
    """Read-modify-write manifest.json, keyed by path-relative-to-root."""
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for p, entry in entries.items():
        manifest[p.relative_to(root).as_posix()] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Persistence functions
# ---------------------------------------------------------------------------

def persist_universe_master(
    universe: pd.DataFrame,
    snapshot: str,
    paths=PATHS,
) -> dict[str, Path]:
    """Write the master fundamentals CSV and update the manifest.

    Filename contract (read by §4.3 which re-writes it with liquidity columns):
        data/processed/fundamentals_{snapshot}.csv
    """
    fund_path = paths.PROCESSED / f"fundamentals_{snapshot}.csv"
    universe.to_csv(fund_path)

    _write_manifest(
        paths.DATA / "manifest.json",
        paths.ROOT,
        {fund_path: _manifest_entry(fund_path, universe, snapshot)},
    )
    return {"fundamentals": fund_path}


def persist_panels(
    prices: pd.DataFrame,
    volume: pd.DataFrame,
    returns: pd.DataFrame,
    universe: pd.DataFrame,
    snapshot: str,
    paths=PATHS,
) -> dict[str, Path]:
    """Write price/volume/returns parquet panels and re-write the universe CSV.

    Filename contract (NB02 reads these by name):
        data/processed/prices_{snapshot}.parquet
        data/processed/volume_{snapshot}.parquet
        data/processed/returns_{snapshot}.parquet
        data/processed/fundamentals_{snapshot}.csv   (re-write with liquidity cols)
    """
    manifest_path = paths.DATA / "manifest.json"
    entries: dict[Path, dict] = {}
    written: dict[str, Path]  = {}

    for name, df in [("prices", prices), ("volume", volume), ("returns", returns)]:
        p = paths.PROCESSED / f"{name}_{snapshot}.parquet"
        df.to_parquet(p)
        entries[p] = _manifest_entry(p, df, snapshot, timeseries=True)
        written[name] = p

    fund_path = paths.PROCESSED / f"fundamentals_{snapshot}.csv"
    universe.to_csv(fund_path)
    entries[fund_path] = _manifest_entry(fund_path, universe, snapshot)
    written["fundamentals"] = fund_path

    _write_manifest(manifest_path, paths.ROOT, entries)
    return written


def persist_earnings(
    earnings: pd.DataFrame,
    snapshot: str,
    paths=PATHS,
) -> dict[str, Path]:
    """Write the earnings parquet and update the manifest.

    Filename contract (NB02/NB03 read this by name):
        data/processed/earnings_{snapshot}.parquet
    """
    earn_path = paths.PROCESSED / f"earnings_{snapshot}.parquet"
    earnings.to_parquet(earn_path)

    entry = _manifest_entry(earn_path, earnings, snapshot)
    if len(earnings) and earnings["report_date"].notna().any():
        entry["date_min"] = str(earnings["report_date"].min().date())
        entry["date_max"] = str(earnings["report_date"].max().date())

    _write_manifest(
        paths.DATA / "manifest.json",
        paths.ROOT,
        {earn_path: entry},
    )
    return {"earnings": earn_path}


def persist_universe_tier(
    universe_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
    returns_df: pd.DataFrame | None = None,
    *,
    tier: str,
    snapshot: str | None = None,
    paths=PATHS,
) -> dict[str, Path]:
    """Write the NB02 universe export files for one tier and update the manifest.

    Filename contract — these are the stable names NB02 reads by name:

    tier="strict" (institutional):
        data/processed/universe_institutional_fundamentals.csv
        data/processed/tickers_institutional.csv

    tier="loose" (pairs engine):
        data/processed/universe_pairs_fundamentals.csv
        data/processed/tickers_pairs.csv
        data/processed/prices_pairs.csv
        data/processed/returns_pairs.csv

    universe_df must already be filtered to the desired tickers; its index is
    used as the Ticker list.  For the loose tier, prices_df and returns_df
    must be provided (columns = final_tickers).
    """
    if tier not in ("strict", "loose"):
        raise ValueError(f"tier must be 'strict' or 'loose', got {tier!r}")

    tickers = list(universe_df.index)
    entries: dict[Path, dict] = {}
    written: dict[str, Path]  = {}

    if tier == "strict":
        fund_p  = paths.PROCESSED / "universe_institutional_fundamentals.csv"
        tick_p  = paths.PROCESSED / "tickers_institutional.csv"

        universe_df.to_csv(fund_p, na_rep="NaN")
        pd.Series(tickers, name="Ticker").to_csv(tick_p, index=False)

        entries[fund_p] = _manifest_entry(fund_p, universe_df, snapshot)
        entries[tick_p] = _manifest_entry(tick_p, pd.Series(tickers), snapshot)
        written = {"universe_institutional_fundamentals": fund_p,
                   "tickers_institutional": tick_p}

    else:  # loose
        if prices_df is None or returns_df is None:
            raise ValueError("prices_df and returns_df are required for tier='loose'")

        fund_p  = paths.PROCESSED / "universe_pairs_fundamentals.csv"
        tick_p  = paths.PROCESSED / "tickers_pairs.csv"
        px_p    = paths.PROCESSED / "prices_pairs.csv"
        ret_p   = paths.PROCESSED / "returns_pairs.csv"

        universe_df.to_csv(fund_p, na_rep="NaN")
        pd.Series(tickers, name="Ticker").to_csv(tick_p, index=False)
        prices_df.to_csv(px_p,  na_rep="NaN")
        returns_df.to_csv(ret_p, na_rep="NaN")

        entries[fund_p] = _manifest_entry(fund_p, universe_df, snapshot)
        entries[tick_p] = _manifest_entry(tick_p, pd.Series(tickers), snapshot)
        entries[px_p]   = _manifest_entry(px_p,  prices_df,  snapshot)
        entries[ret_p]  = _manifest_entry(ret_p, returns_df, snapshot)
        written = {
            "universe_pairs_fundamentals": fund_p,
            "tickers_pairs":               tick_p,
            "prices_pairs":                px_p,
            "returns_pairs":               ret_p,
        }

    _write_manifest(paths.DATA / "manifest.json", paths.ROOT, entries)
    return written


# ---------------------------------------------------------------------------
# Pull orchestrators
# ---------------------------------------------------------------------------

_EARN_COLS = ["Ticker", "report_date", "period_date", "before_after_market",
              "currency", "eps_actual", "eps_estimate", "surprise_pct"]


def pull_universe_fundamentals(client, tickers, ticker_to_country, *,
                                snapshot: str, offline: bool, processed_dir):
    """Pull or replay fundamentals across the active universe.

    OFFLINE_MODE: reads fundamentals_{snapshot}.csv (index_col='Ticker') from processed_dir.
    Online: loops, calls client.fundamentals(f'{t}.US') + flatten_fundamentals;
            collects quality flags and error messages by ticker.

    Returns (universe_df indexed by 'Ticker', quality_dict, errors_dict).
    Preserves the exact lift-and-shift behavior from NB01 §3.2.
    """
    if offline:
        universe = pd.read_csv(processed_dir / f"fundamentals_{snapshot}.csv",
                               index_col="Ticker")
        print(f"[OFFLINE] loaded universe from fundamentals_{snapshot}.csv: {universe.shape}")
        return universe, {}, {}

    rows: list[dict]         = []
    quality: dict[str, list] = {}
    errors: dict[str, str]   = {}

    for t in tqdm(tickers, desc="fundamentals"):
        code_us = f"{t}.US"
        try:
            payload = client.fundamentals(code_us)
        except Exception as e:
            errors[t] = f"{type(e).__name__}: {e}"
            continue
        row, flags = flatten_fundamentals(t, payload, ticker_to_country[t])
        rows.append(row)
        if flags:
            quality[t] = flags

    universe = pd.DataFrame(rows).set_index("Ticker")
    return universe, quality, errors


def pull_eod_panels(client, tickers, *, from_date: str,
                    snapshot: str, offline: bool, processed_dir):
    """Pull or replay EOD price/volume/returns panels.

    OFFLINE: reads prices_{snapshot}.parquet / volume_{snapshot}.parquet /
             returns_{snapshot}.parquet.
    Online: per ticker → client.eod(f'{t}.US', from_date=...) → typed DataFrame
            → panel assembly.
    Panel index.name = "Date" (NB02 reads prices_pairs.csv with index_col="Date").
    Returns (prices, volume, returns, errors_dict).
    """
    if offline:
        prices  = pd.read_parquet(processed_dir / f"prices_{snapshot}.parquet")
        volume  = pd.read_parquet(processed_dir / f"volume_{snapshot}.parquet")
        returns = pd.read_parquet(processed_dir / f"returns_{snapshot}.parquet")
        print(f"[OFFLINE] loaded panels from data/processed/: prices {prices.shape}")
        for _panel in (prices, volume, returns):
            _panel.index.name = "Date"
        return prices, volume, returns, {}

    px_frames: dict  = {}
    eod_errors: dict = {}

    for t in tqdm(tickers, desc="eod"):
        code_us = f"{t}.US"
        try:
            bars = client.eod(code_us, from_date=from_date)
        except Exception as e:
            eod_errors[t] = f"{type(e).__name__}: {e}"
            continue
        if not bars:
            eod_errors[t] = "empty response"
            continue
        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df["date"])
        for col in ("open", "high", "low", "close", "adjusted_close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "adjusted_close" in df.columns and df["adjusted_close"].notna().any():
            df["price"] = df["adjusted_close"]
        else:
            df["price"] = df["close"]
        px_frames[t] = df.set_index("date").sort_index()

    prices  = pd.DataFrame({t: f["price"]  for t, f in px_frames.items()}).sort_index()
    volume  = pd.DataFrame({t: f["volume"] for t, f in px_frames.items()}).sort_index()
    returns = prices.pct_change()

    for _panel in (prices, volume, returns):
        _panel.index.name = "Date"

    return prices, volume, returns, eod_errors


def pull_earnings_calendar(client, tickers, *, from_date: str,
                            snapshot: str, offline: bool, processed_dir):
    """Pull or replay the earnings calendar across the active universe.

    OFFLINE: reads earnings_{snapshot}.parquet.
    Online: per ticker → client.earnings_calendar → normalize → typed long-form
            DataFrame.
    Returns (earnings_df, earn_counts_dict, errors_dict).
    """
    if offline:
        earnings    = pd.read_parquet(processed_dir / f"earnings_{snapshot}.parquet")
        earn_counts = earnings["Ticker"].value_counts().to_dict()
        print(f"[OFFLINE] loaded earnings from earnings_{snapshot}.parquet: {earnings.shape}")
        return earnings, earn_counts, {}

    earn_rows: list   = []
    earn_errors: dict = {}
    earn_counts: dict = {}

    for t in tqdm(tickers, desc="earnings"):
        code_us = f"{t}.US"
        try:
            payload = client.earnings_calendar(from_date=from_date, symbols=code_us)
        except Exception as e:
            earn_errors[t] = f"{type(e).__name__}: {e}"
            continue
        events = payload.get("earnings", []) if isinstance(payload, dict) else []
        earn_counts[t] = len(events)
        for ev in events:
            earn_rows.append({
                "Ticker":              t,
                "report_date":         ev.get("report_date"),
                "period_date":         ev.get("date"),
                "before_after_market": ev.get("before_after_market"),
                "currency":            ev.get("currency"),
                "eps_actual":          ev.get("actual"),
                "eps_estimate":        ev.get("estimate"),
                "surprise_pct":        ev.get("percent"),
            })

    earnings = pd.DataFrame(earn_rows, columns=_EARN_COLS)
    for c in ("report_date", "period_date"):
        earnings[c] = pd.to_datetime(earnings[c], errors="coerce")
    for c in ("eps_actual", "eps_estimate", "surprise_pct"):
        earnings[c] = pd.to_numeric(earnings[c], errors="coerce")
    earnings = earnings.sort_values(["Ticker", "report_date"]).reset_index(drop=True)
    return earnings, earn_counts, earn_errors
