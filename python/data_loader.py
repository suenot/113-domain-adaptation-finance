"""
Data Loading and Feature Engineering
=====================================

Provides data ingestion from crypto exchanges (Bybit) and simulated market
data generators for both stock-like and crypto-like price series. Includes
a feature generator that computes standard technical indicators used as
inputs to domain adaptation models.

Classes
-------
Kline
    Immutable representation of a single candlestick bar.
BybitClient
    Synchronous REST client for the Bybit v5 market-data API.
SimulatedDataGenerator
    Generates synthetic OHLCV data that mimics stock or crypto properties.
FeatureGenerator
    Transforms raw Kline sequences into feature matrices and binary labels.

Functions
---------
klines_to_dataframe
    Convert a list of Kline objects into a pandas DataFrame.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Kline data structure
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Kline:
    """Single candlestick (OHLCV) bar.

    Parameters
    ----------
    timestamp : int
        Unix timestamp in milliseconds for the bar open.
    open : float
        Opening price.
    high : float
        Highest price during the bar.
    low : float
        Lowest price during the bar.
    close : float
        Closing price.
    volume : float
        Traded volume during the bar.
    """

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


# ---------------------------------------------------------------------------
# Bybit REST client
# ---------------------------------------------------------------------------

class BybitClient:
    """Synchronous client for fetching kline data from the Bybit v5 API.

    Parameters
    ----------
    base_url : str, default ``"https://api.bybit.com"``
        Base URL for the Bybit REST API.

    Examples
    --------
    >>> client = BybitClient()
    >>> klines = client.fetch_klines("BTCUSDT", interval="60", limit=100)
    >>> len(klines) <= 100
    True
    """

    def __init__(self, base_url: str = "https://api.bybit.com") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "DomainAdaptationFinance/0.1",
        })

    # ---- public ----------------------------------------------------------

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "60",
        limit: int = 200,
        category: str = "linear",
    ) -> List[Kline]:
        """Fetch historical kline/candlestick data from Bybit.

        Uses the ``/v5/market/kline`` endpoint.  Results are returned in
        **chronological** order (oldest first).

        Parameters
        ----------
        symbol : str
            Trading pair symbol, e.g. ``"BTCUSDT"``.
        interval : str, default ``"60"``
            Kline interval.  Supported values include ``"1"``, ``"5"``,
            ``"15"``, ``"60"``, ``"240"``, ``"D"``, ``"W"``.
        limit : int, default 200
            Number of bars to retrieve (max 1000 per Bybit docs).
        category : str, default ``"linear"``
            Product category: ``"linear"``, ``"inverse"``, or ``"spot"``.

        Returns
        -------
        List[Kline]
            Chronologically ordered list of candlestick bars.

        Raises
        ------
        RuntimeError
            If the API returns a non-zero return code.
        requests.RequestException
            On network-level failures.
        """
        url = f"{self.base_url}/v5/market/kline"
        params = {
            "category": category,
            "symbol": symbol,
            "interval": interval,
            "limit": min(limit, 1000),
        }

        response = self._session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("retCode", -1) != 0:
            raise RuntimeError(
                f"Bybit API error {data.get('retCode')}: {data.get('retMsg')}"
            )

        raw_list = data.get("result", {}).get("list", [])
        klines: List[Kline] = []
        for row in raw_list:
            # Bybit returns: [timestamp, open, high, low, close, volume, turnover]
            klines.append(
                Kline(
                    timestamp=int(row[0]),
                    open=float(row[1]),
                    high=float(row[2]),
                    low=float(row[3]),
                    close=float(row[4]),
                    volume=float(row[5]),
                )
            )

        # Bybit returns newest-first; reverse to chronological order.
        klines.reverse()
        return klines


# ---------------------------------------------------------------------------
# Simulated data generators
# ---------------------------------------------------------------------------

class SimulatedDataGenerator:
    """Generate synthetic OHLCV data for testing and demonstration.

    The generator produces two distinct statistical regimes:

    * **Stock-like** data: moderate drift, lower volatility, volume that
      follows a log-normal distribution.
    * **Crypto-like** data: near-zero drift, higher volatility, heavier tails,
      and occasional volume spikes.

    Both use geometric Brownian motion with regime-specific parameters so that
    domain adaptation across the two is a non-trivial transfer-learning task.
    """

    def generate_stock_data(
        self,
        n_samples: int = 500,
        seed: Optional[int] = None,
    ) -> List[Kline]:
        """Generate stock-like synthetic OHLCV data.

        Parameters
        ----------
        n_samples : int, default 500
            Number of candlestick bars to generate.
        seed : int or None, default None
            Random seed for reproducibility.

        Returns
        -------
        List[Kline]
            Chronologically ordered synthetic klines.
        """
        rng = np.random.default_rng(seed)
        return self._generate(
            rng=rng,
            n_samples=n_samples,
            initial_price=100.0,
            drift=0.0002,          # positive daily drift
            volatility=0.015,      # ~1.5% daily vol
            volume_mean=1_000_000,
            volume_std=200_000,
            intrabar_noise=0.005,  # OHLC spread within bar
        )

    def generate_crypto_data(
        self,
        n_samples: int = 500,
        seed: Optional[int] = None,
    ) -> List[Kline]:
        """Generate crypto-like synthetic OHLCV data.

        Crypto data has higher volatility, heavier tails, and occasional
        volume spikes compared to the stock generator.

        Parameters
        ----------
        n_samples : int, default 500
            Number of candlestick bars to generate.
        seed : int or None, default None
            Random seed for reproducibility.

        Returns
        -------
        List[Kline]
            Chronologically ordered synthetic klines.
        """
        rng = np.random.default_rng(seed)
        return self._generate(
            rng=rng,
            n_samples=n_samples,
            initial_price=30_000.0,
            drift=0.00005,         # near-zero drift
            volatility=0.035,      # ~3.5% daily vol
            volume_mean=500,
            volume_std=300,
            intrabar_noise=0.015,  # wider OHLC spread
        )

    # ---- private ---------------------------------------------------------

    @staticmethod
    def _generate(
        rng: np.random.Generator,
        n_samples: int,
        initial_price: float,
        drift: float,
        volatility: float,
        volume_mean: float,
        volume_std: float,
        intrabar_noise: float,
    ) -> List[Kline]:
        """Core GBM-based price path generator.

        Parameters
        ----------
        rng : np.random.Generator
            Numpy random generator instance.
        n_samples : int
            Number of bars.
        initial_price : float
            Starting close price.
        drift : float
            Per-bar expected log-return.
        volatility : float
            Per-bar standard deviation of log-returns.
        volume_mean : float
            Mean volume (before noise).
        volume_std : float
            Standard deviation of volume noise.
        intrabar_noise : float
            Controls how far high/low deviate from open/close.

        Returns
        -------
        List[Kline]
        """
        # Log returns via GBM
        log_returns = rng.normal(
            loc=drift - 0.5 * volatility ** 2,
            scale=volatility,
            size=n_samples,
        )
        # Cumulative price path (close prices)
        close_prices = initial_price * np.exp(np.cumsum(log_returns))

        # Construct open as previous close (shifted by 1)
        open_prices = np.empty_like(close_prices)
        open_prices[0] = initial_price
        open_prices[1:] = close_prices[:-1]

        # High / low: random extension beyond max/min of open and close
        bar_max = np.maximum(open_prices, close_prices)
        bar_min = np.minimum(open_prices, close_prices)
        high_ext = rng.exponential(scale=intrabar_noise, size=n_samples)
        low_ext = rng.exponential(scale=intrabar_noise, size=n_samples)

        high_prices = bar_max * (1.0 + high_ext)
        low_prices = bar_min * (1.0 - low_ext)
        # Ensure low > 0
        low_prices = np.maximum(low_prices, bar_min * 0.5)

        # Volume
        volumes = np.abs(rng.normal(loc=volume_mean, scale=volume_std, size=n_samples))

        # Timestamps: one bar per hour starting from a fixed epoch
        base_ts = 1_700_000_000_000  # ~Nov 2023 in ms
        timestamps = base_ts + np.arange(n_samples) * 3_600_000

        klines: List[Kline] = []
        for i in range(n_samples):
            klines.append(
                Kline(
                    timestamp=int(timestamps[i]),
                    open=round(float(open_prices[i]), 4),
                    high=round(float(high_prices[i]), 4),
                    low=round(float(low_prices[i]), 4),
                    close=round(float(close_prices[i]), 4),
                    volume=round(float(volumes[i]), 2),
                )
            )

        return klines


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

class FeatureGenerator:
    """Technical-indicator feature generator for domain adaptation models.

    Computes a fixed set of normalised features from raw OHLCV data:

    1. **Returns** -- log return of the close price.
    2. **Volatility** -- rolling standard deviation of returns.
    3. **Momentum** -- cumulative return over the look-back window.
    4. **RSI** -- Relative Strength Index (Wilder smoothing).
    5. **Volume ratio** -- current volume / rolling mean volume.
    6. **MA ratio** -- close / rolling mean close (trend signal).
    7. **Bollinger position** -- where close sits within 2-sigma bands.

    All features are z-score normalised across the output array so that
    different financial instruments share comparable scales -- a prerequisite
    for effective domain adaptation.

    Parameters
    ----------
    window_size : int, default 20
        Look-back window for rolling statistics.
    """

    N_FEATURES = 7  # number of features produced

    def __init__(self, window_size: int = 20) -> None:
        self.window_size = window_size

    # ---- public ----------------------------------------------------------

    def generate_features(self, klines: List[Kline]) -> np.ndarray:
        """Compute feature matrix from a list of klines.

        Rows corresponding to the initial ``window_size`` bars are discarded
        because rolling statistics are undefined there.

        Parameters
        ----------
        klines : List[Kline]
            Chronologically ordered candlestick bars.

        Returns
        -------
        np.ndarray
            Feature matrix of shape ``(n_valid_bars, 7)`` where
            ``n_valid_bars = len(klines) - window_size``.

        Raises
        ------
        ValueError
            If fewer than ``window_size + 1`` klines are provided.
        """
        closes = np.array([k.close for k in klines], dtype=np.float64)
        volumes = np.array([k.volume for k in klines], dtype=np.float64)
        n = len(closes)
        w = self.window_size

        if n < w + 1:
            raise ValueError(
                f"Need at least {w + 1} klines, got {n}."
            )

        # --- raw series ---------------------------------------------------
        log_returns = np.diff(np.log(closes))  # length n-1

        # Pad log_returns to length n (first element = 0)
        log_returns_full = np.concatenate([[0.0], log_returns])

        # Rolling volatility (std of returns)
        volatility = self._rolling_std(log_returns_full, w)

        # Momentum: cumulative return over window
        momentum = self._rolling_sum(log_returns_full, w)

        # RSI (14-period by convention, but we use window_size for consistency)
        rsi = self._compute_rsi(closes, w)

        # Volume ratio: current volume / rolling mean volume
        vol_ma = self._rolling_mean(volumes, w)
        # Avoid division by zero
        vol_ma = np.where(vol_ma == 0, 1.0, vol_ma)
        volume_ratio = volumes / vol_ma

        # MA ratio: close / SMA
        close_ma = self._rolling_mean(closes, w)
        close_ma = np.where(close_ma == 0, 1.0, close_ma)
        ma_ratio = closes / close_ma

        # Bollinger position: (close - SMA) / (2 * rolling_std)
        close_std = self._rolling_std(closes, w)
        close_std = np.where(close_std == 0, 1.0, close_std)
        bollinger_pos = (closes - close_ma) / (2.0 * close_std)

        # --- assemble & trim to valid range [w:] -------------------------
        features = np.column_stack([
            log_returns_full[w:],
            volatility[w:],
            momentum[w:],
            rsi[w:],
            volume_ratio[w:],
            ma_ratio[w:],
            bollinger_pos[w:],
        ])

        # Z-score normalisation per feature column
        features = self._zscore(features)

        return features.astype(np.float32)

    def generate_labels(self, klines: List[Kline]) -> np.ndarray:
        """Generate binary classification labels.

        The label for bar *t* is ``1`` if the next-bar return is positive
        (i.e. ``close[t+1] > close[t]``), and ``0`` otherwise.  The last bar
        receives a label of ``0`` because there is no future return.

        The first ``window_size`` bars are dropped to align with
        :meth:`generate_features`.

        Parameters
        ----------
        klines : List[Kline]
            Chronologically ordered candlestick bars.

        Returns
        -------
        np.ndarray
            Binary label array of shape ``(n_valid_bars,)`` with dtype
            ``float32``, aligned with the output of :meth:`generate_features`.
        """
        closes = np.array([k.close for k in klines], dtype=np.float64)
        n = len(closes)
        w = self.window_size

        # Next-bar return sign
        labels = np.zeros(n, dtype=np.float32)
        labels[:-1] = (closes[1:] > closes[:-1]).astype(np.float32)

        return labels[w:]

    # ---- private helpers -------------------------------------------------

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Simple rolling mean using a cumulative-sum trick."""
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        rolling = (cs[window:] - cs[:-window]) / window
        # Pad front with first valid value to keep length == len(arr)
        pad = np.full(window - 1, rolling[0])
        return np.concatenate([pad, rolling])

    @staticmethod
    def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling standard deviation (population)."""
        n = len(arr)
        result = np.zeros(n, dtype=np.float64)
        for i in range(window - 1, n):
            result[i] = np.std(arr[i - window + 1 : i + 1])
        # Fill early values with zero
        return result

    @staticmethod
    def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
        """Rolling sum."""
        cs = np.cumsum(arr)
        cs = np.insert(cs, 0, 0.0)
        rolling = cs[window:] - cs[:-window]
        pad = np.full(window - 1, rolling[0])
        return np.concatenate([pad, rolling])

    @staticmethod
    def _compute_rsi(closes: np.ndarray, window: int) -> np.ndarray:
        """Relative Strength Index using exponential (Wilder) smoothing."""
        n = len(closes)
        rsi = np.full(n, 50.0, dtype=np.float64)  # neutral default

        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        if len(gains) < window:
            return rsi

        # Seed with SMA
        avg_gain = np.mean(gains[:window])
        avg_loss = np.mean(losses[:window])

        for i in range(window, len(deltas)):
            avg_gain = (avg_gain * (window - 1) + gains[i]) / window
            avg_loss = (avg_loss * (window - 1) + losses[i]) / window
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

        # Normalise RSI to [0, 1] range
        rsi = rsi / 100.0
        return rsi

    @staticmethod
    def _zscore(arr: np.ndarray) -> np.ndarray:
        """Per-column z-score normalisation with zero-std guard."""
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        std = np.where(std == 0, 1.0, std)
        return (arr - mean) / std


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def klines_to_dataframe(klines: List[Kline]) -> pd.DataFrame:
    """Convert a list of :class:`Kline` objects to a pandas DataFrame.

    Parameters
    ----------
    klines : List[Kline]
        Chronologically ordered kline bars.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``timestamp``, ``open``, ``high``, ``low``,
        ``close``, ``volume`` and a ``datetime`` column parsed from
        the millisecond timestamp.
    """
    df = pd.DataFrame([
        {
            "timestamp": k.timestamp,
            "open": k.open,
            "high": k.high,
            "low": k.low,
            "close": k.close,
            "volume": k.volume,
        }
        for k in klines
    ])
    if not df.empty:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("datetime")
    return df
