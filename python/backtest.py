"""
Backtesting Engine for Domain-Adapted Trading Models
=====================================================

Provides an event-driven backtesting framework that evaluates domain-adapted
models on historical (or simulated) OHLCV data.  The engine supports
configurable transaction costs, slippage, position sizing, and risk-management
rules (stop-loss and take-profit).

Key performance metrics are computed automatically:

* **Sharpe ratio** -- risk-adjusted return using daily log returns.
* **Sortino ratio** -- downside-risk-adjusted return.
* **Maximum drawdown** -- largest peak-to-trough equity decline.
* **Win rate** -- fraction of profitable trades.

Classes
-------
Trade
    Record of a single completed round-trip trade.
BacktestConfig
    Configuration parameters for the backtesting engine.
BacktestResults
    Container for aggregate performance statistics.
BacktestEngine
    Core engine that steps through bars, generates signals, and tracks P&L.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

from .data_loader import FeatureGenerator, Kline


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Record of a single completed round-trip trade.

    Attributes
    ----------
    entry_price : float
        Price at which the position was opened (after slippage).
    exit_price : float
        Price at which the position was closed (after slippage).
    entry_time : int
        Timestamp (ms) when the position was opened.
    exit_time : int
        Timestamp (ms) when the position was closed.
    direction : str
        ``"long"`` or ``"short"``.
    pnl : float
        Realised profit/loss for the trade (after commissions).
    """

    entry_price: float
    exit_price: float
    entry_time: int
    exit_time: int
    direction: str
    pnl: float


@dataclass
class BacktestConfig:
    """Configuration for the backtesting engine.

    Attributes
    ----------
    initial_balance : float
        Starting account balance in quote currency.
    commission : float
        One-way commission rate (e.g. 0.001 = 0.1 %).
    slippage : float
        Simulated slippage as a fraction of price (e.g. 0.0005 = 5 bps).
    max_position_size : float
        Maximum fraction of equity risked per trade (e.g. 0.1 = 10 %).
    stop_loss_pct : float
        Stop-loss trigger as a percentage move against the position.
    take_profit_pct : float
        Take-profit trigger as a percentage move in favour.
    min_holding_bars : int
        Minimum number of bars to hold before allowing an exit signal.
    """

    initial_balance: float = 100_000.0
    commission: float = 0.001
    slippage: float = 0.0005
    max_position_size: float = 0.1
    stop_loss_pct: float = 0.02
    take_profit_pct: float = 0.04
    min_holding_bars: int = 1


@dataclass
class BacktestResults:
    """Aggregate results from a backtest run.

    Attributes
    ----------
    trades : List[Trade]
        All completed round-trip trades.
    total_pnl : float
        Sum of P&L across all trades.
    sharpe_ratio : float
        Annualised Sharpe ratio (assuming 252 trading days).
    sortino_ratio : float
        Annualised Sortino ratio (downside deviation only).
    max_drawdown : float
        Maximum peak-to-trough drawdown as a positive fraction (e.g. 0.15 = 15 %).
    win_rate : float
        Fraction of trades with positive P&L.
    total_return_pct : float
        Total return as a percentage of initial balance.
    n_trades : int
        Total number of completed trades.
    avg_trade_pnl : float
        Average P&L per trade.
    profit_factor : float
        Gross profit / gross loss (inf if no losing trades).
    """

    trades: List[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    total_return_pct: float = 0.0
    n_trades: int = 0
    avg_trade_pnl: float = 0.0
    profit_factor: float = 0.0

    def summary(self) -> str:
        """Return a formatted multi-line summary string."""
        lines = [
            "=" * 50,
            "         Backtest Results Summary",
            "=" * 50,
            f"  Total trades:       {self.n_trades}",
            f"  Win rate:           {self.win_rate:.2%}",
            f"  Total P&L:          {self.total_pnl:>12.2f}",
            f"  Total return:       {self.total_return_pct:>+.2f}%",
            f"  Avg trade P&L:      {self.avg_trade_pnl:>12.2f}",
            f"  Profit factor:      {self.profit_factor:>12.2f}",
            f"  Sharpe ratio:       {self.sharpe_ratio:>12.3f}",
            f"  Sortino ratio:      {self.sortino_ratio:>12.3f}",
            f"  Max drawdown:       {self.max_drawdown:.2%}",
            "=" * 50,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """Event-driven backtesting engine for domain-adapted models.

    The engine iterates bar-by-bar through historical data, generates
    trading signals via a PyTorch model, and simulates order execution
    with realistic transaction costs and slippage.

    Parameters
    ----------
    config : BacktestConfig or None
        Backtesting configuration.  If ``None``, defaults are used.

    Examples
    --------
    >>> from domain_adaptation_finance import (
    ...     SimulatedDataGenerator, FeatureGenerator,
    ...     DomainAdaptationModel, BacktestEngine, BacktestConfig,
    ... )
    >>> gen = SimulatedDataGenerator()
    >>> klines = gen.generate_stock_data(300, seed=42)
    >>> model = DomainAdaptationModel(input_size=7, hidden_size=32)
    >>> engine = BacktestEngine(BacktestConfig(initial_balance=50_000))
    >>> results = engine.run(model, klines, FeatureGenerator())
    >>> print(results.summary())
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    # ---- public ----------------------------------------------------------

    def run(
        self,
        model: torch.nn.Module,
        klines: List[Kline],
        feature_generator: FeatureGenerator,
    ) -> BacktestResults:
        """Execute a full backtest.

        Parameters
        ----------
        model : torch.nn.Module
            A trained model that accepts a feature tensor and returns label
            logits via its ``feature_extractor`` and ``label_predictor``
            (i.e. a :class:`DomainAdaptationModel`).
        klines : List[Kline]
            Historical OHLCV bars (chronological order).
        feature_generator : FeatureGenerator
            Feature generator to transform klines into model inputs.

        Returns
        -------
        BacktestResults
            Comprehensive performance statistics.
        """
        cfg = self.config

        # Generate features and align klines
        features = feature_generator.generate_features(klines)
        # klines are trimmed by window_size at the front
        offset = feature_generator.window_size
        aligned_klines = klines[offset:]

        # Generate model signals
        signals = self._generate_signals(model, features)

        # Simulate trading
        trades: List[Trade] = []
        balance = cfg.initial_balance
        position: Optional[dict] = None  # {direction, entry_price, entry_time, entry_bar}

        for i in range(len(signals)):
            current_kline = aligned_klines[i]
            signal = signals[i]

            if position is not None:
                # Check exit conditions
                bars_held = i - position["entry_bar"]
                should_exit = False
                exit_price = current_kline.close

                if position["direction"] == "long":
                    pct_move = (current_kline.close - position["entry_price"]) / position["entry_price"]
                    # Stop loss
                    if current_kline.low <= position["entry_price"] * (1 - cfg.stop_loss_pct):
                        exit_price = position["entry_price"] * (1 - cfg.stop_loss_pct)
                        should_exit = True
                    # Take profit
                    elif current_kline.high >= position["entry_price"] * (1 + cfg.take_profit_pct):
                        exit_price = position["entry_price"] * (1 + cfg.take_profit_pct)
                        should_exit = True
                    # Signal reversal (only after minimum hold)
                    elif signal == -1 and bars_held >= cfg.min_holding_bars:
                        should_exit = True
                else:  # short
                    pct_move = (position["entry_price"] - current_kline.close) / position["entry_price"]
                    # Stop loss
                    if current_kline.high >= position["entry_price"] * (1 + cfg.stop_loss_pct):
                        exit_price = position["entry_price"] * (1 + cfg.stop_loss_pct)
                        should_exit = True
                    # Take profit
                    elif current_kline.low <= position["entry_price"] * (1 - cfg.take_profit_pct):
                        exit_price = position["entry_price"] * (1 - cfg.take_profit_pct)
                        should_exit = True
                    # Signal reversal
                    elif signal == 1 and bars_held >= cfg.min_holding_bars:
                        should_exit = True

                if should_exit:
                    # Apply slippage to exit
                    if position["direction"] == "long":
                        exit_price *= (1 - cfg.slippage)
                    else:
                        exit_price *= (1 + cfg.slippage)

                    # Calculate P&L
                    pos_size = balance * cfg.max_position_size
                    if position["direction"] == "long":
                        gross_pnl = pos_size * (exit_price - position["entry_price"]) / position["entry_price"]
                    else:
                        gross_pnl = pos_size * (position["entry_price"] - exit_price) / position["entry_price"]

                    # Commissions (entry + exit)
                    commission_cost = pos_size * cfg.commission * 2
                    net_pnl = gross_pnl - commission_cost

                    trade = Trade(
                        entry_price=position["entry_price"],
                        exit_price=exit_price,
                        entry_time=position["entry_time"],
                        exit_time=current_kline.timestamp,
                        direction=position["direction"],
                        pnl=net_pnl,
                    )
                    trades.append(trade)
                    balance += net_pnl
                    position = None

            # Open new position if flat and have a signal
            if position is None and signal != 0:
                direction = "long" if signal == 1 else "short"
                entry_price = current_kline.close

                # Apply slippage to entry
                if direction == "long":
                    entry_price *= (1 + cfg.slippage)
                else:
                    entry_price *= (1 - cfg.slippage)

                position = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "entry_time": current_kline.timestamp,
                    "entry_bar": i,
                }

        # Force close any open position at the end
        if position is not None and len(aligned_klines) > 0:
            last_kline = aligned_klines[-1]
            exit_price = last_kline.close
            if position["direction"] == "long":
                exit_price *= (1 - cfg.slippage)
            else:
                exit_price *= (1 + cfg.slippage)

            pos_size = balance * cfg.max_position_size
            if position["direction"] == "long":
                gross_pnl = pos_size * (exit_price - position["entry_price"]) / position["entry_price"]
            else:
                gross_pnl = pos_size * (position["entry_price"] - exit_price) / position["entry_price"]

            commission_cost = pos_size * cfg.commission * 2
            net_pnl = gross_pnl - commission_cost

            trade = Trade(
                entry_price=position["entry_price"],
                exit_price=exit_price,
                entry_time=position["entry_time"],
                exit_time=last_kline.timestamp,
                direction=position["direction"],
                pnl=net_pnl,
            )
            trades.append(trade)
            balance += net_pnl

        return self._calculate_metrics(trades, cfg.initial_balance)

    # ---- private ---------------------------------------------------------

    def _generate_signals(
        self,
        model: torch.nn.Module,
        features: np.ndarray,
    ) -> List[int]:
        """Generate trading signals from model predictions.

        The model is run in evaluation mode.  Class 1 maps to a long signal
        (+1), class 0 maps to a short signal (-1).  If the model's confidence
        is below 55 % for either class, the signal is neutral (0).

        Parameters
        ----------
        model : torch.nn.Module
            Trained domain-adapted model.
        features : np.ndarray
            Feature matrix of shape ``(n_bars, n_features)``.

        Returns
        -------
        List[int]
            Signal per bar: ``+1`` (long), ``-1`` (short), or ``0`` (flat).
        """
        model.eval()
        confidence_threshold = 0.55

        x = torch.tensor(features, dtype=torch.float32)

        with torch.no_grad():
            # Support both DomainAdaptationModel and plain models
            if hasattr(model, "feature_extractor") and hasattr(model, "label_predictor"):
                feats = model.feature_extractor(x)
                logits = model.label_predictor(feats)
            else:
                logits = model(x)

            probs = torch.softmax(logits, dim=1)

        signals: List[int] = []
        for i in range(len(probs)):
            prob_up = probs[i, 1].item() if probs.size(1) > 1 else probs[i, 0].item()
            prob_down = 1.0 - prob_up

            if prob_up >= confidence_threshold:
                signals.append(1)
            elif prob_down >= confidence_threshold:
                signals.append(-1)
            else:
                signals.append(0)

        return signals

    @staticmethod
    def _calculate_metrics(
        trades: List[Trade],
        initial_balance: float,
    ) -> BacktestResults:
        """Compute aggregate performance metrics from a list of trades.

        Parameters
        ----------
        trades : List[Trade]
            Completed round-trip trades.
        initial_balance : float
            Starting account balance.

        Returns
        -------
        BacktestResults
            Filled-in results dataclass.
        """
        if not trades:
            return BacktestResults(
                trades=trades,
                total_pnl=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_return_pct=0.0,
                n_trades=0,
                avg_trade_pnl=0.0,
                profit_factor=0.0,
            )

        pnls = np.array([t.pnl for t in trades])
        total_pnl = float(pnls.sum())
        n_trades = len(trades)
        avg_pnl = total_pnl / n_trades

        # Win rate
        wins = int((pnls > 0).sum())
        win_rate = wins / n_trades

        # Total return
        total_return_pct = (total_pnl / initial_balance) * 100.0

        # Profit factor
        gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
        gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Equity curve for drawdown and ratios
        equity = np.zeros(n_trades + 1)
        equity[0] = initial_balance
        for i, pnl in enumerate(pnls):
            equity[i + 1] = equity[i] + pnl

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / np.where(peak == 0, 1.0, peak)
        max_drawdown = float(drawdowns.max())

        # Sharpe ratio (annualised, assuming ~252 trading days / trades-per-year)
        if len(pnls) > 1 and pnls.std() > 0:
            # Normalise returns by account balance for ratio calculation
            returns = pnls / initial_balance
            annualisation = math.sqrt(min(252, n_trades))
            sharpe_ratio = float(returns.mean() / returns.std() * annualisation)
        else:
            sharpe_ratio = 0.0

        # Sortino ratio (downside deviation only)
        if len(pnls) > 1:
            returns = pnls / initial_balance
            downside = returns[returns < 0]
            if len(downside) > 0 and downside.std() > 0:
                annualisation = math.sqrt(min(252, n_trades))
                sortino_ratio = float(returns.mean() / downside.std() * annualisation)
            else:
                # No downside -- perfectly positive
                sortino_ratio = float("inf") if returns.mean() > 0 else 0.0
        else:
            sortino_ratio = 0.0

        return BacktestResults(
            trades=trades,
            total_pnl=total_pnl,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_return_pct=total_return_pct,
            n_trades=n_trades,
            avg_trade_pnl=avg_pnl,
            profit_factor=profit_factor,
        )
