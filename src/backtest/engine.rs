//! # Backtesting Engine
//!
//! Simulates trading strategy execution on historical data with realistic
//! market conditions including commissions and slippage.
//!
//! ## Metrics Computed
//!
//! - **Total P&L**: Net profit/loss after all costs
//! - **Sharpe Ratio**: Risk-adjusted return (annualized)
//! - **Sortino Ratio**: Downside risk-adjusted return (annualized)
//! - **Maximum Drawdown**: Largest peak-to-trough decline
//! - **Win Rate**: Percentage of profitable trades
//! - **Profit Factor**: Gross profits / gross losses
//! - **Average Trade**: Mean P&L per trade
//!
//! ## Example
//!
//! ```rust,no_run
//! use domain_adaptation_trading::backtest::engine::{BacktestEngine, BacktestConfig};
//! use domain_adaptation_trading::trading::strategy::{AdaptiveStrategy, RiskParameters};
//! use domain_adaptation_trading::model::network::DomainAdaptationModel;
//! use domain_adaptation_trading::data::bybit::BybitClient;
//!
//! let config = BacktestConfig::default();
//! let engine = BacktestEngine::new(config);
//!
//! let model = DomainAdaptationModel::new(6, 32, 1);
//! let strategy = AdaptiveStrategy::new(model, 20, RiskParameters::default());
//!
//! let client = BybitClient::new();
//! let klines = client.simulated_klines("BTCUSDT", 500);
//!
//! let results = engine.run(&strategy, &klines).unwrap();
//! println!("{}", results);
//! ```

use crate::data::bybit::Kline;
use crate::trading::strategy::{AdaptiveStrategy, TradeRecord};
use crate::{DomainAdaptationError, Result};

/// Configuration for the backtesting engine.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Starting account balance in quote currency (e.g., USDT)
    pub initial_balance: f64,
    /// Commission rate per trade as a fraction (e.g., 0.001 = 0.1%)
    pub commission: f64,
    /// Slippage as a fraction of price (e.g., 0.0005 = 0.05%)
    pub slippage: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_balance: 10_000.0,
            commission: 0.001,  // 0.1% (typical for crypto)
            slippage: 0.0005,   // 0.05%
        }
    }
}

/// A single trade with costs applied.
#[derive(Debug, Clone)]
pub struct Trade {
    /// Price at which the trade was entered (adjusted for slippage)
    pub entry_price: f64,
    /// Price at which the trade was exited (adjusted for slippage)
    pub exit_price: f64,
    /// Timestamp or index of entry
    pub entry_time: u64,
    /// Timestamp or index of exit
    pub exit_time: u64,
    /// Direction: 1.0 for long, -1.0 for short
    pub direction: f64,
    /// Net P&L after commissions and slippage
    pub pnl: f64,
    /// Gross P&L before costs
    pub gross_pnl: f64,
    /// Total costs (commission + slippage)
    pub costs: f64,
    /// Holding period in number of bars
    pub holding_period: usize,
}

/// Comprehensive backtesting results.
#[derive(Debug, Clone)]
pub struct BacktestResults {
    /// All completed trades
    pub trades: Vec<Trade>,
    /// Net total profit/loss after all costs
    pub total_pnl: f64,
    /// Gross total profit/loss before costs
    pub gross_pnl: f64,
    /// Total trading costs
    pub total_costs: f64,
    /// Annualized Sharpe ratio (assuming 252 trading days)
    pub sharpe_ratio: f64,
    /// Annualized Sortino ratio (using downside deviation)
    pub sortino_ratio: f64,
    /// Maximum drawdown as a fraction (e.g., 0.1 = 10%)
    pub max_drawdown: f64,
    /// Win rate as a fraction (e.g., 0.6 = 60%)
    pub win_rate: f64,
    /// Profit factor: gross profits / gross losses
    pub profit_factor: f64,
    /// Average P&L per trade
    pub avg_trade_pnl: f64,
    /// Number of total trades
    pub total_trades: usize,
    /// Number of winning trades
    pub winning_trades: usize,
    /// Number of losing trades
    pub losing_trades: usize,
    /// Final account balance
    pub final_balance: f64,
    /// Return on initial investment as a fraction
    pub total_return: f64,
}

impl std::fmt::Display for BacktestResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "==========================================")?;
        writeln!(f, "         Backtest Results")?;
        writeln!(f, "==========================================")?;
        writeln!(f, "Total Trades:     {}", self.total_trades)?;
        writeln!(f, "Winning Trades:   {}", self.winning_trades)?;
        writeln!(f, "Losing Trades:    {}", self.losing_trades)?;
        writeln!(f, "------------------------------------------")?;
        writeln!(f, "Total Return:     {:.2}%", self.total_return * 100.0)?;
        writeln!(f, "Final Balance:    {:.2}", self.final_balance)?;
        writeln!(f, "Total P&L:        {:.2}", self.total_pnl)?;
        writeln!(f, "Gross P&L:        {:.2}", self.gross_pnl)?;
        writeln!(f, "Total Costs:      {:.2}", self.total_costs)?;
        writeln!(f, "------------------------------------------")?;
        writeln!(f, "Win Rate:         {:.1}%", self.win_rate * 100.0)?;
        writeln!(f, "Avg Trade P&L:    {:.4}", self.avg_trade_pnl)?;
        writeln!(f, "Profit Factor:    {:.2}", self.profit_factor)?;
        writeln!(f, "------------------------------------------")?;
        writeln!(f, "Sharpe Ratio:     {:.3}", self.sharpe_ratio)?;
        writeln!(f, "Sortino Ratio:    {:.3}", self.sortino_ratio)?;
        writeln!(f, "Max Drawdown:     {:.2}%", self.max_drawdown * 100.0)?;
        writeln!(f, "==========================================")
    }
}

/// Backtesting engine that simulates strategy execution on historical data.
///
/// Applies commissions and slippage to trade executions and computes
/// comprehensive performance metrics.
pub struct BacktestEngine {
    /// Backtesting configuration
    config: BacktestConfig,
}

impl BacktestEngine {
    /// Creates a new backtesting engine with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Backtesting parameters (initial balance, commission, slippage)
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Creates a backtesting engine with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(BacktestConfig::default())
    }

    /// Runs the backtest, executing the strategy on historical kline data.
    ///
    /// # Arguments
    ///
    /// * `strategy` - The trading strategy to evaluate
    /// * `klines` - Historical OHLCV data sorted chronologically
    ///
    /// # Returns
    ///
    /// Comprehensive backtesting results including all trades and metrics.
    ///
    /// # Errors
    ///
    /// Returns [`DomainAdaptationError::BacktestError`] if the strategy
    /// execution fails or produces invalid results.
    pub fn run(&self, strategy: &AdaptiveStrategy, klines: &[Kline]) -> Result<BacktestResults> {
        if klines.is_empty() {
            return Err(DomainAdaptationError::BacktestError(
                "No kline data provided".to_string(),
            ));
        }

        // Execute the strategy to get raw trade records
        let raw_trades = strategy.execute(klines)?;

        // Apply costs and convert to Trade objects
        let trades: Vec<Trade> = raw_trades
            .iter()
            .map(|rt| self.apply_costs(rt, klines))
            .collect();

        // Calculate metrics
        self.calculate_metrics(&trades)
    }

    /// Applies commission and slippage costs to a raw trade record.
    fn apply_costs(&self, raw_trade: &TradeRecord, klines: &[Kline]) -> Trade {
        // Apply slippage: entry price worsened, exit price worsened
        let entry_slippage = raw_trade.entry_price * self.config.slippage;
        let exit_slippage = raw_trade.exit_price * self.config.slippage;

        let entry_price = if raw_trade.direction > 0.0 {
            raw_trade.entry_price + entry_slippage // Long: buy higher
        } else {
            raw_trade.entry_price - entry_slippage // Short: sell lower
        };

        let exit_price = if raw_trade.direction > 0.0 {
            raw_trade.exit_price - exit_slippage // Long: sell lower
        } else {
            raw_trade.exit_price + exit_slippage // Short: buy higher
        };

        // Position size from raw trade
        let position_value = self.config.initial_balance * 0.1; // Risk 10% per trade
        let size = position_value / entry_price;

        // Gross P&L
        let gross_pnl = raw_trade.direction * (exit_price - entry_price) * size;

        // Commission costs
        let entry_commission = entry_price * size * self.config.commission;
        let exit_commission = exit_price * size * self.config.commission;
        let total_costs = entry_commission + exit_commission + (entry_slippage + exit_slippage) * size;

        let net_pnl = gross_pnl - entry_commission - exit_commission;

        // Timestamps
        let entry_time = if raw_trade.entry_index < klines.len() {
            klines[raw_trade.entry_index].timestamp
        } else {
            0
        };
        let exit_time = if raw_trade.exit_index < klines.len() {
            klines[raw_trade.exit_index].timestamp
        } else {
            0
        };

        Trade {
            entry_price,
            exit_price,
            entry_time,
            exit_time,
            direction: raw_trade.direction,
            pnl: net_pnl,
            gross_pnl,
            costs: total_costs,
            holding_period: raw_trade.exit_index.saturating_sub(raw_trade.entry_index),
        }
    }

    /// Calculates comprehensive performance metrics from completed trades.
    fn calculate_metrics(&self, trades: &[Trade]) -> Result<BacktestResults> {
        if trades.is_empty() {
            return Ok(BacktestResults {
                trades: Vec::new(),
                total_pnl: 0.0,
                gross_pnl: 0.0,
                total_costs: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                avg_trade_pnl: 0.0,
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                final_balance: self.config.initial_balance,
                total_return: 0.0,
            });
        }

        let total_pnl: f64 = trades.iter().map(|t| t.pnl).sum();
        let gross_pnl: f64 = trades.iter().map(|t| t.gross_pnl).sum();
        let total_costs: f64 = trades.iter().map(|t| t.costs).sum();

        let winning_trades = trades.iter().filter(|t| t.pnl > 0.0).count();
        let losing_trades = trades.iter().filter(|t| t.pnl <= 0.0).count();
        let total_trades = trades.len();

        let win_rate = if total_trades > 0 {
            winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let avg_trade_pnl = total_pnl / total_trades as f64;

        // Profit factor
        let gross_profits: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
        let gross_losses: f64 = trades
            .iter()
            .filter(|t| t.pnl <= 0.0)
            .map(|t| t.pnl.abs())
            .sum();
        let profit_factor = if gross_losses > 0.0 {
            gross_profits / gross_losses
        } else if gross_profits > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };

        // Compute equity curve for drawdown and risk metrics
        let returns: Vec<f64> = trades
            .iter()
            .map(|t| t.pnl / self.config.initial_balance)
            .collect();

        let sharpe_ratio = Self::compute_sharpe_ratio(&returns);
        let sortino_ratio = Self::compute_sortino_ratio(&returns);
        let max_drawdown = Self::compute_max_drawdown(&returns, self.config.initial_balance);

        let final_balance = self.config.initial_balance + total_pnl;
        let total_return = total_pnl / self.config.initial_balance;

        Ok(BacktestResults {
            trades: trades.to_vec(),
            total_pnl,
            gross_pnl,
            total_costs,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown,
            win_rate,
            profit_factor,
            avg_trade_pnl,
            total_trades,
            winning_trades,
            losing_trades,
            final_balance,
            total_return,
        })
    }

    /// Computes the annualized Sharpe ratio.
    ///
    /// `Sharpe = sqrt(252) * mean(returns) / std(returns)`
    ///
    /// Assumes returns are per-trade, not per-period.
    fn compute_sharpe_ratio(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        // Annualize: assume roughly 252 trading days / average trades per year
        let annualization = 252.0_f64.sqrt();
        annualization * mean / std
    }

    /// Computes the annualized Sortino ratio.
    ///
    /// Like Sharpe but only penalizes downside volatility:
    /// `Sortino = sqrt(252) * mean(returns) / downside_deviation`
    fn compute_sortino_ratio(returns: &[f64]) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;

        // Downside deviation: std of negative returns only
        let downside_sq: f64 = returns
            .iter()
            .filter(|&&r| r < 0.0)
            .map(|r| r.powi(2))
            .sum();
        let downside_dev = (downside_sq / n).sqrt();

        if downside_dev < 1e-10 {
            return if mean > 0.0 { f64::INFINITY } else { 0.0 };
        }

        let annualization = 252.0_f64.sqrt();
        annualization * mean / downside_dev
    }

    /// Computes the maximum drawdown from an equity curve.
    ///
    /// Maximum drawdown is the largest peak-to-trough decline as a fraction
    /// of the peak value.
    fn compute_max_drawdown(returns: &[f64], initial_balance: f64) -> f64 {
        if returns.is_empty() {
            return 0.0;
        }

        let mut equity = initial_balance;
        let mut peak = initial_balance;
        let mut max_dd = 0.0;

        for &ret in returns {
            equity += ret * initial_balance;
            if equity > peak {
                peak = equity;
            }
            let drawdown = (peak - equity) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }

    /// Returns a reference to the backtest configuration.
    pub fn config(&self) -> &BacktestConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::BybitClient;
    use crate::model::network::DomainAdaptationModel;
    use crate::trading::strategy::RiskParameters;

    #[test]
    fn test_backtest_config_default() {
        let config = BacktestConfig::default();
        assert!((config.initial_balance - 10_000.0).abs() < 1e-10);
        assert!((config.commission - 0.001).abs() < 1e-10);
        assert!((config.slippage - 0.0005).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_ratio_positive() {
        let returns = vec![0.01, 0.02, 0.01, 0.015, 0.005]; // All positive
        let sharpe = BacktestEngine::compute_sharpe_ratio(&returns);
        assert!(sharpe > 0.0, "Sharpe should be positive for all-positive returns");
    }

    #[test]
    fn test_sharpe_ratio_empty() {
        let sharpe = BacktestEngine::compute_sharpe_ratio(&[]);
        assert!((sharpe - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sortino_ratio() {
        let returns = vec![0.01, 0.02, -0.005, 0.015, -0.01];
        let sortino = BacktestEngine::compute_sortino_ratio(&returns);
        assert!(sortino.is_finite());
    }

    #[test]
    fn test_sortino_no_downside() {
        let returns = vec![0.01, 0.02, 0.03];
        let sortino = BacktestEngine::compute_sortino_ratio(&returns);
        assert!(sortino.is_infinite() || sortino > 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        // Scenario: go up, then down, then recover
        let returns = vec![0.1, 0.05, -0.15, -0.05, 0.1, 0.02];
        let dd = BacktestEngine::compute_max_drawdown(&returns, 10_000.0);
        assert!(dd > 0.0, "Max drawdown should be positive");
        assert!(dd <= 1.0, "Max drawdown should be <= 1.0");
    }

    #[test]
    fn test_max_drawdown_no_decline() {
        let returns = vec![0.01, 0.02, 0.03];
        let dd = BacktestEngine::compute_max_drawdown(&returns, 10_000.0);
        assert!((dd - 0.0).abs() < 1e-10, "No drawdown if always increasing");
    }

    #[test]
    fn test_backtest_empty_klines() {
        let engine = BacktestEngine::with_defaults();
        let model = DomainAdaptationModel::new(6, 32, 1);
        let strategy = AdaptiveStrategy::new(model, 20, RiskParameters::default());

        let result = engine.run(&strategy, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_backtest_run() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 200);

        let engine = BacktestEngine::with_defaults();
        let model = DomainAdaptationModel::new(6, 32, 1);
        let strategy = AdaptiveStrategy::new(model, 20, RiskParameters::default());

        let results = engine.run(&strategy, &klines).unwrap();

        // Verify result consistency
        assert_eq!(
            results.total_trades,
            results.winning_trades + results.losing_trades
        );
        assert!((0.0..=1.0).contains(&results.win_rate));
        assert!(results.max_drawdown >= 0.0);
        assert!(results.max_drawdown <= 1.0);
    }

    #[test]
    fn test_results_display() {
        let results = BacktestResults {
            trades: Vec::new(),
            total_pnl: 500.0,
            gross_pnl: 600.0,
            total_costs: 100.0,
            sharpe_ratio: 1.5,
            sortino_ratio: 2.0,
            max_drawdown: 0.05,
            win_rate: 0.6,
            profit_factor: 1.8,
            avg_trade_pnl: 25.0,
            total_trades: 20,
            winning_trades: 12,
            losing_trades: 8,
            final_balance: 10_500.0,
            total_return: 0.05,
        };

        let display = format!("{}", results);
        assert!(display.contains("500.00"));
        assert!(display.contains("1.500"));
        assert!(display.contains("5.00%")); // max drawdown or total return
    }

    #[test]
    fn test_empty_trades_metrics() {
        let engine = BacktestEngine::with_defaults();
        let results = engine.calculate_metrics(&[]).unwrap();

        assert_eq!(results.total_trades, 0);
        assert!((results.total_pnl - 0.0).abs() < 1e-10);
        assert!((results.sharpe_ratio - 0.0).abs() < 1e-10);
        assert!((results.final_balance - 10_000.0).abs() < 1e-10);
    }
}
