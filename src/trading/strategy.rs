//! # Adaptive Trading Strategy
//!
//! Implements a complete trading strategy that combines domain-adapted model
//! predictions with risk management. The strategy manages positions with
//! configurable stop-loss, take-profit, and position sizing parameters.
//!
//! ## Strategy Flow
//!
//! ```text
//! Klines -> Feature Generation -> Signal Generation -> Risk Check -> Position Management
//!                                                                          |
//!                                                     Entry/Exit Decision <-+
//!                                                          |
//!                                                    Trade Execution
//! ```
//!
//! ## Risk Management
//!
//! - **Position sizing**: Limited to `max_position_size` as a fraction of equity
//! - **Stop loss**: Automatic exit when unrealized loss exceeds threshold
//! - **Take profit**: Automatic exit when unrealized profit reaches threshold
//! - **Single position**: Only one position at a time (no pyramiding)

use crate::data::bybit::Kline;
use crate::data::features::FeatureGenerator;
use crate::model::network::DomainAdaptationModel;
use crate::trading::signals::{SignalGenerator, TradingSignal};
use crate::Result;

/// Represents a trading position with entry information and direction.
#[derive(Debug, Clone)]
pub struct Position {
    /// Price at which the position was entered
    pub entry_price: f64,
    /// Position size (number of units)
    pub size: f64,
    /// Direction: 1.0 for long, -1.0 for short
    pub direction: f64,
    /// Index (or timestamp) when the position was entered
    pub entry_index: usize,
}

impl Position {
    /// Computes the unrealized profit/loss at the given price.
    ///
    /// # Arguments
    ///
    /// * `current_price` - Current market price
    ///
    /// # Returns
    ///
    /// Unrealized P&L (positive = profit, negative = loss).
    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        self.direction * (current_price - self.entry_price) * self.size
    }

    /// Computes the unrealized return as a fraction of entry value.
    pub fn unrealized_return(&self, current_price: f64) -> f64 {
        self.direction * (current_price - self.entry_price) / self.entry_price
    }

    /// Returns true if this is a long position.
    pub fn is_long(&self) -> bool {
        self.direction > 0.0
    }
}

/// Risk management parameters for the trading strategy.
#[derive(Debug, Clone)]
pub struct RiskParameters {
    /// Maximum position size as a fraction of equity (e.g., 0.1 = 10%)
    pub max_position_size: f64,
    /// Stop loss as a fraction of entry price (e.g., 0.02 = 2%)
    pub stop_loss: f64,
    /// Take profit as a fraction of entry price (e.g., 0.04 = 4%)
    pub take_profit: f64,
}

impl Default for RiskParameters {
    fn default() -> Self {
        Self {
            max_position_size: 0.1,
            stop_loss: 0.02,
            take_profit: 0.04,
        }
    }
}

/// Record of a completed trade (entry and exit).
#[derive(Debug, Clone)]
pub struct TradeRecord {
    /// Price at which the trade was entered
    pub entry_price: f64,
    /// Price at which the trade was exited
    pub exit_price: f64,
    /// Index when the trade was entered
    pub entry_index: usize,
    /// Index when the trade was exited
    pub exit_index: usize,
    /// Direction: 1.0 for long, -1.0 for short
    pub direction: f64,
    /// Realized profit/loss
    pub pnl: f64,
    /// Reason for exit
    pub exit_reason: ExitReason,
}

/// Reason why a trade was closed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExitReason {
    /// Signal changed (Buy -> Sell or vice versa)
    SignalChange,
    /// Stop loss triggered
    StopLoss,
    /// Take profit triggered
    TakeProfit,
    /// End of data
    EndOfData,
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::SignalChange => write!(f, "Signal Change"),
            ExitReason::StopLoss => write!(f, "Stop Loss"),
            ExitReason::TakeProfit => write!(f, "Take Profit"),
            ExitReason::EndOfData => write!(f, "End of Data"),
        }
    }
}

/// Adaptive trading strategy powered by a domain-adapted model.
///
/// Combines model-based signal generation with risk management rules
/// to produce a complete trading system suitable for backtesting.
///
/// # Example
///
/// ```rust
/// use domain_adaptation_trading::model::network::DomainAdaptationModel;
/// use domain_adaptation_trading::trading::strategy::{AdaptiveStrategy, RiskParameters};
///
/// let model = DomainAdaptationModel::new(6, 32, 1);
/// let risk = RiskParameters {
///     max_position_size: 0.1,
///     stop_loss: 0.02,
///     take_profit: 0.04,
/// };
/// let strategy = AdaptiveStrategy::new(model, 20, risk);
/// ```
pub struct AdaptiveStrategy {
    /// Signal generator wrapping the domain-adapted model
    signal_generator: SignalGenerator,
    /// Feature generator for converting klines to model inputs
    feature_generator: FeatureGenerator,
    /// Risk management parameters
    risk_params: RiskParameters,
}

impl AdaptiveStrategy {
    /// Creates a new adaptive trading strategy.
    ///
    /// # Arguments
    ///
    /// * `model` - Trained domain adaptation model
    /// * `window_size` - Lookback window for feature generation
    /// * `risk_params` - Risk management parameters
    pub fn new(
        model: DomainAdaptationModel,
        window_size: usize,
        risk_params: RiskParameters,
    ) -> Self {
        Self {
            signal_generator: SignalGenerator::with_defaults(model),
            feature_generator: FeatureGenerator::new(window_size),
            risk_params,
        }
    }

    /// Creates a strategy with custom signal thresholds.
    ///
    /// # Arguments
    ///
    /// * `model` - Trained domain adaptation model
    /// * `window_size` - Lookback window for feature generation
    /// * `risk_params` - Risk management parameters
    /// * `buy_threshold` - Probability threshold for Buy signals
    /// * `sell_threshold` - Probability threshold for Sell signals
    pub fn with_thresholds(
        model: DomainAdaptationModel,
        window_size: usize,
        risk_params: RiskParameters,
        buy_threshold: f64,
        sell_threshold: f64,
    ) -> Self {
        Self {
            signal_generator: SignalGenerator::new(model, buy_threshold, sell_threshold),
            feature_generator: FeatureGenerator::new(window_size),
            risk_params,
        }
    }

    /// Executes the strategy on historical kline data.
    ///
    /// Iterates through the kline data, generating signals and managing
    /// positions according to the risk parameters. Returns all completed
    /// trades and any open position at the end.
    ///
    /// # Arguments
    ///
    /// * `klines` - Historical OHLCV data sorted chronologically
    ///
    /// # Returns
    ///
    /// A vector of completed [`TradeRecord`]s.
    ///
    /// # Process
    ///
    /// 1. Generate features from klines
    /// 2. For each feature vector, generate a trading signal
    /// 3. Check risk management rules (stop loss, take profit)
    /// 4. Open/close positions based on signals and risk checks
    pub fn execute(&self, klines: &[Kline]) -> Result<Vec<TradeRecord>> {
        let features = self.feature_generator.generate_features(klines);
        if features.is_empty() {
            return Ok(Vec::new());
        }

        let mut trades: Vec<TradeRecord> = Vec::new();
        let mut current_position: Option<Position> = None;

        // The features start at index window_size in the klines array
        let offset = self.feature_generator.window_size;

        for (i, feature_vec) in features.iter().enumerate() {
            let kline_idx = offset + i;
            let current_price = klines[kline_idx].close;

            // Check risk management for existing positions
            if let Some(ref pos) = current_position {
                let unr_return = pos.unrealized_return(current_price);

                // Check stop loss
                if unr_return <= -self.risk_params.stop_loss {
                    let pnl = pos.unrealized_pnl(current_price);
                    trades.push(TradeRecord {
                        entry_price: pos.entry_price,
                        exit_price: current_price,
                        entry_index: pos.entry_index,
                        exit_index: kline_idx,
                        direction: pos.direction,
                        pnl,
                        exit_reason: ExitReason::StopLoss,
                    });
                    current_position = None;
                    continue;
                }

                // Check take profit
                if unr_return >= self.risk_params.take_profit {
                    let pnl = pos.unrealized_pnl(current_price);
                    trades.push(TradeRecord {
                        entry_price: pos.entry_price,
                        exit_price: current_price,
                        entry_index: pos.entry_index,
                        exit_index: kline_idx,
                        direction: pos.direction,
                        pnl,
                        exit_reason: ExitReason::TakeProfit,
                    });
                    current_position = None;
                    continue;
                }
            }

            // Generate trading signal
            let signal = self.signal_generator.generate_signal(feature_vec)?;

            match signal {
                TradingSignal::Buy => {
                    if let Some(ref pos) = current_position {
                        if !pos.is_long() {
                            // Close short position on Buy signal
                            let pnl = pos.unrealized_pnl(current_price);
                            trades.push(TradeRecord {
                                entry_price: pos.entry_price,
                                exit_price: current_price,
                                entry_index: pos.entry_index,
                                exit_index: kline_idx,
                                direction: pos.direction,
                                pnl,
                                exit_reason: ExitReason::SignalChange,
                            });
                            current_position = None;
                        }
                    }

                    if current_position.is_none() {
                        // Open long position
                        let size = self.risk_params.max_position_size / current_price;
                        current_position = Some(Position {
                            entry_price: current_price,
                            size,
                            direction: 1.0,
                            entry_index: kline_idx,
                        });
                    }
                }
                TradingSignal::Sell => {
                    if let Some(ref pos) = current_position {
                        if pos.is_long() {
                            // Close long position on Sell signal
                            let pnl = pos.unrealized_pnl(current_price);
                            trades.push(TradeRecord {
                                entry_price: pos.entry_price,
                                exit_price: current_price,
                                entry_index: pos.entry_index,
                                exit_index: kline_idx,
                                direction: pos.direction,
                                pnl,
                                exit_reason: ExitReason::SignalChange,
                            });
                            current_position = None;
                        }
                    }

                    if current_position.is_none() {
                        // Open short position
                        let size = self.risk_params.max_position_size / current_price;
                        current_position = Some(Position {
                            entry_price: current_price,
                            size,
                            direction: -1.0,
                            entry_index: kline_idx,
                        });
                    }
                }
                TradingSignal::Hold => {
                    // Do nothing - maintain current position
                }
            }
        }

        // Close any remaining position at end of data
        if let Some(pos) = current_position {
            let last_price = klines.last().map(|k| k.close).unwrap_or(0.0);
            let pnl = pos.unrealized_pnl(last_price);
            trades.push(TradeRecord {
                entry_price: pos.entry_price,
                exit_price: last_price,
                entry_index: pos.entry_index,
                exit_index: klines.len() - 1,
                direction: pos.direction,
                pnl,
                exit_reason: ExitReason::EndOfData,
            });
        }

        Ok(trades)
    }

    /// Returns a reference to the signal generator.
    pub fn signal_generator(&self) -> &SignalGenerator {
        &self.signal_generator
    }

    /// Returns a reference to the risk parameters.
    pub fn risk_params(&self) -> &RiskParameters {
        &self.risk_params
    }

    /// Returns the feature generator's window size.
    pub fn window_size(&self) -> usize {
        self.feature_generator.window_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::bybit::BybitClient;

    #[test]
    fn test_position_pnl_long() {
        let pos = Position {
            entry_price: 100.0,
            size: 1.0,
            direction: 1.0,
            entry_index: 0,
        };

        assert!((pos.unrealized_pnl(110.0) - 10.0).abs() < 1e-10);
        assert!((pos.unrealized_pnl(90.0) - (-10.0)).abs() < 1e-10);
        assert!(pos.is_long());
    }

    #[test]
    fn test_position_pnl_short() {
        let pos = Position {
            entry_price: 100.0,
            size: 1.0,
            direction: -1.0,
            entry_index: 0,
        };

        assert!((pos.unrealized_pnl(90.0) - 10.0).abs() < 1e-10);
        assert!((pos.unrealized_pnl(110.0) - (-10.0)).abs() < 1e-10);
        assert!(!pos.is_long());
    }

    #[test]
    fn test_position_return() {
        let pos = Position {
            entry_price: 100.0,
            size: 1.0,
            direction: 1.0,
            entry_index: 0,
        };

        assert!((pos.unrealized_return(102.0) - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_risk_parameters_default() {
        let risk = RiskParameters::default();
        assert!((risk.max_position_size - 0.1).abs() < 1e-10);
        assert!((risk.stop_loss - 0.02).abs() < 1e-10);
        assert!((risk.take_profit - 0.04).abs() < 1e-10);
    }

    #[test]
    fn test_exit_reason_display() {
        assert_eq!(format!("{}", ExitReason::StopLoss), "Stop Loss");
        assert_eq!(format!("{}", ExitReason::TakeProfit), "Take Profit");
        assert_eq!(format!("{}", ExitReason::SignalChange), "Signal Change");
        assert_eq!(format!("{}", ExitReason::EndOfData), "End of Data");
    }

    #[test]
    fn test_strategy_creation() {
        let model = DomainAdaptationModel::new(6, 32, 1);
        let risk = RiskParameters::default();
        let strategy = AdaptiveStrategy::new(model, 20, risk);
        assert_eq!(strategy.window_size(), 20);
    }

    #[test]
    fn test_strategy_execute() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 200);

        let model = DomainAdaptationModel::new(6, 32, 1);
        let risk = RiskParameters::default();
        let strategy = AdaptiveStrategy::new(model, 20, risk);

        let trades = strategy.execute(&klines).unwrap();
        // Should produce some trades with 200 klines
        // (exact number depends on model initialization)
        // Verify execution completes without panic; trade count depends on model init
        let _ = trades.len();
    }

    #[test]
    fn test_strategy_insufficient_data() {
        let client = BybitClient::new();
        let klines = client.simulated_klines("BTCUSDT", 10);

        let model = DomainAdaptationModel::new(6, 32, 1);
        let risk = RiskParameters::default();
        let strategy = AdaptiveStrategy::new(model, 20, risk);

        let trades = strategy.execute(&klines).unwrap();
        assert!(trades.is_empty());
    }

    #[test]
    fn test_trade_record_fields() {
        let trade = TradeRecord {
            entry_price: 100.0,
            exit_price: 105.0,
            entry_index: 0,
            exit_index: 10,
            direction: 1.0,
            pnl: 5.0,
            exit_reason: ExitReason::TakeProfit,
        };

        assert_eq!(trade.exit_reason, ExitReason::TakeProfit);
        assert!((trade.pnl - 5.0).abs() < 1e-10);
    }
}
