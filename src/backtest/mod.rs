//! # Backtest Module
//!
//! Provides a backtesting engine for evaluating domain-adapted trading strategies
//! on historical data. Computes comprehensive performance metrics including
//! Sharpe ratio, Sortino ratio, maximum drawdown, and win rate.
//!
//! ## Features
//!
//! - Configurable initial balance, commission, and slippage
//! - Detailed trade-by-trade logging
//! - Risk-adjusted return metrics
//! - Equity curve computation for drawdown analysis

pub mod engine;
