//! # Domain Adaptation for Finance
//!
//! This crate implements domain adaptation techniques for adapting trading models
//! across different financial domains (e.g., from equities to crypto, from one
//! exchange to another, or from one market regime to another).
//!
//! ## Key Techniques
//!
//! - **DANN (Domain-Adversarial Neural Network)**: Uses adversarial training with
//!   a gradient reversal layer to learn domain-invariant feature representations.
//!
//! - **MMD (Maximum Mean Discrepancy)**: Minimizes the distribution distance between
//!   source and target domains using kernel-based methods.
//!
//! - **CORAL (Correlation Alignment)**: Aligns the second-order statistics (covariance)
//!   of source and target feature distributions.
//!
//! ## Architecture
//!
//! ```text
//! +-----------+     +-------------------+     +------------------+
//! | Raw Data  | --> | Feature Generator | --> | Domain Adaptation|
//! | (Bybit)   |     | (returns, vol,   |     | (DANN/MMD/CORAL) |
//! +-----------+     |  momentum, RSI)  |     +------------------+
//!                   +-------------------+              |
//!                                                      v
//!                   +-------------------+     +------------------+
//!                   | Backtest Engine   | <-- | Trading Strategy |
//!                   | (metrics, PnL)   |     | (signals, risk)  |
//!                   +-------------------+     +------------------+
//! ```
//!
//! ## Usage
//!
//! ```rust,no_run
//! use domain_adaptation_trading::data::bybit::BybitClient;
//! use domain_adaptation_trading::data::features::FeatureGenerator;
//! use domain_adaptation_trading::adaptation::dann::DANNTrainer;
//!
//! // Fetch market data
//! let client = BybitClient::new();
//! let feature_gen = FeatureGenerator::new(20);
//! ```

pub mod data;
pub mod model;
pub mod adaptation;
pub mod trading;
pub mod backtest;

// Re-export commonly used types
pub use data::bybit::{BybitClient, Kline};
pub use data::features::FeatureGenerator;
pub use model::network::{DomainAdaptationModel, FeatureExtractor, LabelPredictor, DomainClassifier};
pub use adaptation::dann::DANNTrainer;
pub use adaptation::mmd::MMDAdapter;
pub use adaptation::coral::CORALAdapter;
pub use trading::signals::{TradingSignal, SignalGenerator};
pub use trading::strategy::{AdaptiveStrategy, Position};
pub use backtest::engine::{BacktestEngine, BacktestConfig, BacktestResults, Trade};

use thiserror::Error;

/// Errors that can occur during domain adaptation operations.
#[derive(Error, Debug)]
pub enum DomainAdaptationError {
    /// Error during model operations (forward pass, training, etc.)
    #[error("Model error: {0}")]
    ModelError(String),

    /// Error during data loading or preprocessing
    #[error("Data error: {0}")]
    DataError(String),

    /// Error communicating with external APIs (e.g., Bybit)
    #[error("API error: {0}")]
    ApiError(String),

    /// Error during backtesting
    #[error("Backtest error: {0}")]
    BacktestError(String),

    /// Error during domain adaptation procedure
    #[error("Adaptation error: {0}")]
    AdaptationError(String),

    /// Invalid parameter provided to a function
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// A specialized Result type for domain adaptation operations.
pub type Result<T> = std::result::Result<T, DomainAdaptationError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = DomainAdaptationError::ModelError("dimension mismatch".to_string());
        assert_eq!(err.to_string(), "Model error: dimension mismatch");

        let err = DomainAdaptationError::InvalidParameter("negative learning rate".to_string());
        assert_eq!(err.to_string(), "Invalid parameter: negative learning rate");
    }

    #[test]
    fn test_result_type() {
        let ok_result: Result<i32> = Ok(42);
        assert!(ok_result.is_ok());

        let err_result: Result<i32> = Err(DomainAdaptationError::DataError("missing".to_string()));
        assert!(err_result.is_err());
    }
}
