//! # Trading Signal Generation
//!
//! Converts domain-adapted model predictions into actionable trading signals.
//! The signal generator uses configurable thresholds to translate continuous
//! model outputs into discrete Buy/Sell/Hold decisions.
//!
//! ## Signal Logic
//!
//! ```text
//! prediction > buy_threshold  -> Buy
//! prediction < sell_threshold -> Sell
//! otherwise                   -> Hold
//! ```
//!
//! The thresholds create a "dead zone" around 0.5 (the decision boundary for
//! binary classification) to reduce whipsaw trading in uncertain conditions.

use crate::model::network::DomainAdaptationModel;
use crate::Result;

/// Discrete trading signal produced by the signal generator.
///
/// Represents the recommended action at a given point in time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TradingSignal {
    /// Bullish signal: open or increase a long position
    Buy,
    /// Bearish signal: open or increase a short position, or close longs
    Sell,
    /// Neutral signal: maintain current position
    Hold,
}

impl std::fmt::Display for TradingSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TradingSignal::Buy => write!(f, "BUY"),
            TradingSignal::Sell => write!(f, "SELL"),
            TradingSignal::Hold => write!(f, "HOLD"),
        }
    }
}

/// Generates trading signals from a domain-adapted model.
///
/// The signal generator wraps a [`DomainAdaptationModel`] and applies
/// configurable thresholds to convert probabilistic predictions into
/// discrete trading decisions.
///
/// # Example
///
/// ```rust
/// use domain_adaptation_trading::model::network::DomainAdaptationModel;
/// use domain_adaptation_trading::trading::signals::SignalGenerator;
///
/// let model = DomainAdaptationModel::new(6, 32, 1);
/// let signal_gen = SignalGenerator::new(model, 0.6, 0.4);
///
/// let features = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];
/// let signal = signal_gen.generate_signal(&features).unwrap();
/// ```
pub struct SignalGenerator {
    /// The underlying domain adaptation model
    model: DomainAdaptationModel,
    /// Threshold above which a Buy signal is generated (typical: 0.55-0.7)
    buy_threshold: f64,
    /// Threshold below which a Sell signal is generated (typical: 0.3-0.45)
    sell_threshold: f64,
}

impl SignalGenerator {
    /// Creates a new signal generator.
    ///
    /// # Arguments
    ///
    /// * `model` - Trained domain adaptation model
    /// * `buy_threshold` - Prediction threshold for Buy signals (should be > 0.5)
    /// * `sell_threshold` - Prediction threshold for Sell signals (should be < 0.5)
    ///
    /// # Panics
    ///
    /// Panics if `sell_threshold >= buy_threshold`.
    pub fn new(model: DomainAdaptationModel, buy_threshold: f64, sell_threshold: f64) -> Self {
        assert!(
            sell_threshold < buy_threshold,
            "sell_threshold ({}) must be less than buy_threshold ({})",
            sell_threshold,
            buy_threshold
        );
        Self {
            model,
            buy_threshold,
            sell_threshold,
        }
    }

    /// Creates a signal generator with default thresholds (0.6 / 0.4).
    ///
    /// # Arguments
    ///
    /// * `model` - Trained domain adaptation model
    pub fn with_defaults(model: DomainAdaptationModel) -> Self {
        Self::new(model, 0.6, 0.4)
    }

    /// Generates a trading signal from input features.
    ///
    /// The model's sigmoid output is compared against the buy and sell
    /// thresholds to produce a discrete signal.
    ///
    /// # Arguments
    ///
    /// * `features` - Normalized feature vector from the feature generator
    ///
    /// # Returns
    ///
    /// A [`TradingSignal`] indicating the recommended action.
    pub fn generate_signal(&self, features: &[f64]) -> Result<TradingSignal> {
        let prediction = self.model.predict_labels(features)?;
        let prob = prediction[0];

        Ok(self.threshold_signal(prob))
    }

    /// Generates signals for a batch of feature vectors.
    ///
    /// # Arguments
    ///
    /// * `features_batch` - Slice of feature vectors
    ///
    /// # Returns
    ///
    /// A vector of trading signals, one per input.
    pub fn generate_signals_batch(&self, features_batch: &[Vec<f64>]) -> Result<Vec<TradingSignal>> {
        features_batch
            .iter()
            .map(|f| self.generate_signal(f))
            .collect()
    }

    /// Returns the raw model prediction (probability) without thresholding.
    ///
    /// Useful for analyzing model confidence or implementing custom signal logic.
    ///
    /// # Arguments
    ///
    /// * `features` - Normalized feature vector
    ///
    /// # Returns
    ///
    /// The model's sigmoid output in [0, 1].
    pub fn prediction_confidence(&self, features: &[f64]) -> Result<f64> {
        let prediction = self.model.predict_labels(features)?;
        Ok(prediction[0])
    }

    /// Applies threshold logic to convert a probability to a signal.
    fn threshold_signal(&self, probability: f64) -> TradingSignal {
        if probability > self.buy_threshold {
            TradingSignal::Buy
        } else if probability < self.sell_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// Returns a reference to the underlying model.
    pub fn model(&self) -> &DomainAdaptationModel {
        &self.model
    }

    /// Returns the current buy threshold.
    pub fn buy_threshold(&self) -> f64 {
        self.buy_threshold
    }

    /// Returns the current sell threshold.
    pub fn sell_threshold(&self) -> f64 {
        self.sell_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_display() {
        assert_eq!(format!("{}", TradingSignal::Buy), "BUY");
        assert_eq!(format!("{}", TradingSignal::Sell), "SELL");
        assert_eq!(format!("{}", TradingSignal::Hold), "HOLD");
    }

    #[test]
    fn test_signal_equality() {
        assert_eq!(TradingSignal::Buy, TradingSignal::Buy);
        assert_ne!(TradingSignal::Buy, TradingSignal::Sell);
    }

    #[test]
    fn test_signal_generator_creation() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let gen = SignalGenerator::new(model, 0.6, 0.4);
        assert!((gen.buy_threshold() - 0.6).abs() < 1e-10);
        assert!((gen.sell_threshold() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_signal_generator_defaults() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let gen = SignalGenerator::with_defaults(model);
        assert!((gen.buy_threshold() - 0.6).abs() < 1e-10);
        assert!((gen.sell_threshold() - 0.4).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "sell_threshold")]
    fn test_invalid_thresholds() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let _gen = SignalGenerator::new(model, 0.3, 0.7); // sell > buy -> panic
    }

    #[test]
    fn test_threshold_logic() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let gen = SignalGenerator::new(model, 0.6, 0.4);

        assert_eq!(gen.threshold_signal(0.8), TradingSignal::Buy);
        assert_eq!(gen.threshold_signal(0.2), TradingSignal::Sell);
        assert_eq!(gen.threshold_signal(0.5), TradingSignal::Hold);
    }

    #[test]
    fn test_generate_signal() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let gen = SignalGenerator::new(model, 0.6, 0.4);

        let features = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];
        let signal = gen.generate_signal(&features).unwrap();

        // Signal should be one of the three variants
        matches!(signal, TradingSignal::Buy | TradingSignal::Sell | TradingSignal::Hold);
    }

    #[test]
    fn test_batch_signals() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let gen = SignalGenerator::new(model, 0.6, 0.4);

        let batch = vec![
            vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1],
            vec![-0.1, 0.2, -0.3, 0.0, -0.5, 0.1],
        ];

        let signals = gen.generate_signals_batch(&batch).unwrap();
        assert_eq!(signals.len(), 2);
    }

    #[test]
    fn test_prediction_confidence() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let gen = SignalGenerator::new(model, 0.6, 0.4);

        let features = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];
        let confidence = gen.prediction_confidence(&features).unwrap();

        assert!((0.0..=1.0).contains(&confidence), "Confidence should be in [0, 1]");
    }
}
