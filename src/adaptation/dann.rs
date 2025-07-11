//! # DANN (Domain-Adversarial Neural Network) Implementation
//!
//! Implements the Domain-Adversarial Neural Network training procedure from
//! [Ganin et al., 2016](https://arxiv.org/abs/1505.07818), adapted for
//! financial domain adaptation.
//!
//! ## Key Concept: Gradient Reversal
//!
//! The gradient reversal layer (GRL) is the core innovation of DANN. During the
//! forward pass, it acts as an identity function. During backpropagation, it
//! multiplies the gradient by `-lambda`, effectively reversing the gradient
//! direction. This forces the feature extractor to produce features that the
//! domain classifier *cannot* use to distinguish domains, while still being
//! useful for the label predictor.
//!
//! ## Training Loss
//!
//! ```text
//! total_loss = label_loss + lambda * domain_loss
//! ```
//!
//! Where:
//! - `label_loss`: Binary cross-entropy for predicting price direction
//! - `domain_loss`: Binary cross-entropy for domain classification (reversed gradient)
//! - `lambda`: Trade-off parameter controlling adaptation strength
//!
//! ## Usage
//!
//! ```rust
//! use domain_adaptation_trading::adaptation::dann::DANNTrainer;
//! use domain_adaptation_trading::model::network::DomainAdaptationModel;
//!
//! let model = DomainAdaptationModel::new(6, 32, 1);
//! let trainer = DANNTrainer::new(model, 0.001, 0.1);
//! ```

use crate::model::network::DomainAdaptationModel;
use crate::{DomainAdaptationError, Result};
use tracing::{info, debug};

/// Trainer for Domain-Adversarial Neural Networks.
///
/// Orchestrates the adversarial training process between the label predictor
/// and domain classifier, with gradient reversal applied to the feature extractor.
pub struct DANNTrainer {
    /// The domain adaptation model being trained
    pub model: DomainAdaptationModel,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Trade-off parameter for domain adaptation loss.
    /// Higher values encourage more domain-invariant features at the potential
    /// cost of task performance.
    pub lambda_domain: f64,
}

/// Training metrics collected during DANN training.
#[derive(Debug, Clone)]
pub struct DANNMetrics {
    /// Average label prediction loss over the epoch
    pub avg_label_loss: f64,
    /// Average domain classification loss over the epoch
    pub avg_domain_loss: f64,
    /// Combined total loss
    pub avg_total_loss: f64,
    /// Number of training steps completed
    pub steps: usize,
}

impl std::fmt::Display for DANNMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Steps: {} | Label Loss: {:.4} | Domain Loss: {:.4} | Total: {:.4}",
            self.steps, self.avg_label_loss, self.avg_domain_loss, self.avg_total_loss
        )
    }
}

impl DANNTrainer {
    /// Creates a new DANN trainer.
    ///
    /// # Arguments
    ///
    /// * `model` - The domain adaptation model to train
    /// * `learning_rate` - Step size for gradient descent (typical: 0.001)
    /// * `lambda_domain` - Domain adaptation trade-off parameter (typical: 0.1-1.0)
    ///
    /// # Example
    ///
    /// ```rust
    /// use domain_adaptation_trading::adaptation::dann::DANNTrainer;
    /// use domain_adaptation_trading::model::network::DomainAdaptationModel;
    ///
    /// let model = DomainAdaptationModel::new(6, 32, 1);
    /// let trainer = DANNTrainer::new(model, 0.001, 0.1);
    /// ```
    pub fn new(model: DomainAdaptationModel, learning_rate: f64, lambda_domain: f64) -> Self {
        Self {
            model,
            learning_rate,
            lambda_domain,
        }
    }

    /// Performs a single DANN training step.
    ///
    /// Processes one source sample (with label) and one target sample (without label)
    /// through the model, computing both the label prediction loss and domain
    /// classification loss. The gradient reversal is applied internally during
    /// the feature extractor update.
    ///
    /// # Arguments
    ///
    /// * `source_features` - Feature vector from the source domain
    /// * `source_label` - Ground truth label for the source sample
    /// * `target_features` - Feature vector from the target domain (no label needed)
    ///
    /// # Returns
    ///
    /// A tuple `(label_loss, domain_loss)` for monitoring training progress.
    pub fn train_step(
        &mut self,
        source_features: &[f64],
        source_label: &[f64],
        target_features: &[f64],
    ) -> Result<(f64, f64)> {
        // Step 1: Train on source data (has labels + domain label 0)
        let (label_loss, source_domain_loss) = self.model.train_step_dann(
            source_features,
            Some(source_label),
            0.0, // Source domain = 0
            self.learning_rate,
            self.lambda_domain,
        )?;

        // Step 2: Train on target data (no labels + domain label 1)
        let (_, target_domain_loss) = self.model.train_step_dann(
            target_features,
            None,   // No labels for target domain
            1.0,    // Target domain = 1
            self.learning_rate,
            self.lambda_domain,
        )?;

        let domain_loss = (source_domain_loss + target_domain_loss) / 2.0;

        debug!(
            label_loss = label_loss,
            domain_loss = domain_loss,
            "DANN train step"
        );

        Ok((label_loss, domain_loss))
    }

    /// Trains the DANN model for a specified number of epochs.
    ///
    /// Iterates over paired source and target samples, performing adversarial
    /// training at each step. The lambda parameter can optionally be scheduled
    /// to increase over training (curriculum-style).
    ///
    /// # Arguments
    ///
    /// * `source_features` - Feature matrix for the source domain (one vector per sample)
    /// * `source_labels` - Label vector for the source domain
    /// * `target_features` - Feature matrix for the target domain
    /// * `epochs` - Number of complete passes through the data
    ///
    /// # Returns
    ///
    /// Training metrics for the final epoch.
    ///
    /// # Errors
    ///
    /// Returns [`DomainAdaptationError::DataError`] if the source features
    /// and labels have mismatched lengths, or if either dataset is empty.
    pub fn train(
        &mut self,
        source_features: &[Vec<f64>],
        source_labels: &[f64],
        target_features: &[Vec<f64>],
        epochs: usize,
    ) -> Result<DANNMetrics> {
        if source_features.is_empty() || target_features.is_empty() {
            return Err(DomainAdaptationError::DataError(
                "Source and target features must not be empty".to_string(),
            ));
        }

        if source_features.len() != source_labels.len() {
            return Err(DomainAdaptationError::DataError(format!(
                "Source features ({}) and labels ({}) length mismatch",
                source_features.len(),
                source_labels.len()
            )));
        }

        let n_source = source_features.len();
        let n_target = target_features.len();

        let mut final_metrics = DANNMetrics {
            avg_label_loss: 0.0,
            avg_domain_loss: 0.0,
            avg_total_loss: 0.0,
            steps: 0,
        };

        for epoch in 0..epochs {
            let mut epoch_label_loss = 0.0;
            let mut epoch_domain_loss = 0.0;
            let n_steps = n_source.max(n_target);

            // Optional: schedule lambda to increase over training
            // This implements the schedule from the DANN paper:
            // lambda = 2 / (1 + exp(-10 * p)) - 1, where p is progress
            let progress = epoch as f64 / epochs.max(1) as f64;
            let scheduled_lambda =
                self.lambda_domain * (2.0 / (1.0 + (-10.0 * progress).exp()) - 1.0);
            let original_lambda = self.lambda_domain;
            self.lambda_domain = scheduled_lambda.max(0.01); // Ensure minimum lambda

            for step in 0..n_steps {
                let source_idx = step % n_source;
                let target_idx = step % n_target;

                let source_label_vec = vec![source_labels[source_idx]];

                let (ll, dl) = self.train_step(
                    &source_features[source_idx],
                    &source_label_vec,
                    &target_features[target_idx],
                )?;

                epoch_label_loss += ll;
                epoch_domain_loss += dl;
            }

            self.lambda_domain = original_lambda;

            let avg_ll = epoch_label_loss / n_steps as f64;
            let avg_dl = epoch_domain_loss / n_steps as f64;

            final_metrics = DANNMetrics {
                avg_label_loss: avg_ll,
                avg_domain_loss: avg_dl,
                avg_total_loss: avg_ll + self.lambda_domain * avg_dl,
                steps: n_steps,
            };

            if epoch % 10 == 0 || epoch == epochs - 1 {
                info!(
                    epoch = epoch,
                    label_loss = format!("{:.4}", avg_ll),
                    domain_loss = format!("{:.4}", avg_dl),
                    lambda = format!("{:.4}", scheduled_lambda),
                    "DANN training progress"
                );
            }
        }

        Ok(final_metrics)
    }

    /// Evaluates the domain adaptation quality.
    ///
    /// Computes several metrics to assess how well the model has adapted:
    ///
    /// 1. **Source accuracy**: Label prediction accuracy on source domain
    /// 2. **Domain confusion**: How close domain predictions are to 0.5
    ///    (perfect confusion = successful adaptation)
    ///
    /// # Arguments
    ///
    /// * `source_features` - Source domain feature vectors
    /// * `source_labels` - Source domain ground truth labels
    /// * `target_features` - Target domain feature vectors
    ///
    /// # Returns
    ///
    /// A tuple `(source_accuracy, domain_confusion_score)` where:
    /// - `source_accuracy` is in [0, 1] (higher is better)
    /// - `domain_confusion_score` is in [0, 1] (closer to 1 means better adaptation)
    pub fn evaluate(
        &self,
        source_features: &[Vec<f64>],
        source_labels: &[f64],
        target_features: &[Vec<f64>],
    ) -> Result<(f64, f64)> {
        // Source accuracy
        let mut correct = 0;
        for (features, &label) in source_features.iter().zip(source_labels.iter()) {
            let pred = self.model.predict_labels(features)?;
            let pred_class = if pred[0] > 0.5 { 1.0 } else { 0.0 };
            if (pred_class - label).abs() < 0.5 {
                correct += 1;
            }
        }
        let source_accuracy = correct as f64 / source_features.len() as f64;

        // Domain confusion: measure how confused the domain classifier is
        // Perfect adaptation: all domain predictions near 0.5
        let mut domain_confusion = 0.0;
        let total = source_features.len() + target_features.len();

        for features in source_features.iter().chain(target_features.iter()) {
            let domain_pred = self.model.classify_domain(features)?;
            // Confusion = 1 - |pred - 0.5| * 2 (1.0 when pred=0.5, 0.0 when pred=0 or 1)
            let confusion = 1.0 - (domain_pred[0] - 0.5).abs() * 2.0;
            domain_confusion += confusion;
        }
        domain_confusion /= total as f64;

        Ok((source_accuracy, domain_confusion))
    }

    /// Returns a reference to the underlying model.
    pub fn model(&self) -> &DomainAdaptationModel {
        &self.model
    }

    /// Consumes the trainer and returns the trained model.
    pub fn into_model(self) -> DomainAdaptationModel {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_data() -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
        // Simple source data
        let source_features = vec![
            vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1],
            vec![0.2, 0.1, -0.1, 0.3, -0.2, 0.4],
            vec![-0.1, 0.3, 0.2, -0.2, 0.1, 0.0],
            vec![0.3, -0.1, 0.0, 0.1, 0.3, -0.2],
        ];
        let source_labels = vec![1.0, 0.0, 1.0, 0.0];

        // Target data (slightly shifted distribution)
        let target_features = vec![
            vec![0.2, -0.1, 0.4, 0.1, 0.6, 0.0],
            vec![0.3, 0.2, 0.0, 0.4, -0.1, 0.5],
            vec![0.0, 0.4, 0.3, -0.1, 0.2, 0.1],
            vec![0.4, 0.0, 0.1, 0.2, 0.4, -0.1],
        ];

        (source_features, source_labels, target_features)
    }

    #[test]
    fn test_dann_trainer_creation() {
        let model = DomainAdaptationModel::new(6, 32, 1);
        let trainer = DANNTrainer::new(model, 0.001, 0.1);
        assert!((trainer.learning_rate - 0.001).abs() < 1e-10);
        assert!((trainer.lambda_domain - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_train_step() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let mut trainer = DANNTrainer::new(model, 0.001, 0.1);

        let source = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];
        let label = vec![1.0];
        let target = vec![0.2, -0.1, 0.4, 0.1, 0.6, 0.0];

        let (ll, dl) = trainer.train_step(&source, &label, &target).unwrap();
        assert!(ll >= 0.0, "Label loss should be non-negative");
        assert!(dl >= 0.0, "Domain loss should be non-negative");
    }

    #[test]
    fn test_train_epochs() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let mut trainer = DANNTrainer::new(model, 0.01, 0.1);
        let (source_features, source_labels, target_features) = create_test_data();

        let metrics = trainer
            .train(&source_features, &source_labels, &target_features, 5)
            .unwrap();

        assert!(metrics.avg_label_loss >= 0.0);
        assert!(metrics.avg_domain_loss >= 0.0);
        assert!(metrics.steps > 0);
    }

    #[test]
    fn test_evaluate() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let trainer = DANNTrainer::new(model, 0.001, 0.1);
        let (source_features, source_labels, target_features) = create_test_data();

        let (accuracy, confusion) = trainer
            .evaluate(&source_features, &source_labels, &target_features)
            .unwrap();

        assert!((0.0..=1.0).contains(&accuracy));
        assert!((0.0..=1.0).contains(&confusion));
    }

    #[test]
    fn test_empty_data_error() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let mut trainer = DANNTrainer::new(model, 0.001, 0.1);

        let result = trainer.train(&[], &[], &[vec![0.1; 6]], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_labels_error() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let mut trainer = DANNTrainer::new(model, 0.001, 0.1);

        let source = vec![vec![0.1; 6], vec![0.2; 6]];
        let labels = vec![1.0]; // Mismatch: 2 features, 1 label
        let target = vec![vec![0.3; 6]];

        let result = trainer.train(&source, &labels, &target, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_into_model() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let trainer = DANNTrainer::new(model, 0.001, 0.1);
        let model = trainer.into_model();
        assert_eq!(model.input_size, 6);
    }

    #[test]
    fn test_metrics_display() {
        let metrics = DANNMetrics {
            avg_label_loss: 0.5432,
            avg_domain_loss: 0.6789,
            avg_total_loss: 0.6111,
            steps: 100,
        };
        let display = format!("{}", metrics);
        assert!(display.contains("0.5432"));
        assert!(display.contains("100"));
    }
}
