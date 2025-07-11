//! # CORAL (Correlation Alignment) Adaptation
//!
//! Implements the CORAL (CORrelation ALignment) domain adaptation method from
//! [Sun et al., 2016](https://arxiv.org/abs/1607.01719).
//!
//! CORAL aligns the second-order statistics (covariance matrices) of source and
//! target feature distributions. This is based on the observation that domain shift
//! often manifests as changes in feature correlations rather than means.
//!
//! ## Mathematical Formulation
//!
//! The CORAL loss is defined as the squared Frobenius norm of the difference
//! between source and target covariance matrices:
//!
//! ```text
//! L_CORAL = (1 / 4d^2) * ||C_S - C_T||_F^2
//! ```
//!
//! Where:
//! - `C_S`: Covariance matrix of source features
//! - `C_T`: Covariance matrix of target features
//! - `d`: Feature dimension
//! - `||.||_F`: Frobenius norm
//!
//! ## Advantages
//!
//! - No adversarial training required (simpler and more stable)
//! - Computationally efficient (only requires covariance computation)
//! - Works well when correlation structure is the primary domain difference
//! - Particularly useful for financial data where asset correlations vary across regimes

use crate::model::network::DomainAdaptationModel;
use crate::{DomainAdaptationError, Result};
use tracing::{info, debug};

/// Domain adaptation using CORAL (Correlation Alignment).
///
/// Trains the model to produce features whose covariance structure matches
/// between source and target domains.
pub struct CORALAdapter {
    /// The domain adaptation model being trained
    pub model: DomainAdaptationModel,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Weight for the CORAL loss term relative to the prediction loss
    pub coral_weight: f64,
}

/// Training metrics for CORAL adaptation.
#[derive(Debug, Clone)]
pub struct CORALMetrics {
    /// Average prediction loss on source domain
    pub avg_label_loss: f64,
    /// CORAL loss (covariance alignment measure)
    pub coral_loss: f64,
    /// Combined total loss
    pub avg_total_loss: f64,
    /// Number of epochs completed
    pub epochs_completed: usize,
}

impl std::fmt::Display for CORALMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Epochs: {} | Label Loss: {:.4} | CORAL: {:.6} | Total: {:.4}",
            self.epochs_completed, self.avg_label_loss, self.coral_loss, self.avg_total_loss
        )
    }
}

impl CORALAdapter {
    /// Creates a new CORAL adapter.
    ///
    /// # Arguments
    ///
    /// * `model` - The domain adaptation model to train
    /// * `learning_rate` - Step size for gradient descent
    /// * `coral_weight` - Weight for the CORAL loss term (typical: 0.01-1.0).
    ///   Higher values enforce stricter covariance alignment.
    ///
    /// # Example
    ///
    /// ```rust
    /// use domain_adaptation_trading::adaptation::coral::CORALAdapter;
    /// use domain_adaptation_trading::model::network::DomainAdaptationModel;
    ///
    /// let model = DomainAdaptationModel::new(6, 32, 1);
    /// let adapter = CORALAdapter::new(model, 0.001, 0.5);
    /// ```
    pub fn new(model: DomainAdaptationModel, learning_rate: f64, coral_weight: f64) -> Self {
        Self {
            model,
            learning_rate,
            coral_weight,
        }
    }

    /// Computes the covariance matrix of a set of feature vectors.
    ///
    /// Uses the unbiased estimator (dividing by n-1).
    ///
    /// # Arguments
    ///
    /// * `features` - Slice of feature vectors, all the same dimension
    ///
    /// # Returns
    ///
    /// The covariance matrix as a 2D vector (row-major).
    ///
    /// # Errors
    ///
    /// Returns an error if the features slice is empty or has fewer than 2 samples.
    pub fn compute_covariance(features: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        if features.len() < 2 {
            return Err(DomainAdaptationError::DataError(
                "Need at least 2 samples to compute covariance".to_string(),
            ));
        }

        let n = features.len() as f64;
        let d = features[0].len();

        // Compute mean vector
        let mut mean = vec![0.0; d];
        for f in features {
            for (j, &val) in f.iter().enumerate() {
                mean[j] += val / n;
            }
        }

        // Compute covariance matrix: C[i][j] = (1/(n-1)) * sum((x_i - mean_i)(x_j - mean_j))
        let mut cov = vec![vec![0.0; d]; d];
        for f in features {
            for i in 0..d {
                let diff_i = f[i] - mean[i];
                for j in 0..d {
                    let diff_j = f[j] - mean[j];
                    cov[i][j] += diff_i * diff_j;
                }
            }
        }

        // Normalize by (n-1) for unbiased estimator
        let norm = (n - 1.0).max(1.0);
        for row in &mut cov {
            for val in row.iter_mut() {
                *val /= norm;
            }
        }

        Ok(cov)
    }

    /// Computes the CORAL loss between source and target covariance matrices.
    ///
    /// CORAL loss = (1 / 4d^2) * ||C_S - C_T||_F^2
    ///
    /// Where ||.||_F is the Frobenius norm (square root of sum of squared elements).
    ///
    /// # Arguments
    ///
    /// * `source_cov` - Source domain covariance matrix
    /// * `target_cov` - Target domain covariance matrix
    ///
    /// # Returns
    ///
    /// The CORAL loss value. Lower values indicate better alignment.
    pub fn coral_loss(source_cov: &[Vec<f64>], target_cov: &[Vec<f64>]) -> Result<f64> {
        if source_cov.is_empty() || target_cov.is_empty() {
            return Err(DomainAdaptationError::DataError(
                "Covariance matrices must not be empty".to_string(),
            ));
        }

        let d = source_cov.len();
        if d != target_cov.len() {
            return Err(DomainAdaptationError::DataError(format!(
                "Covariance matrix dimension mismatch: {} vs {}",
                d,
                target_cov.len()
            )));
        }

        // Compute squared Frobenius norm of difference
        let mut frobenius_sq = 0.0;
        for i in 0..d {
            for j in 0..d {
                let diff = source_cov[i][j] - target_cov[i][j];
                frobenius_sq += diff * diff;
            }
        }

        // Normalize by 4d^2 as per CORAL paper
        let d_f = d as f64;
        Ok(frobenius_sq / (4.0 * d_f * d_f))
    }

    /// Computes the Frobenius norm of a matrix.
    ///
    /// ||A||_F = sqrt(sum(a_ij^2))
    pub fn frobenius_norm(matrix: &[Vec<f64>]) -> f64 {
        let sum_sq: f64 = matrix
            .iter()
            .flat_map(|row| row.iter())
            .map(|&x| x * x)
            .sum();
        sum_sq.sqrt()
    }

    /// Trains the model using CORAL domain adaptation.
    ///
    /// The training objective combines:
    /// 1. Prediction loss on labeled source data
    /// 2. CORAL loss to align covariance structures
    ///
    /// # Arguments
    ///
    /// * `source_features` - Source domain input features
    /// * `source_labels` - Source domain labels
    /// * `target_features` - Target domain input features (unlabeled)
    /// * `epochs` - Number of training epochs
    ///
    /// # Returns
    ///
    /// Training metrics from the final epoch.
    pub fn adapt(
        &mut self,
        source_features: &[Vec<f64>],
        source_labels: &[f64],
        target_features: &[Vec<f64>],
        epochs: usize,
    ) -> Result<CORALMetrics> {
        if source_features.is_empty() || target_features.is_empty() {
            return Err(DomainAdaptationError::DataError(
                "Source and target features must not be empty".to_string(),
            ));
        }

        if source_features.len() != source_labels.len() {
            return Err(DomainAdaptationError::DataError(
                "Source features and labels length mismatch".to_string(),
            ));
        }

        let mut final_metrics = CORALMetrics {
            avg_label_loss: 0.0,
            coral_loss: 0.0,
            avg_total_loss: 0.0,
            epochs_completed: 0,
        };

        for epoch in 0..epochs {
            let mut epoch_label_loss = 0.0;

            // Step 1: Train on source data with labels
            for (features, &label) in source_features.iter().zip(source_labels.iter()) {
                let label_vec = vec![label];
                let (ll, _) = self.model.train_step_dann(
                    features,
                    Some(&label_vec),
                    0.0,
                    self.learning_rate,
                    0.0, // No DANN domain loss; using CORAL instead
                )?;
                epoch_label_loss += ll;
            }

            // Step 2: Compute CORAL loss on hidden features
            let source_hidden: Vec<Vec<f64>> = source_features
                .iter()
                .map(|f| self.model.forward_features(f))
                .collect::<Result<Vec<_>>>()?;

            let target_hidden: Vec<Vec<f64>> = target_features
                .iter()
                .map(|f| self.model.forward_features(f))
                .collect::<Result<Vec<_>>>()?;

            let source_cov = Self::compute_covariance(&source_hidden)?;
            let target_cov = Self::compute_covariance(&target_hidden)?;
            let c_loss = Self::coral_loss(&source_cov, &target_cov)?;

            // Step 3: Apply CORAL gradient to feature extractor
            self.apply_coral_gradient(
                source_features,
                target_features,
                &source_cov,
                &target_cov,
            )?;

            let avg_ll = epoch_label_loss / source_features.len() as f64;
            let total = avg_ll + self.coral_weight * c_loss;

            final_metrics = CORALMetrics {
                avg_label_loss: avg_ll,
                coral_loss: c_loss,
                avg_total_loss: total,
                epochs_completed: epoch + 1,
            };

            if epoch % 10 == 0 || epoch == epochs - 1 {
                info!(
                    epoch = epoch,
                    label_loss = format!("{:.4}", avg_ll),
                    coral_loss = format!("{:.6}", c_loss),
                    total_loss = format!("{:.4}", total),
                    "CORAL adaptation progress"
                );
            }

            debug!(epoch = epoch, "CORAL epoch completed");
        }

        Ok(final_metrics)
    }

    /// Applies an approximate CORAL gradient to the feature extractor.
    ///
    /// The gradient is derived from the difference between source and target
    /// covariance matrices. For each sample, it pushes the feature extractor
    /// to produce representations with more aligned covariance structure.
    fn apply_coral_gradient(
        &mut self,
        source_features: &[Vec<f64>],
        target_features: &[Vec<f64>],
        source_cov: &[Vec<f64>],
        target_cov: &[Vec<f64>],
    ) -> Result<()> {
        let d = source_cov.len();

        // Compute covariance difference matrix
        let mut cov_diff = vec![vec![0.0; d]; d];
        for i in 0..d {
            for j in 0..d {
                cov_diff[i][j] = source_cov[i][j] - target_cov[i][j];
            }
        }

        let diff_norm = Self::frobenius_norm(&cov_diff);
        if diff_norm < 1e-10 {
            return Ok(()); // Covariances already aligned
        }

        // Scale the gradient
        let scale = self.learning_rate * self.coral_weight / (diff_norm + 1e-8);

        // Apply gradient to reduce covariance difference
        // For source features: push toward target covariance
        let n_update = source_features.len().min(target_features.len());
        for i in 0..n_update {
            // Compute source hidden features
            let source_hidden = self.model.forward_features(&source_features[i])?;

            // Pseudo-gradient: cov_diff * h_source (direction to reduce mismatch)
            let mut grad = vec![0.0; d];
            for row in 0..d {
                for col in 0..d {
                    grad[row] += cov_diff[row][col] * source_hidden[col];
                }
                grad[row] *= scale;
            }

            self.model
                .feature_extractor
                .backward(&source_features[i], &grad, self.learning_rate)?;
        }

        Ok(())
    }

    /// Evaluates the current CORAL alignment between domains.
    ///
    /// # Arguments
    ///
    /// * `source_features` - Raw source domain features
    /// * `target_features` - Raw target domain features
    ///
    /// # Returns
    ///
    /// The CORAL loss in the current feature space.
    pub fn evaluate_coral(
        &self,
        source_features: &[Vec<f64>],
        target_features: &[Vec<f64>],
    ) -> Result<f64> {
        let source_hidden: Vec<Vec<f64>> = source_features
            .iter()
            .map(|f| self.model.forward_features(f))
            .collect::<Result<Vec<_>>>()?;

        let target_hidden: Vec<Vec<f64>> = target_features
            .iter()
            .map(|f| self.model.forward_features(f))
            .collect::<Result<Vec<_>>>()?;

        let source_cov = Self::compute_covariance(&source_hidden)?;
        let target_cov = Self::compute_covariance(&target_hidden)?;

        Self::coral_loss(&source_cov, &target_cov)
    }

    /// Returns a reference to the underlying model.
    pub fn model(&self) -> &DomainAdaptationModel {
        &self.model
    }

    /// Consumes the adapter and returns the trained model.
    pub fn into_model(self) -> DomainAdaptationModel {
        self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_identity() {
        // Features that should produce approximately identity covariance
        let features = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![-1.0, 0.0, 0.0],
            vec![0.0, -1.0, 0.0],
            vec![0.0, 0.0, -1.0],
        ];

        let cov = CORALAdapter::compute_covariance(&features).unwrap();
        assert_eq!(cov.len(), 3);
        assert_eq!(cov[0].len(), 3);

        // Diagonal elements should be positive
        for i in 0..3 {
            assert!(cov[i][i] > 0.0);
        }
    }

    #[test]
    fn test_covariance_symmetric() {
        let features = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        let cov = CORALAdapter::compute_covariance(&features).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (cov[i][j] - cov[j][i]).abs() < 1e-10,
                    "Covariance should be symmetric"
                );
            }
        }
    }

    #[test]
    fn test_covariance_insufficient_samples() {
        let features = vec![vec![1.0, 2.0, 3.0]]; // Only 1 sample
        assert!(CORALAdapter::compute_covariance(&features).is_err());
    }

    #[test]
    fn test_coral_loss_identical() {
        let cov = vec![
            vec![1.0, 0.5],
            vec![0.5, 1.0],
        ];

        let loss = CORALAdapter::coral_loss(&cov, &cov).unwrap();
        assert!((loss - 0.0).abs() < 1e-10, "CORAL loss of identical matrices should be 0");
    }

    #[test]
    fn test_coral_loss_different() {
        let source_cov = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let target_cov = vec![
            vec![2.0, 0.5],
            vec![0.5, 2.0],
        ];

        let loss = CORALAdapter::coral_loss(&source_cov, &target_cov).unwrap();
        assert!(loss > 0.0, "CORAL loss of different matrices should be positive");
    }

    #[test]
    fn test_coral_loss_dimension_mismatch() {
        let source = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let target = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]];

        assert!(CORALAdapter::coral_loss(&source, &target).is_err());
    }

    #[test]
    fn test_frobenius_norm() {
        let matrix = vec![
            vec![3.0, 0.0],
            vec![0.0, 4.0],
        ];
        let norm = CORALAdapter::frobenius_norm(&matrix);
        assert!((norm - 5.0).abs() < 1e-10, "Frobenius norm of diag(3,4) should be 5");
    }

    #[test]
    fn test_coral_adapt() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let mut adapter = CORALAdapter::new(model, 0.01, 0.5);

        let source = vec![
            vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1],
            vec![0.2, 0.1, -0.1, 0.3, -0.2, 0.4],
            vec![-0.1, 0.3, 0.2, -0.2, 0.1, 0.0],
        ];
        let labels = vec![1.0, 0.0, 1.0];
        let target = vec![
            vec![0.3, 0.0, 0.5, 0.2, 0.7, 0.1],
            vec![0.4, 0.3, 0.1, 0.5, 0.0, 0.6],
            vec![0.1, 0.5, 0.4, 0.0, 0.3, 0.2],
        ];

        let metrics = adapter.adapt(&source, &labels, &target, 5).unwrap();
        assert!(metrics.avg_label_loss >= 0.0);
        assert!(metrics.coral_loss >= 0.0);
        assert_eq!(metrics.epochs_completed, 5);
    }

    #[test]
    fn test_evaluate_coral() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let adapter = CORALAdapter::new(model, 0.001, 0.5);

        let source = vec![vec![0.1; 6], vec![0.2; 6], vec![0.3; 6]];
        let target = vec![vec![0.5; 6], vec![0.6; 6], vec![0.7; 6]];

        let loss = adapter.evaluate_coral(&source, &target).unwrap();
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_coral_empty_data() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let mut adapter = CORALAdapter::new(model, 0.001, 0.5);

        let result = adapter.adapt(&[], &[], &[vec![0.1; 6]], 5);
        assert!(result.is_err());
    }
}
