//! # Maximum Mean Discrepancy (MMD) Adaptation
//!
//! Implements domain adaptation using Maximum Mean Discrepancy, a kernel-based
//! distance metric between probability distributions. MMD measures the distance
//! between the mean embeddings of two distributions in a Reproducing Kernel
//! Hilbert Space (RKHS).
//!
//! ## Mathematical Formulation
//!
//! Given source features `X_s` and target features `X_t`, the empirical MMD is:
//!
//! ```text
//! MMD^2(X_s, X_t) = 1/n_s^2 * sum_{i,j} k(x_s^i, x_s^j)
//!                  + 1/n_t^2 * sum_{i,j} k(x_t^i, x_t^j)
//!                  - 2/(n_s*n_t) * sum_{i,j} k(x_s^i, x_t^j)
//! ```
//!
//! Where `k` is a Gaussian (RBF) kernel: `k(x, y) = exp(-||x-y||^2 / (2*sigma^2))`
//!
//! ## Training Objective
//!
//! ```text
//! total_loss = label_loss + mmd_weight * MMD^2(features_source, features_target)
//! ```
//!
//! The model is trained to minimize both the prediction loss on labeled source data
//! and the distribution distance between source and target feature representations.

use crate::model::network::DomainAdaptationModel;
use crate::{DomainAdaptationError, Result};
use tracing::{info, debug};

/// Domain adaptation using Maximum Mean Discrepancy.
///
/// Trains the model to produce features that minimize the MMD distance
/// between source and target domains while maintaining good prediction
/// performance on the source domain.
pub struct MMDAdapter {
    /// Bandwidth parameter for the Gaussian kernel.
    /// Controls the sensitivity to distribution differences.
    /// Smaller values make the kernel more sensitive to local structure.
    pub kernel_bandwidth: f64,
    /// The domain adaptation model being trained
    pub model: DomainAdaptationModel,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Weight for the MMD regularization term
    pub mmd_weight: f64,
}

/// Training metrics for MMD adaptation.
#[derive(Debug, Clone)]
pub struct MMDMetrics {
    /// Average prediction loss on source domain
    pub avg_label_loss: f64,
    /// Average MMD distance between domains
    pub avg_mmd_distance: f64,
    /// Combined total loss
    pub avg_total_loss: f64,
    /// Number of training epochs completed
    pub epochs_completed: usize,
}

impl std::fmt::Display for MMDMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Epochs: {} | Label Loss: {:.4} | MMD: {:.6} | Total: {:.4}",
            self.epochs_completed, self.avg_label_loss, self.avg_mmd_distance, self.avg_total_loss
        )
    }
}

impl MMDAdapter {
    /// Creates a new MMD adapter.
    ///
    /// # Arguments
    ///
    /// * `model` - The domain adaptation model to train
    /// * `kernel_bandwidth` - Gaussian kernel bandwidth (sigma). Typical: 1.0.
    ///   Use the median heuristic: set to the median pairwise distance.
    /// * `mmd_weight` - Weight for the MMD penalty term (typical: 0.1-1.0)
    /// * `learning_rate` - Step size for gradient descent
    ///
    /// # Example
    ///
    /// ```rust
    /// use domain_adaptation_trading::adaptation::mmd::MMDAdapter;
    /// use domain_adaptation_trading::model::network::DomainAdaptationModel;
    ///
    /// let model = DomainAdaptationModel::new(6, 32, 1);
    /// let adapter = MMDAdapter::new(model, 1.0, 0.5, 0.001);
    /// ```
    pub fn new(
        model: DomainAdaptationModel,
        kernel_bandwidth: f64,
        mmd_weight: f64,
        learning_rate: f64,
    ) -> Self {
        Self {
            kernel_bandwidth,
            model,
            learning_rate,
            mmd_weight,
        }
    }

    /// Computes the Gaussian (RBF) kernel between two feature vectors.
    ///
    /// `k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))`
    ///
    /// # Arguments
    ///
    /// * `x` - First feature vector
    /// * `y` - Second feature vector
    /// * `bandwidth` - Kernel bandwidth parameter (sigma)
    ///
    /// # Returns
    ///
    /// Kernel value in (0, 1].
    pub fn gaussian_kernel(x: &[f64], y: &[f64], bandwidth: f64) -> f64 {
        let sq_dist: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum();
        (-sq_dist / (2.0 * bandwidth * bandwidth)).exp()
    }

    /// Computes the squared Maximum Mean Discrepancy between two sets of features.
    ///
    /// Uses the unbiased estimator of MMD^2 with a Gaussian kernel.
    ///
    /// # Arguments
    ///
    /// * `source_features` - Feature vectors from the source domain
    /// * `target_features` - Feature vectors from the target domain
    ///
    /// # Returns
    ///
    /// The squared MMD distance. Values close to 0 indicate similar distributions.
    ///
    /// # Errors
    ///
    /// Returns an error if either feature set is empty.
    pub fn compute_mmd(
        &self,
        source_features: &[Vec<f64>],
        target_features: &[Vec<f64>],
    ) -> Result<f64> {
        if source_features.is_empty() || target_features.is_empty() {
            return Err(DomainAdaptationError::DataError(
                "Feature sets must not be empty for MMD computation".to_string(),
            ));
        }

        let n_s = source_features.len() as f64;
        let n_t = target_features.len() as f64;

        // Compute source-source kernel sum
        let mut ss_sum = 0.0;
        for i in 0..source_features.len() {
            for j in 0..source_features.len() {
                if i != j {
                    ss_sum += Self::gaussian_kernel(
                        &source_features[i],
                        &source_features[j],
                        self.kernel_bandwidth,
                    );
                }
            }
        }
        let ss_term = ss_sum / (n_s * (n_s - 1.0).max(1.0));

        // Compute target-target kernel sum
        let mut tt_sum = 0.0;
        for i in 0..target_features.len() {
            for j in 0..target_features.len() {
                if i != j {
                    tt_sum += Self::gaussian_kernel(
                        &target_features[i],
                        &target_features[j],
                        self.kernel_bandwidth,
                    );
                }
            }
        }
        let tt_term = tt_sum / (n_t * (n_t - 1.0).max(1.0));

        // Compute source-target kernel sum
        let mut st_sum = 0.0;
        for s in source_features {
            for t in target_features {
                st_sum += Self::gaussian_kernel(s, t, self.kernel_bandwidth);
            }
        }
        let st_term = 2.0 * st_sum / (n_s * n_t);

        Ok((ss_term + tt_term - st_term).max(0.0))
    }

    /// Trains the model using MMD-based domain adaptation.
    ///
    /// Alternates between:
    /// 1. Minimizing prediction loss on labeled source data
    /// 2. Minimizing MMD distance between source and target feature representations
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
    ) -> Result<MMDMetrics> {
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

        let mut final_metrics = MMDMetrics {
            avg_label_loss: 0.0,
            avg_mmd_distance: 0.0,
            avg_total_loss: 0.0,
            epochs_completed: 0,
        };

        for epoch in 0..epochs {
            let mut epoch_label_loss = 0.0;

            // Step 1: Train on source data with labels
            for (i, (features, &label)) in source_features
                .iter()
                .zip(source_labels.iter())
                .enumerate()
            {
                let label_vec = vec![label];
                // Use DANN train step with zero lambda to train only on labels
                let (ll, _) = self.model.train_step_dann(
                    features,
                    Some(&label_vec),
                    0.0, // Source domain
                    self.learning_rate,
                    0.0, // No domain adaptation here; we use MMD instead
                )?;
                epoch_label_loss += ll;

                debug!(
                    epoch = epoch,
                    sample = i,
                    label_loss = format!("{:.4}", ll),
                    "MMD source training step"
                );
            }

            // Step 2: Compute MMD between extracted features
            let source_hidden: Vec<Vec<f64>> = source_features
                .iter()
                .map(|f| self.model.forward_features(f))
                .collect::<Result<Vec<_>>>()?;

            let target_hidden: Vec<Vec<f64>> = target_features
                .iter()
                .map(|f| self.model.forward_features(f))
                .collect::<Result<Vec<_>>>()?;

            let mmd_distance = self.compute_mmd(&source_hidden, &target_hidden)?;

            // Step 3: Apply MMD gradient to feature extractor
            // We approximate the MMD gradient by computing a small perturbation
            // and using it to nudge the feature extractor toward lower MMD
            self.apply_mmd_gradient(source_features, target_features, mmd_distance)?;

            let avg_ll = epoch_label_loss / source_features.len() as f64;
            let total = avg_ll + self.mmd_weight * mmd_distance;

            final_metrics = MMDMetrics {
                avg_label_loss: avg_ll,
                avg_mmd_distance: mmd_distance,
                avg_total_loss: total,
                epochs_completed: epoch + 1,
            };

            if epoch % 10 == 0 || epoch == epochs - 1 {
                info!(
                    epoch = epoch,
                    label_loss = format!("{:.4}", avg_ll),
                    mmd = format!("{:.6}", mmd_distance),
                    total_loss = format!("{:.4}", total),
                    "MMD adaptation progress"
                );
            }
        }

        Ok(final_metrics)
    }

    /// Applies an approximate MMD gradient to the feature extractor.
    ///
    /// Uses a numerical gradient approach: for each source and target sample,
    /// compute a pseudo-gradient that pushes the hidden representations closer
    /// together, scaled by the MMD weight.
    fn apply_mmd_gradient(
        &mut self,
        source_features: &[Vec<f64>],
        target_features: &[Vec<f64>],
        mmd_value: f64,
    ) -> Result<()> {
        // Scale the adaptation step by the MMD value
        let adaptation_lr = self.learning_rate * self.mmd_weight * mmd_value.sqrt().min(1.0);

        if adaptation_lr < 1e-10 {
            return Ok(()); // MMD already very small, skip
        }

        // Compute mean hidden features for source and target
        let source_hidden: Vec<Vec<f64>> = source_features
            .iter()
            .map(|f| self.model.forward_features(f))
            .collect::<Result<Vec<_>>>()?;

        let target_hidden: Vec<Vec<f64>> = target_features
            .iter()
            .map(|f| self.model.forward_features(f))
            .collect::<Result<Vec<_>>>()?;

        let hidden_size = self.model.hidden_size;

        // Compute mean difference between source and target hidden representations
        let mut mean_diff = vec![0.0; hidden_size];
        let n_s = source_hidden.len() as f64;
        let n_t = target_hidden.len() as f64;

        for s in &source_hidden {
            for (j, &val) in s.iter().enumerate() {
                mean_diff[j] += val / n_s;
            }
        }
        for t in &target_hidden {
            for (j, &val) in t.iter().enumerate() {
                mean_diff[j] -= val / n_t;
            }
        }

        // Apply gradient to push feature extractor toward reducing the mean difference
        // For source samples: gradient pushes features toward target mean
        let n_update = source_features.len().min(target_features.len());
        for i in 0..n_update {
            // Pseudo-gradient: direction that reduces the mean difference
            let grad: Vec<f64> = mean_diff.iter().map(|&d| d * adaptation_lr).collect();

            self.model
                .feature_extractor
                .backward(&source_features[i], &grad, adaptation_lr)?;
        }

        Ok(())
    }

    /// Computes the MMD distance using the current model's feature representations.
    ///
    /// This is useful for monitoring adaptation progress without training.
    ///
    /// # Arguments
    ///
    /// * `source_features` - Raw source domain features
    /// * `target_features` - Raw target domain features
    ///
    /// # Returns
    ///
    /// The MMD distance between the domains in feature space.
    pub fn evaluate_mmd(
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

        self.compute_mmd(&source_hidden, &target_hidden)
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
    fn test_gaussian_kernel_identical() {
        let x = vec![1.0, 2.0, 3.0];
        let k = MMDAdapter::gaussian_kernel(&x, &x, 1.0);
        assert!((k - 1.0).abs() < 1e-10, "Kernel of identical points should be 1.0");
    }

    #[test]
    fn test_gaussian_kernel_distant() {
        let x = vec![0.0, 0.0];
        let y = vec![10.0, 10.0];
        let k = MMDAdapter::gaussian_kernel(&x, &y, 1.0);
        assert!(k < 0.01, "Kernel of distant points should be near 0");
    }

    #[test]
    fn test_gaussian_kernel_bandwidth() {
        let x = vec![0.0, 0.0];
        let y = vec![1.0, 1.0];
        let k_narrow = MMDAdapter::gaussian_kernel(&x, &y, 0.1);
        let k_wide = MMDAdapter::gaussian_kernel(&x, &y, 10.0);
        assert!(k_wide > k_narrow, "Wider bandwidth should give higher kernel value");
    }

    #[test]
    fn test_mmd_same_distribution() {
        let model = DomainAdaptationModel::new(3, 8, 1);
        let adapter = MMDAdapter::new(model, 1.0, 0.5, 0.001);

        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
        ];

        let mmd = adapter.compute_mmd(&data, &data).unwrap();
        // MMD of identical sets should be 0 (unbiased estimator may differ slightly)
        assert!(mmd < 0.1, "MMD of identical sets should be near 0, got {}", mmd);
    }

    #[test]
    fn test_mmd_different_distributions() {
        let model = DomainAdaptationModel::new(3, 8, 1);
        let adapter = MMDAdapter::new(model, 1.0, 0.5, 0.001);

        let source = vec![
            vec![0.0, 0.0, 0.0],
            vec![0.1, 0.1, 0.1],
            vec![-0.1, -0.1, -0.1],
        ];

        let target = vec![
            vec![5.0, 5.0, 5.0],
            vec![5.1, 5.1, 5.1],
            vec![4.9, 4.9, 4.9],
        ];

        let mmd = adapter.compute_mmd(&source, &target).unwrap();
        assert!(mmd > 0.0, "MMD of different distributions should be positive");
    }

    #[test]
    fn test_mmd_adapt() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let mut adapter = MMDAdapter::new(model, 1.0, 0.5, 0.01);

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
        assert!(metrics.avg_mmd_distance >= 0.0);
        assert_eq!(metrics.epochs_completed, 5);
    }

    #[test]
    fn test_mmd_empty_data() {
        let model = DomainAdaptationModel::new(3, 8, 1);
        let adapter = MMDAdapter::new(model, 1.0, 0.5, 0.001);

        let result = adapter.compute_mmd(&[], &[vec![1.0, 2.0, 3.0]]);
        assert!(result.is_err());
    }

    #[test]
    fn test_evaluate_mmd() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let adapter = MMDAdapter::new(model, 1.0, 0.5, 0.001);

        let source = vec![vec![0.1; 6], vec![0.2; 6]];
        let target = vec![vec![0.5; 6], vec![0.6; 6]];

        let mmd = adapter.evaluate_mmd(&source, &target).unwrap();
        assert!(mmd >= 0.0);
    }
}
