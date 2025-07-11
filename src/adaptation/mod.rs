//! # Adaptation Module
//!
//! Implements three domain adaptation techniques for transferring trading models
//! across different financial domains:
//!
//! ## DANN (Domain-Adversarial Neural Network)
//!
//! Learns domain-invariant features through adversarial training. A gradient
//! reversal layer ensures the feature extractor learns representations that
//! are useful for the task but indistinguishable across domains.
//!
//! **Best for**: When you have labeled source data and unlabeled target data,
//! and the feature distributions differ significantly between domains.
//!
//! ## MMD (Maximum Mean Discrepancy)
//!
//! Minimizes the statistical distance between source and target feature
//! distributions using a kernel-based metric. Directly penalizes distribution
//! mismatch during training.
//!
//! **Best for**: When domains have smooth, continuous distribution shifts
//! (e.g., gradual market regime changes).
//!
//! ## CORAL (Correlation Alignment)
//!
//! Aligns the second-order statistics (covariance matrices) of source and
//! target features. Simpler and faster than DANN, effective when the
//! feature correlations are the primary domain shift.
//!
//! **Best for**: When domains differ mainly in feature correlations rather
//! than means (e.g., cross-asset adaptation where correlation structure varies).

pub mod dann;
pub mod mmd;
pub mod coral;
