//! # Model Module
//!
//! Provides neural network architectures for domain adaptation in financial
//! applications. The core model follows the DANN (Domain-Adversarial Neural
//! Network) architecture with three components:
//!
//! 1. **Feature Extractor** - Shared layers that learn domain-invariant representations
//! 2. **Label Predictor** - Task-specific head for predicting trading signals
//! 3. **Domain Classifier** - Adversarial head for distinguishing source/target domains
//!
//! The gradient reversal layer between the feature extractor and domain classifier
//! encourages the feature extractor to learn representations that are useful for
//! the task but indistinguishable across domains.

pub mod network;
