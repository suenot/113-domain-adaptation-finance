//! # Neural Network Implementation
//!
//! Implements a simple feedforward neural network suitable for domain adaptation
//! in financial applications. The architecture is designed for the DANN framework
//! with separate feature extraction, label prediction, and domain classification
//! components.
//!
//! ## Architecture
//!
//! ```text
//!                          +-------------------+
//!                          | Feature Extractor |
//!                          | (shared layers)   |
//!                          +---------+---------+
//!                                    |
//!                     +--------------+--------------+
//!                     |                             |
//!              +------+------+            +---------+---------+
//!              |   Label     |            | Gradient Reversal |
//!              |  Predictor  |            |      Layer        |
//!              +------+------+            +---------+---------+
//!                     |                             |
//!                  Prediction              +--------+--------+
//!                                          |    Domain       |
//!                                          |   Classifier    |
//!                                          +--------+--------+
//!                                                   |
//!                                            Domain Label
//! ```
//!
//! ## Weight Initialization
//!
//! Weights are initialized using Xavier/Glorot uniform initialization:
//! `W ~ U(-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out)))`

use rand::Rng;
use crate::{DomainAdaptationError, Result};

/// A single fully-connected (dense) neural network layer.
///
/// Performs the linear transformation `output = input * W + b` followed by
/// an optional ReLU activation.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Weight matrix of shape `[input_size, output_size]`, stored row-major.
    pub weights: Vec<Vec<f64>>,
    /// Bias vector of length `output_size`.
    pub biases: Vec<f64>,
    /// Number of input neurons.
    pub input_size: usize,
    /// Number of output neurons.
    pub output_size: usize,
}

impl Layer {
    /// Creates a new layer with Xavier-initialized weights and zero biases.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features
    /// * `output_size` - Number of output features
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        // Xavier initialization: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();

        let weights: Vec<Vec<f64>> = (0..input_size)
            .map(|_| {
                (0..output_size)
                    .map(|_| rng.gen_range(-limit..limit))
                    .collect()
            })
            .collect();

        let biases = vec![0.0; output_size];

        Self {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    /// Performs the forward pass: `output = input * W + b`.
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of length `input_size`
    ///
    /// # Returns
    ///
    /// Output vector of length `output_size`, or an error if dimensions mismatch.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        if input.len() != self.input_size {
            return Err(DomainAdaptationError::ModelError(format!(
                "Layer expects input of size {}, got {}",
                self.input_size,
                input.len()
            )));
        }

        let mut output = self.biases.clone();
        for (i, &x) in input.iter().enumerate() {
            for (j, o) in output.iter_mut().enumerate() {
                *o += x * self.weights[i][j];
            }
        }

        Ok(output)
    }

    /// Applies the forward pass followed by ReLU activation.
    ///
    /// ReLU: `f(x) = max(0, x)`
    ///
    /// # Arguments
    ///
    /// * `input` - Input vector of length `input_size`
    pub fn forward_relu(&self, input: &[f64]) -> Result<Vec<f64>> {
        let output = self.forward(input)?;
        Ok(output.into_iter().map(|x| x.max(0.0)).collect())
    }

    /// Applies a simplified gradient update to the layer weights.
    ///
    /// Updates weights using: `W -= learning_rate * grad`
    ///
    /// # Arguments
    ///
    /// * `input` - The input that was used in the forward pass
    /// * `output_grad` - Gradient of the loss w.r.t. the layer output
    /// * `learning_rate` - Step size for the update
    ///
    /// # Returns
    ///
    /// Gradient of the loss w.r.t. the layer input (for backpropagation).
    pub fn backward(
        &mut self,
        input: &[f64],
        output_grad: &[f64],
        learning_rate: f64,
    ) -> Result<Vec<f64>> {
        if output_grad.len() != self.output_size {
            return Err(DomainAdaptationError::ModelError(
                "Gradient dimension mismatch".to_string(),
            ));
        }

        // Compute input gradient for backprop to previous layers
        let mut input_grad = vec![0.0; self.input_size];
        for i in 0..self.input_size {
            for j in 0..self.output_size {
                input_grad[i] += self.weights[i][j] * output_grad[j];
            }
        }

        // Update weights: W -= lr * outer(input, output_grad)
        for i in 0..self.input_size {
            for j in 0..self.output_size {
                self.weights[i][j] -= learning_rate * input[i] * output_grad[j];
            }
        }

        // Update biases
        for j in 0..self.output_size {
            self.biases[j] -= learning_rate * output_grad[j];
        }

        Ok(input_grad)
    }
}

/// Shared feature extraction network.
///
/// Transforms raw input features into a domain-invariant hidden representation.
/// This component is shared between the label predictor and domain classifier
/// in the DANN architecture.
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    /// First hidden layer
    pub layer1: Layer,
    /// Second hidden layer
    pub layer2: Layer,
}

impl FeatureExtractor {
    /// Creates a new feature extractor with two hidden layers.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Dimension of input features
    /// * `hidden_size` - Dimension of hidden representations
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self {
            layer1: Layer::new(input_size, hidden_size),
            layer2: Layer::new(hidden_size, hidden_size),
        }
    }

    /// Extracts features from the input through two ReLU layers.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw feature vector
    ///
    /// # Returns
    ///
    /// Hidden representation vector of size `hidden_size`.
    pub fn forward(&self, input: &[f64]) -> Result<Vec<f64>> {
        let h1 = self.layer1.forward_relu(input)?;
        self.layer2.forward_relu(&h1)
    }

    /// Performs backward pass through the feature extractor.
    ///
    /// # Arguments
    ///
    /// * `input` - Original input to the forward pass
    /// * `grad` - Gradient from downstream components
    /// * `learning_rate` - Step size
    pub fn backward(
        &mut self,
        input: &[f64],
        grad: &[f64],
        learning_rate: f64,
    ) -> Result<Vec<f64>> {
        // Recompute intermediate activations for backward pass
        let h1 = self.layer1.forward_relu(input)?;

        // Apply ReLU derivative mask to grad for layer2
        let h2_pre = self.layer2.forward(&h1)?;
        let grad_relu2: Vec<f64> = grad
            .iter()
            .zip(h2_pre.iter())
            .map(|(&g, &h)| if h > 0.0 { g } else { 0.0 })
            .collect();

        let grad_h1 = self.layer2.backward(&h1, &grad_relu2, learning_rate)?;

        // Apply ReLU derivative mask for layer1
        let h1_pre = self.layer1.forward(input)?;
        let grad_relu1: Vec<f64> = grad_h1
            .iter()
            .zip(h1_pre.iter())
            .map(|(&g, &h)| if h > 0.0 { g } else { 0.0 })
            .collect();

        self.layer1.backward(input, &grad_relu1, learning_rate)
    }
}

/// Task-specific label prediction head.
///
/// Takes the hidden features from the [`FeatureExtractor`] and produces
/// task predictions (e.g., price direction, return magnitude).
#[derive(Debug, Clone)]
pub struct LabelPredictor {
    /// Hidden to output layer
    pub layer: Layer,
}

impl LabelPredictor {
    /// Creates a new label predictor.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Dimension of the input (from feature extractor)
    /// * `output_size` - Number of output classes or regression targets
    pub fn new(hidden_size: usize, output_size: usize) -> Self {
        Self {
            layer: Layer::new(hidden_size, output_size),
        }
    }

    /// Predicts labels from hidden features.
    ///
    /// Applies a sigmoid activation for binary classification output in [0, 1].
    ///
    /// # Arguments
    ///
    /// * `features` - Hidden feature vector from the feature extractor
    pub fn forward(&self, features: &[f64]) -> Result<Vec<f64>> {
        let logits = self.layer.forward(features)?;
        // Sigmoid activation for binary classification
        Ok(logits.into_iter().map(|x| sigmoid(x)).collect())
    }

    /// Performs backward pass through the label predictor.
    ///
    /// # Arguments
    ///
    /// * `features` - Input features used in forward pass
    /// * `predictions` - Output from forward pass
    /// * `targets` - Ground truth labels
    /// * `learning_rate` - Step size
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. input features and the binary cross-entropy loss value.
    pub fn backward(
        &mut self,
        features: &[f64],
        predictions: &[f64],
        targets: &[f64],
        learning_rate: f64,
    ) -> Result<(Vec<f64>, f64)> {
        // Binary cross-entropy loss gradient: pred - target
        let grad: Vec<f64> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| p - t)
            .collect();

        // Compute loss for monitoring
        let loss = binary_cross_entropy(predictions, targets);

        let input_grad = self.layer.backward(features, &grad, learning_rate)?;
        Ok((input_grad, loss))
    }
}

/// Domain classification head for adversarial training.
///
/// Attempts to distinguish between source domain (label 0) and target domain
/// (label 1) features. In DANN training, the gradient from this classifier
/// is reversed before being applied to the feature extractor, encouraging
/// domain-invariant features.
#[derive(Debug, Clone)]
pub struct DomainClassifier {
    /// First classification layer
    pub layer1: Layer,
    /// Output layer
    pub layer2: Layer,
}

impl DomainClassifier {
    /// Creates a new domain classifier with two layers.
    ///
    /// # Arguments
    ///
    /// * `hidden_size` - Dimension of input features (from feature extractor)
    pub fn new(hidden_size: usize) -> Self {
        let intermediate = hidden_size / 2;
        let intermediate = intermediate.max(2);
        Self {
            layer1: Layer::new(hidden_size, intermediate),
            layer2: Layer::new(intermediate, 1),
        }
    }

    /// Classifies the domain of the input features.
    ///
    /// # Returns
    ///
    /// Sigmoid probability: values near 0.0 indicate source domain,
    /// values near 1.0 indicate target domain.
    pub fn forward(&self, features: &[f64]) -> Result<Vec<f64>> {
        let h = self.layer1.forward_relu(features)?;
        let logits = self.layer2.forward(&h)?;
        Ok(logits.into_iter().map(|x| sigmoid(x)).collect())
    }

    /// Performs backward pass through the domain classifier.
    ///
    /// # Arguments
    ///
    /// * `features` - Input features from the feature extractor
    /// * `predictions` - Domain predictions from forward pass
    /// * `targets` - Domain labels (0.0 = source, 1.0 = target)
    /// * `learning_rate` - Step size
    ///
    /// # Returns
    ///
    /// Gradient w.r.t. input features and the domain classification loss.
    pub fn backward(
        &mut self,
        features: &[f64],
        predictions: &[f64],
        targets: &[f64],
        learning_rate: f64,
    ) -> Result<(Vec<f64>, f64)> {
        // BCE gradient
        let grad_output: Vec<f64> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| p - t)
            .collect();

        let loss = binary_cross_entropy(predictions, targets);

        // Backward through layer2
        let h = self.layer1.forward_relu(features)?;
        let grad_h = self.layer2.backward(&h, &grad_output, learning_rate)?;

        // Apply ReLU derivative
        let h_pre = self.layer1.forward(features)?;
        let grad_relu: Vec<f64> = grad_h
            .iter()
            .zip(h_pre.iter())
            .map(|(&g, &h_val)| if h_val > 0.0 { g } else { 0.0 })
            .collect();

        let input_grad = self.layer1.backward(features, &grad_relu, learning_rate)?;
        Ok((input_grad, loss))
    }
}

/// Complete Domain Adaptation Model combining all three components.
///
/// This model integrates the feature extractor, label predictor, and domain
/// classifier into a single coherent architecture for domain-adversarial training.
///
/// # Training Flow
///
/// 1. Source data: feature_extractor -> label_predictor (minimize label loss)
/// 2. Both domains: feature_extractor -> gradient_reversal -> domain_classifier
///    (minimize domain loss, but reversed gradient makes features domain-invariant)
#[derive(Debug, Clone)]
pub struct DomainAdaptationModel {
    /// Shared feature extraction layers
    pub feature_extractor: FeatureExtractor,
    /// Task-specific prediction head
    pub label_predictor: LabelPredictor,
    /// Domain classification head (for adversarial training)
    pub domain_classifier: DomainClassifier,
    /// Dimension of the hidden representation
    pub hidden_size: usize,
    /// Dimension of the input features
    pub input_size: usize,
    /// Dimension of the output predictions
    pub output_size: usize,
}

impl DomainAdaptationModel {
    /// Creates a new domain adaptation model.
    ///
    /// # Arguments
    ///
    /// * `input_size` - Number of input features (e.g., 6 for the default feature set)
    /// * `hidden_size` - Dimension of hidden representations (e.g., 32, 64)
    /// * `output_size` - Number of prediction outputs (e.g., 1 for binary classification)
    ///
    /// # Example
    ///
    /// ```rust
    /// use domain_adaptation_trading::model::network::DomainAdaptationModel;
    ///
    /// let model = DomainAdaptationModel::new(6, 32, 1);
    /// ```
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            feature_extractor: FeatureExtractor::new(input_size, hidden_size),
            label_predictor: LabelPredictor::new(hidden_size, output_size),
            domain_classifier: DomainClassifier::new(hidden_size),
            hidden_size,
            input_size,
            output_size,
        }
    }

    /// Extracts hidden features from raw input.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw feature vector of dimension `input_size`
    ///
    /// # Returns
    ///
    /// Hidden representation of dimension `hidden_size`.
    pub fn forward_features(&self, input: &[f64]) -> Result<Vec<f64>> {
        self.feature_extractor.forward(input)
    }

    /// Predicts task labels from raw input.
    ///
    /// Passes the input through the feature extractor and then the label predictor.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw feature vector
    ///
    /// # Returns
    ///
    /// Prediction vector of dimension `output_size`.
    pub fn predict_labels(&self, input: &[f64]) -> Result<Vec<f64>> {
        let features = self.feature_extractor.forward(input)?;
        self.label_predictor.forward(&features)
    }

    /// Classifies the domain of the input.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw feature vector
    ///
    /// # Returns
    ///
    /// Domain probability (0 = source, 1 = target).
    pub fn classify_domain(&self, input: &[f64]) -> Result<Vec<f64>> {
        let features = self.feature_extractor.forward(input)?;
        self.domain_classifier.forward(&features)
    }

    /// Performs a complete DANN forward pass returning all outputs.
    ///
    /// This method computes features, label predictions, and domain
    /// classifications in a single forward pass (sharing the feature
    /// extraction computation).
    ///
    /// In the backward pass, the gradient from the domain classifier
    /// is reversed (multiplied by -lambda) before being applied to
    /// the feature extractor. This is the core of the DANN training
    /// procedure.
    ///
    /// # Arguments
    ///
    /// * `input` - Raw feature vector
    ///
    /// # Returns
    ///
    /// A tuple of `(features, label_predictions, domain_predictions)`.
    pub fn forward_dann(&self, input: &[f64]) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let features = self.feature_extractor.forward(input)?;
        let label_pred = self.label_predictor.forward(&features)?;
        let domain_pred = self.domain_classifier.forward(&features)?;
        Ok((features, label_pred, domain_pred))
    }

    /// Performs a DANN training step with gradient reversal.
    ///
    /// This is the core training procedure for domain-adversarial learning:
    ///
    /// 1. Forward pass through all components
    /// 2. Compute label prediction loss (on source data only)
    /// 3. Compute domain classification loss (on both domains)
    /// 4. Backward pass with gradient reversal for the domain classifier
    ///
    /// # Arguments
    ///
    /// * `input` - Raw feature vector
    /// * `label_target` - Optional ground truth label (None for target domain)
    /// * `domain_target` - Domain label (0.0 = source, 1.0 = target)
    /// * `learning_rate` - Step size for weight updates
    /// * `lambda_domain` - Weight for domain adaptation loss (controls gradient reversal strength)
    ///
    /// # Returns
    ///
    /// A tuple of `(label_loss, domain_loss)`. Label loss is 0.0 for target domain samples.
    pub fn train_step_dann(
        &mut self,
        input: &[f64],
        label_target: Option<&[f64]>,
        domain_target: f64,
        learning_rate: f64,
        lambda_domain: f64,
    ) -> Result<(f64, f64)> {
        // Forward pass
        let features = self.feature_extractor.forward(input)?;
        let label_pred = self.label_predictor.forward(&features)?;
        let domain_pred = self.domain_classifier.forward(&features)?;

        let domain_target_vec = vec![domain_target];

        // Label predictor backward (only for source domain with labels)
        let mut label_loss = 0.0;
        let mut label_grad = vec![0.0; self.hidden_size];

        if let Some(targets) = label_target {
            let (lg, ll) = self.label_predictor.backward(
                &features,
                &label_pred,
                targets,
                learning_rate,
            )?;
            label_loss = ll;
            label_grad = lg;
        }

        // Domain classifier backward
        let (domain_grad, domain_loss) = self.domain_classifier.backward(
            &features,
            &domain_pred,
            &domain_target_vec,
            learning_rate,
        )?;

        // Combine gradients with gradient reversal for domain component
        // The key insight: we NEGATE the domain gradient before passing to feature extractor
        let combined_grad: Vec<f64> = label_grad
            .iter()
            .zip(domain_grad.iter())
            .map(|(&lg, &dg)| lg - lambda_domain * dg) // Gradient reversal: subtract domain grad
            .collect();

        // Update feature extractor with combined gradient
        self.feature_extractor
            .backward(input, &combined_grad, learning_rate)?;

        Ok((label_loss, domain_loss))
    }

    /// Makes a batch prediction for multiple inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - Slice of feature vectors
    ///
    /// # Returns
    ///
    /// Vector of prediction vectors, one per input.
    pub fn predict_batch(&self, inputs: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        inputs
            .iter()
            .map(|input| self.predict_labels(input))
            .collect()
    }
}

/// Sigmoid activation function: `1 / (1 + exp(-x))`.
///
/// Clamps input to [-500, 500] to prevent overflow.
pub fn sigmoid(x: f64) -> f64 {
    let x = x.clamp(-500.0, 500.0);
    1.0 / (1.0 + (-x).exp())
}

/// Binary cross-entropy loss.
///
/// `L = -1/n * sum(t * ln(p) + (1-t) * ln(1-p))`
///
/// Predictions are clamped to [epsilon, 1-epsilon] to prevent log(0).
pub fn binary_cross_entropy(predictions: &[f64], targets: &[f64]) -> f64 {
    let eps = 1e-7;
    let n = predictions.len() as f64;

    predictions
        .iter()
        .zip(targets.iter())
        .map(|(&p, &t)| {
            let p = p.clamp(eps, 1.0 - eps);
            -(t * p.ln() + (1.0 - t) * (1.0 - p).ln())
        })
        .sum::<f64>()
        / n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_forward() {
        let layer = Layer::new(3, 2);
        let input = vec![1.0, 2.0, 3.0];
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_layer_dimension_mismatch() {
        let layer = Layer::new(3, 2);
        let input = vec![1.0, 2.0]; // Wrong size
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_layer_relu() {
        let mut layer = Layer::new(2, 3);
        // Set weights to produce some negative outputs
        layer.weights = vec![vec![1.0, -1.0, 0.5], vec![-0.5, 1.0, -1.0]];
        layer.biases = vec![0.0, 0.0, 0.0];

        let input = vec![1.0, 1.0];
        let output = layer.forward_relu(&input).unwrap();

        for &val in &output {
            assert!(val >= 0.0, "ReLU output should be non-negative");
        }
    }

    #[test]
    fn test_feature_extractor() {
        let fe = FeatureExtractor::new(6, 16);
        let input = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];
        let features = fe.forward(&input).unwrap();
        assert_eq!(features.len(), 16);
    }

    #[test]
    fn test_label_predictor() {
        let lp = LabelPredictor::new(16, 1);
        let features = vec![0.1; 16];
        let prediction = lp.forward(&features).unwrap();
        assert_eq!(prediction.len(), 1);
        assert!((0.0..=1.0).contains(&prediction[0]), "Sigmoid output should be in [0,1]");
    }

    #[test]
    fn test_domain_classifier() {
        let dc = DomainClassifier::new(16);
        let features = vec![0.1; 16];
        let domain_pred = dc.forward(&features).unwrap();
        assert_eq!(domain_pred.len(), 1);
        assert!((0.0..=1.0).contains(&domain_pred[0]));
    }

    #[test]
    fn test_full_model_forward() {
        let model = DomainAdaptationModel::new(6, 32, 1);
        let input = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];

        let features = model.forward_features(&input).unwrap();
        assert_eq!(features.len(), 32);

        let labels = model.predict_labels(&input).unwrap();
        assert_eq!(labels.len(), 1);

        let domain = model.classify_domain(&input).unwrap();
        assert_eq!(domain.len(), 1);
    }

    #[test]
    fn test_dann_forward() {
        let model = DomainAdaptationModel::new(6, 32, 1);
        let input = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];

        let (features, labels, domain) = model.forward_dann(&input).unwrap();
        assert_eq!(features.len(), 32);
        assert_eq!(labels.len(), 1);
        assert_eq!(domain.len(), 1);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let preds = vec![0.9, 0.1];
        let targets = vec![1.0, 0.0];
        let loss = binary_cross_entropy(&preds, &targets);
        assert!(loss < 0.2, "Loss should be low for correct predictions");

        let preds_bad = vec![0.1, 0.9];
        let loss_bad = binary_cross_entropy(&preds_bad, &targets);
        assert!(loss_bad > loss, "Wrong predictions should have higher loss");
    }

    #[test]
    fn test_train_step() {
        let mut model = DomainAdaptationModel::new(6, 16, 1);
        let input = vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1];
        let target = vec![1.0];

        let result = model.train_step_dann(&input, Some(&target), 0.0, 0.01, 0.1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_prediction() {
        let model = DomainAdaptationModel::new(6, 16, 1);
        let inputs = vec![
            vec![0.1, -0.2, 0.3, 0.0, 0.5, -0.1],
            vec![-0.1, 0.2, -0.3, 0.0, -0.5, 0.1],
        ];

        let predictions = model.predict_batch(&inputs).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_layer_backward() {
        let mut layer = Layer::new(3, 2);
        let input = vec![1.0, 2.0, 3.0];
        let _output = layer.forward(&input).unwrap();
        let grad = vec![0.1, -0.1];

        let input_grad = layer.backward(&input, &grad, 0.01).unwrap();
        assert_eq!(input_grad.len(), 3);
    }
}
