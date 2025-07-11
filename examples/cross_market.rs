//! # Cross-Market Domain Adaptation Example
//!
//! Demonstrates adapting a trading model trained on one market (equities/SPY)
//! to another market (crypto/BTCUSDT) using all three adaptation techniques:
//!
//! 1. **DANN** - Adversarial domain-invariant features
//! 2. **MMD** - Distribution distance minimization
//! 3. **CORAL** - Covariance structure alignment
//!
//! The example compares the performance of adapted vs non-adapted models
//! to show the benefit of domain adaptation for cross-market transfer.

use domain_adaptation_trading::data::bybit::BybitClient;
use domain_adaptation_trading::data::features::FeatureGenerator;
use domain_adaptation_trading::model::network::DomainAdaptationModel;
use domain_adaptation_trading::adaptation::dann::DANNTrainer;
use domain_adaptation_trading::adaptation::mmd::MMDAdapter;
use domain_adaptation_trading::adaptation::coral::CORALAdapter;

fn main() {
    println!("==============================================");
    println!("  Chapter 92: Cross-Market Adaptation");
    println!("  Equities (SPY) -> Crypto (BTCUSDT)");
    println!("==============================================\n");

    let client = BybitClient::new();
    let feature_gen = FeatureGenerator::new(20);

    // ---------------------------------------------------------------
    // Generate market data for two different domains
    // ---------------------------------------------------------------
    println!("[1/6] Generating market data...");

    // Source domain: Equity-like market (lower volatility)
    let source_klines = client.simulated_klines("SPY", 400);
    let source_features = feature_gen.generate_features(&source_klines);
    let source_labels = feature_gen.generate_labels(&source_klines);
    println!("  Source (SPY):     {} samples", source_features.len());

    // Target domain: Crypto market (higher volatility, different dynamics)
    let target_klines = client.simulated_klines("BTCUSDT", 400);
    let target_features = feature_gen.generate_features(&target_klines);
    let target_labels = feature_gen.generate_labels(&target_klines);
    println!("  Target (BTCUSDT): {} samples", target_features.len());

    // Align lengths
    let min_len = source_features.len()
        .min(target_features.len())
        .min(source_labels.len())
        .min(target_labels.len());

    let source_features = source_features[..min_len].to_vec();
    let source_labels = source_labels[..min_len].to_vec();
    let target_features = target_features[..min_len].to_vec();
    let target_labels = target_labels[..min_len].to_vec();

    // ---------------------------------------------------------------
    // Baseline: Non-adapted model (trained only on source)
    // ---------------------------------------------------------------
    println!("\n[2/6] Training baseline model (no adaptation)...");
    let baseline_model = DomainAdaptationModel::new(6, 32, 1);
    let mut baseline_trainer = DANNTrainer::new(baseline_model, 0.01, 0.0); // lambda=0 = no adaptation

    baseline_trainer
        .train(&source_features, &source_labels, &target_features, 30)
        .expect("Baseline training failed");

    let baseline_accuracy = evaluate_target_accuracy(
        baseline_trainer.model(),
        &target_features,
        &target_labels,
    );
    println!("  Baseline target accuracy: {:.1}%", baseline_accuracy * 100.0);

    // ---------------------------------------------------------------
    // Method 1: DANN Adaptation
    // ---------------------------------------------------------------
    println!("\n[3/6] Training DANN-adapted model...");
    let dann_model = DomainAdaptationModel::new(6, 32, 1);
    let mut dann_trainer = DANNTrainer::new(dann_model, 0.01, 0.5);

    let dann_metrics = dann_trainer
        .train(&source_features, &source_labels, &target_features, 30)
        .expect("DANN training failed");

    let dann_accuracy = evaluate_target_accuracy(
        dann_trainer.model(),
        &target_features,
        &target_labels,
    );
    println!("  DANN training: {}", dann_metrics);
    println!("  DANN target accuracy: {:.1}%", dann_accuracy * 100.0);

    let (_, dann_confusion) = dann_trainer
        .evaluate(&source_features, &source_labels, &target_features)
        .expect("DANN evaluation failed");
    println!("  DANN domain confusion: {:.3}", dann_confusion);

    // ---------------------------------------------------------------
    // Method 2: MMD Adaptation
    // ---------------------------------------------------------------
    println!("\n[4/6] Training MMD-adapted model...");
    let mmd_model = DomainAdaptationModel::new(6, 32, 1);
    let mut mmd_adapter = MMDAdapter::new(mmd_model, 1.0, 0.5, 0.01);

    let mmd_metrics = mmd_adapter
        .adapt(&source_features, &source_labels, &target_features, 30)
        .expect("MMD adaptation failed");

    let mmd_accuracy = evaluate_target_accuracy(
        mmd_adapter.model(),
        &target_features,
        &target_labels,
    );
    println!("  MMD training: {}", mmd_metrics);
    println!("  MMD target accuracy: {:.1}%", mmd_accuracy * 100.0);

    let mmd_distance = mmd_adapter
        .evaluate_mmd(&source_features, &target_features)
        .expect("MMD evaluation failed");
    println!("  MMD distance (post-adaptation): {:.6}", mmd_distance);

    // ---------------------------------------------------------------
    // Method 3: CORAL Adaptation
    // ---------------------------------------------------------------
    println!("\n[5/6] Training CORAL-adapted model...");
    let coral_model = DomainAdaptationModel::new(6, 32, 1);
    let mut coral_adapter = CORALAdapter::new(coral_model, 0.01, 0.5);

    let coral_metrics = coral_adapter
        .adapt(&source_features, &source_labels, &target_features, 30)
        .expect("CORAL adaptation failed");

    let coral_accuracy = evaluate_target_accuracy(
        coral_adapter.model(),
        &target_features,
        &target_labels,
    );
    println!("  CORAL training: {}", coral_metrics);
    println!("  CORAL target accuracy: {:.1}%", coral_accuracy * 100.0);

    let coral_loss = coral_adapter
        .evaluate_coral(&source_features, &target_features)
        .expect("CORAL evaluation failed");
    println!("  CORAL loss (post-adaptation): {:.6}", coral_loss);

    // ---------------------------------------------------------------
    // Summary comparison
    // ---------------------------------------------------------------
    println!("\n[6/6] Summary Comparison");
    println!("==============================================");
    println!("  Method        | Target Accuracy | Metric");
    println!("  --------------|-----------------|----------");
    println!(
        "  Baseline      | {:>13.1}%  | N/A",
        baseline_accuracy * 100.0
    );
    println!(
        "  DANN          | {:>13.1}%  | confusion={:.3}",
        dann_accuracy * 100.0, dann_confusion
    );
    println!(
        "  MMD           | {:>13.1}%  | distance={:.4}",
        mmd_accuracy * 100.0, mmd_distance
    );
    println!(
        "  CORAL         | {:>13.1}%  | cov_loss={:.4}",
        coral_accuracy * 100.0, coral_loss
    );
    println!("==============================================");

    println!("\nNote: Results vary due to random initialization.");
    println!("In practice, adaptation typically improves target performance");
    println!("when there is a genuine distribution shift between domains.");
}

/// Evaluates model accuracy on target domain data.
///
/// Since target labels are available (for evaluation only, not training),
/// we compute the classification accuracy.
fn evaluate_target_accuracy(
    model: &DomainAdaptationModel,
    features: &[Vec<f64>],
    labels: &[f64],
) -> f64 {
    let mut correct = 0;
    let total = features.len().min(labels.len());

    for i in 0..total {
        if let Ok(pred) = model.predict_labels(&features[i]) {
            let pred_class = if pred[0] > 0.5 { 1.0 } else { 0.0 };
            if (pred_class - labels[i]).abs() < 0.5 {
                correct += 1;
            }
        }
    }

    correct as f64 / total as f64
}
