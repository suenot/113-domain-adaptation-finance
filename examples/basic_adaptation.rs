//! # Basic Domain Adaptation Example
//!
//! Demonstrates the fundamental workflow of domain adaptation for finance:
//!
//! 1. Generate simulated source and target domain data
//! 2. Extract features from both domains
//! 3. Train a DANN model to adapt from source to target
//! 4. Evaluate adaptation quality by measuring domain confusion
//!
//! This example uses simulated data with controlled domain shift to clearly
//! illustrate the effect of domain adaptation.

use domain_adaptation_trading::data::bybit::BybitClient;
use domain_adaptation_trading::data::features::FeatureGenerator;
use domain_adaptation_trading::model::network::DomainAdaptationModel;
use domain_adaptation_trading::adaptation::dann::DANNTrainer;

fn main() {
    println!("==============================================");
    println!("  Chapter 92: Domain Adaptation for Finance");
    println!("  Basic DANN Adaptation Example");
    println!("==============================================\n");

    // Initialize the Bybit client (using simulated data)
    let client = BybitClient::new();
    let feature_gen = FeatureGenerator::new(20);

    // ---------------------------------------------------------------
    // Step 1: Generate source domain data (e.g., BTC in normal regime)
    // ---------------------------------------------------------------
    println!("[1/5] Generating source domain data (BTCUSDT - normal market)...");
    let source_klines = client.simulated_klines("BTCUSDT", 300);
    let source_features = feature_gen.generate_features(&source_klines);
    let source_labels = feature_gen.generate_labels(&source_klines);
    println!("  Source: {} samples, {} features each",
        source_features.len(), source_features.first().map_or(0, |f| f.len()));

    // ---------------------------------------------------------------
    // Step 2: Generate target domain data (e.g., BTC in volatile regime)
    // ---------------------------------------------------------------
    println!("[2/5] Generating target domain data (BTCUSDT - volatile regime)...");
    let target_klines = client.simulated_klines_with_shift("BTCUSDT", 300, 0.001, 1.5);
    let target_features = feature_gen.generate_features(&target_klines);
    println!("  Target: {} samples, {} features each",
        target_features.len(), target_features.first().map_or(0, |f| f.len()));

    // Ensure we have matching lengths for training
    let min_len = source_features.len().min(target_features.len()).min(source_labels.len());
    let source_features = &source_features[..min_len];
    let source_labels = &source_labels[..min_len];
    let target_features = &target_features[..min_len];

    // ---------------------------------------------------------------
    // Step 3: Create and train the DANN model
    // ---------------------------------------------------------------
    println!("\n[3/5] Training DANN model...");
    println!("  Architecture: 6 -> 32 -> 32 (feature extractor)");
    println!("                32 -> 1 (label predictor)");
    println!("                32 -> 16 -> 1 (domain classifier)");
    println!("  Learning rate: 0.01, Lambda: 0.3\n");

    let model = DomainAdaptationModel::new(6, 32, 1);
    let mut trainer = DANNTrainer::new(model, 0.01, 0.3);

    // Evaluate before training
    let (pre_accuracy, pre_confusion) = trainer
        .evaluate(source_features, source_labels, target_features)
        .expect("Pre-training evaluation failed");

    println!("  Before training:");
    println!("    Source accuracy:        {:.1}%", pre_accuracy * 100.0);
    println!("    Domain confusion score: {:.3}", pre_confusion);

    // Train for multiple epochs
    let epochs = 50;
    println!("\n  Training for {} epochs...", epochs);
    let metrics = trainer
        .train(
            &source_features.to_vec(),
            &source_labels.to_vec(),
            &target_features.to_vec(),
            epochs,
        )
        .expect("DANN training failed");

    println!("  Training complete: {}", metrics);

    // ---------------------------------------------------------------
    // Step 4: Evaluate after training
    // ---------------------------------------------------------------
    println!("\n[4/5] Evaluating adaptation quality...");
    let (post_accuracy, post_confusion) = trainer
        .evaluate(source_features, source_labels, target_features)
        .expect("Post-training evaluation failed");

    println!("  After training:");
    println!("    Source accuracy:        {:.1}%", post_accuracy * 100.0);
    println!("    Domain confusion score: {:.3}", post_confusion);

    // ---------------------------------------------------------------
    // Step 5: Compare predictions on target domain
    // ---------------------------------------------------------------
    println!("\n[5/5] Sample predictions on target domain:");
    let model = trainer.model();

    for i in 0..5.min(target_features.len()) {
        let pred = model.predict_labels(&target_features[i])
            .expect("Prediction failed");
        let domain = model.classify_domain(&target_features[i])
            .expect("Domain classification failed");

        println!(
            "  Sample {}: Prediction={:.3}, Domain={:.3} (0=source, 1=target)",
            i, pred[0], domain[0]
        );
    }

    println!("\n==============================================");
    println!("  Interpretation:");
    println!("  - Domain confusion near 1.0 = good adaptation");
    println!("    (classifier cannot distinguish domains)");
    println!("  - Source accuracy maintained = features still useful");
    println!("==============================================");
}
