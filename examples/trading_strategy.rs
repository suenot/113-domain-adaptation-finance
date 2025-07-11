//! # Trading Strategy with Domain Adaptation Example
//!
//! Demonstrates a complete workflow combining domain adaptation with
//! a trading strategy and backtesting:
//!
//! 1. Generate training data from a source market (equities)
//! 2. Train a DANN model to adapt to the target market (crypto)
//! 3. Create an adaptive trading strategy using the trained model
//! 4. Backtest the strategy on target market data
//! 5. Report comprehensive performance metrics
//!
//! This example shows how domain adaptation can be used in practice
//! to transfer trading knowledge from one market to another.

use domain_adaptation_trading::data::bybit::BybitClient;
use domain_adaptation_trading::data::features::FeatureGenerator;
use domain_adaptation_trading::model::network::DomainAdaptationModel;
use domain_adaptation_trading::adaptation::dann::DANNTrainer;
use domain_adaptation_trading::trading::strategy::{AdaptiveStrategy, RiskParameters};
use domain_adaptation_trading::backtest::engine::{BacktestEngine, BacktestConfig};

fn main() {
    println!("==============================================");
    println!("  Chapter 92: Domain-Adapted Trading Strategy");
    println!("  Backtest with DANN Adaptation");
    println!("==============================================\n");

    let client = BybitClient::new();
    let feature_gen = FeatureGenerator::new(20);

    // ---------------------------------------------------------------
    // Step 1: Generate training data
    // ---------------------------------------------------------------
    println!("[1/6] Generating training data...");

    // Source domain: equity-like market for training
    let source_klines = client.simulated_klines("SPY", 500);
    let source_features = feature_gen.generate_features(&source_klines);
    let source_labels = feature_gen.generate_labels(&source_klines);
    println!("  Source domain (SPY): {} training samples", source_features.len());

    // Target domain: crypto market (adaptation target)
    let target_train_klines = client.simulated_klines("BTCUSDT", 500);
    let target_features = feature_gen.generate_features(&target_train_klines);
    println!("  Target domain (BTCUSDT): {} adaptation samples", target_features.len());

    // Align lengths
    let min_len = source_features.len()
        .min(target_features.len())
        .min(source_labels.len());

    let source_features = source_features[..min_len].to_vec();
    let source_labels = source_labels[..min_len].to_vec();
    let target_features = target_features[..min_len].to_vec();

    // ---------------------------------------------------------------
    // Step 2: Train DANN model
    // ---------------------------------------------------------------
    println!("\n[2/6] Training DANN model for domain adaptation...");
    println!("  Architecture: input(6) -> hidden(32) -> hidden(32) -> output(1)");
    println!("  Learning rate: 0.01, Lambda: 0.3, Epochs: 50");

    let model = DomainAdaptationModel::new(6, 32, 1);
    let mut trainer = DANNTrainer::new(model, 0.01, 0.3);

    let metrics = trainer
        .train(&source_features, &source_labels, &target_features, 50)
        .expect("DANN training failed");

    println!("\n  Training results: {}", metrics);

    let (accuracy, confusion) = trainer
        .evaluate(&source_features, &source_labels, &target_features)
        .expect("Evaluation failed");
    println!("  Source accuracy:  {:.1}%", accuracy * 100.0);
    println!("  Domain confusion: {:.3}", confusion);

    // ---------------------------------------------------------------
    // Step 3: Create adaptive trading strategy
    // ---------------------------------------------------------------
    println!("\n[3/6] Creating adaptive trading strategy...");

    let trained_model = trainer.into_model();

    let risk_params = RiskParameters {
        max_position_size: 0.10,   // Risk 10% of equity per trade
        stop_loss: 0.02,           // 2% stop loss
        take_profit: 0.04,         // 4% take profit (2:1 reward/risk)
    };

    println!("  Risk Parameters:");
    println!("    Max position size: {:.0}%", risk_params.max_position_size * 100.0);
    println!("    Stop loss:         {:.1}%", risk_params.stop_loss * 100.0);
    println!("    Take profit:       {:.1}%", risk_params.take_profit * 100.0);

    let adapted_strategy = AdaptiveStrategy::new(trained_model, 20, risk_params.clone());

    // ---------------------------------------------------------------
    // Step 4: Generate test data for backtesting
    // ---------------------------------------------------------------
    println!("\n[4/6] Generating out-of-sample test data...");
    let test_klines = client.simulated_klines("BTCUSDT", 1000);
    println!("  Test period: {} klines (BTCUSDT)", test_klines.len());

    // ---------------------------------------------------------------
    // Step 5: Backtest adapted strategy
    // ---------------------------------------------------------------
    println!("\n[5/6] Running backtest with adapted strategy...");

    let config = BacktestConfig {
        initial_balance: 10_000.0,
        commission: 0.001,    // 0.1% commission (typical for crypto)
        slippage: 0.0005,     // 0.05% slippage
    };

    println!("  Backtest Config:");
    println!("    Initial balance: ${:.0}", config.initial_balance);
    println!("    Commission:      {:.2}%", config.commission * 100.0);
    println!("    Slippage:        {:.3}%", config.slippage * 100.0);

    let engine = BacktestEngine::new(config.clone());
    let adapted_results = engine
        .run(&adapted_strategy, &test_klines)
        .expect("Adapted backtest failed");

    println!("\n  Adapted Strategy Results:");
    println!("{}", adapted_results);

    // ---------------------------------------------------------------
    // Step 6: Compare with non-adapted baseline
    // ---------------------------------------------------------------
    println!("[6/6] Comparing with non-adapted baseline...\n");

    // Train a model without domain adaptation
    let baseline_model = DomainAdaptationModel::new(6, 32, 1);
    let mut baseline_trainer = DANNTrainer::new(baseline_model, 0.01, 0.0); // No adaptation

    baseline_trainer
        .train(&source_features, &source_labels, &target_features, 50)
        .expect("Baseline training failed");

    let baseline_model = baseline_trainer.into_model();
    let baseline_strategy = AdaptiveStrategy::new(baseline_model, 20, risk_params);

    let baseline_engine = BacktestEngine::new(config);
    let baseline_results = baseline_engine
        .run(&baseline_strategy, &test_klines)
        .expect("Baseline backtest failed");

    println!("  Baseline Strategy Results:");
    println!("{}", baseline_results);

    // ---------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------
    println!("==============================================");
    println!("  Side-by-Side Comparison");
    println!("==============================================");
    println!("  {:20} | {:>12} | {:>12}", "Metric", "Adapted", "Baseline");
    println!("  {:20} | {:>12} | {:>12}", "--------------------", "------------", "------------");
    println!(
        "  {:20} | {:>12} | {:>12}",
        "Total Trades",
        adapted_results.total_trades,
        baseline_results.total_trades
    );
    println!(
        "  {:20} | {:>11.2}% | {:>11.2}%",
        "Total Return",
        adapted_results.total_return * 100.0,
        baseline_results.total_return * 100.0
    );
    println!(
        "  {:20} | {:>12.2} | {:>12.2}",
        "Total P&L",
        adapted_results.total_pnl,
        baseline_results.total_pnl
    );
    println!(
        "  {:20} | {:>11.1}% | {:>11.1}%",
        "Win Rate",
        adapted_results.win_rate * 100.0,
        baseline_results.win_rate * 100.0
    );
    println!(
        "  {:20} | {:>12.3} | {:>12.3}",
        "Sharpe Ratio",
        adapted_results.sharpe_ratio,
        baseline_results.sharpe_ratio
    );
    println!(
        "  {:20} | {:>12.3} | {:>12.3}",
        "Sortino Ratio",
        adapted_results.sortino_ratio,
        baseline_results.sortino_ratio
    );
    println!(
        "  {:20} | {:>11.2}% | {:>11.2}%",
        "Max Drawdown",
        adapted_results.max_drawdown * 100.0,
        baseline_results.max_drawdown * 100.0
    );
    println!(
        "  {:20} | {:>12.2} | {:>12.2}",
        "Profit Factor",
        adapted_results.profit_factor,
        baseline_results.profit_factor
    );
    println!("==============================================");

    // Print some sample trades
    if !adapted_results.trades.is_empty() {
        println!("\nSample Adapted Trades (first 5):");
        println!("  {:>8} | {:>8} | {:>5} | {:>10} | {:>8}",
            "Entry", "Exit", "Dir", "P&L", "Periods");
        println!("  {:>8} | {:>8} | {:>5} | {:>10} | {:>8}",
            "--------", "--------", "-----", "----------", "--------");

        for trade in adapted_results.trades.iter().take(5) {
            let dir = if trade.direction > 0.0 { "LONG" } else { "SHORT" };
            println!(
                "  {:>8.2} | {:>8.2} | {:>5} | {:>10.4} | {:>8}",
                trade.entry_price,
                trade.exit_price,
                dir,
                trade.pnl,
                trade.holding_period
            );
        }
    }

    println!("\nNote: This uses simulated data. In production, connect to");
    println!("the Bybit API for real market data using BybitClient::fetch_klines().");
}
