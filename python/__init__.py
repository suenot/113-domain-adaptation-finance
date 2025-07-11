"""
Chapter 92: Domain Adaptation for Finance
==========================================

This package implements domain adaptation techniques for adapting trading
models across financial domains. It provides three core adaptation methods:

- **DANN** (Domain-Adversarial Neural Network): Uses a gradient reversal layer
  to learn domain-invariant feature representations by adversarial training
  against a domain classifier.

- **MMD** (Maximum Mean Discrepancy): Minimizes the distribution distance
  between source and target domain features using a kernel-based metric.

- **CORAL** (Correlation Alignment): Aligns second-order statistics (covariance)
  of source and target feature distributions.

Modules
-------
data_loader
    Data ingestion from exchanges and simulated sources, plus feature engineering
    for technical indicators used as model inputs.

domain_adapter
    Neural network architectures and training routines for DANN, MMD, and CORAL
    domain adaptation approaches.

backtest
    Event-driven backtesting engine for evaluating adapted models on historical
    price data with realistic transaction costs and risk controls.

Example
-------
>>> from domain_adaptation_finance import (
...     SimulatedDataGenerator, FeatureGenerator,
...     DomainAdaptationModel, DANNTrainer,
...     BacktestEngine, BacktestConfig,
... )
>>> gen = SimulatedDataGenerator()
>>> source_klines = gen.generate_stock_data(500)
>>> target_klines = gen.generate_crypto_data(500)
"""

__version__ = "0.1.0"

from .data_loader import (
    Kline,
    BybitClient,
    SimulatedDataGenerator,
    FeatureGenerator,
    klines_to_dataframe,
)

from .domain_adapter import (
    FeatureExtractor,
    LabelPredictor,
    DomainClassifier,
    DomainAdaptationModel,
    GradientReversalFunction,
    DANNTrainer,
    MMDAdapter,
    CORALAdapter,
)

from .backtest import (
    Trade,
    BacktestConfig,
    BacktestResults,
    BacktestEngine,
)

__all__ = [
    # data_loader
    "Kline",
    "BybitClient",
    "SimulatedDataGenerator",
    "FeatureGenerator",
    "klines_to_dataframe",
    # domain_adapter
    "FeatureExtractor",
    "LabelPredictor",
    "DomainClassifier",
    "DomainAdaptationModel",
    "GradientReversalFunction",
    "DANNTrainer",
    "MMDAdapter",
    "CORALAdapter",
    # backtest
    "Trade",
    "BacktestConfig",
    "BacktestResults",
    "BacktestEngine",
]
