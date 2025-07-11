"""
Domain Adaptation Models for Financial Transfer Learning
=========================================================

Implements three domain adaptation strategies for transferring trading models
across financial regimes (e.g. equities to crypto, developed to emerging
markets, low-vol to high-vol periods):

1. **DANN** -- Domain-Adversarial Neural Network (Ganin et al., 2016).
   A gradient reversal layer forces the shared feature extractor to learn
   representations that are discriminative for the label task but invariant
   to the source/target domain.

2. **MMD** -- Maximum Mean Discrepancy adaptation.
   A kernel-based distributional distance is added as a regulariser to
   align source and target feature distributions in RKHS.

3. **CORAL** -- Correlation Alignment (Sun & Saenko, 2016).
   Matches second-order statistics (covariance matrices) of source and
   target features so that learned representations share similar
   correlation structure.

All three methods share the same :class:`FeatureExtractor` /
:class:`LabelPredictor` backbone so results are directly comparable.

Classes
-------
FeatureExtractor
    Shared trunk that maps raw features to a hidden representation.
LabelPredictor
    Classification head (long/short signal prediction).
DomainClassifier
    Binary head that predicts source vs. target domain.
GradientReversalFunction
    Autograd function that reverses gradients during backpropagation.
DomainAdaptationModel
    Full DANN architecture combining the three heads above.
DANNTrainer
    End-to-end training loop for the DANN approach.
MMDAdapter
    Training loop with MMD-based feature alignment.
CORALAdapter
    Training loop with CORAL-based covariance alignment.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Neural-network building blocks
# ---------------------------------------------------------------------------

class FeatureExtractor(nn.Module):
    """Shared feature extraction trunk.

    Architecture: ``input -> Linear -> BN -> ReLU -> Linear -> BN -> ReLU``

    Parameters
    ----------
    input_size : int
        Dimensionality of the raw feature vector (e.g. 7 technical
        indicators).
    hidden_size : int
        Width of the hidden layers and the output representation.
    """

    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(batch, input_size)``.

        Returns
        -------
        torch.Tensor
            Feature tensor of shape ``(batch, hidden_size)``.
        """
        return self.net(x)


class LabelPredictor(nn.Module):
    """Task-specific prediction head (binary classification).

    Parameters
    ----------
    hidden_size : int
        Size of the feature vector coming from :class:`FeatureExtractor`.
    output_size : int
        Number of output classes.  For a long/short signal use ``2``;
        for a single-sigmoid probability use ``1``.
    """

    def __init__(self, hidden_size: int = 64, output_size: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict labels from features.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(batch, hidden_size)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(batch, output_size)``.
        """
        return self.net(features)


class DomainClassifier(nn.Module):
    """Domain discrimination head.

    Outputs a single logit representing the probability that the input
    belongs to the *source* domain.

    Parameters
    ----------
    hidden_size : int
        Width of the incoming feature vector.
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Classify domain.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(batch, hidden_size)``.

        Returns
        -------
        torch.Tensor
            Scalar logit of shape ``(batch, 1)``.
        """
        return self.net(features)


# ---------------------------------------------------------------------------
# Gradient reversal
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """Gradient Reversal Layer (GRL).

    During the forward pass the input is passed through unchanged.  During
    backpropagation the gradient is multiplied by ``-lambda_`` so that the
    upstream feature extractor receives *adversarial* gradients from the
    domain classifier.

    This is the core mechanism of DANN: by reversing the domain-classification
    gradient, the feature extractor is encouraged to produce representations
    that *confuse* the domain classifier -- i.e. domain-invariant features.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        """Identity forward pass; stores ``lambda_`` for backward.

        Parameters
        ----------
        ctx : FunctionCtx
            Autograd context object.
        x : torch.Tensor
            Input tensor.
        lambda_ : float
            Scaling factor for the reversed gradient.

        Returns
        -------
        torch.Tensor
            The unchanged input ``x``.
        """
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        """Reverse the gradient, scaled by ``-lambda_``.

        Parameters
        ----------
        ctx : FunctionCtx
            Contains the stored ``lambda_``.
        grad_output : torch.Tensor
            Incoming gradient from downstream.

        Returns
        -------
        Tuple[torch.Tensor, None]
            Negated and scaled gradient; ``None`` for ``lambda_`` (no grad).
        """
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    """Functional interface for :class:`GradientReversalFunction`."""
    return GradientReversalFunction.apply(x, lambda_)


# ---------------------------------------------------------------------------
# Full DANN model
# ---------------------------------------------------------------------------

class DomainAdaptationModel(nn.Module):
    """Domain-Adversarial Neural Network (DANN) for financial data.

    Combines a shared :class:`FeatureExtractor`, a :class:`LabelPredictor`,
    and a :class:`DomainClassifier` connected via a gradient reversal layer.

    Parameters
    ----------
    input_size : int
        Raw feature dimensionality.
    hidden_size : int, default 64
        Width of the shared hidden representation.
    output_size : int, default 2
        Number of label classes (2 = long / short).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        output_size: int = 2,
    ) -> None:
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_size, hidden_size)
        self.label_predictor = LabelPredictor(hidden_size, output_size)
        self.domain_classifier = DomainClassifier(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all three heads.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_size)``.
        alpha : float, default 1.0
            Gradient-reversal scaling factor.  Typically annealed from 0 to 1
            during training.

        Returns
        -------
        label_pred : torch.Tensor
            Label logits of shape ``(batch, output_size)``.
        domain_pred : torch.Tensor
            Domain logit of shape ``(batch, 1)``.
        """
        features = self.feature_extractor(x)
        label_pred = self.label_predictor(features)

        reversed_features = gradient_reversal(features, alpha)
        domain_pred = self.domain_classifier(reversed_features)

        return label_pred, domain_pred

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the shared feature representation (no prediction heads).

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_size)``.

        Returns
        -------
        torch.Tensor
            Hidden features of shape ``(batch, hidden_size)``.
        """
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
        return features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: return predicted class probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(batch, input_size)``.

        Returns
        -------
        torch.Tensor
            Softmax probabilities of shape ``(batch, output_size)``.
        """
        self.eval()
        with torch.no_grad():
            features = self.feature_extractor(x)
            logits = self.label_predictor(features)
            probs = torch.softmax(logits, dim=1)
        return probs


# ---------------------------------------------------------------------------
# DANN Trainer
# ---------------------------------------------------------------------------

class DANNTrainer:
    """Training loop for the DANN domain adaptation model.

    Uses a linear schedule to anneal ``alpha`` (the GRL weight) from 0 to 1
    over the course of training, following the original Ganin et al. recipe.

    Parameters
    ----------
    model : DomainAdaptationModel
        The DANN model to train.
    lr : float, default 1e-3
        Learning rate for Adam.
    lambda_domain : float, default 1.0
        Weighting factor for the domain classification loss relative to
        the label prediction loss.
    device : str, default ``"cpu"``
        Torch device string.
    """

    def __init__(
        self,
        model: DomainAdaptationModel,
        lr: float = 1e-3,
        lambda_domain: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.lambda_domain = lambda_domain
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.label_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.BCEWithLogitsLoss()

    # ---- public ----------------------------------------------------------

    def train_epoch(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        alpha: float = 1.0,
    ) -> Dict[str, float]:
        """Run one training epoch over paired source/target mini-batches.

        Parameters
        ----------
        source_loader : DataLoader
            Yields ``(features, labels)`` tuples for the source domain.
        target_loader : DataLoader
            Yields ``(features,)`` tuples for the target domain (no labels).
        alpha : float, default 1.0
            Current gradient-reversal scaling factor.

        Returns
        -------
        Dict[str, float]
            Dictionary with keys ``label_loss``, ``domain_loss``, and
            ``total_loss`` (averaged over mini-batches).
        """
        self.model.train()
        running_label = 0.0
        running_domain = 0.0
        running_total = 0.0
        n_batches = 0

        target_iter = iter(target_loader)

        for source_batch in source_loader:
            source_x, source_y = source_batch[0].to(self.device), source_batch[1].to(self.device)

            # Cycle through target data if it's shorter
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)

            target_x = target_batch[0].to(self.device)

            # Combine source + target for domain classification
            combined_x = torch.cat([source_x, target_x], dim=0)

            # Domain labels: source=1, target=0
            domain_labels = torch.cat([
                torch.ones(source_x.size(0), 1),
                torch.zeros(target_x.size(0), 1),
            ]).to(self.device)

            # Forward pass
            label_pred, domain_pred = self.model(combined_x, alpha=alpha)

            # Label loss (source only)
            source_label_pred = label_pred[: source_x.size(0)]
            label_loss = self.label_criterion(source_label_pred, source_y.long())

            # Domain loss (both domains)
            domain_loss = self.domain_criterion(domain_pred, domain_labels)

            # Combined loss
            total_loss = label_loss + self.lambda_domain * domain_loss

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_label += label_loss.item()
            running_domain += domain_loss.item()
            running_total += total_loss.item()
            n_batches += 1

        n_batches = max(n_batches, 1)
        return {
            "label_loss": running_label / n_batches,
            "domain_loss": running_domain / n_batches,
            "total_loss": running_total / n_batches,
        }

    def train(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        target_data: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 64,
    ) -> List[Dict[str, float]]:
        """Full training run with GRL annealing.

        Parameters
        ----------
        source_data : np.ndarray
            Source domain features, shape ``(n_source, n_features)``.
        source_labels : np.ndarray
            Source domain labels, shape ``(n_source,)``.
        target_data : np.ndarray
            Target domain features, shape ``(n_target, n_features)``.
        n_epochs : int, default 50
            Number of training epochs.
        batch_size : int, default 64
            Mini-batch size.

        Returns
        -------
        List[Dict[str, float]]
            Per-epoch loss dictionaries (see :meth:`train_epoch`).
        """
        source_dataset = TensorDataset(
            torch.tensor(source_data, dtype=torch.float32),
            torch.tensor(source_labels, dtype=torch.float32),
        )
        target_dataset = TensorDataset(
            torch.tensor(target_data, dtype=torch.float32),
        )

        source_loader = DataLoader(
            source_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )
        target_loader = DataLoader(
            target_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        history: List[Dict[str, float]] = []

        for epoch in range(n_epochs):
            # Linear annealing of alpha from 0 -> 1
            progress = epoch / max(n_epochs - 1, 1)
            alpha = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0  # sigmoid schedule

            epoch_losses = self.train_epoch(source_loader, target_loader, alpha=alpha)
            epoch_losses["alpha"] = alpha
            epoch_losses["epoch"] = epoch
            history.append(epoch_losses)

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(
                    f"  [DANN] Epoch {epoch + 1:>3d}/{n_epochs}  "
                    f"label={epoch_losses['label_loss']:.4f}  "
                    f"domain={epoch_losses['domain_loss']:.4f}  "
                    f"alpha={alpha:.3f}"
                )

        return history

    def evaluate(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate label prediction accuracy and loss.

        Parameters
        ----------
        data : np.ndarray
            Feature matrix, shape ``(n_samples, n_features)``.
        labels : np.ndarray
            Ground-truth labels, shape ``(n_samples,)``.

        Returns
        -------
        Dict[str, float]
            Contains ``accuracy``, ``loss``, ``precision``, ``recall``.
        """
        self.model.eval()
        x = torch.tensor(data, dtype=torch.float32).to(self.device)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)

        with torch.no_grad():
            features = self.model.feature_extractor(x)
            logits = self.model.label_predictor(features)
            loss = self.label_criterion(logits, y).item()
            preds = torch.argmax(logits, dim=1)
            correct = (preds == y).float()
            accuracy = correct.mean().item()

            # Per-class metrics for binary case
            tp = ((preds == 1) & (y == 1)).float().sum().item()
            fp = ((preds == 1) & (y == 0)).float().sum().item()
            fn = ((preds == 0) & (y == 1)).float().sum().item()

            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)

        return {
            "accuracy": accuracy,
            "loss": loss,
            "precision": precision,
            "recall": recall,
        }


# ---------------------------------------------------------------------------
# MMD Adapter
# ---------------------------------------------------------------------------

class MMDAdapter:
    """Maximum Mean Discrepancy (MMD) domain adaptation.

    Adds a Gaussian-kernel MMD penalty between source and target feature
    distributions to the label prediction loss.  This encourages the feature
    extractor to produce representations whose distributions match across
    domains without requiring an explicit domain classifier.

    Parameters
    ----------
    model : DomainAdaptationModel
        A DANN model (only the feature extractor and label predictor are
        used; the domain classifier is ignored).
    kernel_bandwidth : float, default 1.0
        Bandwidth parameter ``sigma`` for the Gaussian kernel.
    mmd_weight : float, default 1.0
        Weighting factor for the MMD loss relative to the label loss.
    lr : float, default 1e-3
        Learning rate for Adam.
    device : str, default ``"cpu"``
        Torch device string.
    """

    def __init__(
        self,
        model: DomainAdaptationModel,
        kernel_bandwidth: float = 1.0,
        mmd_weight: float = 1.0,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.kernel_bandwidth = kernel_bandwidth
        self.mmd_weight = mmd_weight

        # Only optimise feature extractor + label predictor
        params = list(model.feature_extractor.parameters()) + list(
            model.label_predictor.parameters()
        )
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
        self.label_criterion = nn.CrossEntropyLoss()

    # ---- kernel & MMD ----------------------------------------------------

    def gaussian_kernel(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Gaussian (RBF) kernel matrix between two sets of samples.

        .. math::

            K(x, y) = \\exp\\!\\left(-\\frac{\\|x - y\\|^2}{2\\sigma^2}\\right)

        Parameters
        ----------
        source : torch.Tensor
            Shape ``(n, d)``.
        target : torch.Tensor
            Shape ``(m, d)``.

        Returns
        -------
        torch.Tensor
            Kernel matrix of shape ``(n, m)``.
        """
        n = source.size(0)
        m = target.size(0)
        # Pairwise squared distances
        source_sq = source.pow(2).sum(dim=1, keepdim=True)  # (n, 1)
        target_sq = target.pow(2).sum(dim=1, keepdim=True)  # (m, 1)
        dist = source_sq + target_sq.t() - 2.0 * source @ target.t()  # (n, m)
        return torch.exp(-dist / (2.0 * self.kernel_bandwidth ** 2))

    def compute_mmd(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Unbiased estimate of squared MMD.

        .. math::

            \\text{MMD}^2 = \\frac{1}{n^2} \\sum K(x_s, x_s')
                          + \\frac{1}{m^2} \\sum K(x_t, x_t')
                          - \\frac{2}{nm} \\sum K(x_s, x_t)

        Parameters
        ----------
        source_features : torch.Tensor
            Shape ``(n, d)``.
        target_features : torch.Tensor
            Shape ``(m, d)``.

        Returns
        -------
        torch.Tensor
            Scalar MMD^2 loss.
        """
        k_ss = self.gaussian_kernel(source_features, source_features)
        k_tt = self.gaussian_kernel(target_features, target_features)
        k_st = self.gaussian_kernel(source_features, target_features)

        n = source_features.size(0)
        m = target_features.size(0)

        mmd = k_ss.sum() / (n * n) + k_tt.sum() / (m * m) - 2.0 * k_st.sum() / (n * m)
        return mmd

    # ---- training --------------------------------------------------------

    def train(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        target_data: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 64,
    ) -> List[Dict[str, float]]:
        """Train with MMD regularisation.

        Parameters
        ----------
        source_data : np.ndarray
            Source features, shape ``(n_source, n_features)``.
        source_labels : np.ndarray
            Source labels, shape ``(n_source,)``.
        target_data : np.ndarray
            Target features (unlabelled), shape ``(n_target, n_features)``.
        n_epochs : int, default 50
            Number of epochs.
        batch_size : int, default 64
            Mini-batch size.

        Returns
        -------
        List[Dict[str, float]]
            Per-epoch loss history.
        """
        source_dataset = TensorDataset(
            torch.tensor(source_data, dtype=torch.float32),
            torch.tensor(source_labels, dtype=torch.float32),
        )
        target_dataset = TensorDataset(
            torch.tensor(target_data, dtype=torch.float32),
        )

        source_loader = DataLoader(
            source_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )
        target_loader = DataLoader(
            target_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        history: List[Dict[str, float]] = []

        for epoch in range(n_epochs):
            self.model.train()
            running_label = 0.0
            running_mmd = 0.0
            running_total = 0.0
            n_batches = 0

            target_iter = iter(target_loader)

            for source_batch in source_loader:
                source_x = source_batch[0].to(self.device)
                source_y = source_batch[1].to(self.device).long()

                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)

                target_x = target_batch[0].to(self.device)

                # Extract features
                source_feat = self.model.feature_extractor(source_x)
                target_feat = self.model.feature_extractor(target_x)

                # Label loss (source only)
                label_logits = self.model.label_predictor(source_feat)
                label_loss = self.label_criterion(label_logits, source_y)

                # MMD loss
                mmd_loss = self.compute_mmd(source_feat, target_feat)

                total_loss = label_loss + self.mmd_weight * mmd_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_label += label_loss.item()
                running_mmd += mmd_loss.item()
                running_total += total_loss.item()
                n_batches += 1

            n_batches = max(n_batches, 1)
            epoch_info = {
                "epoch": epoch,
                "label_loss": running_label / n_batches,
                "mmd_loss": running_mmd / n_batches,
                "total_loss": running_total / n_batches,
            }
            history.append(epoch_info)

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(
                    f"  [MMD]  Epoch {epoch + 1:>3d}/{n_epochs}  "
                    f"label={epoch_info['label_loss']:.4f}  "
                    f"mmd={epoch_info['mmd_loss']:.4f}"
                )

        return history


# ---------------------------------------------------------------------------
# CORAL Adapter
# ---------------------------------------------------------------------------

class CORALAdapter:
    """Correlation Alignment (CORAL) domain adaptation.

    Aligns second-order statistics (covariance matrices) of source and target
    feature distributions.  The CORAL loss is the squared Frobenius norm of
    the difference between the two covariance matrices, normalised by the
    feature dimensionality.

    Parameters
    ----------
    model : DomainAdaptationModel
        Model whose feature extractor and label predictor will be trained.
    coral_weight : float, default 1.0
        Weighting factor for the CORAL loss.
    lr : float, default 1e-3
        Learning rate for Adam.
    device : str, default ``"cpu"``
        Torch device string.
    """

    def __init__(
        self,
        model: DomainAdaptationModel,
        coral_weight: float = 1.0,
        lr: float = 1e-3,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.device = torch.device(device)
        self.coral_weight = coral_weight

        params = list(model.feature_extractor.parameters()) + list(
            model.label_predictor.parameters()
        )
        self.optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
        self.label_criterion = nn.CrossEntropyLoss()

    # ---- CORAL loss ------------------------------------------------------

    @staticmethod
    def compute_covariance(features: torch.Tensor) -> torch.Tensor:
        """Compute the sample covariance matrix of a feature batch.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(n, d)`` where ``n`` is batch size and ``d`` is feature
            dimensionality.

        Returns
        -------
        torch.Tensor
            Covariance matrix of shape ``(d, d)``.
        """
        n = features.size(0)
        mean = features.mean(dim=0, keepdim=True)
        centered = features - mean
        cov = (centered.t() @ centered) / max(n - 1, 1)
        return cov

    def coral_loss(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the CORAL loss between source and target features.

        .. math::

            L_{CORAL} = \\frac{1}{4d^2} \\|C_S - C_T\\|_F^2

        Parameters
        ----------
        source_features : torch.Tensor
            Shape ``(n_source, d)``.
        target_features : torch.Tensor
            Shape ``(n_target, d)``.

        Returns
        -------
        torch.Tensor
            Scalar CORAL loss.
        """
        d = source_features.size(1)
        cov_source = self.compute_covariance(source_features)
        cov_target = self.compute_covariance(target_features)

        diff = cov_source - cov_target
        loss = (diff * diff).sum() / (4.0 * d * d)
        return loss

    # ---- training --------------------------------------------------------

    def train(
        self,
        source_data: np.ndarray,
        source_labels: np.ndarray,
        target_data: np.ndarray,
        n_epochs: int = 50,
        batch_size: int = 64,
    ) -> List[Dict[str, float]]:
        """Train with CORAL regularisation.

        Parameters
        ----------
        source_data : np.ndarray
            Source features, shape ``(n_source, n_features)``.
        source_labels : np.ndarray
            Source labels, shape ``(n_source,)``.
        target_data : np.ndarray
            Target features (unlabelled), shape ``(n_target, n_features)``.
        n_epochs : int, default 50
            Number of epochs.
        batch_size : int, default 64
            Mini-batch size.

        Returns
        -------
        List[Dict[str, float]]
            Per-epoch loss history.
        """
        source_dataset = TensorDataset(
            torch.tensor(source_data, dtype=torch.float32),
            torch.tensor(source_labels, dtype=torch.float32),
        )
        target_dataset = TensorDataset(
            torch.tensor(target_data, dtype=torch.float32),
        )

        source_loader = DataLoader(
            source_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )
        target_loader = DataLoader(
            target_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
        )

        history: List[Dict[str, float]] = []

        for epoch in range(n_epochs):
            self.model.train()
            running_label = 0.0
            running_coral = 0.0
            running_total = 0.0
            n_batches = 0

            target_iter = iter(target_loader)

            for source_batch in source_loader:
                source_x = source_batch[0].to(self.device)
                source_y = source_batch[1].to(self.device).long()

                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)

                target_x = target_batch[0].to(self.device)

                # Feature extraction
                source_feat = self.model.feature_extractor(source_x)
                target_feat = self.model.feature_extractor(target_x)

                # Label loss
                label_logits = self.model.label_predictor(source_feat)
                label_loss = self.label_criterion(label_logits, source_y)

                # CORAL loss
                c_loss = self.coral_loss(source_feat, target_feat)

                total_loss = label_loss + self.coral_weight * c_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_label += label_loss.item()
                running_coral += c_loss.item()
                running_total += total_loss.item()
                n_batches += 1

            n_batches = max(n_batches, 1)
            epoch_info = {
                "epoch": epoch,
                "label_loss": running_label / n_batches,
                "coral_loss": running_coral / n_batches,
                "total_loss": running_total / n_batches,
            }
            history.append(epoch_info)

            if (epoch + 1) % max(1, n_epochs // 5) == 0:
                print(
                    f"  [CORAL] Epoch {epoch + 1:>3d}/{n_epochs}  "
                    f"label={epoch_info['label_loss']:.4f}  "
                    f"coral={epoch_info['coral_loss']:.4f}"
                )

        return history
