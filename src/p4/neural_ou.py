"""Neural pair discovery and OU estimation helpers.

These models are intentionally small. The autoencoder is used as a nonlinear
dimensionality reduction step for cross-sectional clustering, while the neural
OU estimator is a lightweight sequence model anchored by classical OU dynamics.
It should be treated as a flexible estimator, not as evidence of superior
forecasting power.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F

from p4.ou_estimator import OUEstimator


def _preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _seed_everything(seed: int = 0) -> np.random.Generator:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)


class PairDiscoveryAutoencoder(nn.Module):
    """Small fully connected autoencoder for cross-sectional pair discovery."""

    def __init__(self, input_dim: int, latent_dim: int = 2) -> None:
        super().__init__()
        if input_dim < 4:
            raise ValueError("input_dim must be at least 4.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")

        hidden_1 = max(12, min(32, input_dim // 8))
        hidden_2 = max(6, min(16, hidden_1 // 2))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, input_dim),
        )
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))

    def set_normalization(self, feature_mean: torch.Tensor, feature_std: torch.Tensor) -> None:
        self.feature_mean.copy_(feature_mean)
        self.feature_std.copy_(torch.clamp(feature_std, min=1e-6))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.feature_mean) / self.feature_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.feature_std + self.feature_mean

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def reconstruct_asset_matrix(self, asset_matrix: NDArray | torch.Tensor) -> NDArray[np.float64]:
        self.eval()
        tensor = torch.as_tensor(asset_matrix, dtype=torch.float32, device=self.feature_mean.device)
        with torch.no_grad():
            normalized = self.normalize(tensor)
            reconstructed = self.denormalize(self.forward(normalized))
        return reconstructed.detach().cpu().numpy().astype(float)


def train_autoencoder(
    returns_matrix: NDArray,
    latent_dim: int = 2,
    epochs: int = 200,
    lr: float = 1e-3,
) -> tuple[PairDiscoveryAutoencoder, NDArray[np.float64]]:
    """Train the autoencoder on asset return histories and return latent coordinates."""

    returns = np.asarray(returns_matrix, dtype=float)
    if returns.ndim != 2:
        raise ValueError("returns_matrix must be a 2-D array of shape [time, assets].")
    if returns.shape[0] < 20 or returns.shape[1] < 2:
        raise ValueError("returns_matrix must contain at least 20 observations and 2 assets.")
    if epochs < 1 or lr <= 0.0:
        raise ValueError("epochs must be positive and lr must be > 0.")

    _seed_everything(0)
    device = _preferred_device()
    asset_matrix = returns.T.astype(np.float32)
    model = PairDiscoveryAutoencoder(input_dim=asset_matrix.shape[1], latent_dim=latent_dim).to(device)

    feature_mean = torch.as_tensor(asset_matrix.mean(axis=0), dtype=torch.float32, device=device)
    feature_std = torch.as_tensor(asset_matrix.std(axis=0), dtype=torch.float32, device=device)
    model.set_normalization(feature_mean, feature_std)

    inputs = torch.as_tensor(asset_matrix, dtype=torch.float32, device=device)
    normalized_inputs = model.normalize(inputs)
    sample_weights = torch.as_tensor(asset_matrix.std(axis=1), dtype=torch.float32, device=device).unsqueeze(1)
    sample_weights = sample_weights / torch.clamp(sample_weights.mean(), min=1e-6)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        reconstructed = model(normalized_inputs)
        error = reconstructed - normalized_inputs
        mse_loss = F.mse_loss(reconstructed, normalized_inputs)
        weighted_penalty = (error.pow(2) * sample_weights).mean()
        loss = mse_loss + 0.25 * weighted_penalty
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        latent = model.encode(normalized_inputs).detach().cpu().numpy().astype(float)
    return model, latent


def discover_spreads_via_clustering(latent_coords: NDArray, n_clusters: int = 10) -> list[tuple[int, int]]:
    """Cluster latent coordinates and select one nearest-neighbour pair per cluster."""

    coords = np.asarray(latent_coords, dtype=float)
    if coords.ndim != 2:
        raise ValueError("latent_coords must be a 2-D array.")
    if coords.shape[0] < 2:
        return []
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")

    n_clusters = min(int(n_clusters), coords.shape[0])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(coords)

    pairs: list[tuple[int, int]] = []
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(labels == cluster_id)
        if members.size < 2:
            continue
        best_pair: tuple[int, int] | None = None
        best_distance = float("inf")
        for idx in range(members.size - 1):
            for jdx in range(idx + 1, members.size):
                left = int(members[idx])
                right = int(members[jdx])
                distance = float(np.linalg.norm(coords[left] - coords[right]))
                if distance < best_distance:
                    best_distance = distance
                    best_pair = tuple(sorted((left, right)))
        if best_pair is not None:
            pairs.append(best_pair)
    return pairs


class NeuralOUEstimator(nn.Module):
    """Tiny LSTM-based OU parameter regressor."""

    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim == 2:
            sequence = sequence.unsqueeze(-1)
        _, (hidden, _) = self.lstm(sequence)
        raw = self.head(hidden[-1])
        theta = F.softplus(raw[..., 0]) + 1e-4
        mu = raw[..., 1]
        sigma = F.softplus(raw[..., 2]) + 1e-4
        return torch.stack((theta, mu, sigma), dim=-1)


def _simulate_ou_batch(
    *,
    length: int,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    theta = rng.uniform(0.03, 0.35, size=n_paths).astype(np.float32)
    mu = rng.uniform(-1.0, 1.0, size=n_paths).astype(np.float32)
    sigma = rng.uniform(0.05, 0.5, size=n_paths).astype(np.float32)
    phi = np.exp(-theta)
    innovation_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta)) / np.maximum(2.0 * theta, 1e-6))

    paths = np.zeros((n_paths, length), dtype=np.float32)
    paths[:, 0] = mu + rng.normal(scale=sigma, size=n_paths).astype(np.float32)
    for t in range(1, length):
        shock = rng.normal(scale=innovation_std, size=n_paths).astype(np.float32)
        paths[:, t] = mu + phi * (paths[:, t - 1] - mu) + shock

    params = np.column_stack([theta, mu, sigma]).astype(np.float32)
    return paths, params


def fit_neural_ou(spread: pd.Series, epochs: int = 100) -> dict[str, float | str]:
    """Estimate OU parameters with a small neural model anchored by classical OU fit."""

    if epochs < 1:
        raise ValueError("epochs must be positive.")
    clean = pd.Series(spread, dtype=float).dropna()
    if len(clean) < 50:
        raise ValueError("spread must contain at least 50 observations.")

    rng = _seed_everything(0)
    device = _preferred_device()
    model = NeuralOUEstimator(hidden_size=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    pretrain_epochs = max(10, min(40, epochs // 2))
    synthetic_paths, synthetic_params = _simulate_ou_batch(length=len(clean), n_paths=96, rng=rng)
    synthetic_x = torch.as_tensor(synthetic_paths[:, :, None], dtype=torch.float32, device=device)
    synthetic_y = torch.as_tensor(synthetic_params, dtype=torch.float32, device=device)

    model.train()
    for _ in range(pretrain_epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(synthetic_x)
        loss = (
            F.mse_loss(pred[:, 0], synthetic_y[:, 0])
            + 0.5 * F.mse_loss(pred[:, 1], synthetic_y[:, 1])
            + F.mse_loss(pred[:, 2], synthetic_y[:, 2])
        )
        loss.backward()
        optimizer.step()

    anchor = OUEstimator().fit(clean)
    anchor_tensor = torch.as_tensor(
        [anchor["kappa"], anchor["mu"], anchor["sigma"]],
        dtype=torch.float32,
        device=device,
    )
    scale_tensor = torch.as_tensor(
        [
            max(float(anchor["kappa"]), 1e-3),
            max(abs(float(anchor["mu"])), 1.0),
            max(float(anchor["sigma"]), 1e-3),
        ],
        dtype=torch.float32,
        device=device,
    )
    observed = torch.as_tensor(clean.to_numpy(dtype=np.float32)[None, :, None], dtype=torch.float32, device=device)
    x_prev = observed[:, :-1, 0]
    x_next = observed[:, 1:, 0]

    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        pred = model(observed)[0]
        theta, mu, sigma = pred
        phi = torch.exp(-theta)
        conditional_mean = mu + phi * (x_prev - mu)
        innovation_var = sigma.pow(2) * (1.0 - torch.exp(-2.0 * theta)) / (2.0 * theta + 1e-6)
        residual = x_next - conditional_mean

        dynamics_loss = residual.pow(2).mean() + (residual.var(unbiased=False) - innovation_var).pow(2)
        anchor_loss = (((pred - anchor_tensor) / scale_tensor) ** 2).mean()
        loss = dynamics_loss + 0.2 * anchor_loss
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        estimate = model(observed)[0].detach().cpu().numpy().astype(float)

    # The neural head is intentionally treated as a flexible refinement around
    # the classical OU fit rather than a free-floating estimator.
    anchor_weight = 0.75
    theta = float(
        np.exp(
            anchor_weight * math.log(max(float(anchor["kappa"]), 1e-6))
            + (1.0 - anchor_weight) * math.log(max(float(estimate[0]), 1e-6))
        )
    )
    mu = float(anchor_weight * float(anchor["mu"]) + (1.0 - anchor_weight) * float(estimate[1]))
    sigma = float(
        np.exp(
            anchor_weight * math.log(max(float(anchor["sigma"]), 1e-6))
            + (1.0 - anchor_weight) * math.log(max(float(estimate[2]), 1e-6))
        )
    )
    return {
        "theta": theta,
        "mu": mu,
        "sigma": sigma,
        "half_life": float(math.log(2.0) / max(theta, 1e-8)),
        "fitted_via": "neural_ou",
        "device": device.type,
    }
