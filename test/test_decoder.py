import math

import pytest
import torch

from src.model.decoder import (
    _expected_log_determinant,
    _expected_log_mixture,
    _expected_mahalanobis,
)


def test_expected_mahalanobis() -> None:
    """Test the expected mahalanobis distance function."""

    def _compute_ground_truth(
        value: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor,
        scale: torch.Tensor,
        nu: torch.Tensor,
    ) -> torch.Tensor:
        n_dim = value.size(-1)
        diff = value.unsqueeze(-2) - mu.unsqueeze(0)  # shape: (N, K, D)
        mahalanobis: torch.Tensor = torch.matmul(
            diff.unsqueeze(-2),
            torch.matmul(scale.unsqueeze(0), diff.unsqueeze(-1)),
        )
        mahalanobis = nu.unsqueeze(0) * mahalanobis.squeeze((-2, -1))
        out = n_dim / beta.unsqueeze(0) + mahalanobis

        return out

    # Test case 1: n_data = 2, n_dim = 2, n_mixture = 1
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mu = torch.tensor([[1.0, 2.0]])
    beta = torch.tensor([1.0])
    scale = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    nu = torch.tensor([1.0])
    out = _expected_mahalanobis(
        obs=value,
        eta=mu[None, ...],
        beta=beta[None, ...],
        scale_tril=torch.linalg.cholesky(scale)[None, ...],
        nu=nu[None, ...],
    )
    assert out == pytest.approx(
        _compute_ground_truth(value, mu, beta, scale, nu)
    )

    # Test case 2: n_data = 3, n_dim = 2, n_mixture = 2
    value = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    mu = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    beta = torch.tensor([1.0, 2.0])
    scale = torch.tensor([[[1.0, 0.5], [0.5, 1.0]], [[2.0, 0.3], [0.3, 2.0]]])
    nu = torch.tensor([1.0, 2.0])
    out = _expected_mahalanobis(
        obs=value,
        eta=mu[None, ...],
        beta=beta[None, ...],
        scale_tril=torch.linalg.cholesky(scale)[None, ...],
        nu=nu[None, ...],
    )
    assert out == pytest.approx(
        _compute_ground_truth(value, mu, beta, scale, nu)
    )

    # Test case 2: n_data = 5, n_dim = 3, n_mixture = 3
    value = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
        ]
    )
    mu = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    beta = torch.tensor([1.0, 2.0, 3.0])
    scale = torch.tensor(
        [
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.1], [0.3, 0.1, 1.0]],
            [[2.0, 0.5, 0.7], [0.5, 2.0, 0.1], [0.7, 0.1, 2.0]],
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        ],
    )
    nu = torch.tensor([1.0, 2.0, 3.0])
    out = _expected_mahalanobis(
        obs=value,
        eta=mu[None, ...],
        beta=beta[None, ...],
        scale_tril=torch.linalg.cholesky(scale)[None, ...],
        nu=nu[None, ...],
    )
    assert out == pytest.approx(
        _compute_ground_truth(value, mu, beta, scale, nu)
    )


def test_expected_log_determinant() -> None:
    """Test the expected log determinant function."""

    def _compute_ground_truth(
        scale: torch.Tensor, nu: torch.Tensor
    ) -> torch.Tensor:
        n_mixture, _, n_dim = scale.shape[-3:]
        out = torch.zeros((n_mixture,))
        for k, s in enumerate(scale):
            out[k] += sum(
                torch.digamma((nu[k] + 1 - i) / 2) for i in range(1, n_dim + 1)
            )
            out[k] += math.log(2) * n_dim
            sign, logabsdet = torch.linalg.slogdet(s)
            out[k] += sign * logabsdet

        return out

    # Test case 1: n_mixture = 1, n_dim = 2
    scale = torch.tensor([[[1.0, 0.5], [0.5, 1.0]]])
    nu = torch.tensor([2.0])
    out = _expected_log_determinant(
        scale_tril=torch.linalg.cholesky(scale), nu=nu
    )
    assert out == pytest.approx(_compute_ground_truth(scale, nu))

    # Test case 2: n_mixture = 2, n_dim = 3
    scale = torch.tensor(
        [
            [[1.0, 0.5, 0.3], [0.5, 1.0, 0.1], [0.3, 0.1, 1.0]],
            [[2.0, 0.5, 0.7], [0.5, 2.0, 0.1], [0.7, 0.1, 2.0]],
        ]
    )
    nu = torch.tensor([4.0, 5.0])
    out = _expected_log_determinant(
        scale_tril=torch.linalg.cholesky(scale), nu=nu
    )
    assert out == pytest.approx(_compute_ground_truth(scale, nu))


def test_expected_log_mixture() -> None:
    """Test the expected log mixture function."""

    def _compute_ground_truth(alpha: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(alpha)
        for ind, alpha_val in enumerate(alpha):
            out[ind] += torch.digamma(alpha_val) - torch.digamma(
                alpha.sum(axis=-1)
            )

        return out

    # Test case 1: n_mixture = 1
    alpha = torch.tensor([0.5])
    out = _expected_log_mixture(alpha=alpha)
    assert out == pytest.approx(_compute_ground_truth(alpha))

    # Test case 2: n_mixture = 2
    alpha = torch.tensor([0.5, 0.3])
    out = _expected_log_mixture(alpha=alpha)
    assert out == pytest.approx(_compute_ground_truth(alpha))

    # Test case 3: n_mixture = 3
    alpha = torch.tensor([0.5, 0.3, 0.2])
    out = _expected_log_mixture(alpha=alpha)
    assert out == pytest.approx(_compute_ground_truth(alpha))
