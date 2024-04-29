import pytest
import torch

from src.model._functions import mvdigamma, squared_frobenius


def test_squared_frobenius() -> None:
    """Test the squared Frobenius norm function."""

    def _compute_ground_truth(value: torch.Tensor) -> torch.Tensor:
        out = torch.matmul(value, value.transpose(-2, -1))
        out = out.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=False)
        return out

    # Case 1: random tensor of shape [2, 3]
    value = torch.rand(2, 3)
    out = squared_frobenius(value=value)
    assert out == pytest.approx(_compute_ground_truth(value=value))

    # Case 2: random tensor of shape [2, 3, 3]
    value = torch.rand(2, 3, 3)
    out = squared_frobenius(value=value)
    assert out == pytest.approx(_compute_ground_truth(value=value))

    # Case 3: used for computing trace with cholesky decomposition
    lhs = torch.tensor([[[3, 1], [1, 2]], [[7, 4], [4, 6]]]).float()
    rhs = torch.tensor([[[5, 2], [2, 4]], [[9, 1], [1, 3]]]).float()
    out = squared_frobenius(
        value=torch.matmul(
            torch.linalg.cholesky(lhs, upper=True), torch.linalg.cholesky(rhs)
        )
    )
    assert out == pytest.approx(
        torch.diagonal(lhs.matmul(rhs), dim1=-2, dim2=-1).sum(dim=-1)
    )

    # Case 4: used for computing trace of inverse matrix multiplication
    out = squared_frobenius(
        value=torch.linalg.solve_triangular(
            torch.linalg.cholesky(lhs),
            torch.linalg.cholesky(rhs),
            upper=False,
        )
    )
    assert out == pytest.approx(
        torch.linalg.inv(lhs)
        .matmul(rhs)
        .diagonal(dim1=-2, dim2=-1)
        .sum(dim=-1)
    )


def test_mvdigamma() -> None:
    """Test the multivariate digamma function."""

    def _compute_ground_truth(value: torch.Tensor, dim: int) -> torch.Tensor:
        out = torch.zeros_like(value)
        for ind, val in enumerate(value):
            for i in range(1, int(dim + 1)):
                out[ind] += torch.digamma(val + 0.5 * (1 - i))

        return out

    # Test exception
    with pytest.raises(Exception):
        mvdigamma(value=torch.Tensor([1.0]), dim=0.0)
        mvdigamma(value=torch.Tensor([1.0]), dim=2.0)

    # Case 1: value = [4.0],  dim = 2.0
    value = torch.Tensor([4.0])
    out = mvdigamma(value=value, dim=2.0)
    assert out == pytest.approx(_compute_ground_truth(value=value, dim=2.0))

    # Case 2: value = [2.0, 3.0, 4.0],  dim = 3.0
    value = torch.Tensor([2.0, 3.0, 4.0])
    out = mvdigamma(value=value, dim=3.0)
    assert out == pytest.approx(_compute_ground_truth(value=value, dim=3.0))

    # Case 3: value = [3.0, 4.0],  dim = 4.0
    value = torch.Tensor([3.0, 4.0])
    out = mvdigamma(value=value, dim=4.0)
    assert out == pytest.approx(_compute_ground_truth(value=value, dim=4.0))
