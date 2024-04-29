# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Decoder components for trajectory planner."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import FloatTensor, Tensor, nn
from torch_geometric.nn.inits import reset

from ._constants import TIKHONOV_REGULARIZATION
from ._functions import mvdigamma
from .distributions.multivariate_student import MultivariateStudent
from .layers import MLP
from .typing import StateDict

__all__ = ["GenerativeNet", "InferenceNet", "Decoder"]


# ----------- Helper functions -----------
def _expected_mahalanobis(
    obs: Tensor, eta: Tensor, beta: Tensor, scale_tril: Tensor, nu: Tensor
) -> Tensor:
    """Computes the expected Mahalanobis distance.

    Args:
        obs (Tensor): The observation tensor of shape `[N, D]`.
        eta (Tensor): The mean tensor of shape `[N, K, D]`.
        beta (Tensor): The concentration tensor of shape `[N, K,]`.
        scale_tril (Tensor): The lower triangular Cholesky factor of the scale
            matrix of shape `[N, K, D, D]`.
        nu (Tensor): The degrees of freedom tensor of shape `[N, K]`.

    Returns:
        Tensor: The expected Mahalanobis distance of shape `[N, K]`.
    """
    # sanity checks
    n_dim = obs.size(-1)
    n_mixture = eta.size(-2)
    assert (
        eta.shape[-2:] == (n_mixture, n_dim)
        and beta.size(-1) == n_mixture
        and scale_tril.shape[-3:] == (n_mixture, n_dim, n_dim)
        and nu.size(-1) == n_mixture
    ), ValueError("Invalid input shapes.")

    if obs.ndim == eta.ndim - 1:
        # align the dimensions if missing mixture dimension
        obs = obs.unsqueeze(-2)

    # compute the Mahalanobis distance
    beta_term = n_dim / beta
    diff = obs - eta
    mahalanobis: Tensor = torch.sum(
        torch.square(
            torch.linalg.matmul(
                scale_tril.transpose(-2, -1), diff.unsqueeze(-1)
            )
        ),
        dim=(-2, -1),
    )
    mahalanobis = nu * mahalanobis

    return beta_term + mahalanobis


def _expected_log_determinant(scale_tril: Tensor, nu: Tensor) -> Tensor:
    """Computes the expected log-determinant of the precision matrix.

    Args:
        scale_tril (Tensor): The lower triangular Cholesky factor of the scale
            matrix of shape `[N, K, D, D]`.
        nu (Tensor): The degrees of freedom tensor of shape `[N, K]`.

    Returns:
        Tensor: The expected log-determinant of shape `[N, K]`.
    """
    # sanity checks
    n_mixture, n_dim, _ = scale_tril.size()[-3:]
    assert nu.shape[-1] == n_mixture, ValueError("Invalid input shapes.")

    # compute the expected log-determinant
    logdet = 2 * torch.sum(
        torch.diagonal(scale_tril, dim1=-2, dim2=-1).log(), dim=-1
    )
    return mvdigamma(nu.div(2), n_dim) + n_dim * math.log(2) + logdet


def _expected_log_mixture(alpha: Tensor) -> Tensor:
    """Computes the expected logarithm of mixture weights.

    Args:
        alpha (Tensor): The concentration tensor of shape `[N, K]`.

    Returns:
        Tensor: The expected logarithm of mixture weights of shape `[N, K]`.
    """
    # sanity checks
    assert torch.all(alpha > 0), ValueError("Invalid concentration.")

    # compute the expected log-mixture weights
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    out = torch.digamma(alpha) - torch.digamma(sum_alpha)

    return out


# ----------- Loss function components -----------
def compute_kl_divergence_normal_wishart(
    p_eta: Tensor,
    p_beta: Tensor,
    p_psi_tril: Tensor,
    p_nu: Tensor,
    q_eta: Tensor,
    q_beta: Tensor,
    q_psi_tril: Tensor,
    q_nu: Tensor,
) -> Tensor:
    n_dim = q_eta.size(-1)

    diff = p_eta - q_eta
    mahalanobis = torch.sum(
        torch.square(
            torch.linalg.matmul(
                q_psi_tril.transpose(-2, -1), diff.unsqueeze(-1)
            )
        ),
        dim=(-2, -1),
    )  # shape: (N, K)
    mahalanobis = p_beta * q_nu * mahalanobis
    logdet = (
        2
        * p_nu
        * (
            p_psi_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
            - q_psi_tril.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
        )
    )
    trace = q_nu * torch.sum(
        torch.square(
            torch.diagonal(
                torch.linalg.solve_triangular(
                    p_psi_tril, q_psi_tril, upper=False
                ),
                dim1=-2,
                dim2=-1,
            )
        ),
        dim=-1,
    )

    # compute the KL divergence between two Normal distributions
    beta_term = n_dim * (p_beta / q_beta - (p_beta.log() - q_beta.log()) - 1)
    kl_div_mean = torch.mul(beta_term + mahalanobis, 0.5)

    # compute the KL divergence between two Wishart distributions
    mvlgamma_term = 2 * (
        torch.mvlgamma(p_nu / 2, n_dim) - torch.mvlgamma(q_nu / 2, n_dim)
    )
    mvdigamma_term = (q_nu - p_nu) * mvdigamma(q_nu / 2, n_dim)
    kl_div_precision = torch.mul(
        trace + logdet + mvlgamma_term + mvdigamma_term - q_nu * n_dim, 0.5
    )

    return kl_div_mean + kl_div_precision


def compute_circular_iou(
    selected_samples: Tensor,
    other_samples: Tensor,
    selected_radius: float = 2.5,
    other_radius: Optional[float] = None,
) -> Tensor:
    """Compute the IoU between two sets of samples regarding circular buffers.

    Args:
        selected_samples (Tensor): Selected samples of shape
            :math:`[num_agents, num_samples, 2]`.
        other_samples (Tensor): Other samples of shape
            :math:`[num_agents, num_samples, 2]`.
        selected_radius (float): Radius of the circular buffer for the selected
            samples in meters. Defaults to `2.5`.
        other_radius (Optional[float]): Radius of the circular buffer for the
            other samples in meters. If `None`, use `selected_radius`.
            Defaults to `None`.

    Returns:
        Tensor: The IoU between the two sets of samples.
    """
    if other_radius is None:
        other_radius = selected_radius
    if selected_samples.ndim == other_samples.ndim - 1:
        selected_samples = selected_samples.unsqueeze(-2)

    assert selected_samples.ndim == 3 and selected_samples.size(-1) == 2
    assert other_samples.ndim == 3 and other_samples.size(-1) == 2

    d = torch.cdist(other_samples, selected_samples, p=2).squeeze(-1)
    y = 0.5 * torch.sqrt(
        (-d + selected_radius + other_radius)
        * (d + selected_radius - other_radius)
        * (d - selected_radius + other_radius)
        * (d + selected_radius + other_radius)
    )
    area_intersection = (
        selected_radius**2
        * torch.acos(
            (d.pow(2) + selected_radius**2 - other_radius**2)
            / (2 * d * selected_radius)
        )
        + other_radius**2
        * torch.acos(
            (d.pow(2) - selected_radius**2 + other_radius**2)
            / (2 * d * other_radius)
        )
        - y
    )
    area_union = (
        selected_radius**2 * math.pi
        + other_radius**2 * math.pi
        - area_intersection
    )
    iou = area_intersection / area_union

    # handle extreme cases: no intersection or full intersection
    iou[torch.isclose(d, torch.tensor(0.0))] = 1.0
    iou[d >= selected_radius + other_radius] = 0.0

    return iou


def get_posterior_predictive(
    psi: Tensor, nu: Tensor, mu: Tensor, beta: Tensor
) -> MultivariateStudent:
    """Construct the posterior predictive distribution.

    Args:
        psi (Tensor): Scale matrix for the goal precision posterior.
        nu (Tensor): Degrees of freedom for the goal precision posterior.
        mu (Tensor): Mean vector for the goal mean posterior.
        beta (Tensor): Concentration for the goal mean posterior.

    Returns:
        MultivariateStudent: Posterior predictive distribution.
    """
    # compute student-t distribution parameters
    df = nu - 1.0
    _w = (beta + 1) / beta / df
    scale = _w[..., None, None] * torch.cholesky_inverse(psi)
    ppd = MultivariateStudent(df=df, loc=mu, scale=scale)

    return ppd


class GenerativeNet(nn.Module):
    """Prior network for trajectory generation."""

    # ----------- public attributes ----------- #
    num_mixtures: int
    """int: Number of mixture components in the goal prior distribution."""

    # ----------- private attributes ----------- #
    _alpha: nn.Parameter
    """nn.Parameter: Concentrations in the symmetry Dirichlet prior."""
    _mu: nn.Parameter
    """nn.Parameter: Mean for the goal mean Normal prior."""
    _beta: nn.Parameter
    """nn.Parameter: Concentration for the goal mean Normal prior."""
    _scale_tril: nn.Parameter
    """nn.Parameter: Scale matrix parameters for the goal precision prior."""
    _nu: nn.Parameter
    """nn.Parameter: Degrees of freedom for the goal precision prior."""

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        num_mixtures: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # save arguments
        self.num_mixtures = num_mixtures

        # build goal prior
        self._mu = nn.Parameter(torch.zeros(2), requires_grad=True)
        self._scale_tril = nn.Parameter(torch.zeros(3), requires_grad=True)
        self._z_proxy = MLP(
            in_features=in_features,
            hidden_size=64,
            out_feature=num_mixtures,
            for_graph=False,
            has_norm=True,
        )

        # non-trainable parameters
        self._alpha = nn.Parameter(
            alpha * torch.ones(num_mixtures), requires_grad=False
        )
        self._prior_beta = nn.Parameter(0.01 * torch.ones(1), False)
        self._prior_nu = nn.Parameter(4 * torch.ones(1), False)

    def forward(self) -> StateDict:
        with torch.no_grad():
            psi_tril, nu = self.forward_precision()
            mu, beta = self.forward_mean()

            return {
                "alpha": self.alpha,
                "psi_tril": psi_tril,
                "nu": nu,
                "mu": mu,
                "beta": beta,
            }

    def forward_precision(self) -> Tuple[FloatTensor, FloatTensor]:
        """Forward pass of the latent goal precision prior network.

        Returns:
            Tuple[FloatTensor, FloatTensor]: Parameter psi and nu of the goal
                precision prior distribution.
        """
        # forward pass parameter psi
        a, b, c = self._scale_tril.chunk(3, dim=-1)
        z = torch.zeros_like(a)
        psi_tril = torch.stack(
            (
                torch.cat([nn.functional.softplus(a), b], -1),
                torch.cat([z, nn.functional.softplus(c)], -1),
            ),
            dim=-1,
        )

        # apply Tikhonov regularization
        _eye = torch.eye(
            psi_tril.size(-1), device=psi_tril.device, dtype=psi_tril.dtype
        )
        psi = psi_tril.matmul(psi_tril.transpose(-2, -1))
        psi_tril = torch.linalg.cholesky(psi + _eye * TIKHONOV_REGULARIZATION)

        # preprocess the nu parameter
        nu = self._prior_nu

        return psi_tril, nu

    def forward_mean(self) -> Tuple[FloatTensor, FloatTensor]:
        """Forward pass of the goal mean prior network.

        Returns:
            Tuple[FloatTensor, FloatTensor]: Parameter mu and beta of the goal
                mean prior distribution.
        """
        # forward pass
        mu = self._mu

        # preprocess the beta parameter
        beta = self._prior_beta

        return mu, beta

    @property
    def alpha(self) -> nn.Parameter:
        """nn.Parameter: Concentrations in the symmetry Dirichlet prior."""
        return nn.functional.softplus(self._alpha)


class InferenceNet(nn.Module):
    """Inference network for trajectory recognition."""

    # ----------- public attributes ----------- #
    mean_in_features: int
    """int: Input feature dimensionality for mean network."""
    precision_in_features: int
    """int: Input feature dimensionality for precision network."""
    hidden_size: int
    """int: Hidden feature dimensionality for the MLPs."""
    num_mixtures: int
    """int: Number of mixture components in the goal posterior distribution."""

    # ----------- private attributes ----------- #
    _alpha: nn.Parameter
    """nn.Parameter: Learnable concentrations for the Dirichlet posterior."""
    _mu_network: nn.ModuleDict[str, nn.Linear]
    """nn.ModuleDict: Network parameterize posterior mean vector."""
    _beta: nn.Parameter
    """nn.Parameter: Parameter beta for the goal mean posterior."""
    _scale_tril_network: nn.ModuleDict[str, nn.Linear]
    """nn.ModuleDict: Network parameterize posterior scale matrix."""
    _nu: nn.Parameter
    """nn.Parameter: Parameter nu for the goal precision posterior."""

    def __init__(
        self,
        mean_in_features: int,
        precision_in_features: int,
        hidden_size: int = 64,
        num_mixtures: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # save arguments
        self.mean_in_features = mean_in_features
        self.precision_in_features = precision_in_features
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures

        # build goal posterior network
        self._alpha = nn.Parameter(torch.randn(num_mixtures))
        self._mu_network = nn.ModuleDict(
            {
                f"mean_head_{i+1}": nn.Linear(
                    in_features=self.mean_in_features, out_features=2
                )
                for i in range(num_mixtures)
            }
        )
        self._scale_tril_network = nn.ModuleDict(
            {
                f"scale_head_{i+1}": nn.Linear(
                    in_features=self.precision_in_features, out_features=3
                )
                for i in range(num_mixtures)
            }
        )
        self._beta = nn.Parameter(torch.zeros(num_mixtures))
        self._nu = nn.Parameter(torch.zeros(num_mixtures))

        self.reset_parameters()

    def forward(
        self,
        context_x: Tensor,
        agent_x: Tensor,
        goal: Tensor,
        gen_net: GenerativeNet,
    ) -> StateDict:
        """Forward pass of the inference network.

        Args:
            context_x (Tensor): Context features of shape
                :math:`[n_agents, n_context_features]`.
            agent_x (Tensor): Agent interaction features of shape
                :math:`[n_agents, n_agent_features]`.
            goal (Tensor): Ground-truth goal of shape :math:`[n_agents, 2]`.
            gen_net (GenerativeNet): Generative network for computing the ELBO
                loss during training.

        Returns:
            StateDict: Posterior distributions.
        """
        assert self.num_mixtures == gen_net.num_mixtures, (
            "Unmatched number of mixture components between "
            "the inference and generative networks!"
        )
        x_c, x_a, goal = context_x.float(), agent_x.float(), goal.float()

        # step 1: compute and sample from posterior distributions
        q_psi_tril, q_nu = self.forward_precision(x=x_a)
        q_mu, q_beta = self.forward_mean(x=x_c)

        # step 2: apply E-step and compute the responsibilities
        e_log_pi = _expected_log_mixture(alpha=self.alpha)
        e_log_determinant = _expected_log_determinant(
            scale_tril=q_psi_tril, nu=q_nu
        )
        e_log_mahalanobis = _expected_mahalanobis(
            obs=goal, eta=q_mu, beta=q_beta, scale_tril=q_psi_tril, nu=q_nu
        )
        with torch.no_grad():
            # for E-step, detach the parameters when computing the assignment
            unnormalized_logits = e_log_pi + 0.5 * (
                e_log_determinant
                + goal.size(-1) * math.log(2 * math.pi)
                - e_log_mahalanobis
            )
            resp = nn.functional.softmax(unnormalized_logits, dim=-1)

        # step 3: apply M-step and compute the ELBO loss
        with torch.enable_grad():
            prior_params = gen_net.forward()
            reconstruction_loss = torch.sum(
                resp
                * torch.mul(
                    -e_log_determinant
                    + e_log_mahalanobis
                    + goal.size(-1) * math.log(2 * math.pi),
                    0.5,
                ),
            )
            kl_dirichlet = (
                torch.lgamma(self.alpha.sum(dim=-1))
                - torch.lgamma(self.alpha).sum(dim=-1)
                - torch.lgamma(prior_params["alpha"].sum(dim=-1))
                + torch.lgamma(prior_params["alpha"]).sum(dim=-1)
                + torch.sum(
                    (self.alpha - prior_params["alpha"])
                    * (
                        torch.digamma(self.alpha)
                        - torch.digamma(self.alpha.sum())
                    )
                )
            )
            kl_categorical = torch.sum(
                resp
                * (
                    unnormalized_logits
                    - unnormalized_logits.logsumexp(dim=-1, keepdim=True)
                    - e_log_pi
                ),
                dim=-1,
            )
            kl_normal_wishart = compute_kl_divergence_normal_wishart(
                p_eta=prior_params["mu"],
                p_beta=prior_params["beta"],
                p_psi_tril=prior_params["psi_tril"],
                p_nu=prior_params["nu"],
                q_eta=q_mu,
                q_beta=q_beta,
                q_psi_tril=q_psi_tril,
                q_nu=q_nu,
            ).sum(dim=-1)

        return {
            "reconstruction_loss": reconstruction_loss,
            "kl_div_pi": kl_dirichlet,
            "kl_div_z": kl_categorical,
            "kl_div_nw": kl_normal_wishart,
        }

    def forward_precision(self, x: Tensor) -> Tuple[FloatTensor, FloatTensor]:
        """Forward pass of the goal precision posterior network.

        Args:
            x (Tensor): Posterior input features of shape
                :math:`[n_agents, n_agent_features]`.

        Returns:
            Tuple[FloatTensor, FloatTensor]: Parameter psi and nu of the goal
                precision posterior distribution.
        """
        assert x.size(-1) == self.precision_in_features

        # forward pass
        out: List[Tensor] = []
        for _, layer in self._scale_tril_network.items():
            assert isinstance(layer, nn.Linear)
            out.append(layer.forward(x).unsqueeze(-2))
        out = torch.cat(out, dim=-2)
        a, b, c = out.chunk(3, dim=-1)
        z = torch.zeros_like(a)
        psi_tril = torch.stack(
            (
                torch.cat([nn.functional.softplus(a), b], -1),
                torch.cat([z, nn.functional.softplus(c)], -1),
            ),
            dim=-1,
        )

        # apply Tikhonov regularization
        _eye = torch.eye(
            psi_tril.size(-1), device=psi_tril.device, dtype=psi_tril.dtype
        )
        psi = psi_tril.matmul(psi_tril.transpose(-2, -1))
        psi_tril = torch.linalg.cholesky(psi + _eye * TIKHONOV_REGULARIZATION)

        # preprocess the nu parameter
        dim = psi_tril.size(-1)
        nu = nn.functional.softplus(self._nu) + (dim + 2)
        nu = nu.expand_as(psi_tril[..., 0, 0])

        return psi_tril, nu

    def forward_mean(self, x: Tensor) -> Tuple[FloatTensor, FloatTensor]:
        """Forward pass of the goal mean posterior network.

        Args:
            x (Tensor): Context features of shape
                :math:`[n_agents, n_context_features]`.

        Returns:
            Tuple[FloatTensor, FloatTensor]: Parameter mu and beta of the goal
                mean posterior distribution.
        """
        assert x.size(-1) == self.mean_in_features

        # forward pass
        mu: List[Tensor] = []
        for _, layer in self._mu_network.items():
            assert isinstance(layer, nn.Linear)
            mu.append(layer.forward(x).unsqueeze(-2))
        mu = torch.cat(mu, dim=-2)

        # preprocess the beta parameter
        beta = nn.functional.softplus(self._beta)
        beta = beta.expand_as(mu[..., 0])

        return mu, beta

    def reset_parameters(self) -> None:
        reset(self._mu_network)
        reset(self._scale_tril_network)

    @property
    def alpha(self) -> nn.Parameter:
        """nn.Parameter: Concentrations for the Dirichlet posterior."""
        return nn.functional.softplus(self._alpha)


class Decoder(nn.Module):
    """Decoder module wrapping the inference and generative networks."""

    # ----------- public attributes ----------- #
    horizon: int
    """int: Prediction horizon as in number of frames."""
    num_modals: int
    """int: Number of trajectory modals to predict."""
    iou_radius: float
    """float: Radius for the IOU computation."""
    iou_threshold: float
    """float: IOU threshold for NMS sampling of the candidate goals."""
    grid_size: int
    """int: Grid size in number of samples along x- and y-axis."""
    gamma: float
    """float: Focusing weight for the focal loss."""

    # ----------- private modules ----------- #
    _generative_net: GenerativeNet
    """GenerativeNet: Generative network for modeling joint priors."""
    _inference_net: InferenceNet
    """InferenceNet: Inference network for modeling joint posteriors."""

    def __init__(
        self,
        context_in_features: int,
        agent_in_features: int,
        hidden_size: int = 64,
        num_mixtures: int = 3,
        horizon: int = 30,
        num_modals: int = 6,
        grid_size: float = 0.5,
        iou_radius: float = 1.4,
        iou_threshold: float = 0.0,
        alpha: float = 1.0,
        gamma: float = 4.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # save arguments
        self.num_modals = num_modals
        self.horizon = horizon
        self.gamma = gamma

        self.grid_size = grid_size
        self.iou_radius = iou_radius
        if iou_threshold < 0.0 or iou_threshold > 1.0:
            raise ValueError(
                f"Invalid IOU threshold {iou_threshold}! "
                "It should be in the range [0, 1]."
            )
        self.iou_threshold = iou_threshold

        # build network modules
        self._generative_net = GenerativeNet(
            alpha=alpha,
            in_features=context_in_features,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
        )
        self._inference_net = InferenceNet(
            mean_in_features=context_in_features,
            precision_in_features=agent_in_features,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
        )

    def forward(
        self,
        context_x: Tensor,
        agent_x: Tensor,
        grid_size: Optional[float] = None,
        iou_radius: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        sampling_mode: Optional[str] = None,
    ) -> StateDict:
        """Forward pass of the decoder module.

        Args:
            context_x (Tensor): Context features of the surrounding with shape
                :math:`[n_agents, n_context_features]`.
            agent_x (Tensor): Agent interaction features of the shape
                :math:`[n_agents, n_agent_features]`.
            grid_size (Optional[float]): Grid size in number of grid points
                along half x- or y-axis. If provided, override the initial
                value. Defaults to `None`.
            iou_radius (Optional[float]): Radius for the IoU computation.
                If provided, override the initial value. Defaults to `None`.
            iou_threshold (Optional[float]): IoU threshold for NMS sampling
                of the candidate goals. If provided, override the initial
                value. Defaults to `None`.
            sampling_mode (Optional[str]): Sampling mode for the intention
                prediction. If `None`, use 'proxy-best'. Defaults to `None`.

        Returns:
            StateDict: Distributions parameters and goal predictions.
        """
        sampling_mode = sampling_mode or "proxy-best"
        assert context_x.ndim == agent_x.ndim == 2
        assert context_x.size(0) == agent_x.size(0), (
            "Unmatched number of agents between context and agent features!"
            f"Got {context_x.size(0)} and {agent_x.size(0)}."
        )
        num_batch = context_x.size(0)

        if self.training:
            raise RuntimeError(
                "For training, please call 'forward_bayes_train' or "
                "'forward_proxy_train' methods!"
            )

        prediction: Dict[str, Union[Tensor, float]] = {}
        # For inference, apply NMS sampling and forward pass
        with torch.no_grad():
            q_psi_tril, q_nu = self._inference_net.forward_precision(x=agent_x)
            q_mu, q_beta = self._inference_net.forward_mean(x=context_x)
            ppd = get_posterior_predictive(
                psi=q_psi_tril.matmul(q_psi_tril.transpose(-2, -1)),
                nu=q_nu,
                mu=q_mu,
                beta=q_beta,
            )

            # generate dense samples
            samples = ppd.sample((100 - q_mu.size(-2),))
            samples = samples.permute(0, 2, 1, 3).reshape((-1, num_batch, 2))
            samples = torch.cat([samples, q_mu.transpose(0, 1)], dim=0)

            # evaluate the probability of each candidate
            alpha = self._generative_net.alpha + self.inference_net.alpha
            mixture = alpha / alpha.sum(dim=-1, keepdim=True)
            sample_probs = ppd.log_prob(samples.unsqueeze(-2)).exp()
            sample_probs = torch.sum(mixture * sample_probs, dim=-1)

            # re-organize shape dimensions
            samples = samples.permute(1, 0, 2)  # [n_agents, n_samples, 2]
            sample_probs = sample_probs.permute(1, 0)  # [n_agents, n_samples]

            # apply NMS to sample the top-k candidates
            cand, cand_probs = [], []
            valid_mask = torch.ones_like(sample_probs)
            while len(cand) < self.num_modals:
                # select the sample with the highest probability
                idx = torch.argmax(sample_probs * valid_mask.float(), dim=-1)
                mask = torch.zeros(
                    *sample_probs.shape,
                    device=context_x.device,
                    dtype=torch.bool,
                )
                mask[torch.arange(sample_probs.size(0)), idx] = True
                selected_sampels = samples[mask]
                cand.append(selected_sampels.unsqueeze(-2))
                cand_probs.append(sample_probs[mask].unsqueeze(-1))

                # compute IoU between the selected samples and others
                iou = compute_circular_iou(
                    selected_samples=selected_sampels,
                    other_samples=samples,
                    selected_radius=iou_radius or self.iou_radius,
                    other_radius=iou_radius or self.iou_radius,
                )
                iou_mask = iou > (iou_threshold or self.iou_threshold)

                # update the valid mask
                valid_mask[iou_mask] = -1.0
                valid_mask[mask] = -1.0
            cand = torch.cat(cand, dim=-2)
            cand_probs = torch.cat(cand_probs, dim=-1).softmax(dim=-1)
            cand_probs = torch.softmax(cand_probs, dim=-1)

            prediction.update(
                {
                    "intention": cand.detach(),
                    "confidence": cand_probs.detach(),
                }
            )
        return prediction

    def forward_bayes_train(
        self,
        context_x: Tensor,
        agent_x: Tensor,
        trajectory: Tensor,
    ) -> StateDict:
        """Forward pass the Bayesian Mixture model for training.

        Args:
            context_x (Tensor): Context features of the surrounding with shape
                :math:`[n_agents, n_context_features]`.
            agent_x (Tensor): Agent interaction features of the shape
                :math:`[n_agents, n_agent_features]`.
            trajectory (Tensor): Ground-truth trajectory of shape
                :math:`[n_agents, num_timestep, 2]`.

        Returns:
            StateDict: Distribution parameters and ELBO loss.
        """
        assert context_x.ndim == agent_x.ndim == 2
        assert context_x.size(0) == agent_x.size(0), (
            "Unmatched number of agents between context and agent features!"
            f"Got {context_x.size(0)} and {agent_x.size(0)}."
        )

        with torch.no_grad():
            goal = trajectory[..., -1, 0:2]

        inf_output = self._inference_net.forward(
            context_x=context_x,
            agent_x=agent_x,
            goal=goal,  # scale down
            gen_net=self._generative_net,
        )

        # post-processing distribution losses
        rec_loss = inf_output["reconstruction_loss"]
        kl_div_pi = inf_output["kl_div_pi"]
        kl_div_z = inf_output["kl_div_z"]
        kl_div_nw = inf_output["kl_div_nw"]
        with torch.no_grad():
            kl_divergence = kl_div_pi + kl_div_nw + kl_div_z

        kl_loss = kl_div_pi + kl_div_z.sum() + kl_div_nw.sum()
        loss = rec_loss + kl_loss

        return {
            "loss": loss,
            "reconstruction_loss": rec_loss.mean(),
            "kl_divergence": kl_divergence.mean(),
            "kl_div_pi": kl_div_pi.mean().item(),
            "kl_div_z": kl_div_z.mean().item(),
            "kl_div_nw": kl_div_nw.mean().item(),
        }

    @property
    def generative_net(self) -> GenerativeNet:
        """GenerativeNet: Generative network for modeling joint priors."""
        return self._generative_net

    @property
    def inference_net(self) -> InferenceNet:
        """InferenceNet: Inference network for modeling joint posteriors."""
        return self._inference_net
