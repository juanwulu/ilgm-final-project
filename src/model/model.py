# Copyright (c) 2024, Juanwu Lu.
# Released under the BSD 3-Clause License.
# Please see the LICENSE file that should have been included as part of this
# project source code.
"""Fast-adaptive Motion Prediction model."""
from __future__ import annotations

from itertools import chain
from typing import Any, Optional

import torch
from lightning.pytorch import LightningModule
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torchmetrics import MeanMetric, MetricCollection, MinMetric

from ..data.base_data import BaseData
from ..data.metrics import (
    MinAverageDisplacementError,
    MinFinalDisplacementError,
    MissRate,
)
from .decoder import Decoder
from .encoder import Encoder
from .layers import MLP
from .typing import StateDict

__all__ = ["GNeVANetwork", "GNeVALightningModule"]


class GNeVANetwork(nn.Module):
    """Goal-based Neural Variational Agent Network."""

    # ----------- public attributes ------------ #
    encoder: Encoder
    """Encoder: Encoder module for feature encoding."""
    decoder: Decoder
    """Decoder: Variational probabilistic model for intention generation."""
    num_goal_heads: int
    """int: Number of mixtures for intention generation."""
    num_modals: int
    """int: Number of modalities."""
    goal_predictor: MLP
    """MLP: Goal predictor module for initial guess of goal."""
    planner: nn.LSTM
    """nn.LSTM: Planner for generating intermediate motion states."""
    prediction_horizon: int
    """int: Prediction horizon."""

    # ----------- private attributes ----------- #
    _train_step: int
    """int: Current training step."""

    def __init__(
        self,
        # encoder arguments
        map_in_features: int,
        map_hidden_size: int,
        map_num_layers: int,
        motion_in_features: int,
        motion_hidden_size: int,
        motion_num_layers: int,
        num_encoder_heads: int,
        # decoder arguments
        intention_hidden_size: int,
        observation_horizon: int,
        prediction_horizon: int,
        num_goal_heads: int,
        # other arguments
        map_out_size: Optional[int] = None,
        motion_out_size: Optional[int] = None,
        num_modals: int = 6,
        dropout: float = 0.1,
        grid_size: float = 0.5,
        iou_radius: float = 1.0,
        iou_threshold: float = 0.5,
        alpha: float = 1.0,
        gamma: float = 4.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Construct a `GNeVANetwork` model instance."""
        super().__init__()

        self.num_goal_heads = num_goal_heads
        self.num_modals = num_modals
        self.prediction_horizon = prediction_horizon

        # encoder modules
        self.encoder = Encoder(
            map_in_features=map_in_features,
            map_hidden_size=map_hidden_size,
            map_out_size=map_out_size,
            map_num_layers=map_num_layers,
            motion_in_features=motion_in_features,
            motion_hidden_size=motion_hidden_size,
            motion_out_size=motion_out_size,
            motion_num_layers=motion_num_layers,
            dropout=dropout,
            num_heads=num_encoder_heads,
            horizon=observation_horizon,
        )

        # decoder modules
        self.decoder = Decoder(
            context_in_features=self.encoder.out_feature,
            agent_in_features=self.encoder.out_feature,
            motion_in_features=motion_in_features,
            hidden_size=intention_hidden_size,
            num_mixtures=num_goal_heads,
            horizon=prediction_horizon,
            grid_size=grid_size,
            iou_radius=iou_radius,
            iou_threshold=iou_threshold,
            alpha=alpha,
            gamma=gamma,
        )

        # planner module
        self.planner = nn.ModuleDict(
            {
                "lstm": nn.LSTM(
                    input_size=2,
                    hidden_size=self.encoder.motionnet.out_feature,
                    batch_first=True,
                ),
                "mlp": nn.Sequential(
                    nn.Linear(self.encoder.motionnet.out_feature, 128),
                    nn.SiLU(),
                    nn.Linear(128, 128),
                    nn.SiLU(),
                    nn.Linear(128, 2),
                ),
            }
        )

    def forward(
        self,
        data: BaseData,
        grid_size: Optional[float] = None,
        iou_radius: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        sampling_mode: Optional[str] = None,
    ) -> StateDict:
        assert data.is_valid(), "Invalid data!"
        assert not self.training, "Model is in training mode!"
        assert sampling_mode in [
            "prior_random",
            "prior_mll",
            "posterior_random",
            "posterior_map",
            None,
        ], f"Invalid sampling mode: {sampling_mode}."
        sampling_mode = sampling_mode or "posterior_map"
        with torch.no_grad():
            # ----------- encoder forward ----------- #
            encoder_output = self.encoder.forward(data=data)

            # ----------- decoder forward ----------- #
            prediction = self.decoder.forward(
                context_x=encoder_output["context_feats"],
                agent_x=encoder_output["interaction_feats"],
                grid_size=grid_size,
                iou_radius=iou_radius,
                iou_threshold=iou_threshold,
                sampling_mode=sampling_mode,
            )
            prediction["current_state"] = encoder_output["current_state"]

            # ----------- planner forward ----------- #
            with torch.no_grad():
                goal = prediction["intention"]
                goal = goal.unsqueeze(-2).expand(
                    -1, -1, self.prediction_horizon, -1
                )
                hidden = encoder_output["target_feats"].unsqueeze(0)
                output = torch.zeros(
                    size=goal.shape[0:2]
                    + (
                        self.prediction_horizon,
                        self.encoder.motionnet.out_feature,
                    ),
                    device=goal.device,
                    dtype=goal.dtype,
                )
                for i in range(goal.size(1)):
                    output[:, i] = self.planner.lstm.forward(
                        input=goal[:, i],
                        hx=(hidden, torch.zeros_like(hidden)),
                    )[0]
                trajectory = self.planner.mlp.forward(output).cumsum(dim=-2)
                prediction["trajectory"] = trajectory

            return prediction

    def forward_train_bayes(self, data: BaseData) -> StateDict:
        assert data.is_valid(), "Invalid data!"
        assert self.training, "Model is not in training mode!"
        with torch.set_grad_enabled(mode=self.training):
            # ----------- encoder forward ----------- #
            encoder_output = self.encoder.forward(data=data)

            # ----------- decoder forward ----------- #
            prediction = self.decoder.forward_bayes_train(
                context_x=encoder_output["context_feats"],
                agent_x=encoder_output["interaction_feats"],
                trajectory=data.get("y_motion", None),
            )

            return prediction

    def forward_train_planner(self, data: BaseData) -> StateDict:
        assert data.is_valid(), "Invalid data!"
        assert self.training, "Model is not in training mode!"
        # ----------- encoder forward ----------- #
        with torch.no_grad():
            encoder_output = self.encoder.forward(data=data)

        # ----------- planner forward ----------- #
        goal = data.y_motion[..., -1:, 0:2]
        goal = goal.expand(-1, self.prediction_horizon, -1)
        hidden = encoder_output["target_feats"].detach().unsqueeze(0)
        output, _ = self.planner.lstm.forward(
            input=goal,
            hx=(hidden, torch.zeros_like(hidden)),
        )
        trajectory = self.planner.mlp.forward(output).cumsum(dim=-2)
        traj_loss = nn.functional.huber_loss(
            input=trajectory[..., 0:2],
            target=data.y_motion[..., 0:2],
        )

        return {
            "loss": traj_loss,
            "trajectory_loss": traj_loss.item(),
        }

    def __str__(self) -> str:
        attr_str: str = ", ".join(
            [
                f"encoder={self.encoder}",
                f"decoder={self.decoder}",
                f"planner={self.planner}",
                f"num_modals={self.num_modals}",
            ]
        )
        return f"<{self.__class__.__name__}({attr_str}) at {hex(id(self))}>"

    def __repr__(self) -> str:
        return str(self)


class GNeVALightningModule(LightningModule):
    """Wrapper class for FAMOPNet to be used with pytorch lightning."""

    network: GNeVANetwork
    """GNeVANetwork: GNeVANetwork model."""

    def __init__(
        self,
        network_kwargs: dict[str, Any],
    ) -> None:
        super().__init__()

        self.automatic_optimization = False
        self.save_hyperparameters(logger=False)
        self.network = GNeVANetwork(**network_kwargs)

        # metric objects for calculating and averaging accuracy across batches
        self.val_made = MinAverageDisplacementError()
        self.val_mfde = MinFinalDisplacementError()
        self.val_mr = MissRate()

        # for tracking learning rate
        self.trackers = MetricCollection(
            {
                "bayes": MeanMetric(),
                "planner": MeanMetric(),
            }
        )

        # for averaging loss across batches
        self.train_losses = MetricCollection(
            {
                "loss": MeanMetric(),
                "reconstruction_loss": MeanMetric(),
                "kl_divergence": MeanMetric(),
                "kl_div_pi": MeanMetric(),
                "kl_div_z": MeanMetric(),
                "kl_div_nw": MeanMetric(),
                "goal_loss": MeanMetric(),
                "trajectory_loss": MeanMetric(),
            }
        )

        # for tracking best so far validation final displacement error
        self.val_mfde_best = MinMetric()

    def forward(self, data: BaseData, *args, **kwargs) -> StateDict:
        return self.network.forward(data=data, *args, **kwargs)

    def on_train_start(self) -> None:
        # by default lightning executes validation step sanity checks before
        # training starts, so it's worth to make sure validation metrics don't
        # store results from these checks
        self.val_mfde.reset()
        self.val_mr.reset()
        self.val_mfde_best.reset()

    def training_step(self, batch: BaseData) -> None:
        assert isinstance(batch, BaseData)
        opt_bayes, opt_planner = self.optimizers()
        sch_bayes, sch_planner = self.lr_schedulers()

        res = {}
        total_loss = 0.0

        opt_bayes.zero_grad()
        output = self.network.forward_train_bayes(data=batch)
        self.manual_backward(output["loss"])
        opt_bayes.step()
        sch_bayes.step()
        total_loss += output["loss"].item()
        res.update(output)

        opt_planner.zero_grad()
        output = self.network.forward_train_planner(data=batch)
        self.manual_backward(output["loss"])
        opt_planner.step()
        sch_planner.step()
        total_loss += output["loss"].item()
        res.update(output)

        # update and log losses
        res["loss"] = total_loss
        for key, value in self.train_losses.items():
            if key in res:
                value(res[key])
                self.log(
                    f"train/{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )

        # update and log learning rate
        self.trackers["bayes"](opt_bayes.param_groups[0]["lr"])
        self.trackers["planner"](opt_planner.param_groups[0]["lr"])
        self.log(
            "lr/bayes", self.trackers["bayes"], on_epoch=False, on_step=True
        )
        self.log(
            "lr/planner",
            self.trackers["planner"],
            on_epoch=False,
            on_step=True,
        )

        return output["loss"]

    def on_train_epoch_end(self) -> None:
        pass

    def validation_step(self, batch: BaseData, batch_idx: int) -> None:
        assert isinstance(batch, BaseData)
        output = self.forward(data=batch)

        # update and log metrics
        self.val_made(preds=output["trajectory"], target=batch.y_motion)
        self.val_mfde(preds=output["trajectory"], target=batch.y_motion)
        self.val_mr(
            preds=output["trajectory"],
            target=batch.y_motion,
            anchor=batch.anchor,
            batch=batch.get("y_motion_batch"),
        )

    def on_validation_epoch_end(self) -> None:
        self.log("val/made", self.val_made, sync_dist=True, prog_bar=True)
        self.log("val/mfde", self.val_mfde, sync_dist=True, prog_bar=True)
        self.log("val/mr", self.val_mr, sync_dist=True, prog_bar=True)

        mfde = self.val_mfde.compute()
        self.val_mfde_best(mfde)
        # log `val_mfde_best` as a value through `.compute()` method,
        # instead of as a metric object,
        # otherwise metric would be reset after each epoch.
        self.log(
            "val/mfde_best",
            self.val_mfde_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Any, batch_idx: int) -> StateDict:
        assert isinstance(batch, BaseData)
        output = self.forward(batch)  # noqa: F841

        return output

    def on_test_epoch_end(self) -> None:
        pass

    def configure_optimizers(self) -> Any:
        bayes_optimizer = AdamW(
            params=chain(
                self.network.encoder.parameters(),
                self.network.decoder.generative_net.parameters(),
                self.network.decoder.inference_net.parameters(),
            ),
            lr=1e-3,
            weight_decay=1e-2,
        )
        planner_optimizer = AdamW(
            params=self.network.planner.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )
        bayes_scheduler = SequentialLR(
            optimizer=bayes_optimizer,
            schedulers=[
                LambdaLR(
                    optimizer=bayes_optimizer,
                    lr_lambda=lambda step: min(1.0, max(0.0, step / 1000.0)),
                ),
                CosineAnnealingLR(
                    optimizer=bayes_optimizer,
                    T_max=200000,
                    eta_min=3e-7,
                ),
                LambdaLR(
                    optimizer=bayes_optimizer,
                    lr_lambda=lambda step: 3e-7
                    / bayes_optimizer.defaults["lr"],
                ),
            ],
            milestones=[1000, 200000],
        )
        planner_scheduler = SequentialLR(
            optimizer=planner_optimizer,
            schedulers=[
                LambdaLR(
                    optimizer=planner_optimizer,
                    lr_lambda=lambda step: min(1.0, max(0.0, step / 1000.0)),
                ),
                CosineAnnealingLR(
                    planner_optimizer, T_max=200000, eta_min=3e-7
                ),
                LambdaLR(
                    optimizer=planner_optimizer,
                    lr_lambda=lambda step: 3e-7
                    / planner_optimizer.defaults["lr"],
                ),
            ],
            milestones=[1000, 200000],
        )

        return (
            {
                "optimizer": bayes_optimizer,
                "lr_scheduler": {
                    "scheduler": bayes_scheduler,
                    "monitor": "val/mfde_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
            {
                "optimizer": planner_optimizer,
                "lr_scheduler": {
                    "scheduler": planner_scheduler,
                    "monitor": "val/mfde_best",
                    "interval": "epoch",
                    "frequency": 1,
                },
            },
        )
