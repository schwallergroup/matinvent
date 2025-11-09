from typing import Dict, Literal, Optional, Tuple, TypeVar
import torch

from mattergen.common.loss import MaterialsLoss
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption, apply
from mattergen.diffusion.data.batched_data import BatchedData

T = TypeVar("T", bound=BatchedData)


class SampleLoss(MaterialsLoss):
    def __init__(
        self,
        reduce: Literal["sum", "mean"] = "sum",
        d3pm_hybrid_lambda: float = 0.01,
        include_pos: bool = True,
        include_cell: bool = True,
        include_atomic_numbers: bool = True,
        weights: Optional[Dict[str, float]] = None,
    ):
        if weights is None:
            weights = {
                "atomic_numbers": 1.0,
                "cell": 1.0,
                "pos": 0.1,
            }
        super().__init__(
            reduce=reduce,
            d3pm_hybrid_lambda=d3pm_hybrid_lambda,
            include_pos=include_pos,
            include_cell=include_cell,
            include_atomic_numbers=include_atomic_numbers,
            weights=weights,
        )

    def __call__(
        self,
        *,
        multi_corruption: MultiCorruption[T],
        batch: T,
        noisy_batch: T,
        score_model_output: T,
        t: torch.Tensor,
        node_is_unmasked: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_idx = {k: batch.get_batch_idx(k) for k in self.loss_fns.keys()}
        node_is_unmasked = {k: node_is_unmasked for k in self.loss_fns.keys()}

        # Dict[str, torch.Tensor]
        # Keys are field names and values are loss per sample, with shape (batch_size,).
        loss_per_sample_per_field = apply(
            fns=self.loss_fns,
            corruption=multi_corruption.corruptions,
            x=batch,
            noisy_x=noisy_batch,
            score_model_output=score_model_output,
            batch_idx=batch_idx,
            broadcast=dict(t=t, batch_size=batch.get_batch_size(), batch=batch),
            node_is_unmasked=node_is_unmasked,
        )
        assert set([v.shape for v in loss_per_sample_per_field.values()]) == {
            (batch.get_batch_size(),)
        }, "All losses should have shape (batch_size,)."
        # Aggregate losses per field over samples.
        scalar_loss_per_field = {k: v.mean() for k, v in loss_per_sample_per_field.items()}

        # Dict[str, torch.Tensor], dictionary containing metrics to be logged,
        metrics_dict = scalar_loss_per_field
        # This is the loss that is used for backpropagation (after mean aggregation over samples).
        # Shape: (batch_size,)
        agg_loss = torch.stack(
            [self.loss_weights[k] * v for k, v in loss_per_sample_per_field.items()], dim=0
        ).sum(0)

        return (
            agg_loss,
            metrics_dict,
        )
