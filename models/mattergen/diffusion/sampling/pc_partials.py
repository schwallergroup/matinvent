# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Protocol

from models.mattergen.diffusion.corruption.corruption import Corruption
from models.mattergen.diffusion.corruption.sde_lib import ScoreFunction
from models.mattergen.diffusion.sampling.predictors import Predictor
from models.mattergen.diffusion.sampling.predictors_correctors import LangevinCorrector


class PredictorPartial(Protocol):
    def __call__(self, *, corruption: Corruption, score_fn: ScoreFunction | None) -> Predictor:
        raise NotImplementedError


class CorrectorPartial(Protocol):
    def __call__(
        self, *, corruption: Corruption, n_steps: int, score_fn: ScoreFunction | None
    ) -> LangevinCorrector:
        raise NotImplementedError
