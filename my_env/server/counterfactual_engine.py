# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Counterfactual Engine for the OpenBoardroom Environment.

A small fixed-weight PyTorch MLP that takes CompanyState and a proposed
decision string and forecasts alternate outcomes for what-if reasoning.
Weights are initialized deterministically via torch.manual_seed — no
training loop is involved.
"""

import hashlib
import json
import math
from typing import Any, Dict

import torch
import torch.nn as nn

try:
    from ..models import CompanyState
except ImportError:
    from models import CompanyState

# Number of features from CompanyState.to_tensor_input()
_STATE_FEATURES = 6
# Extra features for hash-based decision encoding
_DECISION_FEATURES = 4
_INPUT_DIM = _STATE_FEATURES + _DECISION_FEATURES
_OUTPUT_DIM = 3  # revenue_delta, churn_delta, user_growth_delta


def _encode_decision(decision: str, params: Dict[str, Any]) -> list[float]:
    """Stable SHA-256 encoding of decision + params into ``_DECISION_FEATURES`` floats.

    Uses ``hashlib`` (not Python ``hash()``) so the same inputs yield identical
    vectors across processes and runs (reproducible counterfactuals).
    """
    if not isinstance(params, dict):
        params = {}
    payload = json.dumps(
        {"decision": decision, "parameters": params},
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return [digest[i] / 255.0 for i in range(_DECISION_FEATURES)]


class _CounterfactualMLP(nn.Module):
    """Linear(input_dim, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, output_dim)"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CounterfactualEngine:
    """Fixed-weight MLP counterfactual simulator.

    Parameters
    ----------
    seed : int
        Seed passed to ``torch.manual_seed`` so that the randomly-initialised
        weights are deterministic and reproducible.
    """

    def __init__(self, seed: int) -> None:
        torch.manual_seed(seed)
        self.model = _CounterfactualMLP(_INPUT_DIM, _OUTPUT_DIM)
        # Freeze all parameters — no training
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def simulate(
        self,
        company_state: CompanyState,
        decision: str,
        params: Dict,
    ) -> Dict[str, float]:
        """Run a what-if simulation.

        Parameters
        ----------
        company_state : CompanyState
            Current company metrics.
        decision : str
            Free-form decision description.
        params : Dict
            Additional decision parameters.

        Returns
        -------
        Dict[str, float]
            Keys: ``projected_revenue_delta``, ``projected_churn_delta``,
            ``projected_user_growth_delta``.  Each value is a finite float.
        """
        state_features = company_state.to_tensor_input()
        decision_features = _encode_decision(decision, params)
        input_vector = state_features + decision_features

        x = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = self.model(x).squeeze(0)

        # Defensive: clamp any non-finite values
        values = output.tolist()
        values = [v if math.isfinite(v) else 0.0 for v in values]

        return {
            "projected_revenue_delta": values[0],
            "projected_churn_delta": values[1],
            "projected_user_growth_delta": values[2],
        }
