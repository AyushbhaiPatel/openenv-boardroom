# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Noise Injector for the OpenBoardroom Environment.

Applies difficulty-scaled data corruption to observation data dictionaries.
Easy difficulty passes data through unchanged; medium adds Gaussian noise
and occasional NaN values; hard adds larger noise, NaN values, and one
deliberately misleading signal.

All noise is seeded via numpy for full determinism (Req 13.4).
"""

from copy import deepcopy
from typing import Any, Dict

import math
import numpy as np


class NoiseInjector:
    """Injects difficulty-scaled noise into observation data dictionaries.

    Noise levels by difficulty tier:
        - **easy**: No noise — data passes through unchanged.
        - **medium**: Gaussian perturbation (±5–10%), occasional NaN.
        - **hard**: Gaussian perturbation (±15–20%), NaN, plus one
          deliberately misleading signal that reverses a metric's direction.

    All randomness is driven by a seeded ``numpy.random.Generator`` so that
    identical seeds produce identical noise patterns (Req 13.4).
    """

    def __init__(self, seed: int, difficulty: str) -> None:
        """Initialise the injector.

        Args:
            seed: Integer seed for the numpy random generator.
            difficulty: One of ``"easy"``, ``"medium"``, ``"hard"``.
        """
        self._rng = np.random.default_rng(seed)
        self._difficulty = difficulty

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a (possibly corrupted) copy of *data*.

        The original dictionary is never mutated.  Noise is applied only to
        values that are numeric (``int`` or ``float``).  Non-numeric values
        are passed through unchanged.

        Args:
            data: A flat dictionary of metric names to values.

        Returns:
            A new dictionary with noise applied according to the difficulty.
        """
        if self._difficulty == "easy":
            return deepcopy(data)

        if self._difficulty == "medium":
            return self._inject_medium(data)

        if self._difficulty == "hard":
            return self._inject_hard(data)

        # Unknown difficulty — treat as easy (no noise).
        return deepcopy(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inject_medium(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Medium noise: ±5–10% Gaussian perturbation, occasional NaN."""
        result: Dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # ~10% chance of injecting NaN
                if self._rng.random() < 0.10:
                    result[key] = float("nan")
                else:
                    noise_pct = self._rng.normal(0, 0.075)  # std ~7.5% (centre of 5–10%)
                    noise_pct = float(np.clip(noise_pct, -0.10, 0.10))
                    noisy = value * (1.0 + noise_pct)
                    result[key] = type(value)(noisy) if isinstance(value, int) else noisy
            else:
                result[key] = deepcopy(value)
        return result

    def _inject_hard(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Hard noise: ±15–20% Gaussian, NaN, plus one misleading signal."""
        result: Dict[str, Any] = {}
        numeric_keys = [
            k for k, v in data.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]

        # Pick one numeric key to receive the misleading signal.
        misleading_key: str | None = None
        if numeric_keys:
            misleading_key = self._rng.choice(numeric_keys)

        for key, value in data.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                # ~15% chance of NaN (higher than medium)
                if self._rng.random() < 0.15:
                    result[key] = float("nan")
                elif key == misleading_key:
                    # Misleading signal: flip the sign of the value's
                    # deviation from a neutral baseline so the observed
                    # trend appears opposite to reality.
                    result[key] = self._mislead(value)
                else:
                    noise_pct = self._rng.normal(0, 0.175)  # std ~17.5% (centre of 15–20%)
                    noise_pct = float(np.clip(noise_pct, -0.20, 0.20))
                    noisy = value * (1.0 + noise_pct)
                    result[key] = type(value)(noisy) if isinstance(value, int) else noisy
            else:
                result[key] = deepcopy(value)

        # If the misleading key ended up as NaN (from the 15% chance),
        # ensure we still have a misleading signal by overwriting it.
        if misleading_key is not None and (
            isinstance(result.get(misleading_key), float)
            and math.isnan(result[misleading_key])
        ):
            result[misleading_key] = self._mislead(data[misleading_key])

        return result

    @staticmethod
    def _mislead(value: float | int) -> float:
        """Return a value whose direction is opposite to the original.

        For positive values the result is pushed toward zero (inverted
        direction) but never goes negative, since business metrics like
        revenue, MAU, and support_load cannot be negative in reality.
        For zero the result is a small positive offset.
        """
        if value == 0:
            return 0.01
        if value > 0:
            # Push toward zero: keep 70% reduction but floor at 0.0
            return max(0.0, float(value * 0.3))
        # Negative values (rare): flip toward positive
        return float(abs(value) * 0.3)
