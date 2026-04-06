# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for ExplanationGrader."""

import importlib
import sys
import os

# Direct import to avoid broken server/__init__.py chain
_server_dir = os.path.join(os.path.dirname(__file__), "..", "server")
_spec = importlib.util.spec_from_file_location(
    "explanation_grader",
    os.path.join(_server_dir, "explanation_grader.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ExplanationGrader = _mod.ExplanationGrader


class TestExplanationGrader:
    """Unit tests for the ExplanationGrader heuristic scorer."""

    def setup_method(self) -> None:
        self.grader = ExplanationGrader()
        self.ctx: dict = {"difficulty": "medium", "objective": "test"}

    # ------------------------------------------------------------------
    # Basic range and determinism
    # ------------------------------------------------------------------

    def test_empty_explanation_scores_zero(self) -> None:
        assert self.grader.grade("", self.ctx) == 0.0

    def test_whitespace_only_scores_zero(self) -> None:
        assert self.grader.grade("   ", self.ctx) == 0.0

    def test_score_in_valid_range(self) -> None:
        score = self.grader.grade("revenue is up 10%", self.ctx)
        assert 0.0 <= score <= 1.0

    def test_deterministic_same_input(self) -> None:
        text = "Revenue might drop. The analyst warned about churn."
        s1 = self.grader.grade(text, self.ctx)
        s2 = self.grader.grade(text, self.ctx)
        assert s1 == s2

    # ------------------------------------------------------------------
    # Property 12: Monotonicity — all-three > none
    # ------------------------------------------------------------------

    def test_rich_explanation_beats_empty_content(self) -> None:
        """An explanation with data, uncertainty, AND stakeholder refs
        must score strictly higher than one with none of these."""
        rich = (
            "Based on revenue data showing 15% churn increase, "
            "this might risk further decline. "
            "The analyst and CEO both raised concerns."
        )
        bare = "I think we should do something about the situation."
        assert self.grader.grade(rich, self.ctx) > self.grader.grade(bare, self.ctx)

    def test_no_dimensions_scores_low(self) -> None:
        score = self.grader.grade("Let us proceed with the plan.", self.ctx)
        assert score < 0.2

    # ------------------------------------------------------------------
    # Individual dimension checks
    # ------------------------------------------------------------------

    def test_data_evidence_increases_score(self) -> None:
        without = "We should act now."
        with_data = "Revenue is $1.2M and churn is at 8%."
        assert self.grader.grade(with_data, self.ctx) > self.grader.grade(without, self.ctx)

    def test_uncertainty_increases_score(self) -> None:
        without = "We will succeed."
        with_hedge = "We might succeed, but there is risk."
        assert self.grader.grade(with_hedge, self.ctx) > self.grader.grade(without, self.ctx)

    def test_stakeholder_increases_score(self) -> None:
        without = "We should cut costs."
        with_stake = "The analyst and CEO have different viewpoints on cutting costs."
        assert self.grader.grade(with_stake, self.ctx) > self.grader.grade(without, self.ctx)
