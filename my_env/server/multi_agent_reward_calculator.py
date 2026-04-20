"""Multi-agent reward calculator for the boardroom environment."""


class MultiAgentRewardCalculator:
    """Computes multi-agent-specific reward components for the boardroom environment."""

    def compute_present_evidence_reward(self, actor_state, evidence_delta: float) -> float:
        """Returns 0.05 + 0.10 * evidence_delta clamped to [0.05, 0.15]."""
        reward = 0.05 + 0.10 * evidence_delta
        return max(0.05, min(0.15, reward))

    def compute_negotiate_reward(self, lobby_reduction: float) -> float:
        """Returns 0.05 if lobby_reduction > 0 else 0.0."""
        return 0.05 if lobby_reduction > 0 else 0.0

    def compute_board_vote_reward(self, approvals: int) -> float:
        """Returns +0.2 if approvals >= 2, -0.2 otherwise."""
        return 0.2 if approvals >= 2 else -0.2

    def compute_risk_alert_reward(self, decision_addresses_alert: bool, alert_active: bool) -> float:
        """Returns +0.15 if decision_addresses_alert, -0.15 if alert active and not addressed, 0.0 if no alert."""
        if decision_addresses_alert:
            return 0.15
        if alert_active and not decision_addresses_alert:
            return -0.15
        return 0.0

    def compute_hidden_metric_reveal_reward(self, revealed: bool) -> float:
        """Returns +0.3 if revealed else 0.0."""
        return 0.3 if revealed else 0.0
