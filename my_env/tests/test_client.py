"""Tests for the BoardroomEnv client serialization/deserialization."""

from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from my_env.client import BoardroomEnv
from my_env.models import BoardroomAction, BoardroomObservation


class TestStepPayload:
    """Tests for _step_payload serialization."""

    def test_serializes_action_type_and_parameters(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        action = BoardroomAction(
            action_type="query_data",
            parameters={"metric": "revenue"},
        )
        payload = client._step_payload(action)
        assert payload == {
            "action_type": "query_data",
            "parameters": {"metric": "revenue"},
        }

    def test_serializes_empty_parameters(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        action = BoardroomAction(action_type="analyze_trend")
        payload = client._step_payload(action)
        assert payload == {
            "action_type": "analyze_trend",
            "parameters": {},
        }

    def test_serializes_make_decision_with_explanation(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        action = BoardroomAction(
            action_type="make_decision",
            parameters={
                "decision": "increase_ad_spend",
                "parameters": {"amount": 50000},
                "explanation": "Revenue is declining due to high churn.",
            },
        )
        payload = client._step_payload(action)
        assert payload["action_type"] == "make_decision"
        assert payload["parameters"]["explanation"] == "Revenue is declining due to high churn."


class TestParseResult:
    """Tests for _parse_result deserialization."""

    def test_parses_full_observation(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        payload = {
            "observation": {
                "data_tables": {"revenue": [100000, 120000]},
                "stakeholder_feedback": "Growth looks strong.",
                "simulation_results": {"projected_revenue_delta": 0.05},
                "quarter": 2,
                "step_count": 5,
                "metadata": {"seed": 42},
            },
            "reward": 0.25,
            "done": False,
        }
        result = client._parse_result(payload)

        assert isinstance(result, StepResult)
        obs = result.observation
        assert isinstance(obs, BoardroomObservation)
        assert obs.data_tables == {"revenue": [100000, 120000]}
        assert obs.stakeholder_feedback == "Growth looks strong."
        assert obs.simulation_results == {"projected_revenue_delta": 0.05}
        assert obs.quarter == 2
        assert obs.step_count == 5
        assert obs.done is False
        assert obs.reward == 0.25
        assert obs.metadata == {"seed": 42}
        assert result.reward == 0.25
        assert result.done is False

    def test_parses_minimal_observation(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        payload = {
            "observation": {},
            "reward": None,
            "done": True,
        }
        result = client._parse_result(payload)

        obs = result.observation
        assert obs.data_tables == {}
        assert obs.stakeholder_feedback is None
        assert obs.simulation_results is None
        assert obs.quarter == 1
        assert obs.step_count == 0
        assert obs.done is True
        assert result.done is True

    def test_parses_done_episode_with_audit_trail(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        payload = {
            "observation": {
                "data_tables": {},
                "quarter": 3,
                "step_count": 20,
                "metadata": {
                    "audit_trail": [{"step": 1, "action_type": "query_data"}],
                    "final_score": 0.75,
                },
            },
            "reward": 0.75,
            "done": True,
        }
        result = client._parse_result(payload)

        assert result.done is True
        assert result.reward == 0.75
        assert "audit_trail" in result.observation.metadata


class TestParseState:
    """Tests for _parse_state deserialization."""

    def test_parses_state(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        payload = {"episode_id": "ep-123", "step_count": 7}
        state = client._parse_state(payload)

        assert isinstance(state, State)
        assert state.episode_id == "ep-123"
        assert state.step_count == 7

    def test_parses_state_with_defaults(self):
        client = BoardroomEnv.__new__(BoardroomEnv)
        payload = {}
        state = client._parse_state(payload)

        assert state.episode_id is None
        assert state.step_count == 0
