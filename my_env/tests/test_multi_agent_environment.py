"""Unit tests for MultiAgentBoardroomEnvironment (task 11.1)."""

import pytest
from my_env.models import BoardroomAction
from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment


@pytest.fixture()
def env_easy():
    env = MultiAgentBoardroomEnvironment()
    env.reset(seed=42, difficulty="easy")
    return env


@pytest.fixture()
def env_hard():
    env = MultiAgentBoardroomEnvironment()
    env.reset(seed=42, difficulty="hard")
    return env


class TestResetInitialisesActorStates:
    def test_three_distinct_actor_states(self, env_easy):
        states = env_easy._actor_states
        assert set(states.keys()) == {"ceo", "cfo", "risk_officer"}
        assert states["ceo"].name == "ceo"
        assert states["cfo"].name == "cfo"
        assert states["risk_officer"].name == "risk_officer"

    def test_easy_defaults(self, env_easy):
        assert env_easy._actor_states["cfo"].lobby_pressure == 0.0
        assert env_easy._actor_states["cfo"].evidence_weight == 0.0
        assert env_easy._actor_states["risk_officer"].alert_active is False

    def test_hard_defaults(self, env_hard):
        ceo = env_hard._actor_states["ceo"]
        assert ceo.hidden_agenda == "launch"
        assert "support_load" in ceo.hidden_metrics
        assert "release_risk" in ceo.hidden_metrics
        assert ceo.stance == "pro_launch"

    def test_actor_messages_in_reset_obs(self):
        env = MultiAgentBoardroomEnvironment()
        obs = env.reset(seed=1, difficulty="medium")
        assert "actor_messages" in obs.metadata
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}


class TestCeoHardDifficultyHiddenAgenda:
    def test_hidden_agenda_is_launch_on_hard(self, env_hard):
        assert env_hard._actor_states["ceo"].hidden_agenda == "launch"

    def test_hidden_agenda_not_launch_on_easy(self, env_easy):
        assert env_easy._actor_states["ceo"].hidden_agenda != "launch"


class TestCeoSuppressesMetricsHard:
    def test_suppresses_support_load_before_evidence(self, env_hard):
        obs = env_hard.step(BoardroomAction(
            action_type="query_data",
            parameters={"metric": "support_load"},
        ))
        assert obs.data_tables.get("support_load") == "[SUPPRESSED — data unavailable at this time]"

    def test_no_suppression_on_easy(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="query_data",
            parameters={"metric": "support_load"},
        ))
        assert obs.data_tables.get("support_load") != "[SUPPRESSED — data unavailable at this time]"


class TestHiddenMetricRevealSequence:
    def test_reveal_after_contradiction_evidence(self, env_hard):
        # Present contradicting evidence to CEO first
        env_hard.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "ceo", "metric": "support_load", "value": 0.95,
                        "interpretation": "data shows risk evidence"},
        ))
        assert env_hard._ceo_contradiction_flag is True
        # Now query the suppressed metric — should reveal
        obs = env_hard.step(BoardroomAction(
            action_type="query_data",
            parameters={"metric": "support_load"},
        ))
        assert obs.data_tables.get("support_load") != "[SUPPRESSED — data unavailable at this time]"


class TestRevisionChance:
    def test_second_make_decision_resolves(self, env_hard):
        # Force rejection: CEO rejects delay, CFO pro_launch, risk officer alert
        env_hard._actor_states["ceo"].hidden_agenda = "launch"
        env_hard._actor_states["cfo"].stance = "pro_launch"
        env_hard._actor_states["cfo"].lobby_pressure = 0.9
        env_hard._actor_states["risk_officer"].alert_active = True
        obs1 = env_hard.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "delay launch", "parameters": {}, "explanation": "too risky"},
        ))
        if obs1.done:
            return  # Already resolved
        obs2 = env_hard.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "delay launch again", "parameters": {}, "explanation": "still risky"},
        ))
        assert obs2.done is True


class TestRiskOfficerIntelUnlock:
    def test_intel_unlocked_after_alert_and_evidence(self, env_hard):
        env_hard._actor_states["risk_officer"].alert_active = True
        obs = env_hard.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "risk_officer", "metric": "support_load",
                        "value": 0.95, "interpretation": "support load risk data"},
        ))
        assert env_hard._actor_states["risk_officer"].intel_unlocked is True
        assert "risk_intel" in obs.data_tables
        assert "alert_threshold" in obs.data_tables["risk_intel"]


class TestBoardVoteBreakdownInMetadata:
    def test_board_vote_in_metadata(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        assert "board_vote" in obs.metadata
        assert set(obs.metadata["board_vote"].keys()) == {"ceo", "cfo", "risk_officer"}


class TestActorMessagesInEveryObservation:
    def test_actor_messages_after_query_data(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="query_data", parameters={"metric": "revenue"},
        ))
        assert "actor_messages" in obs.metadata
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}

    def test_actor_messages_after_present_evidence(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="present_evidence",
            parameters={"target": "cfo", "metric": "revenue", "value": 100, "interpretation": "data"},
        ))
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}

    def test_actor_messages_after_negotiate(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="negotiate",
            parameters={"target": "ceo", "position": "let's be cautious"},
        ))
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}

    def test_actor_messages_after_make_decision(self, env_easy):
        obs = env_easy.step(BoardroomAction(
            action_type="make_decision",
            parameters={"decision": "reduce churn", "parameters": {}, "explanation": "churn is the bottleneck"},
        ))
        assert set(obs.metadata["actor_messages"].keys()) == {"ceo", "cfo", "risk_officer"}
