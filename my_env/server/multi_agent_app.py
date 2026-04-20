# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone FastAPI application for the MultiAgentBoardroomEnvironment.

Usage:
    uvicorn server.multi_agent_app:app --reload --host 0.0.0.0 --port 8001
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from my_env.models import BoardroomAction, BoardroomObservation
    from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment
except ImportError:  # pragma: no cover — inside Docker PYTHONPATH=/app/env
    from models import BoardroomAction, BoardroomObservation
    from multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment


app = create_app(
    MultiAgentBoardroomEnvironment,
    BoardroomAction,
    BoardroomObservation,
    env_name="multi_agent_boardroom",
    max_concurrent_envs=1,
)
