# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""OpenBoardroom environment server components."""

try:
    from my_env.server.boardroom_environment import BoardroomEnvironment
    from my_env.server.multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment
except ImportError:  # pragma: no cover — inside Docker
    from boardroom_environment import BoardroomEnvironment
    from multi_agent_boardroom_environment import MultiAgentBoardroomEnvironment

__all__ = ["BoardroomEnvironment", "MultiAgentBoardroomEnvironment"]
