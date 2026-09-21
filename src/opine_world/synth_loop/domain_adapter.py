"""The interface a game domain implements to work with the engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DomainAdapter(ABC):
    """What a game domain must provide to the engine."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short name of the domain."""
        ...

    @abstractmethod
    def write_replay_buffer(self, transitions: list, workspace_dir: Path) -> None:
        """Write the recorded transitions into the synthesis workspace."""
        ...

    @abstractmethod
    def write_test_runner(
        self, workspace_dir: Path, structure: str = "free",
    ) -> None:
        """Write the script that checks a world model against the replay buffer."""
        ...

    @abstractmethod
    def write_initial_data(self, initial_state: Any, workspace_dir: Path) -> None:
        """Write any starting data the world model may load."""
        ...

    @abstractmethod
    def format_code_stub(self) -> str:
        """Return the starting game_engine.py."""
        ...

    @abstractmethod
    def format_synthesis_prompt(
        self,
        workspace_dir: str,
        test_runner_path: str,
        project_root: str,
    ) -> str:
        """Return the prompt for the synthesis agent."""
        ...

    def format_goal_description(self, mission: str | None = None, **kwargs) -> str:
        """Return the goal text for context.txt."""
        if mission:
            return f"Goal: {mission}"
        return "Goal: unknown, infer from reward signals in the transition data."

    def format_transitions_for_context(
        self, transitions: list, max_examples: int = 10,
        mission: str | None = None,
    ) -> str:
        """Return the first transitions as text for context.txt."""
        sections = []

        goal_desc = self.format_goal_description(mission)
        if goal_desc:
            sections.append(f"# Goal\n{goal_desc}\n")

        sections.append("# Observed Transitions")
        for t in transitions[:max_examples]:
            sections.append(repr(t))

        return "\n\n".join(sections)
