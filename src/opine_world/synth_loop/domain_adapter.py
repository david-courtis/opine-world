"""Abstract domain adapter interface for the synthesis engine."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class DomainAdapter(ABC):
    """Interface each domain must implement to plug into the synthesis engine."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...


    @abstractmethod
    def write_replay_buffer(self, transitions: list, workspace_dir: Path) -> None:
        ...

    @abstractmethod
    def write_test_runner(self, workspace_dir: Path) -> None:
        ...

    @abstractmethod
    def write_initial_data(self, initial_state: Any, workspace_dir: Path) -> None:
        ...


    @abstractmethod
    def format_code_stub(self) -> str:
        ...

    @abstractmethod
    def format_synthesis_prompt(
        self,
        workspace_dir: str,
        test_runner_path: str,
        project_root: str,
    ) -> str:
        ...


    def format_goal_description(self, mission: str | None = None, **kwargs) -> str:
        """Return a natural-language goal description for context.txt."""
        if mission:
            return f"Goal: {mission}"
        return "Goal: unknown, infer from reward signals in the transition data."


    def format_transitions_for_context(
        self, transitions: list, max_examples: int = 10,
        mission: str | None = None,
    ) -> str:
        sections = []

        goal_desc = self.format_goal_description(mission)
        if goal_desc:
            sections.append(f"# Goal\n{goal_desc}\n")

        sections.append("# Observed Transitions")
        for t in transitions[:max_examples]:
            sections.append(repr(t))

        return "\n\n".join(sections)
