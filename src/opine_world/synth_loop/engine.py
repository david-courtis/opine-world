"""Core synthesis engine: the main loop that any domain plugs into."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import aliases as aliases_mod
from .click_utils import action_id_of, action_label, is_click
from .domain_adapter import DomainAdapter
from .epistemic import dump_epistemic_matrix
from .ontology import Ontology
from .planner import (
    PlanResult,
    object_states_equal,
    plan_from_model,
)
from .planner import (
    load_model as load_planner_model,
)
from .prompt_safety import sanitize_model_visible_text
from .runlog import RunLog
from .spriteless_eta import refresh_spriteless_diagnostics


class EnvironmentInterface(Protocol):
    """Protocol for any environment the engine can drive."""

    def reset(self) -> dict: ...
    def step(self, action_id: int) -> tuple[dict, float, bool]: ...
    def get_available_actions(self) -> list[int]: ...
    def get_level_index(self) -> int: ...
    def get_mission(self) -> str | None: ...
    def extract_state(self) -> list[dict]: ...
    def describe_state(self, state: list[dict]) -> str: ...
    def compute_diff(self, before: list[dict], after: list[dict]) -> str: ...


class AnalyzerNoPlanError(RuntimeError):
    """Raised before env.step when the analyzer cannot produce a valid action."""


def _frame_pixel_diff(before, after, max_cells: int = 30) -> str:
    """Return a one-line summary of palette cells that changed between two frames.

    Returns "Nothing changed" when frames are equal or either is None."""
    if before is None or after is None:
        return "Nothing changed"
    try:
        b = before.tolist() if hasattr(before, "tolist") else before
        a = after.tolist() if hasattr(after, "tolist") else after
    except Exception:
        return "Nothing changed"
    if len(b) != len(a):
        return f"frame shape changed: {len(b)}x? -> {len(a)}x?"
    cells: list[str] = []
    total = 0
    for r in range(len(b)):
        rb, ra = b[r], a[r]
        if hasattr(rb, "tolist"):
            rb = rb.tolist()
        if hasattr(ra, "tolist"):
            ra = ra.tolist()
        if len(rb) != len(ra):
            return f"frame row {r} width changed: {len(rb)} -> {len(ra)}"
        for c in range(len(rb)):
            if int(rb[c]) != int(ra[c]):
                total += 1
                if len(cells) < max_cells:
                    cells.append(f"({r},{c})={int(ra[c])}")
    if total == 0:
        return "Nothing changed"
    if total <= max_cells:
        return f"{total} cells changed: " + ", ".join(cells)
    return (
        f"{total} cells changed; first {max_cells}: "
        + ", ".join(cells)
    )


def _aux_prompt(name: str) -> str:
    """Read an analyzer auxiliary prompt (prompts/analyzer/shared/<name>). Returns '' on miss."""
    try:
        return (
            Path(__file__).resolve().parent
            / "prompts" / "analyzer" / "shared" / name
        ).read_text(encoding="utf-8")
    except Exception:
        return ""


@dataclass
class TransitionRecord:
    """A single observed transition with full context."""
    before_state: list[dict]
    action_id: int
    action_name: str
    after_state: list[dict]
    diff_text: str
    reward: float
    done: bool
    timestep: int
    level: int
    before_frame: list[list[int]] | None = None
    after_frame: list[list[int]] | None = None


class EngineLogger:
    """Structured logger that writes to both stdout and a log file."""

    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._path = log_dir / "engine.log"
        if not self._path.exists():
            self._path.write_text("")
        self._step = 0

    def log(self, category: str, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}][{self._step:04d}][{category:12s}] {msg}"
        print(line)
        with open(self._path, "a") as f:
            f.write(line + "\n")

    def advance(self):
        self._step += 1

    def section(self, title: str):
        sep = "=" * 60
        self.log("", sep)
        self.log("", title)
        self.log("", sep)


@dataclass
class EngineConfig:
    """Configuration for the synthesis engine."""
    max_actions: int = 300
    synthesis_interval: int = 30
    ontology_measure_interval: int = 5
    synthesis_model: str = "claude-opus-4-7[1m]"
    synthesis_effort: str = "max"
    synthesis_max_turns: int = 100
    synthesis_timeout: int = 0
    min_transitions_for_synthesis: int = 10
    synthesis_defer_min_moves_after_divergence: int = 12
    synthesis_defer_max_errors: int = 6
    synthesis_defer_min_action_plans_after_divergence: int = 4
    seed: int = 42
    goal_hint: str = ""
    output_dir: str | Path = "results/run"

    agentic_consumer_model: str = "claude-opus-4-7"
    agentic_consumer_effort: str = "max"
    agentic_consumer_max_turns: int = 30
    agentic_consumer_timeout_s: int = 1800
    agentic_consumer_max_retries: int = 5

    subprocess_sandbox: bool = True

    debug: bool = False

    escalate_to_opus_after: int = 20
    escalation_synthesis_model: str = "opus"

    epistemic_alpha_0: float = 1.0
    epistemic_beta_0: float = 1.0
    epistemic_kappa: float = 2.0
    epistemic_sort_by: str = "thompson"

    synth_mode: str = "free"

    crystallisation_enabled: bool = False
    crystallisation_alias_min_score: int = 5
    crystallisation_alias_margin: int = 3
    crystallisation_modal_frac_threshold: float = 0.95
    crystallisation_stratum_n_min: int = 3

    frames_only: bool = False

    backend: str = "claude"
    codex_model: str = "gpt-5.5"
    codex_effort: str = "high"
    codex_home: str | None = None
    codex_image: str = "codex-agent"
    codex_network: str = "codex-filtered"
    codex_gateway: str | None = None
    synth_continue: bool = False

    claude_isolation: str = "docker"
    claude_image: str = "claude-agent"
    claude_network: str = "claude-filtered"
    claude_gateway: str | None = None
    claude_docker_memory: str = "12g"
    claude_docker_cpus: str = "2.0"
    claude_docker_pids_limit: str = "512"

    snapshot_dir: str | Path | None = None
    stop_and_snapshot_at_step: int | None = None

    synth_simplify_interval: int = 4
    critique_enabled: bool = False
    critique_interval: int = 5
    critique_repair_enabled: bool = False
    critique_repair_rounds: int = 1
    critique_recheck_after_repair: bool = True

    animation_analysis_enabled: bool = False
    animation_analysis_max_events: int = 8
    animation_analysis_timeout_s: int = 600

    planner_enabled: bool = True
    planner_autonomous: bool = False
    planner_after_levels_completed: int = 1
    planner_max_depth: int = 0
    planner_max_nodes: int = 0
    planner_timeout_s: int = 30
    planner_retry_interval: int = 10
    planner_max_click_targets: int = 0
    planner_require_completed_verification: bool = True
    planner_verify_max_levels: int = 0


class SynthesisEngine:
    """Domain-agnostic synthesis engine: explore, record transitions, and synthesize a world model."""

    def __init__(
        self,
        env: EnvironmentInterface,
        adapter: DomainAdapter,
        config: EngineConfig | None = None,
    ):
        self.env = env
        self.adapter = adapter
        self.config = config or EngineConfig()

        self.output_dir = Path(self.config.output_dir).resolve()
        self.logger = EngineLogger(self.output_dir)
        self.run_log = RunLog(self.output_dir / "run_log.txt")
        self.frames_dir = self.output_dir / "frames"
        try:
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            actions_path = self.frames_dir / "actions.jsonl"
            if not actions_path.exists():
                actions_path.write_text("")
        except Exception:
            pass
        try:
            import numpy as _np
            em_seed = self.output_dir / "epistemic_matrix.json"
            if not em_seed.exists():
                dump_epistemic_matrix(
                    [], em_seed,
                    alpha_0=self.config.epistemic_alpha_0,
                    beta_0=self.config.epistemic_beta_0,
                    kappa=self.config.epistemic_kappa,
                    sort_by=self.config.epistemic_sort_by,
                    rng=_np.random.default_rng(self.config.seed),
                )
        except Exception:
            pass
        self.ontology = Ontology(
            alpha_0=self.config.epistemic_alpha_0,
            kappa=self.config.epistemic_kappa,
        )
        try:
            ont_seed = self.output_dir / "ontology_error.json"
            if not ont_seed.exists():
                self.ontology.dump(ont_seed)
        except Exception:
            pass

        try:
            ss_seed = self.output_dir / "synth_status.json"
            if not ss_seed.exists():
                ss_seed.write_text(
                    json.dumps({
                        "synthesis_count": 0,
                        "model_present": False,
                        "engine_step": 0,
                        "n_transitions": 0,
                        "synth_learnings": "",
                        "critique_findings": "",
                        "critique_response": "",
                        "animation_findings": "",
                        "shared_model_updates": "",
                        "handoff_files": {
                            "world_model": "world_model.md",
                            "shared_model_updates": "shared_model_updates.md",
                            "current_level_reasoning_log": (
                                "level_1_reasoning_log.md"
                            ),
                            "current_level_report": "level_1_report.md",
                        },
                    }, indent=2)
                )
        except Exception:
            pass

        self.replay_buffer: list[TransitionRecord] = []
        self.rng = np.random.default_rng(self.config.seed)
        self.current_level = 0
        self.total_reward = 0.0
        self.levels_completed = 0
        self.game_won = False

        self.world_model_doc: str = ""

        self.goal_hypothesis_code: str | None = None
        self.goal_hypothesis_synthesis_run: int = 0
        self.goal_in_english: str | None = None
        self.synth_learnings: str = ""
        self.critique_findings: str = ""
        self.critique_response: str = ""
        self.animation_findings: str = ""
        self.shared_model_updates: str = ""
        self._animation_analysis_count: int = 0
        self._codex_synth_session_id: str | None = None
        self._ensure_shared_model_artifacts(current_level=self.current_level)

        self.synthesis_count = 0
        self.best_transition_accuracy = 0.0
        self.best_reward_accuracy = 0.0
        self.last_synthesis_step = -self.config.synthesis_interval
        self._pending_synthesis_step: int | None = None

        self.known_types: dict[str, dict] = {}
        self.level_states: dict[int, list[dict]] = {}
        self.level_frames: dict[int, list[list[int]]] = {}

        self.type_aliases: dict[str, list[dict]] = {}

        self._llm_plan: list[int] = []
        self._llm_plan_origin_step: int = -1
        self._llm_plan_no_effect_streak: int = 0

        self._warmup_queue: list[dict] = []

        self._consecutive_failed_syntheses: int = 0

        self._game_over_streak: int = 0
        self._steps_since_analyzer: list[dict] = []
        self._model_error_first_step: int | None = None
        self._model_error_last_step: int | None = None
        self._model_error_count: int = 0
        self._model_error_action_plan_count: int = 0
        self._model_error_level_completed: bool = False
        self._last_synthesis_level: int = int(self.current_level)
        self._last_synthesis_gate_log: str = ""
        self._current_action_plan_source: str | None = None
        self._div_model = None
        self._div_reward_model = None
        self._div_model_round: int = -1
        self._div_mask = frozenset()

        self._planner_queue: list[Any] = []
        self._planner_trace: list[dict] = []
        self._planner_expectation: dict | None = None
        self._planner_plan_origin_step: int = -1
        self._planner_blocked_round: int = -1
        self._planner_retry_after_step: int = 0
        self._planner_last_status: dict = {"ok": False, "reason": "not_run"}
        self._planner_last_gate_reason: str = ""
        self._planner_model = None
        self._planner_model_round: int = -1
        self._planner_consistency_key: tuple[int, int] | None = None
        self._planner_consistent: bool = False
        self._planner_verification_key: tuple[int, int, int] | None = None
        self._planner_verified: bool = False

        self.crystallised: bool = False
        self.crystallisation_step: int | None = None
        self.crystallised_partition: dict[str, str] = {}
        self.crystallised_scope_extra: dict[str, None] = {}

        self._last_plan_hint: str | None = None

        from .agentic_consumer import AgenticConsumer
        _codex = self.config.backend == "codex"
        self.agentic_consumer = AgenticConsumer(
            model=(self.config.codex_model if _codex
                   else self.config.agentic_consumer_model),
            max_turns=self.config.agentic_consumer_max_turns,
            timeout_s=self.config.agentic_consumer_timeout_s,
            log_dir=self.output_dir / "analyzer_logs",
            sandbox=self.config.subprocess_sandbox,
            engine_output_dir=self.output_dir,
            effort=(self.config.codex_effort if _codex
                    else self.config.agentic_consumer_effort),
            backend=self.config.backend,
            codex_home=self.config.codex_home,
            codex_image=self.config.codex_image,
            codex_network=self.config.codex_network,
            codex_gateway=self.config.codex_gateway,
            claude_isolation=self.config.claude_isolation,
            claude_image=self.config.claude_image,
            claude_network=self.config.claude_network,
            claude_gateway=self.config.claude_gateway,
            claude_docker_memory=self.config.claude_docker_memory,
            claude_docker_cpus=self.config.claude_docker_cpus,
            claude_docker_pids_limit=self.config.claude_docker_pids_limit,
        )

        self._resume_step: int | None = None
        self._actions_taken: list[int] = []

        self._stopped_for_snapshot: bool = False
        self._snapshot_completed_step: int | None = None

    def run(self) -> dict:
        """Run the full engine loop until game won or budget exhausted. Returns a summary dict."""
        start_step = self._resume_step or 0
        self.logger.section(
            "SYNTHESIS ENGINE START"
            if start_step == 0
            else f"SYNTHESIS ENGINE RESUME (from step {start_step})"
        )
        model_label = (
            self.config.codex_model
            if self.config.backend == "codex"
            else self.config.synthesis_model
        )
        self.logger.log(
            "CONFIG",
            f"max_actions={self.config.max_actions}, "
            f"synthesis_interval={self.config.synthesis_interval}, "
            f"backend={self.config.backend}, model={model_label}",
        )

        if start_step == 0:
            state = self.env.extract_state()
            self.current_level = self.env.get_level_index()
            self.level_states[self.current_level] = copy.deepcopy(state)
            if self.config.frames_only:
                try:
                    gf = getattr(self.env, "get_frame", None)
                    f0 = gf() if gf is not None else None
                    if f0 is not None:
                        self.level_frames[self.current_level] = (
                            f0.tolist() if hasattr(f0, "tolist") else f0
                        )
                except Exception:
                    pass
            self._discover_types(state)

            mission = self.env.get_mission()
            actions = self.env.get_available_actions()

            self.logger.log("INIT", f"Level {self.current_level + 1}, "
                            f"{len(state)} objects, "
                            f"actions={actions}")
            if not self.config.frames_only:
                self.logger.log(
                    "INIT", f"State: {self.env.describe_state(state)}"
                )
            if mission:
                self.logger.log("INIT", f"Mission: {mission}")

            self._update_world_model_doc(state, mission)
            self._ensure_shared_model_artifacts(
                current_level=self.current_level
            )

            if (
                self.config.frames_only
                and self.current_level not in self.level_frames
            ):
                try:
                    gf = getattr(self.env, "get_frame", None)
                    f0 = gf() if gf is not None else None
                    if f0 is not None:
                        self.level_frames[self.current_level] = (
                            f0.tolist() if hasattr(f0, "tolist") else f0
                        )
                except Exception:
                    pass

            if self.config.frames_only:
                self.run_log.write_header({
                    "max_actions": self.config.max_actions,
                    "synthesis_model": self.config.synthesis_model,
                    "agentic_consumer_model": (
                        self.config.agentic_consumer_model),
                    "seed": self.config.seed,
                    "available_actions": actions,
                    "mission": mission,
                    "frames_only": True,
                })
            else:
                self.run_log.write_header({
                    "max_actions": self.config.max_actions,
                    "synthesis_model": self.config.synthesis_model,
                    "agentic_consumer_model": (
                        self.config.agentic_consumer_model),
                    "seed": self.config.seed,
                    "available_actions": actions,
                    "mission": mission,
                    "initial_state_desc": self.env.describe_state(state),
                    "initial_state": state,
                })
            self._record_initial_frame()
            if not self.config.frames_only:
                self._queue_click_all_warmup(state, actions, start_step)
        else:
            state = self.env.extract_state()
            mission = self.env.get_mission()
            actions = self.env.get_available_actions()
            self.logger.log(
                "RESUME",
                f"env restored to step {start_step - 1}; "
                f"continuing main loop at step {start_step}"
            )
            self.run_log.append_resume_marker(start_step)
            self._ensure_shared_model_artifacts(
                current_level=self.current_level
            )
            try:
                n_ont = self.ontology.rehydrate(
                    self.output_dir / "ontology_error_trace.jsonl"
                )
                if n_ont:
                    self.logger.log(
                        "ONTOLOGY",
                        f"rehydrated {n_ont} trace records on resume"
                    )
            except Exception as exc:
                self.logger.log(
                    "ONTOLOGY",
                    f"rehydrate failed: {type(exc).__name__}: {exc}"
                )

        last_step = start_step - 1
        for step in range(start_step, self.config.max_actions):
            last_step = step
            self.logger.advance()
            self._cur_step = step

            if self.game_won:
                self.logger.log("DONE", "Game won!")
                break

            if self._pending_synthesis_step is not None:
                pending_step = int(self._pending_synthesis_step)
                self.logger.section(
                    f"PENDING SYNTHESIS RESUME (from step {pending_step})"
                )
                self._pending_synthesis_step = None
                if self._should_synthesize(pending_step):
                    self._run_synthesis(pending_step, mission, state)
                else:
                    self.logger.log(
                        "SYNTHESIS",
                        "skipped stale pending synthesis; no fresh "
                        "post-plan model mismatch remains",
                    )
                self._save_checkpoint(pending_step)

            if (
                self._snapshot_due(step)
                and not self._warmup_queue
                and not self._planner_queue
                and not (self._llm_plan and self._plan_is_valid(step))
                and not (
                    hasattr(self.env, "is_game_over")
                    and self.env.is_game_over()
                )
            ):
                self._stopped_for_snapshot = True
                self._take_snapshot(step - 1, reason="stop_at_step")
                self.logger.log(
                    "SNAPSHOT",
                    f"stopping at clean analyzer boundary: completed_step="
                    f"{step - 1}, queued plan len={len(self._llm_plan)}, "
                    f"queued planner len={len(self._planner_queue)}, "
                    f"target={self.config.stop_and_snapshot_at_step}",
                )
                break

            action = self._choose_action(state, actions, step)
            action_id = action_id_of(action)
            pre_frame = None
            try:
                gf = getattr(self.env, "get_frame", None)
                if gf is not None:
                    pre_frame = gf()
            except Exception:
                pre_frame = None
            action_name = action_label(action, frame=pre_frame)
            self._actions_taken.append(action)

            before_state = copy.deepcopy(state)
            level_before = self.env.get_level_index()

            self.logger.log("ACTION", f"{action_name}")
            new_state, reward, done = self.env.step(action)
            state = self.env.extract_state()

            level_after = self.env.get_level_index()
            if self.config.frames_only:
                diff = "(frames_only)"
            else:
                diff = aliases_mod.annotate_text(
                    self.env.compute_diff(before_state, state),
                    self.type_aliases,
                )

            post_frame = None
            try:
                gf = getattr(self.env, "get_frame", None)
                if gf is not None:
                    post_frame = gf()
            except Exception:
                post_frame = None

            if self.config.frames_only:
                diff = _frame_pixel_diff(pre_frame, post_frame)

            before_frame_serialised: list[list[int]] | None = None
            after_frame_serialised: list[list[int]] | None = None
            if self.config.frames_only:
                if pre_frame is not None:
                    try:
                        before_frame_serialised = pre_frame.tolist() if hasattr(pre_frame, "tolist") else pre_frame
                    except Exception:
                        before_frame_serialised = None
                if post_frame is not None:
                    try:
                        after_frame_serialised = post_frame.tolist() if hasattr(post_frame, "tolist") else post_frame
                    except Exception:
                        after_frame_serialised = None

            transition = TransitionRecord(
                before_state=before_state if not self.config.frames_only else [],
                action_id=action_id,
                action_name=action_name,
                after_state=copy.deepcopy(state) if not self.config.frames_only else [],
                diff_text=diff,
                reward=reward,
                done=done,
                timestep=step,
                level=level_before,
                before_frame=before_frame_serialised,
                after_frame=after_frame_serialised,
            )
            if is_click(action):
                transition.click_x = int(action["x"])
                transition.click_y = int(action["y"])
            self.replay_buffer.append(transition)

            self._record_step_frame(step, transition)

            self._record_execution_divergence(
                step,
                action,
                before_state,
                state,
                pre_frame,
                post_frame,
                reward=reward,
                done=done,
            )
            self._validate_planner_execution(
                step=step,
                action=action,
                actual_state=state,
                actual_frame=post_frame,
                reward=reward,
                done=done,
            )

            ascii_frame = None
            try:
                get_frame = getattr(self.env, "get_frame", None)
                if get_frame is not None:
                    frame = get_frame()
                    if frame is not None:
                        from .vision import frame_to_ascii
                        ascii_frame = frame_to_ascii(frame)
            except Exception:
                ascii_frame = None
            action_sequence_diffs: list[str] = []
            try:
                during = getattr(self.env, "_last_during_frames", None) or []
                if during:
                    from .vision import diff_to_ascii
                    before_f = getattr(self.env, "_last_before_frame", None)
                    after_f = getattr(self.env, "_last_after_frame", None)
                    seq = [before_f] + list(during) + [after_f]
                    seq = [f for f in seq if f is not None]
                    for a, b in zip(seq[:-1], seq[1:]):
                        d = diff_to_ascii(a, b)
                        changed = [
                            f"({r},{c})={ch}"
                            for r, line in enumerate(d.splitlines())
                            for c, ch in enumerate(line)
                            if ch != "."
                        ]
                        if not changed:
                            action_sequence_diffs.append("(no change)")
                        elif len(changed) <= 40:
                            action_sequence_diffs.append(", ".join(changed))
                        else:
                            action_sequence_diffs.append(
                                f"{len(changed)} cells changed; first 20: "
                                + ", ".join(changed[:20])
                            )
            except Exception:
                action_sequence_diffs = []
            if self.config.frames_only:
                rl_state_desc = ""
                rl_after_state: list[dict] = []
            else:
                rl_state_desc = aliases_mod.annotate_text(
                    self.env.describe_state(state), self.type_aliases,
                )
                rl_after_state = state
            self.run_log.append_step(
                step=step,
                action_id=action_id,
                action_name=action_name,
                level=level_before,
                reward=float(reward),
                done=bool(done),
                diff_text=diff,
                state_desc=rl_state_desc,
                after_state=rl_after_state,
                ascii_frame=ascii_frame,
                action_sequence_diffs=action_sequence_diffs or None,
            )

            if self.config.frames_only:
                self._refresh_spriteless_diagnostics(step, reason="step")
            else:
                try:
                    dump_epistemic_matrix(
                        self._serialize_transitions(),
                        self.output_dir / "epistemic_matrix.json",
                        alpha_0=self.config.epistemic_alpha_0,
                        beta_0=self.config.epistemic_beta_0,
                        kappa=self.config.epistemic_kappa,
                        sort_by=self.config.epistemic_sort_by,
                        rng=self.rng,
                    )
                except Exception as exc:
                    self.logger.log(
                        "EPISTEMIC", f"dump failed: {type(exc).__name__}: {exc}"
                    )

            if not self.config.frames_only:
                try:
                    interval = max(1, int(self.config.ontology_measure_interval))
                    if step == start_step or (step % interval == 0):
                        self.ontology.measure(
                            step,
                            self._serialize_transitions(),
                            aliases=self.type_aliases or None,
                        )
                        self.ontology.dump(
                            self.output_dir / "ontology_error.json"
                        )
                        trace_rec = self.ontology._trace[-1]
                        self.ontology.append_trace_line(
                            self.output_dir / "ontology_error_trace.jsonl",
                            trace_rec,
                        )
                        log_msg = (
                            f"eta={trace_rec['eta']:.4f} "
                            f"eta*={trace_rec['eta_star']:.4f} "
                            f"(-{trace_rec['eta_reduction']:.4f} via "
                            f"{trace_rec['best_candidate']}) "
                            f"strata={trace_rec['n_strata']} "
                            f"n={trace_rec['n_transitions']}"
                        )
                        if "eta_extended" in trace_rec:
                            log_msg += (
                                f" | eta_ext={trace_rec['eta_extended']:.4f}"
                                f" (role={trace_rec['eta_role_component']:.4f},"
                                f" eff={trace_rec['eta_effect_component']:.4f},"
                                f" K={trace_rec['n_candidate_roles']})"
                            )
                        self.logger.log("ONTOLOGY", log_msg)
                except Exception as exc:
                    self.logger.log(
                        "ONTOLOGY",
                        f"measure failed: {type(exc).__name__}: {exc}"
                    )

            self.logger.log("OBSERVATION", f"Diff: {diff}")
            if reward > 0:
                self.total_reward += reward
                self.logger.log("REWARD", f"reward={reward}, total={self.total_reward}")

            if diff == "Nothing changed":
                self._llm_plan_no_effect_streak += 1
            else:
                self._llm_plan_no_effect_streak = 0

            if reward > 0 or level_after > level_before:
                if self._llm_plan:
                    self.logger.log(
                        "ANALYZER",
                        f"score-change flush: cleared {len(self._llm_plan)} "
                        f"queued actions (reward={reward}, level_advance="
                        f"{level_after > level_before})"
                    )
                    self._llm_plan.clear()
                if self._planner_queue:
                    self.logger.log(
                        "PLANNER",
                        f"score-change flush: cleared "
                        f"{len(self._planner_queue)} queued planner action(s)"
                    )
                self._clear_planner_plan()

            self._record_completed_action_plan_after_divergence(step)

            if level_after > level_before:
                if self._model_error_first_step is not None:
                    self._model_error_level_completed = True
                self.levels_completed += 1
                self.current_level = level_after
                self.logger.log("LEVEL", f"Advanced to level {level_after + 1} "
                                f"({self.levels_completed} completed)")
                self.run_log.append_level_advance(step, level_before, level_after)
                self._last_plan_hint = None
                self._new_level_pending = True
                self._level_start_step = step + 1
                self._trouble1_next = 100
                self._trouble2_next = 200

                state = self.env.extract_state()
                self.level_states[self.current_level] = copy.deepcopy(state)
                if self.config.frames_only:
                    try:
                        gf = getattr(self.env, "get_frame", None)
                        f0 = gf() if gf is not None else None
                        if f0 is not None:
                            self.level_frames[self.current_level] = (
                                f0.tolist() if hasattr(f0, "tolist") else f0
                            )
                    except Exception:
                        pass
                self._discover_types(state)

                if not self.config.frames_only:
                    self.logger.log(
                        "LEVEL",
                        f"New state: {self.env.describe_state(state)}",
                    )

                self._update_world_model_doc(state, mission)
                self._ensure_shared_model_artifacts(
                    current_level=self.current_level,
                    completed_level=level_before,
                )

            if hasattr(self.env, 'is_game_won') and self.env.is_game_won():
                self.game_won = True
                self.logger.log("WIN", "Game completed!")
                if self._llm_plan:
                    self.logger.log(
                        "ANALYZER",
                        f"win flush: cleared {len(self._llm_plan)} queued actions"
                    )
                    self._llm_plan.clear()
                if self._planner_queue:
                    self.logger.log(
                        "PLANNER",
                        f"win flush: cleared {len(self._planner_queue)} "
                        "queued planner action(s)"
                    )
                self._clear_planner_plan()
                self._save_checkpoint(step)
                break

            if (
                self.config.crystallisation_enabled
                and not self.crystallised
            ):
                partition = self._check_crystallisation(step)
                if partition is not None:
                    self._commit_crystallisation(step, partition)

            if self._should_synthesize(step):
                self._pending_synthesis_step = step
                self._save_checkpoint(step)
                self._run_synthesis(step, mission, state)
                self._pending_synthesis_step = None

            self._save_checkpoint(step)

            if (
                self.config.stop_and_snapshot_at_step is not None
                and step + 1 >= self.config.stop_and_snapshot_at_step + 50
            ):
                self._stopped_for_snapshot = True
                self._take_snapshot(step, reason="hard_stop_margin")
                self.logger.log(
                    "SNAPSHOT",
                    f"hard-stop snapshot at end-of-step {step} "
                    f"(no clean boundary within margin of "
                    f"{self.config.stop_and_snapshot_at_step})",
                )
                break

        if (
            not self._stopped_for_snapshot
            and last_step - self.last_synthesis_step >= 5
            and len(self.replay_buffer) > 0
        ):
            final_should_run = False
            if self.synthesis_count == 0:
                final_should_run = True
            elif self._recent_execution_diverged():
                gate = self._synthesis_gate_status(last_step)
                final_should_run = (
                    bool(gate.get("ready"))
                    and not self._model_is_consistent()
                )
            if final_should_run:
                self.logger.section("FINAL SYNTHESIS")
                self._run_synthesis(last_step, mission, state)

        final_step = (
            self._snapshot_completed_step
            if (self._stopped_for_snapshot
                and self._snapshot_completed_step is not None)
            else last_step
        )
        summary = self._build_summary(final_step)
        if not self.config.debug and not self._stopped_for_snapshot:
            self.logger.log(
                "PUBLISH",
                "finalizing clean run output (use --debug to keep the full "
                "operational output)",
            )
            try:
                from .publish import finalize_clean
                finalize_clean(
                    self.output_dir,
                    path_subs=[(str(self.output_dir), "<run>")],
                )
            except Exception as exc:
                self.logger.log(
                    "PUBLISH",
                    f"finalize_clean failed: {type(exc).__name__}: {exc}",
                )
        return summary

    def _queue_click_all_warmup(
        self, state: list[dict], actions: list[int], start_step: int,
    ) -> None:
        """Queue one click per distinct visible non-background sprite at game start.

        Skipped when ACTION6 is not available. ARC-AGI-3 invariant: every level
        with ACTION6 has at least one click that produces a transition.
        """
        if 6 not in actions:
            return
        targets: list[tuple[int, int, str, list[str]]] = []
        seen: set[tuple[int, int]] = set()
        for o in state:
            if not o.get("visible", True):
                continue
            if o.get("w", 0) >= 64 and o.get("h", 0) >= 64:
                continue
            if "ihdgageizm" in (o.get("tags") or []):
                continue
            dx = int(o.get("display_x", o.get("x", 0)))
            dy = int(o.get("display_y", o.get("y", 0)))
            dw = int(o.get("display_w", o.get("w", 1)))
            dh = int(o.get("display_h", o.get("h", 1)))
            cx = max(0, min(63, dx + max(0, dw // 2)))
            cy = max(0, min(63, dy + max(0, dh // 2)))
            if (cx, cy) in seen:
                continue
            seen.add((cx, cy))
            targets.append((cx, cy, o.get("name", "?"),
                            list(o.get("tags") or [])))

        if not targets:
            return

        self._warmup_queue = [
            {"action_id": 6, "x": cx, "y": cy} for cx, cy, _, _ in targets
        ]
        target_summary = "; ".join(
            f"{name}@display({cx},{cy})" for cx, cy, name, _ in targets
        )
        self.run_log.append_note(
            step=start_step,
            source="engine",
            text=(
                f"AUTO-CLICK WARMUP: queueing {len(targets)} click "
                f"action(s) at game start, one per distinct visible "
                f"non-background sprite. ARC-3 invariant: every level "
                f"on which ACTION6 is available has at least one "
                f"click target that produces a transition. These "
                f"clicks are recorded as normal transitions under "
                f"phi_1 (synth must predict their effects). Analyzer: "
                f"review the resulting STEP blocks "
                f"{start_step}..{start_step + len(targets) - 1} on "
                f"your first call to learn which click(s) had effect. "
                f"Targets: {target_summary}"
            ),
        )
        self.logger.log(
            "WARMUP",
            f"queued {len(targets)} click-all warmup actions"
            f" (steps {start_step}..{start_step + len(targets) - 1})",
        )

    def _choose_action(
        self, state: list[dict], actions: list[int], step: int,
    ) -> Any | None:
        """Choose next action.

        Priority: GAME-OVER recovery, WARMUP queue, C3 planner queue,
        analyzer queue, fresh C3 plan, fresh analyzer call, hard failure.
        """
        self._current_action_plan_source = None
        if hasattr(self.env, "is_game_over") and self.env.is_game_over():
            self._game_over_streak += 1
            self._llm_plan.clear()
            self._warmup_queue.clear()
            self._clear_planner_plan()
            if self._game_over_streak >= 2:
                self.logger.log(
                    "GAME_OVER",
                    f"level budget exhausted; forcing RESET (level retry) after "
                    f"{self._game_over_streak} frozen step(s)")
                self._game_over_streak = 0
                self._current_action_plan_source = "recovery"
                return 0
            self.logger.log(
                "GAME_OVER",
                "level budget exhausted; board frozen -- agent should RESET (0)")
        else:
            self._game_over_streak = 0

        if self._warmup_queue:
            action = self._warmup_queue.pop(0)
            aid = action_id_of(action)
            if aid in actions:
                self.logger.log(
                    "WARMUP",
                    f"{action_label(action)} "
                    f"(remaining={len(self._warmup_queue)})",
                )
                self._current_action_plan_source = "warmup"
                return action
            return self._choose_action(state, actions, step)

        if getattr(self.config, "planner_autonomous", False):
            planner_action = self._planner_choose_action(
                state, actions, step, allow_new_plan=False,
            )
            if planner_action is not None:
                self._current_action_plan_source = "planner"
                return planner_action

        if self._llm_plan and self._plan_is_valid(step):
            action = self._llm_plan.pop(0)
            aid = action_id_of(action)
            if aid not in actions:
                return self._choose_action(state, actions, step)
            self.logger.log("PLAN",
                f"{action_label(action)} (remaining={len(self._llm_plan)}, "
                f"plan_age={step - self._llm_plan_origin_step})")
            self._current_action_plan_source = "analyzer"
            return action

        if self._llm_plan and not self._plan_is_valid(step):
            self.logger.log("PLAN",
                f"abandoning stale plan ({len(self._llm_plan)} steps left, "
                f"no_effect_streak={self._llm_plan_no_effect_streak}, "
                f"age={step - self._llm_plan_origin_step})")
            self._llm_plan.clear()
            self._llm_plan_no_effect_streak = 0

        if getattr(self.config, "planner_autonomous", False):
            planner_action = self._planner_choose_action(
                state, actions, step, allow_new_plan=True,
            )
            if planner_action is not None:
                self._current_action_plan_source = "planner"
                return planner_action

        ag_action = self._agentic_choose_action(state, actions, step)
        if ag_action is not None:
            self._current_action_plan_source = "analyzer"
            return ag_action

        msg = (
            f"No valid action at step {step}: planner unavailable and analyzer "
            "did not produce a valid plan. Refusing to mutate the environment."
        )
        self.logger.log("ANALYZER", msg)
        try:
            (self.output_dir / "analyzer_failure.json").write_text(
                json.dumps(
                    {
                        "step": step,
                        "completed_step": step - 1,
                        "level": self.current_level,
                        "reason": "no_valid_action",
                        "message": msg,
                        "ts": time.time(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
        except Exception:
            pass
        self._save_checkpoint(step - 1)
        raise AnalyzerNoPlanError(msg)

    def _clear_planner_plan(self) -> None:
        """Drop any queued C3 trace without changing model-level block/cooldown."""
        self._planner_queue = []
        self._planner_trace = []
        self._planner_expectation = None

    def _latest_game_engine_path(self) -> Path:
        return (
            self.output_dir / "synthesis"
            / f"run_{self.synthesis_count:03d}" / "game_engine.py"
        ).resolve()

    def _load_planner_model(self):
        """Import latest game_engine.py for C3 planning, cached by synthesis round."""
        if self.synthesis_count <= 0:
            return None
        if self._planner_model_round == self.synthesis_count:
            return self._planner_model
        self._planner_model = None
        self._planner_model_round = self.synthesis_count
        code_path = self._latest_game_engine_path()
        if not code_path.exists():
            return None
        try:
            self._planner_model = load_planner_model(
                code_path,
                module_name=(
                    f"_planner_model_{self.synthesis_count}_"
                    f"{id(self)}"
                ),
            )
        except Exception as exc:
            self._planner_last_status = {
                "ok": False,
                "reason": f"load failed: {type(exc).__name__}: {exc}",
            }
            self._planner_model = None
        return self._planner_model

    def _planner_model_is_consistent(self) -> bool:
        """Cached replay verification for planner gating."""
        key = (int(self.synthesis_count), len(self.replay_buffer))
        if self._planner_consistency_key == key:
            return bool(self._planner_consistent)
        self._planner_consistency_key = key
        self._planner_consistent = bool(self._model_is_consistent())
        return bool(self._planner_consistent)

    def _planner_scope_tags(self) -> set[str] | None:
        if not self.crystallised:
            return None
        return set(self._compute_scope_tags())

    def _current_planner_state(
        self, state: list[dict],
    ) -> Any | None:
        if not self.config.frames_only:
            return copy.deepcopy(state)
        get_frame = getattr(self.env, "get_frame", None)
        if get_frame is None:
            return None
        try:
            frame = get_frame()
            if frame is None:
                return None
            return frame.tolist() if hasattr(frame, "tolist") else frame
        except Exception:
            return None

    def _verify_planner_on_completed_levels(
        self, actions: list[int], step: int,
    ) -> bool:
        """Baseline1-style planner gate: completed level starts must be solvable.

        This is model-side verification only. It never executes real actions.
        Failures disable C3 for the current synthesis round, leaving exploration
        and CEGIS to continue normally.
        """
        key = (
            int(self.synthesis_count),
            int(self.levels_completed),
            len(self.replay_buffer),
        )
        if self._planner_verification_key == key:
            return bool(self._planner_verified)
        self._planner_verification_key = key
        self._planner_verified = False

        model = self._load_planner_model()
        if model is None:
            return False
        max_levels = int(self.config.planner_verify_max_levels or 0)
        if self.config.frames_only:
            starts = self.level_frames
        else:
            starts = self.level_states
        completed = [
            int(level)
            for level in starts
            if int(level) < int(self.current_level)
        ]
        completed = sorted(completed)
        if max_levels > 0:
            completed = completed[-max_levels:]
        if not completed:
            self._planner_last_status = {
                "ok": False,
                "reason": "no completed level starts to verify",
            }
            return False

        rows: list[dict] = []
        for level in completed:
            start_state = copy.deepcopy(starts[level])
            result = plan_from_model(
                model,
                start_state,
                actions,
                frames_only=self.config.frames_only,
                max_depth=int(self.config.planner_max_depth),
                max_nodes=int(self.config.planner_max_nodes),
                timeout_s=int(self.config.planner_timeout_s),
                max_click_targets=int(self.config.planner_max_click_targets),
            )
            row = {
                "level": int(level),
                **result.summary(),
            }
            rows.append(row)
            if not result.ok:
                self._planner_blocked_round = int(self.synthesis_count)
                self._planner_retry_after_step = (
                    int(step) + int(self.config.planner_retry_interval)
                )
                self._planner_last_status = {
                    "ok": False,
                    "reason": (
                        f"completed-level verification failed on "
                        f"L{level + 1}: {result.reason}"
                    ),
                    "verification": rows,
                }
                self._write_planner_verification(rows, ok=False)
                self.logger.log(
                    "PLANNER",
                    self._planner_last_status["reason"],
                )
                return False

        self._planner_verified = True
        self._planner_last_status = {
            "ok": True,
            "reason": "completed-level verification passed",
            "verification": rows,
        }
        self._write_planner_verification(rows, ok=True)
        self.logger.log(
            "PLANNER",
            f"verified completed level starts: "
            f"{', '.join('L' + str(r['level'] + 1) for r in rows)}",
        )
        return True

    def _write_planner_verification(
        self, rows: list[dict], *, ok: bool,
    ) -> None:
        payload = {
            "ok": bool(ok),
            "synthesis_count": int(self.synthesis_count),
            "levels_completed": int(self.levels_completed),
            "rows": rows,
        }
        paths = [self.output_dir / "planner_verification.json"]
        latest = (
            self.output_dir / "synthesis"
            / f"run_{self.synthesis_count:03d}"
        )
        if latest.exists():
            paths.append(latest / "planner_verification.json")
        for path in paths:
            try:
                path.write_text(json.dumps(payload, indent=2, default=str))
            except Exception:
                pass
        fail_reasons = "; ".join(
            str(r.get("reason", "")) for r in rows if not r.get("ok")
        )
        self._append_planner_feedback({
            "kind": "completed_level_verification",
            "ok": bool(ok),
            "reason": fail_reasons or ("passed" if ok else "failed"),
        })

    def _planner_gate(
        self, actions: list[int], step: int,
    ) -> tuple[bool, str]:
        if not getattr(self.config, "planner_enabled", True):
            return False, "disabled"
        if self.synthesis_count <= 0:
            return False, "no synthesized model"
        if int(self.levels_completed) < int(
            self.config.planner_after_levels_completed
        ):
            return False, "waiting for first completed level"
        if self.config.crystallisation_enabled and not self.crystallised:
            return False, "waiting for crystallisation"
        try:
            if hasattr(self.env, "is_game_over") and self.env.is_game_over():
                return False, "game over"
        except Exception:
            pass
        if self._recent_execution_diverged():
            return False, (
                "model diverged; waiting for CEGIS repair before C3 planner"
            )
        if step < int(self._planner_retry_after_step):
            return False, (
                f"cooldown until step {self._planner_retry_after_step}"
            )
        if int(self._planner_blocked_round) == int(self.synthesis_count):
            return False, "blocked until next synthesis"
        if self._load_planner_model() is None:
            return False, "model import failed"
        if not self._planner_model_is_consistent():
            return False, "model is not replay-consistent"
        if getattr(self.config, "planner_require_completed_verification", True):
            if not self._verify_planner_on_completed_levels(actions, step):
                return False, "completed-level verification failed"
        return True, "ready"

    def _log_planner_gate(self, reason: str) -> None:
        if reason == self._planner_last_gate_reason:
            return
        self._planner_last_gate_reason = reason
        if reason not in ("disabled", "no synthesized model"):
            self.logger.log("PLANNER", f"gated: {reason}")

    def _prime_planner_plan(
        self, state: list[dict], actions: list[int], step: int,
    ) -> bool:
        """Compute and queue a full C3 plan without popping the first action."""
        if self._planner_queue:
            return True
        ok, reason = self._planner_gate(actions, step)
        if not ok:
            self._log_planner_gate(reason)
            return False
        start_state = self._current_planner_state(state)
        if start_state is None:
            self._log_planner_gate("current frame/state unavailable")
            return False
        model = self._load_planner_model()
        if model is None:
            return False

        result: PlanResult = plan_from_model(
            model,
            start_state,
            actions,
            frames_only=self.config.frames_only,
            max_depth=int(self.config.planner_max_depth),
            max_nodes=int(self.config.planner_max_nodes),
            timeout_s=int(self.config.planner_timeout_s),
            max_click_targets=int(self.config.planner_max_click_targets),
        )
        self._planner_last_status = result.summary()
        self._planner_last_gate_reason = ""
        if not result.ok:
            self._planner_retry_after_step = (
                int(step) + int(self.config.planner_retry_interval)
            )
            self.logger.log(
                "PLANNER",
                f"no plan ({result.source}, nodes={result.nodes}, "
                f"depth={result.depth}): {result.reason}",
            )
            self._append_planner_feedback({
                "kind": "live_plan",
                "ok": False,
                "reason": result.reason,
                "source": result.source,
                "nodes": int(result.nodes),
                "depth": int(result.depth),
            })
            return False

        self._planner_queue = [copy.deepcopy(a) for a in result.plan]
        self._planner_trace = [s.to_dict() for s in result.predicted_steps]
        self._planner_expectation = None
        self._planner_plan_origin_step = int(step)
        self._planner_retry_after_step = int(step)
        self.logger.log(
            "PLANNER",
            f"queued {len(self._planner_queue)} action(s) from "
            f"{result.source} (nodes={result.nodes}, depth={result.depth})",
        )
        try:
            self.run_log.append_note(
                step=step,
                source="planner",
                text=(
                    f"plan={self._planner_queue} source={result.source} "
                    f"nodes={result.nodes} depth={result.depth}"
                ),
            )
        except Exception:
            pass
        self._append_planner_feedback({
            "kind": "live_plan",
            "ok": True,
            "source": result.source,
            "plan_len": len(self._planner_queue),
            "nodes": int(result.nodes),
        })
        return True

    def _planner_choose_action(
        self,
        state: list[dict],
        actions: list[int],
        step: int,
        *,
        allow_new_plan: bool,
    ) -> Any | None:
        if not self._planner_queue and allow_new_plan:
            self._prime_planner_plan(state, actions, step)
        while self._planner_queue:
            action = self._planner_queue.pop(0)
            expected = (
                self._planner_trace.pop(0) if self._planner_trace else None
            )
            aid = action_id_of(action)
            if aid not in actions:
                self.logger.log(
                    "PLANNER",
                    f"discarded unavailable action {action!r}",
                )
                self._clear_planner_plan()
                return None
            self._planner_expectation = expected
            self.logger.log(
                "PLANNER",
                f"{action_label(action)} "
                f"(remaining={len(self._planner_queue)}, "
                f"plan_age={step - self._planner_plan_origin_step})",
            )
            return action
        return None

    def _validate_planner_execution(
        self,
        *,
        step: int,
        action: Any,
        actual_state: list[dict],
        actual_frame: Any,
        reward: float,
        done: bool,
    ) -> None:
        expected = self._planner_expectation
        if not expected:
            return
        self._planner_expectation = None

        expected_action = expected.get("action")
        mismatch_reasons: list[str] = []
        try:
            if action_id_of(expected_action) != action_id_of(action):
                mismatch_reasons.append("action changed before execution")
        except Exception:
            mismatch_reasons.append("expected action malformed")

        pred_reward = float(expected.get("reward", 0.0) or 0.0)
        pred_done = bool(expected.get("done", False))
        observed_done = bool(done)
        observed_reward = float(reward or 0.0)
        pred_complete = pred_reward > 0 or pred_done
        obs_complete = observed_reward > 0 or observed_done
        if pred_complete != obs_complete:
            mismatch_reasons.append(
                f"completion mismatch predicted={pred_complete} "
                f"observed={obs_complete}"
            )
        elif pred_done != observed_done:
            mismatch_reasons.append(
                f"done mismatch predicted={pred_done} "
                f"observed={observed_done}"
            )
        elif abs(pred_reward - observed_reward) > 1e-6:
            mismatch_reasons.append(
                f"reward mismatch predicted={pred_reward} "
                f"observed={observed_reward}"
            )

        if not mismatch_reasons and not obs_complete:
            predicted_state = expected.get("next_state")
            if self.config.frames_only:
                if not self._frames_equal(
                    predicted_state,
                    actual_frame,
                    getattr(self._load_planner_model(), "move_counter_mask", None),
                ):
                    mismatch_reasons.append("predicted frame mismatch")
            else:
                if not object_states_equal(
                    predicted_state,
                    actual_state,
                    scope_tags=self._planner_scope_tags(),
                ):
                    mismatch_reasons.append("predicted object state mismatch")

        if not mismatch_reasons:
            self.logger.log("PLANNER", f"validated step {step}")
            return

        self._planner_blocked_round = int(self.synthesis_count)
        self._planner_retry_after_step = (
            int(step) + int(self.config.planner_retry_interval)
        )
        self._clear_planner_plan()
        self._planner_last_status = {
            "ok": False,
            "reason": "; ".join(mismatch_reasons),
            "blocked_round": self._planner_blocked_round,
            "blocked_at_step": int(step),
        }
        if self._steps_since_analyzer:
            self._steps_since_analyzer[-1]["diverged"] = True
        msg = (
            f"aborted on step {step}: "
            f"{self._planner_last_status['reason']}"
        )
        self.logger.log("PLANNER", msg)
        self._append_planner_feedback({
            "kind": "execution_divergence",
            "ok": False,
            "reason": self._planner_last_status["reason"],
            "step": int(step),
        })
        try:
            self.run_log.append_note(
                step=step,
                source="planner",
                text=msg,
            )
        except Exception:
            pass


    def _plan_is_valid(self, step: int) -> bool:
        """Returns False if 3 or more consecutive no-effect steps have occurred."""
        if self._llm_plan_no_effect_streak >= 3:
            return False
        return True

    _ANALYZER_RETRY_NUDGE = (
        "CRITICAL: Your previous response did not result in a valid "
        "next_actions.json. You MUST end by writing next_actions.json "
        "via Bash. Use python:\n"
        "  python3 -c 'import json; "
        'open("next_actions.json","w").write(json.dumps('
        '{"plan":[1,2],"reasoning":"why"}))\'\n'
        "Do NOT skip writing the file. Without next_actions.json the "
        "engine will fail this segment without executing any action."
    )

    def _select_aux_prompt(self, step: int) -> str:
        """Pick a state-triggered analyzer nudge (baseline1-style protocol). '' when
        nothing applies. Priority: game-over > new-level > trouble2/1 > stuck.

        Trouble escalates by steps-on-level (no level advance): trouble1 at +100,
        trouble2 at +200 which also forces a fresh analyzer session (their
        new_session() tunnel-break). Trackers are reset on each level advance."""
        try:
            gover = (hasattr(self.env, "is_game_over")
                     and self.env.is_game_over())
        except Exception:
            gover = False
        if gover:
            return _aux_prompt("death.txt")
        if step == 0:
            self.logger.log("AUX_PROMPT", "initial_diverse_probe")
            return _aux_prompt("initial_diverse_probe.txt")
        if getattr(self, "_new_level_pending", False):
            self._new_level_pending = False
            return _aux_prompt("new_level.txt")
        on_level = step - getattr(self, "_level_start_step", 0)
        if on_level >= getattr(self, "_trouble2_next", 200):
            self._trouble2_next = getattr(self, "_trouble2_next", 200) + 200
            self._trouble1_next = on_level + 100
            reset_on_stuck = (
                os.environ.get(
                    "ARC3_ANALYZER_TROUBLE2_FRESH_SESSION", "0"
                ) == "1"
            )
            if reset_on_stuck:
                try:
                    self.agentic_consumer._needs_fresh_session = True
                except Exception:
                    pass
            self.logger.log(
                "AUX_PROMPT",
                f"trouble2 ({'fresh session' if reset_on_stuck else 'nudge only'})"
                f" on_level={on_level}")
            return _aux_prompt("trouble2.txt")
        if on_level >= getattr(self, "_trouble1_next", 100):
            self._trouble1_next = getattr(self, "_trouble1_next", 100) + 100
            self.logger.log("AUX_PROMPT", f"trouble1 on_level={on_level}")
            return _aux_prompt("trouble1.txt")
        if self._llm_plan_no_effect_streak >= 3:
            return _aux_prompt("stuck.txt")
        return ""

    def _agentic_choose_action(
        self, state: list[dict], actions: list[int], step: int,
    ) -> int | None:
        """Run the analyzer, commit its plan, and return the first action. Returns None on failure."""
        if self.agentic_consumer is None:
            return None

        workspace = self.output_dir / "consumer_workspace"
        notes_path = self.output_dir / "consumer_notes.md"
        synthesis_dir = (
            self.output_dir / "synthesis" / f"run_{self.synthesis_count:03d}"
            if self.synthesis_count > 0 else None
        )

        if not self.config.frames_only:
            for obj in state:
                for tag in obj.get("tags", []):
                    aliases_mod.ensure_seeded(self.type_aliases, tag)
            try:
                workspace.mkdir(parents=True, exist_ok=True)
                aliases_mod.write_workspace_artifact(
                    self.type_aliases, workspace / "type_aliases.json",
                )
                (workspace / "alias_updates.json").unlink(missing_ok=True)
            except Exception as exc:
                self.logger.log("ALIAS", f"workspace write failed: {exc}")
        else:
            workspace.mkdir(parents=True, exist_ok=True)

        self._ensure_shared_model_artifacts(current_level=self.current_level)
        shared_doc_snapshot = self._snapshot_shared_model_artifacts()
        try:
            self._write_synth_status(step)
        except Exception as exc:
            self.logger.log(
                "SYNTH_STATUS",
                f"pre-analyzer write failed: {type(exc).__name__}: {exc}",
            )

        divergence_window = list(self._steps_since_analyzer)
        try:
            divergence_images = self._collect_divergence_images()
        except Exception as exc:
            divergence_images = []
            self.logger.log(
                "DIVERGENCE", f"collect failed: {type(exc).__name__}: {exc}"
            )
        divergence_feedback = self._format_divergence_feedback(
            step, divergence_window,
        )
        animation_notice = self._format_animation_notice(divergence_window)
        operator_msg = self._read_operator_inject()
        if divergence_images:
            self.logger.log(
                "DIVERGENCE",
                f"injecting {len(divergence_images)} frame(s) where the model "
                f"mispredicted the last plan: "
                f"{[ (d['role'], d['step']) for d in divergence_images ]}"
            )

        max_retries = max(1, int(self.config.agentic_consumer_max_retries))
        aux = self._select_aux_prompt(step)
        if aux:
            self.run_log.append_note(step=step, source="aux_prompt",
                                     text=aux.splitlines()[0] if aux else "")
        plan: list[int] = []
        result: dict = {}
        reasoning = ""
        dur: float | str = "?"
        for attempt in range(max_retries):
            extra_parts = []
            if operator_msg:
                extra_parts.append(
                    "\n"
                    + operator_msg
                )
            if attempt > 0:
                extra_parts.append(self._ANALYZER_RETRY_NUDGE)
            elif aux:
                extra_parts.append(aux)
            if divergence_feedback:
                extra_parts.append(divergence_feedback)
            if animation_notice:
                extra_parts.append(animation_notice)
            extra = "\n\n".join(extra_parts)
            ascii_grid = ""
            try:
                get_frame = getattr(self.env, "get_frame", None)
                if get_frame is not None:
                    frame = get_frame()
                    if frame is not None:
                        from .vision import frame_to_ascii
                        ascii_grid = frame_to_ascii(frame)
            except Exception:
                ascii_grid = ""
            current_frame = None
            if self.config.frames_only:
                try:
                    gf = getattr(self.env, "get_frame", None)
                    cf = gf() if gf is not None else None
                    if cf is not None:
                        current_frame = (
                            cf.tolist() if hasattr(cf, "tolist") else cf
                        )
                except Exception:
                    current_frame = None
                consumer_state: list[dict] = []
                consumer_state_desc = ""
            else:
                consumer_state = state
                consumer_state_desc = aliases_mod.annotate_text(
                    self.env.describe_state(state), self.type_aliases,
                )

            result = self.agentic_consumer.choose_actions(
                workspace_dir=workspace,
                run_log_src=self.output_dir / "run_log.txt",
                epistemic_matrix_src=self.output_dir / "epistemic_matrix.json",
                synth_status_src=self._synth_status_path(),
                replay_buffer=self._serialize_transitions(),
                current_state=consumer_state,
                state_desc=consumer_state_desc,
                available_actions=actions,
                synthesis_dir=synthesis_dir,
                notes_persistent=notes_path,
                project_root=str(Path(__file__).resolve().parents[3]),
                moves_remaining=self._get_moves_left(),
                step=step,
                level=self.current_level,
                ascii_grid=ascii_grid,
                score=self.levels_completed,
                extra_user_prompt=extra,
                last_plan_hint=self._last_plan_hint or "",
                recent_history=[
                    {
                        "timestep": t.timestep,
                        "action_id": t.action_id,
                        "action_name": t.action_name,
                        "diff_text": t.diff_text,
                        "reward": t.reward,
                        "level": t.level,
                    }
                    for t in self.replay_buffer[-8:]
                ],
                frames_only=self.config.frames_only,
                current_frame=current_frame,
                game_over=(
                    self.env.is_game_over()
                    if hasattr(self.env, "is_game_over") else False
                ),
                divergence_images=divergence_images,
            )
            shared_summary = self._capture_shared_model_artifacts(
                workspace,
                source="analyzer",
                before_snapshot=shared_doc_snapshot,
            )
            if shared_summary:
                shared_doc_snapshot = self._snapshot_shared_model_artifacts()
                try:
                    self.run_log.append_note(
                        step=step,
                        source="shared_model_docs",
                        text=shared_summary[:1000],
                    )
                except Exception:
                    pass
            plan = []
            for a in result.get("plan", []):
                try:
                    aid = action_id_of(a)
                except (ValueError, TypeError):
                    continue
                if aid not in actions:
                    continue
                if aid == 6 and not (
                    isinstance(a, dict) and "x" in a and "y" in a
                ):
                    continue
                plan.append(a)
            reasoning = (result.get("reasoning") or "")[:160]
            dur = result.get("duration_s", "?")
            if not self.config.frames_only:
                try:
                    updates = aliases_mod.read_updates_file(
                        workspace / "alias_updates.json"
                    )
                    if updates:
                        summary = aliases_mod.apply_updates(
                            self.type_aliases, updates
                        )
                        self.logger.log(
                            "ALIAS",
                            f"applied analyzer updates: "
                            f"+{summary['added']} added, "
                            f"+{summary['upvoted']} upvoted, "
                            f"-{summary['removed']} removed"
                            + (f", errors={summary['errors']}"
                               if summary['errors'] else "")
                        )
                        (workspace / "alias_updates.json").unlink(
                            missing_ok=True
                        )
                except Exception as exc:
                    self.logger.log("ALIAS", f"apply failed: {exc}")
            if plan:
                if attempt > 0:
                    self.logger.log(
                        "ANALYZER",
                        f"recovered on retry {attempt}/{max_retries - 1}"
                    )
                break
            self.logger.log(
                "ANALYZER",
                f"attempt {attempt + 1}/{max_retries} failed "
                f"({result.get('reason', 'unknown')}) in {dur}s"
            )
            if result.get("rate_limited"):
                self.logger.log(
                    "ANALYZER",
                    f"rate-limited after {result.get('rate_limit_wait_s', '?')}s; stopping retries"
                )
                break

        if not plan:
            failure_payload = {
                "step": step,
                "completed_step": step - 1,
                "level": self.current_level,
                "attempts": max_retries,
                "last_reason": result.get("reason", "unknown"),
                "duration_s": dur,
                "rate_limited": bool(result.get("rate_limited")),
                "quota_limited": bool(result.get("quota_limited")),
                "rate_limit_wait_s": result.get("rate_limit_wait_s", 0),
                "message": (
                    "Analyzer failed to produce a valid plan; no environment "
                    "action was executed."
                ),
                "ts": time.time(),
            }
            try:
                (self.output_dir / "analyzer_failure.json").write_text(
                    json.dumps(failure_payload, indent=2, sort_keys=True)
                    + "\n"
                )
            except Exception:
                pass
            self._save_checkpoint(step - 1)
            self.logger.log(
                "ANALYZER",
                f"all {max_retries} attempts failed; hard stop before action"
            )
            raise AnalyzerNoPlanError(
                f"analyzer produced no valid plan at step {step}; "
                "segment stopped before env.step"
            )

        first = plan[0]
        tail = plan[1:]
        self._llm_plan = list(tail)
        self._llm_plan_origin_step = step
        self._llm_plan_no_effect_streak = 0

        if reasoning:
            self._last_plan_hint = reasoning

        self.logger.log(
            "ANALYZER",
            f"plan={plan} ({dur}s) reasoning={reasoning!r}"
        )
        try:
            self.run_log.append_note(
                step=step, source="analyzer",
                text=f"plan={plan} reasoning={reasoning}",
            )
        except Exception:
            pass
        return first

    def _get_moves_left(self) -> int | None:
        """Remaining moves, or None. Not surfaced to the LLM as a number."""
        get_budget = getattr(self.env, "get_move_budget_info", None)
        if get_budget is None:
            return None
        info = get_budget()
        if not info:
            return None
        decr = max(info.get("decrement_per_step", 1), 1)
        remaining = info.get("remaining", 0)
        return remaining // decr if decr else remaining

    def _discover_types(self, state: list[dict]):
        for obj in state:
            for tag in obj.get("tags", []):
                if tag not in self.known_types:
                    self.known_types[tag] = {
                        "name": obj.get("name", tag),
                        "w": obj.get("w"),
                        "h": obj.get("h"),
                        "first_seen_level": self.current_level,
                        "collidable": obj.get("collidable"),
                        "visible": obj.get("visible"),
                    }
                    aliases_mod.ensure_seeded(self.type_aliases, tag)
                    if not self.config.frames_only:
                        self.logger.log(
                            "DISCOVER", f"New type: {tag} "
                            f"(name={obj.get('name')}, "
                            f"{obj.get('w')}x{obj.get('h')})"
                        )

    def _goal_confirmed_on_current_level(self) -> bool:
        """True iff a positive reward has been observed on the current level."""
        cl = self.current_level
        return any(
            getattr(t, "reward", 0.0) > 0 and getattr(t, "level", -1) == cl
            for t in self.replay_buffer
        )

    def _update_world_model_doc(self, state: list[dict], mission: str | None):
        """Rebuild the NL world model document from accumulated knowledge."""
        sections = []

        sections.append("# World Model\n")

        sections.append("## Goal")
        if mission:
            sections.append(f"Mission: {mission}")

        reward_events = [t for t in self.replay_buffer if t.reward > 0]
        if reward_events:
            sections.append("Reward observed at:")
            for t in reward_events:
                sections.append(f"  Step {t.timestep} (level {t.level + 1}): "
                                f"reward={t.reward}, diff: {t.diff_text}")
        elif self.config.goal_hint:
            sections.append(f"No reward observed yet. Hint: {self.config.goal_hint}")
            sections.append("Synthesis must produce a reward_function that hypothesizes "
                            "this reward condition so the exploration LLM can drive the env "
                            "toward it under phi_2 optimism. Even without observing reward, "
                            "hypothesize what state change would earn reward based on the "
                            "game structure and this hint.")
        else:
            sections.append("No reward observed yet. Synthesis must hypothesize what the "
                            "reward condition might be based on the object types and game "
                            "structure. The reward_function should return (1.0, True) when "
                            "the hypothesized condition is met (phi_2 optimism).")

        if self.goal_hypothesis_code and self._goal_confirmed_on_current_level():
            sections.append("\n## Goal Hypothesis (synthesized `reward_function`)")
            sections.append(
                f"From synthesis run {self.goal_hypothesis_synthesis_run}, "
                "CONFIRMED on the current level (a reward was observed "
                "here). Under phi_2 optimism, ASSUME the condition below is "
                "reachable from the current state. Drive the env state to "
                "make this function return `(1.0, True)`."
            )
            sections.append("\n```python")
            sections.append(self.goal_hypothesis_code)
            sections.append("```")
        elif self.goal_hypothesis_code:
            sections.append("\n## Goal Hypothesis (UNCONFIRMED on this level)")
            sections.append(
                f"Carried forward from synthesis run "
                f"{self.goal_hypothesis_synthesis_run}, learned BEFORE any "
                "reward was observed on the current level. The current "
                "level is a distinct environment; this predicate is NOT "
                "known to hold or be reachable here. Treat it as a "
                "candidate prior / hint only, do NOT drive toward it as "
                "an established goal. Explore to discover this level's "
                "actual goal; the synthesized transition model may be used "
                "to simulate your own sub-goals."
            )
            sections.append("\n```python")
            sections.append(self.goal_hypothesis_code)
            sections.append("```")
        else:
            sections.append("\n## Goal Hypothesis")
            sections.append("*Not available yet, synthesis has not produced a "
                            "reward_function. Exploration will be random until "
                            "synthesis run #1 completes.*")

        if not self.config.frames_only:
            sections.append("\n## Object Types")
            for tag, info in sorted(self.known_types.items()):
                best = aliases_mod.best_alias(self.type_aliases, tag)
                alias_str = f" [hypothesised role: {best}]" if best else ""
                sections.append(f"  - {tag}{alias_str}: name={info['name']}, "
                                f"size={info['w']}x{info['h']}, "
                                f"collidable={info['collidable']}, "
                                f"first seen level {info['first_seen_level'] + 1}")

            alias_block = aliases_mod.format_for_world_model_doc(
                self.type_aliases, top_k=3,
            )
            if alias_block:
                sections.append("\n" + alias_block)

        sections.append("\n## Progress")
        sections.append(f"  Levels completed: {self.levels_completed}")
        sections.append(f"  Current level: {self.current_level + 1}")
        sections.append(f"  Total transitions: {len(self.replay_buffer)}")
        sections.append(f"  Total reward: {self.total_reward}")

        if self.synthesis_count > 0:
            sections.append("\n## Synthesis Status")
            sections.append(f"  Synthesis runs: {self.synthesis_count}")
            sections.append(f"  Best transition accuracy: {self.best_transition_accuracy:.0%}")
            sections.append(f"  Best reward accuracy: {self.best_reward_accuracy:.0%}")

        sections.append("\n## Recent Transitions")
        recent = self.replay_buffer[-30:]
        for t in recent:
            if t.diff_text == "Nothing changed":
                sections.append(f"  Step {t.timestep} {t.action_name} (L{t.level+1}): "
                                f"Nothing changed")
            else:
                sections.append(f"  Step {t.timestep} {t.action_name} (L{t.level+1}): "
                                f"{t.diff_text}")
                if t.reward > 0:
                    sections.append(f"    *** REWARD: {t.reward} ***")

        self.world_model_doc = "\n".join(sections)

    SHARED_WORLD_MODEL_FILENAME = "world_model.md"
    SHARED_MODEL_UPDATES_FILENAME = "shared_model_updates.md"

    def _level_reasoning_filename(self, level_idx: int) -> str:
        return f"level_{int(level_idx) + 1}_reasoning_log.md"

    def _level_report_filename(self, level_idx: int) -> str:
        return f"level_{int(level_idx) + 1}_report.md"

    def _shared_world_model_template(self) -> str:
        return (
            "# World Model\n\n"
            "This file is the shared textual world model for the analyzer and "
            "synthesizer. Keep it valid for all solved and partially explored "
            "levels so far. Treat unresolved visual details as evidence, not "
            "noise.\n\n"
            "## Mechanics of the Game\n\n"
            "Describe the inferred mechanics in simple, general terms. Include "
            "entities/object classes, state variables, interaction rules, action "
            "effects, win/loss conditions if known, persistent or remote state "
            "changes, and clearly separated level-specific additions.\n\n"
            "### Ontology\n\n"
            "- Persistent geometry or map structures: unknown.\n"
            "- Object families: unknown.\n"
            "- State variables by family: unknown.\n\n"
            "The ontology should stay as stable as possible across levels. When "
            "a later level introduces new visible elements, first explain how "
            "they fit known families before declaring a genuinely new family.\n\n"
            "## Target of the Game\n\n"
            "Describe the current target hypothesis: what the player is trying "
            "to achieve to complete the level.\n\n"
            "### How the player is expected to infer the target\n\n"
            "Explain the visual clues and logical reasoning that communicate the "
            "objective: motifs, object appearance, map structure, before/after "
            "differences, level transitions, small highlights, remote changes, "
            "or single-pixel markers.\n\n"
            "## Ad Hoc Elements Inventory\n\n"
            "- None recorded yet.\n\n"
            "List every unresolved hack or visual debt concretely: unresolved "
            "pixels, remote effects, temporary cache/mask use, renderer-like "
            "exceptions, level-specific branches, duplicate object families, or "
            "mechanics that are not yet explained by the shared ontology.\n\n"
            "## Newly Introduced But Unexplained Elements\n\n"
            "### Level 1\n\n"
            "- Initial frame inventory not yet recorded.\n\n"
            "For each level, list new visual elements relative to previous "
            "levels that are not yet fully explained: objects, motifs, chambers, "
            "sockets, gates, markers, HUD motifs, animation patterns, changed "
            "pixel clusters, or remote effects.\n"
        )

    def _level_reasoning_template(self, level_number: int) -> str:
        return (
            f"# Level {level_number} Reasoning Log\n\n"
            "Maintain this briefly while solving the level.\n\n"
            "## Hypotheses Tested\n\n"
            "- Not started.\n\n"
            "## Evidence For / Against\n\n"
            "- Not started.\n\n"
            "## Mismatches\n\n"
            "- None recorded yet.\n\n"
            "## Corrections\n\n"
            "- None recorded yet.\n"
        )

    def _level_report_template(self, level_number: int) -> str:
        return (
            f"# Level {level_number} Report\n\n"
            "Status: incomplete.\n\n"
            "After completing this level, summarize what you did, what was easy "
            "to infer, what was difficult to notice, which visual clues mattered, "
            "which false hypotheses slowed progress, and the final form of the "
            "world model.\n"
        )

    def _expected_shared_model_names(
        self,
        *,
        current_level: int | None = None,
        completed_level: int | None = None,
    ) -> set[str]:
        names = {self.SHARED_WORLD_MODEL_FILENAME}
        levels: set[int] = set()
        for idx in (current_level, completed_level):
            if idx is None:
                continue
            try:
                levels.add(max(0, int(idx)))
            except Exception:
                continue
        if not levels:
            try:
                levels.add(max(0, int(getattr(self, "current_level", 0))))
            except Exception:
                levels.add(0)
        for idx in levels:
            names.add(self._level_reasoning_filename(idx))
            names.add(self._level_report_filename(idx))
        return names

    def _iter_shared_model_artifacts(self, base_dir: Path):
        world = base_dir / self.SHARED_WORLD_MODEL_FILENAME
        if world.exists() or world.is_symlink():
            yield world
        for pattern in ("level_*_reasoning_log.md", "level_*_report.md"):
            for path in sorted(base_dir.glob(pattern)):
                if path.exists() or path.is_symlink():
                    yield path
        updates = base_dir / self.SHARED_MODEL_UPDATES_FILENAME
        if updates.exists() or updates.is_symlink():
            yield updates

    def _ensure_shared_model_artifacts(
        self,
        *,
        current_level: int | None = None,
        completed_level: int | None = None,
    ) -> None:
        """Create baseline1-style shared Markdown artifacts if absent."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            world = self.output_dir / self.SHARED_WORLD_MODEL_FILENAME
            if not world.exists():
                world.write_text(self._shared_world_model_template())

            names = self._expected_shared_model_names(
                current_level=current_level,
                completed_level=completed_level,
            )
            for name in sorted(names):
                if name == self.SHARED_WORLD_MODEL_FILENAME:
                    continue
                path = self.output_dir / name
                if path.exists():
                    continue
                if name.endswith("_reasoning_log.md"):
                    level_number = int(name.split("_")[1])
                    path.write_text(self._level_reasoning_template(level_number))
                elif name.endswith("_report.md"):
                    level_number = int(name.split("_")[1])
                    path.write_text(self._level_report_template(level_number))

            updates = self.output_dir / self.SHARED_MODEL_UPDATES_FILENAME
            if not updates.exists():
                updates.write_text(
                    "# Shared Model Updates\n\n"
                    "No synthesis-side shared world-model updates recorded yet.\n"
                )
        except Exception as exc:
            try:
                self.logger.log(
                    "WORLD_MODEL_DOCS",
                    f"ensure failed: {type(exc).__name__}: {exc}",
                )
            except Exception:
                pass

    def _snapshot_shared_model_artifacts(
        self, base_dir: Path | None = None,
    ) -> dict[str, str]:
        base = base_dir or self.output_dir
        snapshot: dict[str, str] = {}
        try:
            for path in self._iter_shared_model_artifacts(base):
                try:
                    snapshot[path.name] = sanitize_model_visible_text(
                        path.read_text()
                    )
                except Exception:
                    continue
        except Exception:
            pass
        return snapshot

    def _stage_shared_model_artifacts(self, ws_dir: Path) -> None:
        """Expose shared model docs in a workspace as writable copies."""
        self._ensure_shared_model_artifacts(current_level=self.current_level)
        for src in self._iter_shared_model_artifacts(self.output_dir):
            dst = ws_dir / src.name
            try:
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.write_text(sanitize_model_visible_text(src.read_text()))
            except Exception as exc:
                try:
                    self.logger.log(
                        "WORLD_MODEL_DOCS",
                        f"{src.name} workspace copy failed: "
                        f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass

    def _format_shared_model_update_summary(
        self,
        *,
        source: str,
        changed: list[str],
        before: dict[str, str],
        after: dict[str, str],
    ) -> str:
        label = "synthesis" if source == "synth" else source
        lines = [
            "# Shared Model Updates",
            "",
            f"Source: {label}",
            f"Synthesis run: {self.synthesis_count}",
            f"Current level: {getattr(self, 'current_level', 0) + 1}",
            "",
            "Changed files:",
        ]
        for name in changed:
            old_lines = set((before.get(name) or "").splitlines())
            additions = [
                ln.strip()
                for ln in (after.get(name) or "").splitlines()
                if ln.strip() and ln not in old_lines
            ]
            snippet = "; ".join(additions[:3])[:500]
            if not snippet:
                snippet = "content changed"
            lines.append(f"- {name}: {snippet}")
        return "\n".join(lines).rstrip() + "\n"

    def _capture_shared_model_artifacts(
        self,
        ws_dir: Path,
        *,
        source: str,
        before_snapshot: dict[str, str] | None = None,
    ) -> str:
        """Copy workspace edits to run-root docs and return a change summary."""
        self._ensure_shared_model_artifacts(current_level=self.current_level)
        before = before_snapshot or self._snapshot_shared_model_artifacts()
        names = set(before)
        names.update(p.name for p in self._iter_shared_model_artifacts(ws_dir))
        names.update(self._expected_shared_model_names(
            current_level=self.current_level,
        ))

        valid_prefixes = ("level_",)
        for name in sorted(names):
            if name in (
                self.SHARED_WORLD_MODEL_FILENAME,
                self.SHARED_MODEL_UPDATES_FILENAME,
            ):
                pass
            elif not (
                name.startswith(valid_prefixes)
                and (
                    name.endswith("_reasoning_log.md")
                    or name.endswith("_report.md")
                )
            ):
                continue

            src = ws_dir / name
            if not (src.exists() or src.is_symlink()):
                continue
            dst = self.output_dir / name
            try:
                if src.resolve() != dst.resolve():
                    text = sanitize_model_visible_text(src.read_text())
                    if text.strip():
                        dst.write_text(text)
            except Exception as exc:
                try:
                    self.logger.log(
                        "WORLD_MODEL_DOCS",
                        f"{source} capture failed for {name}: "
                        f"{type(exc).__name__}: {exc}",
                    )
                except Exception:
                    pass

        after = self._snapshot_shared_model_artifacts()
        changed = [
            name for name in sorted(after)
            if before.get(name) != after.get(name)
            and name != self.SHARED_MODEL_UPDATES_FILENAME
        ]
        if not changed:
            return ""

        summary = self._format_shared_model_update_summary(
            source=source, changed=changed, before=before, after=after,
        )
        try:
            (self.output_dir / self.SHARED_MODEL_UPDATES_FILENAME).write_text(
                sanitize_model_visible_text(summary)
            )
        except Exception:
            pass
        try:
            self.logger.log(
                "WORLD_MODEL_DOCS",
                f"{source} updated {', '.join(changed)}",
            )
        except Exception:
            pass
        return summary

    def _shared_world_model_context(self) -> str:
        self._ensure_shared_model_artifacts(current_level=self.current_level)
        parts: list[str] = []
        for path in sorted(
            self._iter_shared_model_artifacts(self.output_dir),
            key=lambda p: p.name,
        ):
            if path.name == self.SHARED_MODEL_UPDATES_FILENAME:
                continue
            txt = self._read_text_artifact(path, limit=12000)
            if txt:
                parts.append(f"# Shared artifact: {path.name}\n\n{txt}")
        updates = self._read_text_artifact(
            self.output_dir / self.SHARED_MODEL_UPDATES_FILENAME,
            limit=4000,
        )
        if updates:
            parts.append(
                "# Shared artifact: shared_model_updates.md\n\n" + updates
            )
        return "\n\n".join(parts)

    def _observed_changing_tags(self) -> set[str]:
        """Tags whose instances changed position, visibility, rotation, or pixels in any buffer transition."""
        changing: set[str] = set()
        for t in self.replay_buffer:
            before_by_key: dict[tuple, dict] = {}
            for o in t.before_state or []:
                name = o.get("name")
                if name is None:
                    continue
                key = (name, int(o.get("x", 0)), int(o.get("y", 0)))
                before_by_key[key] = o
            after_by_key: dict[tuple, dict] = {}
            for o in t.after_state or []:
                name = o.get("name")
                if name is None:
                    continue
                key = (name, int(o.get("x", 0)), int(o.get("y", 0)))
                after_by_key[key] = o
            for key, bo in before_by_key.items():
                tags = bo.get("tags") or []
                ao = after_by_key.get(key)
                if ao is None:
                    for tag in tags:
                        changing.add(str(tag))
                    continue
                if (bo.get("visible") != ao.get("visible")
                        or int(bo.get("rotation", 0))
                        != int(ao.get("rotation", 0))
                        or bo.get("pixels") != ao.get("pixels")):
                    for tag in tags:
                        changing.add(str(tag))
            for key in set(after_by_key) - set(before_by_key):
                ao = after_by_key[key]
                for tag in ao.get("tags") or []:
                    changing.add(str(tag))
        return changing

    def _check_crystallisation(self, step: int) -> dict[str, str] | None:
        """Evaluate the Prop. 5 trigger. Returns the committed partition {tag: alias} or None."""
        if not self.config.crystallisation_enabled:
            return None
        if self.crystallised:
            return None
        if int(self.levels_completed) < 1:
            return None
        partition = aliases_mod.nondecorative_committed(
            self.type_aliases,
            min_score=int(self.config.crystallisation_alias_min_score),
            min_margin=int(self.config.crystallisation_alias_margin),
        )
        return partition

    def _commit_crystallisation(
        self, step: int, partition: dict[str, str],
    ) -> None:
        """Fire the one-shot Prop. 5 stopping decision. Sets self.crystallised = True permanently."""
        self.crystallised = True
        self.crystallisation_step = int(step)
        self.crystallised_partition = dict(partition)
        changing = self._observed_changing_tags()
        for tag in changing:
            if tag not in self.crystallised_partition:
                self.crystallised_scope_extra[tag] = None

        try:
            (self.output_dir / "crystallisation.json").write_text(
                json.dumps({
                    "step": int(step),
                    "partition": self.crystallised_partition,
                    "scope_extra_changing_tags": sorted(
                        self.crystallised_scope_extra.keys()
                    ),
                    "levels_completed_at_commit": self.levels_completed,
                    "current_level_at_commit": self.current_level,
                    "n_transitions_at_commit": len(self.replay_buffer),
                }, indent=2)
            )
        except Exception as exc:
            self.logger.log(
                "CRYSTAL", f"crystallisation.json write failed: {exc}",
            )

        n_labelled = len(self.crystallised_partition)
        n_extra = len(self.crystallised_scope_extra)
        self.logger.log(
            "CRYSTAL",
            f"COMMIT at step {step}: {n_labelled} labelled tag(s) "
            f"+ {n_extra} observed-changing safety-net tag(s); "
            f"partition={self.crystallised_partition}",
        )

    def _compute_scope_tags(self) -> set[str]:
        """Scope = committed partition union observed-changing tags. Refreshed on every call."""
        scope: set[str] = set(self.crystallised_partition.keys())
        scope.update(self._observed_changing_tags())
        return scope

    def _mark_model_error(
        self,
        step: int,
        reasons: list[str] | tuple[str, ...] | None = None,
        *,
        level_completed: bool = False,
    ) -> None:
        """Accumulate unrepaired world-model divergence debt."""
        if self._model_error_first_step is None:
            self._model_error_first_step = int(step)
        self._model_error_last_step = int(step)
        self._model_error_count += 1
        if level_completed:
            self._model_error_level_completed = True
        reason_txt = "; ".join(str(r) for r in (reasons or []) if r)
        self.logger.log(
            "DIVERGENCE",
            f"model error #{self._model_error_count} at step {step}"
            + (f": {reason_txt}" if reason_txt else ""),
        )

    def _record_completed_action_plan_after_divergence(self, step: int) -> None:
        """Count completed analyzer/planner batches while divergence debt is open."""
        if self._model_error_first_step is None:
            return
        source = self._current_action_plan_source
        if source == "analyzer":
            completed = not self._llm_plan
        elif source == "planner":
            completed = not self._planner_queue
        else:
            completed = False
        if not completed:
            return
        self._model_error_action_plan_count += 1
        self.logger.log(
            "DIVERGENCE",
            "completed action plan after divergence "
            f"#{self._model_error_action_plan_count} "
            f"(source={source}, step={step})",
        )

    def _reset_model_error_debt(self) -> None:
        """Clear divergence debt after a CEGIS attempt or consistency proof."""
        self._model_error_first_step = None
        self._model_error_last_step = None
        self._model_error_count = 0
        self._model_error_action_plan_count = 0
        self._model_error_level_completed = False
        self._last_synthesis_gate_log = ""

    def _synthesis_gate_status(
        self, step: int, ctrl: dict | None = None,
    ) -> dict[str, Any]:
        """Return the delayed-CEGIS gate status for analyzer + engine use."""
        ctrl = ctrl or {}
        active = self._model_error_first_step is not None
        min_moves = max(
            0,
            int(getattr(
                self.config,
                "synthesis_defer_min_moves_after_divergence",
                12,
            ) or 0),
        )
        max_errors = max(
            0,
            int(getattr(
                self.config,
                "synthesis_defer_max_errors",
                6,
            ) or 0),
        )
        min_action_plans = max(
            0,
            int(getattr(
                self.config,
                "synthesis_defer_min_action_plans_after_divergence",
                4,
            ) or 0),
        )
        first = self._model_error_first_step
        moves_since = (
            max(0, int(step) - int(first))
            if active and first is not None else 0
        )
        errors = int(self._model_error_count)
        action_plans = int(self._model_error_action_plan_count)
        moves_left = max(0, min_moves - moves_since)
        errors_left = max(0, max_errors - errors)
        action_plans_left = max(0, min_action_plans - action_plans)
        has_current_level_probe = (
            active
            and int(self.current_level) != int(self._last_synthesis_level)
            and any(
                int(getattr(t, "level", -1)) == int(self.current_level)
                for t in self.replay_buffer
            )
        )
        force_requested = bool(ctrl.get("force_now"))
        ready_reasons: list[str] = []
        if active:
            if force_requested:
                ready_reasons.append("force_now")
            if self._model_error_level_completed:
                ready_reasons.append("level_completed")
            if has_current_level_probe:
                ready_reasons.append("new_level_probe_executed")
            if min_moves <= 0 or moves_since >= min_moves:
                ready_reasons.append("moves_threshold")
            if max_errors <= 0 or errors >= max_errors:
                ready_reasons.append("error_threshold")
            if (
                min_action_plans <= 0
                or action_plans >= min_action_plans
            ):
                ready_reasons.append("action_plan_threshold")
        return {
            "active": active,
            "ready": active and bool(ready_reasons),
            "ready_reasons": ready_reasons,
            "first_error_step": first,
            "last_error_step": self._model_error_last_step,
            "error_count": errors,
            "action_plan_count": action_plans,
            "moves_since_first_error": moves_since,
            "min_moves_after_divergence": min_moves,
            "max_errors_before_synthesis": max_errors,
            "min_action_plans_after_divergence": min_action_plans,
            "moves_until_auto_synthesis": moves_left,
            "errors_until_auto_synthesis": errors_left,
            "action_plans_until_auto_synthesis": action_plans_left,
            "level_completed_since_error": bool(
                self._model_error_level_completed
            ),
            "new_level_probe_executed": bool(has_current_level_probe),
            "force_requested": force_requested,
            "last_synthesis_level": int(self._last_synthesis_level),
        }

    def _format_divergence_feedback(
        self, step: int, window: list[dict],
    ) -> str:
        """Prompt block telling the analyzer how deferred CEGIS is progressing."""
        diverged = [e for e in window if e.get("diverged")]
        status = self._synthesis_gate_status(step)
        if not diverged and not status.get("active"):
            return ""

        lines = [
            "WORLD-MODEL DIVERGENCE FEEDBACK",
            "The executable world model mispredicted recent real-env "
            "outcomes. Attached divergence frames show ground truth for the "
            "first mismatch in the last executed plan and later after-frames.",
        ]
        if diverged:
            lines.append("Divergent executed step(s):")
            for e in diverged[:8]:
                reasons = e.get("reasons") or []
                reason_txt = "; ".join(str(r) for r in reasons) or "mismatch"
                lines.append(f"- step {e.get('step')}: {reason_txt}")
        if status.get("active"):
            if status.get("ready"):
                lines.append(
                    "CEGIS gate: ready after this action sequence drains "
                    f"via {', '.join(status['ready_reasons'])}."
                )
            else:
                lines.append(
                    "CEGIS gate: repair is intentionally deferred. "
                    f"Auto-synthesis opens after {status['moves_until_auto_synthesis']} "
                    "more executed move(s) since the first divergence OR "
                    f"{status['errors_until_auto_synthesis']} more model "
                    "error(s) OR "
                    f"{status['action_plans_until_auto_synthesis']} more "
                    "completed action plan(s), whichever comes first. It also "
                    "opens on level completion or after the first action "
                    "sequence on a brand new level."
                )
            lines.append(
                "If you need the C3 planner now, write synth_control.json with "
                "`force_now: true` and a focused mismatch note; otherwise avoid "
                "relying on the planner until CEGIS has repaired the model."
            )
        lines.append(
            "Inspect any intermediate animation evidence too: check "
            "`animation_events.jsonl` and `../frames/step_<N>_tick_<K>.png` "
            "for the diverging steps. Use a Task/subagent as a second opinion "
            "when animation order or tiny visual changes matter."
        )
        lines.append(
            "Leave concrete questions, interpretations, and requested probes "
            "for the synthesizer in `world_model.md`, the current "
            "`level_N_reasoning_log.md`, `notes.md`, and `synth_control.json` "
            "`focus` if a repair should test a specific hypothesis."
        )
        return "\n".join(lines)

    def _format_animation_notice(self, window: list[dict]) -> str:
        """Proactively tell the analyzer which recently executed steps produced
        multi-tick animation, how many ticks each, and where the frames are --
        so animated mechanics aren't missed even when no divergence flagged them."""
        if not window:
            return ""
        steps = {int(e.get("step", -1)) for e in window}
        diverged = {int(e.get("step", -1)) for e in window if e.get("diverged")}
        manifest = self.frames_dir / "animation_events.jsonl"
        if not manifest.exists():
            return ""
        animated: dict[int, int] = {}
        try:
            for ln in manifest.read_text(errors="ignore").splitlines()[-300:]:
                ln = ln.strip()
                if not ln.startswith("{"):
                    continue
                try:
                    ev = json.loads(ln)
                except Exception:
                    continue
                s = int(ev.get("step", -1))
                if s in steps:
                    animated[s] = len(ev.get("tick_frames") or [])
        except Exception:
            return ""
        if not animated:
            return ""
        parts = []
        for s in sorted(animated):
            tag = " [DIVERGED]" if s in diverged else ""
            parts.append(f"step {s} ({animated[s]} ticks){tag}")
        return (
            "INTERMEDIATE ANIMATION FRAMES -- these recently executed steps "
            "produced multi-tick animation: " + ", ".join(parts) + ". The "
            "frames are at ../frames/step_<NNNN>_tick_<KK>.png. When probing a "
            "hidden or newly-introduced mechanic, or any [DIVERGED] step, READ "
            "those tick frames as images (not just the [ACTION_SEQUENCE] diff "
            "lines) -- the mechanic is frequently visible only mid-animation."
        )

    OPERATOR_INJECT_FILES = ("operator_inject.md", "operator_inject.txt")

    def _read_operator_inject(self) -> str:
        """One-shot operator instruction for the next analyzer call. Drop text
        into <output_dir>/operator_inject.md. It is injected as a high-priority
        block on the next analyzer call and then consumed (renamed) so it fires
        exactly once."""
        for name in self.OPERATOR_INJECT_FILES:
            p = self.output_dir / name
            if not p.exists():
                continue
            try:
                txt = p.read_text().strip()
            except Exception:
                continue
            if not txt:
                continue
            try:
                ts = time.strftime("%Y%m%dT%H%M%S")
                p.rename(self.output_dir / f"operator_inject.consumed_{ts}.md")
            except Exception:
                try:
                    p.unlink()
                except Exception:
                    pass
            return txt
        return ""

    def _should_synthesize(self, step: int) -> bool:
        """Return True if CEGIS should run now.

        After the initial model, synthesis is evidence-gated: it only fires at a
        plan boundary after the model mispredicted a transition/reward in the
        just-executed action sequence. Analyzer ``force_now`` requests are kept
        as focus hints, not as permission to synthesize without a fresh
        execution mismatch.
        """
        if len(self.replay_buffer) < self.config.min_transitions_for_synthesis:
            return False

        if self.config.crystallisation_enabled and not self.crystallised:
            return False

        if self._warmup_queue or self._llm_plan or self._planner_queue:
            return False

        if self.synthesis_count == 0:
            return True

        ctrl = self._read_synth_control()
        defer_until = int(ctrl.get("defer_until_step", -1) or -1)
        if defer_until > step:
            self.logger.log(
                "SYNTH_CONTROL",
                f"deferred until step {defer_until} (analyzer): "
                f"focus={ctrl.get('focus', '')!r}"
            )
            return False

        gate = self._synthesis_gate_status(step, ctrl)
        if ctrl.get("force_now"):
            focus = ctrl.get("focus", "")
            self._consume_synth_control_force()
            if gate.get("active"):
                self.logger.log(
                    "SYNTH_CONTROL",
                    f"force_now opened CEGIS gate after divergence: "
                    f"focus={focus!r}",
                )
            else:
                self.logger.log(
                    "SYNTH_CONTROL",
                    "ignored force_now without a fresh model mismatch; "
                    f"focus retained for next real CEGIS: focus={focus!r}",
                )
                return False

        if not gate.get("active"):
            return False

        if not gate.get("ready"):
            log_key = json.dumps({
                "first": gate.get("first_error_step"),
                "errors": gate.get("error_count"),
                "plans": gate.get("action_plan_count"),
                "moves_left": gate.get("moves_until_auto_synthesis"),
                "errors_left": gate.get("errors_until_auto_synthesis"),
                "plans_left": gate.get("action_plans_until_auto_synthesis"),
            }, sort_keys=True)
            if log_key != self._last_synthesis_gate_log:
                self._last_synthesis_gate_log = log_key
                self.logger.log(
                    "SYNTHESIS",
                    "deferred after divergence: "
                    f"moves_left={gate['moves_until_auto_synthesis']} "
                    f"errors_left={gate['errors_until_auto_synthesis']} "
                    f"plans_left={gate['action_plans_until_auto_synthesis']} "
                    f"errors={gate['error_count']} "
                    f"plans={gate['action_plan_count']} "
                    f"first_step={gate['first_error_step']}",
                )
            return False

        self.logger.log(
            "SYNTHESIS",
            "CEGIS gate ready: " + ", ".join(gate["ready_reasons"]),
        )
        consistent = self._model_is_consistent()
        if consistent:
            self.logger.log(
                "SYNTHESIS",
                "model now verifies against replay; clearing divergence debt",
            )
            self._reset_model_error_debt()
            return False
        return True

    SYNTH_STATUS_FILENAME = "synth_status.json"
    SYNTH_CONTROL_FILENAME = "synth_control.json"

    def _synth_status_path(self) -> Path:
        return self.output_dir / self.SYNTH_STATUS_FILENAME

    def _synth_control_path(self) -> Path:
        return (self.output_dir / "consumer_workspace"
                / self.SYNTH_CONTROL_FILENAME)

    def _write_synth_status(self, step: int) -> None:
        """Write synth_status.json for the analyzer to consult."""
        status = {
            "synthesis_count": self.synthesis_count,
            "last_synthesis_step": self.last_synthesis_step,
            "best_transition_accuracy": round(
                self.best_transition_accuracy, 4),
            "best_reward_accuracy": round(self.best_reward_accuracy, 4),
            "consecutive_failed_syntheses": (
                self._consecutive_failed_syntheses),
            "model_present": self.synthesis_count > 0,
            "goal_hypothesis_run": self.goal_hypothesis_synthesis_run,
            "goal_confirmed_on_current_level": (
                self._goal_confirmed_on_current_level()
            ),
            "goal_hypothesis_snippet": (
                sanitize_model_visible_text(self.goal_hypothesis_code or "")[:800]
            ),
            "goal_in_english": sanitize_model_visible_text(
                self.goal_in_english or ""
            ),
            "synth_learnings": sanitize_model_visible_text(
                self.synth_learnings or ""
            )[:2000],
            "critique_findings": sanitize_model_visible_text(
                self.critique_findings or ""
            )[:2000],
            "critique_response": sanitize_model_visible_text(
                self.critique_response or ""
            )[:2000],
            "animation_findings": sanitize_model_visible_text(
                self.animation_findings or ""
            )[:2000],
            "shared_model_updates": sanitize_model_visible_text(
                self.shared_model_updates or ""
            )[:2000],
            "handoff_files": {
                "world_model": self.SHARED_WORLD_MODEL_FILENAME,
                "shared_model_updates": self.SHARED_MODEL_UPDATES_FILENAME,
                "current_level_reasoning_log": self._level_reasoning_filename(
                    self.current_level
                ),
                "current_level_report": self._level_report_filename(
                    self.current_level
                ),
                "synth_learnings": "synth_learnings.md",
                "critique_findings": "last_critique.md",
                "critique_response": "critique_response.md",
                "animation_findings": "animation_analysis.md",
                "animation_events": "animation_events.jsonl",
            },
            "engine_step": step,
            "n_transitions": len(self.replay_buffer),
            "levels_completed": self.levels_completed,
            "current_level": self.current_level,
            "synthesis_gate": self._synthesis_gate_status(step),
            "planner": {
                "enabled": bool(getattr(self.config, "planner_enabled", True)),
                "after_levels_completed": int(
                    getattr(self.config, "planner_after_levels_completed", 1)
                ),
                "requires_crystallisation": bool(
                    self.config.crystallisation_enabled
                ),
                "requires_completed_verification": bool(
                    getattr(
                        self.config,
                        "planner_require_completed_verification",
                        True,
                    )
                ),
                "queued_actions": len(self._planner_queue),
                "blocked_round": self._planner_blocked_round,
                "retry_after_step": self._planner_retry_after_step,
                "last_status": self._planner_last_status,
            },
        }
        try:
            self._synth_status_path().write_text(
                json.dumps(status, indent=2, default=str)
            )
        except Exception as exc:
            self.logger.log(
                "SYNTH_STATUS",
                f"write failed: {type(exc).__name__}: {exc}"
            )

    def _read_synth_control(self) -> dict:
        """Read analyzer-written synth_control.json. Returns empty dict on miss."""
        path = self._synth_control_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return {}

    def _consume_synth_control_force(self) -> None:
        """Remove force_now from synth_control.json after honoring it."""
        path = self._synth_control_path()
        try:
            data = json.loads(path.read_text())
            if isinstance(data, dict) and data.get("force_now"):
                data.pop("force_now", None)
                path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _record_initial_frame(self) -> None:
        get_frame = getattr(self.env, "get_frame", None)
        if get_frame is None:
            return
        try:
            frame = get_frame()
            if frame is None:
                return
            from .vision import render_frame_to_file
            initial_path = self.frames_dir / "initial.png"
            render_frame_to_file(
                frame, str(initial_path),
                description="initial state",
            )
            shutil.copy2(initial_path, self.frames_dir / "step_0000.png")
        except Exception as exc:
            self.logger.log(
                "FRAMES", f"initial frame render failed: {exc}"
            )

    def _compose_replay(self) -> Path | None:
        """Compile per-step PNGs into a replay video (MP4 via cv2, or GIF fallback). Returns path or None."""
        actions_path = self.frames_dir / "actions.jsonl"
        if not actions_path.exists():
            return None
        frame_files: list[Path] = []
        initial = self.frames_dir / "initial.png"
        if initial.exists():
            frame_files.append(initial)
        try:
            with open(actions_path) as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    s = int(e.get("step", -1))
                    png = self.frames_dir / f"step_{s:04d}.png"
                    if png.exists():
                        frame_files.append(png)
        except Exception:
            return None
        if len(frame_files) < 2:
            return None

        try:
            import cv2
            first = cv2.imread(str(frame_files[0]))
            if first is None:
                raise RuntimeError(f"cv2.imread failed on {frame_files[0]}")
            h, w = first.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            mp4_path = self.output_dir / "replay.mp4"
            writer = cv2.VideoWriter(str(mp4_path), fourcc, 4.0, (w, h))
            if not writer.isOpened():
                raise RuntimeError("cv2 VideoWriter failed to open")
            wrote = 0
            for png in frame_files:
                img = cv2.imread(str(png))
                if img is None:
                    continue
                writer.write(img)
                wrote += 1
            writer.release()
            if wrote >= 2 and mp4_path.exists() and mp4_path.stat().st_size > 0:
                self.logger.log(
                    "REPLAY",
                    f"wrote replay.mp4 ({wrote} frames @ 4fps, "
                    f"{mp4_path.stat().st_size // 1024} KB)"
                )
                return mp4_path
        except Exception as exc:
            self.logger.log("REPLAY", f"mp4 failed: {type(exc).__name__}: {exc}")

        try:
            from PIL import Image
            imgs = [Image.open(p) for p in frame_files]
            gif_path = self.output_dir / "replay.gif"
            imgs[0].save(
                str(gif_path), save_all=True, append_images=imgs[1:],
                duration=250, loop=0, optimize=True,
            )
            self.logger.log(
                "REPLAY",
                f"wrote replay.gif ({len(imgs)} frames, "
                f"{gif_path.stat().st_size // 1024} KB)"
            )
            return gif_path
        except Exception as exc:
            self.logger.log("REPLAY", f"gif fallback failed: {exc}")
            return None

    def _record_step_frame(
        self, step: int, transition: "TransitionRecord",
    ) -> None:
        """Save after-state PNG, per-tick intermediate PNGs, and actions.jsonl manifest entry."""
        get_frame = getattr(self.env, "get_frame", None)
        png_path = self.frames_dir / f"step_{step:04d}.png"
        if get_frame is not None:
            try:
                frame = get_frame()
                if frame is not None:
                    from .vision import render_frame_to_file
                    render_frame_to_file(
                        frame, str(png_path),
                        description=(
                            f"step {step} {transition.action_name} "
                            f"L{transition.level + 1}"
                        ),
                    )
            except Exception as exc:
                self.logger.log(
                    "FRAMES", f"step {step} render failed: {exc}"
                )

        intermediates = getattr(self.env, "_last_during_frames", None) or []
        tick_paths: list[Path] = []
        if intermediates:
            try:
                from .vision import render_frame_to_file
                for k, tick_frame in enumerate(intermediates, start=1):
                    tick_path = (
                        self.frames_dir
                        / f"step_{step:04d}_tick_{k:02d}.png"
                    )
                    render_frame_to_file(
                        tick_frame, str(tick_path),
                        description=(
                            f"step {step} tick {k}/{len(intermediates) + 1} "
                            f"{transition.action_name} "
                            f"L{transition.level + 1} (intermediate)"
                        ),
                    )
                    tick_paths.append(tick_path)
            except Exception as exc:
                self.logger.log(
                    "FRAMES",
                    f"step {step} tick render failed: {exc}"
                )
        if tick_paths:
            event = self._record_animation_event(
                step=step,
                transition=transition,
                tick_paths=tick_paths,
                final_path=png_path,
            )
            self._run_animation_analysis(step, event)
        try:
            with open(self.frames_dir / "actions.jsonl", "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "action_id": transition.action_id,
                    "action_name": transition.action_name,
                    "reward": transition.reward,
                    "done": transition.done,
                    "level": transition.level,
                    "diff_text": transition.diff_text,
                }) + "\n")
        except Exception:
            pass

    def _record_animation_event(
        self,
        *,
        step: int,
        transition: "TransitionRecord",
        tick_paths: list[Path],
        final_path: Path,
    ) -> dict:
        """Append an animated-step manifest entry and return it."""
        before_name = (
            "initial.png" if step == 0 else f"step_{step - 1:04d}.png"
        )
        before_path = self.frames_dir / before_name
        event = {
            "step": int(step),
            "level": int(transition.level),
            "action_id": int(transition.action_id),
            "action_name": transition.action_name,
            "reward": transition.reward,
            "done": bool(transition.done),
            "diff_text": transition.diff_text,
            "before_frame": str(before_path),
            "tick_frames": [str(p) for p in tick_paths],
            "final_frame": str(final_path),
            "note": (
                "Main prediction target is final_frame; tick_frames are "
                "intermediate animation evidence for mechanics."
            ),
        }
        try:
            manifest = self.frames_dir / "animation_events.jsonl"
            with open(manifest, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except Exception as exc:
            self.logger.log(
                "ANIMATION", f"event manifest write failed: {exc}"
            )
        return event

    def _run_animation_analysis(self, step: int, event: dict) -> None:
        """Run an optional backend-matched animation reviewer for an animated step."""
        if not getattr(self.config, "animation_analysis_enabled", False):
            return
        max_events = int(
            getattr(self.config, "animation_analysis_max_events", 0) or 0
        )
        if max_events > 0 and self._animation_analysis_count >= max_events:
            return
        if not event:
            return
        ws_dir = (
            self.output_dir
            / "animation_analysis"
            / f"step_{step:04d}"
        ).resolve()
        try:
            ws_dir.mkdir(parents=True, exist_ok=True)
            (ws_dir / "animation_event.json").write_text(
                json.dumps(event, indent=2, default=str)
            )
            prompt_tmpl = (
                Path(__file__).resolve().parent
                / "prompts" / "synthesizer" / "animation_analysis.md"
            ).read_text(encoding="utf-8")
        except Exception as exc:
            self.logger.log(
                "ANIMATION", f"setup failed for step {step}: {type(exc).__name__}: {exc}"
            )
            return
        paths = []
        for raw in (
            [event.get("before_frame")]
            + list(event.get("tick_frames") or [])
            + [event.get("final_frame")]
        ):
            if not raw:
                continue
            p = Path(raw)
            if p.exists():
                paths.append(p)
        prompt = (
            prompt_tmpl
            .replace("%%STEP%%", str(step))
            .replace("%%EVENT_JSON%%", json.dumps(event, indent=2, default=str))
            .replace("%%WORKSPACE_DIR%%", str(ws_dir))
        )
        self._run_backend_subagent(
            ws_dir=ws_dir,
            prompt=prompt,
            label=f"animation_step_{step:04d}",
            images=paths,
            timeout_s=int(
                getattr(self.config, "animation_analysis_timeout_s", 600)
                or 600
            ),
        )
        out_path = ws_dir / "animation_analysis.md"
        txt = self._read_text_artifact(out_path, limit=8000)
        if not txt:
            self.logger.log(
                "ANIMATION", f"step {step} reviewer produced no animation_analysis.md"
            )
            return
        self._animation_analysis_count += 1
        combined_path = self.output_dir / "animation_analysis.md"
        try:
            prior = ""
            if combined_path.exists():
                prior = combined_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).rstrip()
            block = f"\n\n## Step {step}\n\n{txt.strip()}\n"
            combined = (prior + block).strip() + "\n"
            if len(combined) > 24000:
                combined = (
                    "# Truncated to most recent animation analyses.\n\n"
                    + combined[-23000:]
                )
            combined_path.write_text(combined)
            self.animation_findings = combined[-8000:]
            self.logger.log(
                "ANIMATION", f"step {step} analysis recorded"
            )
        except Exception as exc:
            self.logger.log(
                "ANIMATION",
                f"analysis merge failed for step {step}: {type(exc).__name__}: {exc}",
            )

    def _recent_analyzer_notes(self, n: int = 8) -> str:
        """Return the last N [NOTE source=analyzer] lines from run_log."""
        path = self.output_dir / "run_log.txt"
        if not path.exists():
            return ""
        try:
            lines = path.read_text().splitlines()
        except Exception:
            return ""
        notes = [
            ln for ln in lines
            if ln.startswith("[NOTE step=") and "source=analyzer" in ln
        ]
        return "\n".join(notes[-n:])

    def _append_planner_feedback(self, event: dict) -> None:
        """Append a planner-outcome event to planner_feedback.jsonl (best-effort).

        Captures planner timeouts, no-plan, env divergences, and successes so the
        next synthesis round can read them and judge / improve the planner.
        """
        try:
            rec = {
                "synthesis_count": int(self.synthesis_count),
                "step": int(getattr(self, "_cur_step", -1)),
                "levels_completed": int(self.levels_completed),
                "ts": time.time(),
                **event,
            }
            with open(self.output_dir / "planner_feedback.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass

    def _recent_planner_feedback(self, n: int = 10) -> str:
        """Return the last N planner_feedback.jsonl lines (compact), or ''."""
        path = self.output_dir / "planner_feedback.jsonl"
        if not path.exists():
            return ""
        try:
            lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        except Exception:
            return ""
        return "\n".join(lines[-n:])

    @staticmethod
    def _frames_equal(a, b, mask=None) -> bool:
        """Pixel equality between two 2D palette grids (list or numpy).

        When ``mask`` (a set of (row, col) cells) is given, those cells are
        ignored in the comparison -- used for the synth-declared move-counter
        region the model is permitted not to predict (see _validate_counter_mask).
        """
        if a is None or b is None:
            return False
        try:
            a = a.tolist() if hasattr(a, "tolist") else a
            b = b.tolist() if hasattr(b, "tolist") else b
        except Exception:
            return False
        if len(a) != len(b):
            return False
        for r, (ra, rb) in enumerate(zip(a, b)):
            ra = ra.tolist() if hasattr(ra, "tolist") else ra
            rb = rb.tolist() if hasattr(rb, "tolist") else rb
            if len(ra) != len(rb):
                return False
            for c, (x, y) in enumerate(zip(ra, rb)):
                if mask and (r, c) in mask:
                    continue
                if int(x) != int(y):
                    return False
        return True

    @staticmethod
    def _validate_counter_mask(cells):
        """Validate a synth-declared move-counter mask and return it as a
        frozenset of (row, col), or an EMPTY frozenset if absent/invalid.

        Fail closed: a model may only exclude ONE continuous line of cells at
        most 2 pixels wide (the move-counter HUD strip, whose per-level
        quantization is hard to predict). It may NOT mask an arbitrary region to
        dodge verification of real mechanics.
        """
        try:
            pts = {(int(r), int(c)) for (r, c) in (cells or [])}
        except Exception:
            return frozenset()
        if not pts:
            return frozenset()
        rows = {r for r, _ in pts}
        cols = {c for _, c in pts}
        rspan = max(rows) - min(rows) + 1
        cspan = max(cols) - min(cols) + 1
        if min(rspan, cspan) > 2:
            return frozenset()
        long_span = max(rspan, cspan)
        if len(pts) > 2 * long_span:
            return frozenset()
        long_idx = sorted(cols if cspan >= rspan else rows)
        if long_idx[-1] - long_idx[0] + 1 != len(long_idx):
            return frozenset()
        return frozenset(pts)

    def _load_div_model(self):
        """Import the latest synthesised transition_function in-process, cached by
        synthesis round. Returns the callable or None."""
        if self.synthesis_count == 0:
            return None
        if self._div_model_round == self.synthesis_count:
            return self._div_model
        self._div_model = None
        self._div_reward_model = None
        self._div_mask = frozenset()
        self._div_model_round = self.synthesis_count
        ge = (
            self.output_dir / "synthesis"
            / f"run_{self.synthesis_count:03d}" / "game_engine.py"
        ).resolve()
        if not ge.exists():
            return None
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"_divmodel_{self.synthesis_count}", str(ge),
            )
            mod = importlib.util.module_from_spec(spec)
            mod.copy = __import__("copy")
            spec.loader.exec_module(mod)
            self._div_model = getattr(mod, "transition_function", None)
            self._div_reward_model = getattr(mod, "reward_function", None)
            try:
                mfn = getattr(mod, "move_counter_mask", None)
                self._div_mask = self._validate_counter_mask(
                    mfn() if callable(mfn) else None
                )
            except Exception:
                self._div_mask = frozenset()
        except Exception:
            self._div_model = None
            self._div_reward_model = None
            self._div_mask = frozenset()
        return self._div_model

    def _record_execution_divergence(
        self,
        step: int,
        action,
        before_state,
        actual_state,
        pre_frame,
        post_frame,
        *,
        reward: float = 0.0,
        done: bool = False,
    ) -> None:
        """Record whether the LIVE model mispredicted this just-executed transition.

        Captured now (before CEGIS re-synthesises and heals it) and
        accumulated for the next analyzer call. Checks both transition
        prediction and reward/done, because CEGIS should also fire when the
        dynamics are right but the goal predicate is wrong. Runs the model
        in-process under a short alarm so a pathological model cannot hang the
        engine.
        """
        diverged = False
        reasons: list[str] = []
        try:
            tf = self._load_div_model()
            if tf is not None:
                if (self.config.frames_only and pre_frame is not None
                        and post_frame is not None):
                    pre_list = (
                        pre_frame.tolist() if hasattr(pre_frame, "tolist")
                        else pre_frame
                    )
                    import copy as _copy
                    used_alarm = False
                    old = None
                    try:
                        import signal

                        def _to(*_a):
                            raise TimeoutError()
                        old = signal.signal(signal.SIGALRM, _to)
                        signal.alarm(3)
                        used_alarm = True
                    except Exception:
                        used_alarm = False
                    pred = None
                    try:
                        pred = tf(_copy.deepcopy(pre_list), action)
                    except Exception as exc:
                        diverged = True
                        reasons.append(
                            f"transition_function raised {type(exc).__name__}"
                        )
                    finally:
                        if used_alarm:
                            try:
                                signal.alarm(0)
                                signal.signal(signal.SIGALRM, old)
                            except Exception:
                                pass
                    if not self._frames_equal(
                        pred, post_frame, getattr(self, "_div_mask", None)
                    ):
                        diverged = True
                        reasons.append("predicted frame mismatch")
                    rf = self._div_reward_model
                    if callable(rf):
                        try:
                            pred_r, pred_d = rf(
                                _copy.deepcopy(pre_list),
                                action,
                                _copy.deepcopy(pred),
                            )
                            if (
                                abs(float(pred_r or 0.0)
                                    - float(reward or 0.0)) > 1e-6
                                or bool(pred_d) != bool(done)
                            ):
                                diverged = True
                                reasons.append(
                                    "reward/done mismatch "
                                    f"predicted=({pred_r},{pred_d}) "
                                    f"observed=({reward},{done})"
                                )
                        except Exception:
                            diverged = True
                            reasons.append("reward_function raised")
                elif (
                    not self.config.frames_only
                    and before_state is not None
                    and actual_state is not None
                ):
                    import copy as _copy
                    model_action = action if is_click(action) else action_id_of(action)
                    try:
                        pred = tf(
                            _copy.deepcopy(before_state),
                            _copy.deepcopy(model_action),
                        )
                    except Exception as exc:
                        diverged = True
                        reasons.append(
                            f"transition_function raised {type(exc).__name__}"
                        )
                        pred = None
                    if pred is not None and not object_states_equal(
                        pred,
                        actual_state,
                    ):
                        diverged = True
                        reasons.append("predicted object state mismatch")
                    rf = self._div_reward_model
                    if callable(rf):
                        try:
                            pred_r, pred_d = rf(
                                _copy.deepcopy(before_state),
                                _copy.deepcopy(model_action),
                                _copy.deepcopy(pred),
                            )
                            if (
                                abs(float(pred_r or 0.0)
                                    - float(reward or 0.0)) > 1e-6
                                or bool(pred_d) != bool(done)
                            ):
                                diverged = True
                                reasons.append(
                                    "reward/done mismatch "
                                    f"predicted=({pred_r},{pred_d}) "
                                    f"observed=({reward},{done})"
                                )
                        except Exception:
                            diverged = True
                            reasons.append("reward_function raised")
        except Exception:
            diverged = False
            reasons = []
        if diverged:
            self._mark_model_error(
                step,
                reasons,
                level_completed=(float(reward or 0.0) > 0.0 or bool(done)),
            )
        self._steps_since_analyzer.append({
            "step": int(step),
            "diverged": bool(diverged),
            "reasons": reasons,
        })

    def _recent_execution_diverged(self) -> bool:
        """Whether unrepaired model-error debt is active."""
        return self._model_error_first_step is not None

    def _collect_divergence_images(self, max_images: int = 8) -> list[dict]:
        """Build the divergence-image set for the next analyzer call from the plan
        executed since the last call, then clear the window.

        The first action that diverged contributes its before and after frames.
        Every action after it in the executed plan contributes its after frame
        (once the model is wrong, the rest of the plan ran in an unexpected reality).
        """
        window = self._steps_since_analyzer
        self._steps_since_analyzer = []
        first = next(
            (i for i, e in enumerate(window) if e["diverged"]), None
        )
        if first is None:
            return []
        out: list[dict] = []
        k = window[first]["step"]
        before_png = self.frames_dir / (
            f"step_{k - 1:04d}.png" if k - 1 >= 0 else "initial.png"
        )
        if before_png.exists():
            out.append({"role": "before", "step": k - 1,
                        "path": str(before_png)})
        after_png = self.frames_dir / f"step_{k:04d}.png"
        if after_png.exists():
            out.append({"role": "after", "step": k, "path": str(after_png)})
        for e in window[first + 1:]:
            j = e["step"]
            p = self.frames_dir / f"step_{j:04d}.png"
            if p.exists():
                out.append({"role": "after", "step": j, "path": str(p)})
        return out[:max_images]

    def _refresh_spriteless_diagnostics(
        self, step: int, *, force: bool = False, reason: str = "",
    ) -> None:
        """Refresh ETA/matrix artifacts for frames-only runs using the synth's
        ``extract_objects(frame)`` checklist deliverable.

        This is best-effort and never blocks exploration: a missing or broken
        extractor records a diagnostic artifact and leaves the latest usable
        matrix intact.
        """
        if not self.config.frames_only or self.synthesis_count <= 0:
            return
        if not force:
            interval = max(1, int(self.config.ontology_measure_interval))
            if step % interval != 0:
                return
        latest_ws = (
            self.output_dir / "synthesis"
            / f"run_{self.synthesis_count:03d}"
        ).resolve()
        code_path = latest_ws / "game_engine.py"
        if not code_path.exists():
            return
        try:
            diag_rng = (
                np.random.default_rng(self.config.seed)
                if reason == "resume" else self.rng
            )
            status = refresh_spriteless_diagnostics(
                replay_buffer=self._serialize_transitions(),
                code_path=code_path,
                output_dir=self.output_dir,
                step=step,
                synthesis_count=self.synthesis_count,
                alpha_0=self.config.epistemic_alpha_0,
                beta_0=self.config.epistemic_beta_0,
                kappa=self.config.epistemic_kappa,
                sort_by=self.config.epistemic_sort_by,
                rng=diag_rng,
                max_candidates=self.ontology.max_candidates,
            )
        except Exception as exc:
            self.logger.log(
                "SPRITELESS",
                f"ETA refresh failed: {type(exc).__name__}: {exc}",
            )
            return
        if not status.get("ok"):
            self.logger.log(
                "SPRITELESS",
                "object ETA unavailable"
                + (f" ({reason})" if reason else "")
                + f": {status.get('error', 'unknown error')}",
            )
            return

        latest = status.get("ontology_latest")
        trace_rec = status.get("trace_record")
        if isinstance(latest, dict):
            self.ontology._latest = latest
        if isinstance(trace_rec, dict):
            self.ontology._trace.append(trace_rec)
            self.ontology.append_trace_line(
                self.output_dir / "ontology_error_trace.jsonl",
                trace_rec,
            )
        self.ontology.dump(self.output_dir / "ontology_error.json")
        if isinstance(trace_rec, dict):
            self.logger.log(
                "SPRITELESS",
                f"eta={trace_rec['eta']:.4f} "
                f"eta*={trace_rec['eta_star']:.4f} "
                f"(-{trace_rec['eta_reduction']:.4f} via "
                f"{trace_rec['best_candidate']}) "
                f"types={trace_rec['n_induced_types']} "
                f"frames={trace_rec['n_unique_frames_extracted']} "
                f"reason={reason or 'refresh'}",
            )

    def _model_is_consistent(self) -> bool:
        """Run test_runner on the current code against the full replay buffer. Returns True iff all pass."""
        latest_ws = (
            self.output_dir / "synthesis"
            / f"run_{self.synthesis_count:03d}"
        ).resolve()
        code_path = latest_ws / "game_engine.py"
        if not code_path.exists():
            return False

        if self.config.frames_only:
            self.adapter.write_replay_buffer_frames(
                self._serialize_transitions(), latest_ws,
            )
        else:
            self.adapter.write_replay_buffer(
                self._serialize_transitions(), latest_ws,
            )
        accuracy = self._run_tests(latest_ws)

        import re
        all_pass = True
        for part in accuracy.split("|"):
            m = re.match(r"\s*\w+:\s*(\d+)/(\d+)", part.strip())
            if m and int(m.group(1)) < int(m.group(2)):
                all_pass = False
                break

        if not all_pass:
            self.logger.log("CONSISTENCY", f"INCONSISTENT: {accuracy}")
        return all_pass

    @staticmethod
    def _read_text_artifact(path: Path, *, limit: int = 4000) -> str:
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""
        txt = sanitize_model_visible_text(txt).strip()
        if limit > 0 and len(txt) > limit:
            return txt[:limit] + "\n...[truncated]"
        return txt

    def _run_backend_subagent(
        self,
        *,
        ws_dir: Path,
        prompt: str,
        label: str,
        images: list[Path] | None = None,
        timeout_s: int | None = None,
    ) -> dict:
        """Run a small review subagent on the same backend as synthesis."""
        prompt = sanitize_model_visible_text(prompt)
        safe_label = "".join(
            ch if ch.isalnum() or ch in ("_", "-") else "_"
            for ch in label.lower()
        )
        ws_dir = ws_dir.resolve()
        try:
            (ws_dir / f"{safe_label}_prompt.txt").write_text(prompt)
        except Exception:
            pass
        if self.config.backend == "codex":
            from . import codex_backend as cx
            run_dir = self.output_dir.resolve()
            try:
                rel = ws_dir.relative_to(run_dir)
            except ValueError:
                self.logger.log(
                    label.upper(), f"codex workspace {ws_dir} outside run dir"
                )
                return {"error": "workspace outside run dir"}
            container_images: list[str] = []
            for image in images or []:
                try:
                    image_rel = image.resolve().relative_to(run_dir)
                    container_images.append(f"/run/{image_rel}")
                except Exception:
                    continue
            try:
                res = cx.run_codex_turn(
                    prompt=prompt,
                    workspace_dir=ws_dir,
                    run_dir=run_dir,
                    container_cd=f"/run/{rel}",
                    model=self.config.codex_model,
                    effort=self.config.codex_effort,
                    codex_home=cx.codex_home_path(self.config.codex_home),
                    images=container_images or None,
                    timeout_s=timeout_s,
                    image_name=self.config.codex_image,
                    network=self.config.codex_network,
                    gateway=self.config.codex_gateway,
                    stdout_path=ws_dir / f"{safe_label}_stdout.jsonl",
                    stderr_path=ws_dir / f"{safe_label}_stderr.txt",
                )
                self.logger.log(
                    label.upper(),
                    f"codex subagent: reason={res.get('reason')} "
                    f"rc={res.get('returncode')} dur={res.get('duration_s')}s",
                )
                return res
            except Exception as exc:
                self.logger.log(
                    label.upper(),
                    f"codex subagent failed: {type(exc).__name__}: {exc}",
                )
                return {"error": str(exc)}

        claude = shutil.which("claude")
        if not claude:
            self.logger.log(label.upper(), "Claude CLI not found")
            return {"error": "claude not found"}
        from .sandbox import (
            claude_popen_kwargs,
            describe_claude_resource_limits,
            terminate_process_group,
            wait_with_resource_monitor,
        )
        cmd = [
            claude, "-p", prompt,
            "--model", self.config.synthesis_model,
            "--effort", self.config.synthesis_effort,
            "--max-turns", "8",
            "--output-format", "stream-json", "--verbose",
            "--no-session-persistence",
            "--allowedTools", "Bash(python3:*),Bash(python:*),Read,Grep,Write",
            "--permission-mode", "bypassPermissions",
        ]
        cmd, popen_kwargs, _dname = self._wrap_claude_subprocess(cmd, ws_dir)
        t0 = time.time()
        try:
            with (
                open(ws_dir / f"{safe_label}_stdout.jsonl", "w") as out_f,
                open(ws_dir / f"{safe_label}_stderr.txt", "w") as err_f,
            ):
                self.logger.log(
                    label.upper(),
                    "claude subagent resource limits: "
                    f"{describe_claude_resource_limits()}",
                )
                proc = subprocess.Popen(
                    cmd, stdout=out_f, stderr=err_f,
                    text=True, cwd=str(ws_dir),
                    **popen_kwargs,
                )
                wait_with_resource_monitor(
                    proc, timeout_s=timeout_s, log_fn=self.logger.log,
                )
            dur = round(time.time() - t0, 1)
            self.logger.log(
                label.upper(),
                f"claude subagent rc={proc.returncode} dur={dur}s",
            )
            return {"returncode": proc.returncode, "duration_s": dur}
        except subprocess.TimeoutExpired:
            try:
                terminate_process_group(proc)
            except Exception:
                pass
            self._docker_rm(_dname)
            self.logger.log(label.upper(), f"TIMED OUT after {timeout_s}s")
            return {"returncode": -1, "duration_s": round(time.time() - t0, 1),
                    "reason": "timed_out"}
        except Exception as exc:
            self.logger.log(
                label.upper(), f"failed: {type(exc).__name__}: {exc}"
            )
            return {"error": str(exc), "duration_s": round(time.time() - t0, 1)}

    def _wrap_claude_subprocess(self, cmd: list, ws_dir: Path):
        """Apply the configured claude isolation to a synth-side ``claude`` argv.

        Returns (cmd, popen_kwargs, container_name|None). Mirrors the analyzer's
        AgenticConsumer._wrap_claude_cmd: "docker" runs the claude-agent
        container on the filtered network (returning a name for timeout cleanup).
        "bwrap" (default) keeps the local bubblewrap sandbox. No isolation only
        when subprocess_sandbox is off.
        """
        from .sandbox import claude_popen_kwargs
        if not self.config.subprocess_sandbox:
            return cmd, claude_popen_kwargs(), None
        if self.config.claude_isolation == "docker":
            from .sandbox import wrap_for_docker
            name = (f"arc-claude-synth-{os.getpid()}-"
                    f"{int(time.time() * 1000) % 1_000_000_000}")
            cmd = wrap_for_docker(
                cmd, workspace_dir=ws_dir, engine_output_dir=self.output_dir,
                image=self.config.claude_image,
                network=self.config.claude_network,
                gateway=self.config.claude_gateway,
                container_name=name,
                memory=self.config.claude_docker_memory,
                cpus=self.config.claude_docker_cpus,
                pids_limit=self.config.claude_docker_pids_limit,
            )
            return cmd, {"start_new_session": True}, name
        from .sandbox import wrap_for_sandbox
        cmd = wrap_for_sandbox(
            cmd, workspace_dir=ws_dir, engine_output_dir=self.output_dir,
        )
        return cmd, claude_popen_kwargs(), None

    def _docker_rm(self, name: str | None) -> None:
        """Best-effort force-remove a claude-agent container (timeout cleanup)."""
        if not name:
            return
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=15, check=False,
            )
        except Exception:
            pass

    def _run_critique(self, ws_dir: Path) -> str:
        """Adversarial generalization critique of the just-synthesised game_engine.py.
        Writes findings to output_dir/last_critique.md for the next round's context.
        No-op unless config.critique_enabled. Never blocks synthesis."""
        if not getattr(self.config, "critique_enabled", False):
            return ""
        if not (ws_dir / "game_engine.py").exists():
            return ""
        try:
            crit_prompt = (
                Path(__file__).resolve().parent
                / "prompts" / "synthesizer" / "critique.md"
            ).read_text(encoding="utf-8")
        except Exception:
            return ""
        self._run_backend_subagent(
            ws_dir=ws_dir,
            prompt=crit_prompt,
            label="critique",
            timeout_s=600,
        )
        crit_file = ws_dir / "critique.md"
        if crit_file.exists():
            try:
                self.critique_findings = self._read_text_artifact(
                    crit_file, limit=8000
                )
                (self.output_dir / "last_critique.md").write_text(
                    self.critique_findings
                )
                self.logger.log(
                    "CRITIQUE", "wrote critique findings for next round")
                return self.critique_findings
            except Exception:
                pass
        else:
            self.logger.log("CRITIQUE", "subagent produced no critique.md")
        return ""

    def _critique_interval(self) -> int:
        try:
            return int(getattr(self.config, "critique_interval", 5) or 0)
        except Exception:
            return 0

    def _critique_due_this_round(self) -> bool:
        if not getattr(self.config, "critique_enabled", False):
            return False
        interval = self._critique_interval()
        return interval > 0 and self.synthesis_count > 0 and (
            self.synthesis_count % interval == 0
        )

    @staticmethod
    def _combine_synthesis_results(base: dict, extra: dict) -> dict:
        out = dict(base or {})
        for key in ("duration_s", "rate_limit_wait_s", "attempts"):
            try:
                out[key] = (float(out.get(key, 0) or 0)
                            + float(extra.get(key, 0) or 0))
            except Exception:
                pass
        for key in ("timed_out", "rate_limited"):
            out[key] = bool(out.get(key)) or bool(extra.get(key))
        if extra.get("error"):
            out["error"] = extra.get("error")
        return out

    def _critique_repair_prompt(
        self,
        *,
        critique_text: str,
        accuracy: str,
        round_idx: int,
    ) -> str:
        return f"""You are still the synthesizer for this same ARC-AGI-3 synthesis workspace.

An independent critic just reviewed your current `game_engine.py`. Do not treat
this as a note for a future round if you can fix it from the current evidence.
Before returning control to exploration, repair every valid generalization issue
that can be repaired using the current `context.txt`, `replay_buffer.pkl`, and
`test_runner_output.txt`.

Critique-repair round: {round_idx}
Current verifier result before this repair turn:
{accuracy}

Independent critique:
```text
{critique_text}
```

Required workflow:
1. Read `critique.md`, `game_engine.py`, `context.txt`, and
   `test_runner_output.txt`.
2. Decide which critique findings are valid. If a finding is wrong, reject it
   only with concrete replay-buffer evidence.
3. Edit `game_engine.py` to fix valid issues without adding replay lookup,
   trajectory lookup, known-solution code, or new ad hoc layout hacks.
4. Run `python test_runner.py`. Iterate until `ALL TESTS PASSED` again if at all
   possible.
5. Write `critique_response.md` with sections:
   - `Applied:` concrete fixes made
   - `Rejected:` findings rejected, with evidence
   - `Remaining:` issues that require more real game observations
6. Write/update `synth_learnings.md` for the exploration agent with short
   operational bullets: known mechanics, uncertain hypotheses, high-value probes,
   and avoid-repeat failures.

If the critique exposes a real issue but the current buffer cannot distinguish
the needed rule, keep the verifier passing, state the uncertainty in
`synth_learnings.md`, and request the specific next probe in `Remaining:`.
"""

    def _run_critique_repair_loop(
        self,
        *,
        ws_dir: Path,
        accuracy: str,
        result: dict,
        escalating: bool,
    ) -> tuple[str, dict]:
        if not getattr(self.config, "critique_enabled", False):
            return accuracy, result
        if result.get("quota_limited"):
            return accuracy, result
        if "ALL TESTS PASSED" not in accuracy:
            return accuracy, result
        interval = self._critique_interval()
        if not self._critique_due_this_round():
            self.logger.log(
                "CRITIQUE",
                f"skipped automatic critic on synth round "
                f"{self.synthesis_count} (interval={interval})",
            )
            return accuracy, result

        repair_enabled = bool(
            getattr(self.config, "critique_repair_enabled", True)
        )
        max_repairs = int(getattr(self.config, "critique_repair_rounds", 1) or 0)
        if not repair_enabled or max_repairs <= 0:
            self._run_critique(ws_dir)
            return accuracy, result

        repaired = False
        for repair_idx in range(1, max_repairs + 1):
            critique_text = self._run_critique(ws_dir)
            if not critique_text:
                break
            try:
                (ws_dir / "requires_critique_response.flag").write_text(
                    "Write critique_response.md before ending this synthesis turn.\n"
                )
            except Exception:
                pass
            prompt = self._critique_repair_prompt(
                critique_text=critique_text,
                accuracy=accuracy,
                round_idx=repair_idx,
            )
            self.logger.log(
                "CRITIQUE",
                f"starting in-round repair {repair_idx}/{max_repairs}",
            )
            repair_result = self._invoke_claude(
                ws_dir, prompt, escalating=escalating,
            )
            result = self._combine_synthesis_results(result, repair_result)
            accuracy = self._run_tests(ws_dir)
            self.logger.log(
                "CRITIQUE",
                f"repair {repair_idx}/{max_repairs} verifier result: {accuracy}",
            )
            repaired = True
            if "ALL TESTS PASSED" not in accuracy:
                break

        if (
            repaired
            and "ALL TESTS PASSED" in accuracy
            and bool(getattr(self.config, "critique_recheck_after_repair", True))
        ):
            self.logger.log("CRITIQUE", "final re-check after repair")
            self._run_critique(ws_dir)

        return accuracy, result

    @staticmethod
    def _pickleable_grid(value):
        return value.tolist() if hasattr(value, "tolist") else value

    def _frame_level_cache_entries(self) -> dict[int, list[list[int]]]:
        """Return all observed frames-only level-entry caches.

        Older checkpoints can have a sparse ``level_frames`` map. Reconstruct
        missing entries from reward/done transitions, whose after-frame is the
        next level's hand-authored entry frame in ARC-AGI-3.
        """
        entries: dict[int, list[list[int]]] = {}
        for lvl_idx, frame in self.level_frames.items():
            if frame is None:
                continue
            try:
                entries[int(lvl_idx)] = self._pickleable_grid(frame)
            except Exception:
                continue

        if 0 not in entries:
            for trans in self.replay_buffer:
                if int(getattr(trans, "level", -1)) != 0:
                    continue
                before_frame = getattr(trans, "before_frame", None)
                if before_frame is not None:
                    entries[0] = self._pickleable_grid(before_frame)
                    break

        for trans in self.replay_buffer:
            after_frame = getattr(trans, "after_frame", None)
            if after_frame is None:
                continue
            try:
                is_level_advance = (
                    float(getattr(trans, "reward", 0.0) or 0.0) > 0.0
                    or bool(getattr(trans, "done", False))
                )
                if not is_level_advance:
                    continue
                next_level = int(getattr(trans, "level", -1)) + 1
                if next_level < 0:
                    continue
                entries.setdefault(next_level, self._pickleable_grid(after_frame))
            except Exception:
                continue

        return entries

    def _write_level_initial_caches(self, ws_dir: Path) -> None:
        """Write l<N>_initial.pkl files for every observed level entry.

        These caches are permitted for transition_function on level-entry and
        reset transitions. The verifier statically rejects reward_function cache
        reads, so hiding the current level entry only makes the latest
        level-advance transition impossible to model.
        """
        if self.config.frames_only:
            cache_entries = self._frame_level_cache_entries()
            cache_kind = "level-frame-cache"
        else:
            cache_entries = {
                int(lvl_idx): entry_state
                for lvl_idx, entry_state in self.level_states.items()
                if entry_state is not None
            }
            cache_kind = "level-cache"

        written: list[str] = []
        for lvl_idx, entry in sorted(cache_entries.items()):
            try:
                cache_path = ws_dir / f"l{int(lvl_idx) + 1}_initial.pkl"
                with open(cache_path, "wb") as f:
                    pickle.dump(entry, f)
                written.append(f"L{int(lvl_idx) + 1}")
            except Exception as exc:
                self.logger.log(
                    "SYNTHESIS",
                    f"{cache_kind} write failed for L{int(lvl_idx) + 1}: "
                    f"{type(exc).__name__}: {exc}",
                )

        expected = set(range(0, int(self.current_level) + 1))
        missing = sorted(expected.difference(cache_entries.keys()))
        if missing:
            self.logger.log(
                "SYNTHESIS",
                "WARNING: missing observed level-entry cache(s): "
                + ", ".join(f"L{idx + 1}" for idx in missing),
            )
        elif written:
            self.logger.log(
                "SYNTHESIS",
                f"level-entry caches written: {', '.join(written)}",
            )

    def _run_synthesis(self, step: int, mission: str | None, state: list[dict]):
        self.synthesis_count += 1
        self.last_synthesis_step = step
        self._clear_planner_plan()
        self._planner_model = None
        self._planner_model_round = -1
        self._planner_consistency_key = None
        self._planner_verification_key = None

        self.logger.section(f"SYNTHESIS #{self.synthesis_count}")
        self.logger.log("SYNTHESIS", f"{len(self.replay_buffer)} transitions, "
                        f"levels seen: {list(self.level_states.keys())}")

        ws_dir = (self.output_dir / "synthesis" / f"run_{self.synthesis_count:03d}").resolve()
        ws_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in (
            "last_critique.md",
            "critique.md",
            "critique_response.md",
            "requires_critique_response.flag",
        ):
            stale_path = ws_dir / stale_name
            try:
                if stale_path.exists() or stale_path.is_symlink():
                    stale_path.unlink()
            except Exception:
                pass
        self._ensure_shared_model_artifacts(current_level=self.current_level)
        shared_doc_snapshot = self._snapshot_shared_model_artifacts()

        if self.config.frames_only:
            self.adapter.write_replay_buffer_frames(
                self._serialize_transitions(), ws_dir,
            )
        else:
            self.adapter.write_replay_buffer(
                self._serialize_transitions(), ws_dir,
            )

        self._write_level_initial_caches(ws_dir)

        if not self.config.frames_only:
            first_state = self.level_states.get(
                0, self.replay_buffer[0].before_state,
            )
            self.adapter.write_initial_data(first_state, ws_dir)

        if self.config.frames_only:
            self.adapter.write_test_runner_frames(ws_dir)
        elif self.crystallised:
            scope_tags = sorted(self._compute_scope_tags())
            self.adapter.write_test_runner_crystallised(
                ws_dir, scope_tags=scope_tags,
            )
        else:
            self.adapter.write_test_runner(ws_dir)

        linked_artifacts = {
            "ontology_error.json": self.output_dir / "ontology_error.json",
            "spriteless_object_abstraction.json": (
                self.output_dir / "spriteless_object_abstraction.json"
            ),
            "animation_events.jsonl": (
                self.frames_dir / "animation_events.jsonl"
            ),
            "animation_analysis.md": self.output_dir / "animation_analysis.md",
            "planner_verification.json": (
                self.output_dir / "planner_verification.json"
            ),
        }
        if self._critique_due_this_round():
            linked_artifacts.update({
                "last_critique.md": self.output_dir / "last_critique.md",
            })
        for diag_name, diag_src in linked_artifacts.items():
            if diag_src.exists():
                diag_dst = ws_dir / diag_name
                try:
                    if diag_dst.exists() or diag_dst.is_symlink():
                        diag_dst.unlink()
                    target = os.path.relpath(
                        diag_src.resolve(), diag_dst.parent.resolve()
                    )
                    diag_dst.symlink_to(target)
                except Exception as exc:
                    self.logger.log(
                        "SYNTHESIS",
                        f"{diag_name} symlink failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

        self._stage_shared_model_artifacts(ws_dir)

        try:
            sl_dst = ws_dir / "synth_learnings.md"
            if sl_dst.exists() or sl_dst.is_symlink():
                sl_dst.unlink()
            sl_src = self.output_dir / "synth_learnings.md"
            sl_dst.write_text(
                sanitize_model_visible_text(sl_src.read_text())
                if sl_src.exists() else ""
            )
        except Exception as exc:
            self.logger.log(
                "SYNTHESIS",
                f"synth_learnings.md writable stage failed: "
                f"{type(exc).__name__}: {exc}",
            )

        context = self._shared_world_model_context()
        try:
            legal_actions = sorted(int(a) for a in self.env.get_available_actions())
        except Exception:
            legal_actions = []
        if legal_actions:
            missing_common = [
                a for a in (5, 6) if a not in set(legal_actions)
            ]
            context += (
                "\n\n# Legal actions for this run\n\n"
                f"Current available_actions: {legal_actions}\n"
                "Only these action ids exist for this game/state. Do not "
                "hypothesize, recommend, or encode goal alternatives that "
                "require absent action ids.\n"
            )
            if missing_common:
                context += (
                    f"Specifically, action id(s) {missing_common} are absent "
                    "here; they are not untested mechanics for this run.\n"
                )
        if self.world_model_doc:
            context += (
                "\n\n# Engine Observation Snapshot "
                "(generated from run log; ground-truth summary)\n\n"
                + self.world_model_doc
            )
        if hasattr(self.adapter, 'format_transitions_for_context'):
            formatted = self.adapter.format_transitions_for_context(
                self._serialize_transitions(), max_examples=40, mission=mission,
            )
            context += "\n\n" + formatted

        persistent_notes = ""
        consumer_notes_path = self.output_dir / "consumer_notes.md"
        if consumer_notes_path.exists():
            try:
                persistent_notes = sanitize_model_visible_text(
                    consumer_notes_path.read_text()
                )
            except Exception:
                pass
        recent_notes = sanitize_model_visible_text(
            self._recent_analyzer_notes(n=8)
        )
        ctrl = self._read_synth_control()
        focus = sanitize_model_visible_text(ctrl.get("focus") or "").strip()
        if persistent_notes or recent_notes or focus:
            context += (
                "\n\n"
                "# Analyzer hypotheses (ADVISORY ONLY, verify against the buffer)\n\n"
                "These are natural-language guesses from the action-selector\n"
                "LLM about what's happening in the env. They MAY BE WRONG.\n"
                "The replay buffer is ground truth. Fit the buffer first;\n"
                "use these as hints to test, not rules to encode verbatim.\n"
                "If the buffer contradicts an analyzer hypothesis, trust\n"
                "the buffer.\n\n"
            )
            if focus:
                context += (
                    "## Synthesis focus requested by analyzer\n"
                    f"  {focus}\n\n"
                )
            if persistent_notes:
                context += (
                    "## Analyzer's persistent scratchpad (running theory)\n"
                    "```\n" + persistent_notes.rstrip() + "\n```\n\n"
                )
            if recent_notes:
                context += (
                    "## Recent analyzer reasoning (last 8 plans)\n"
                    "```\n" + recent_notes + "\n```\n"
                )
        if persistent_notes:
            try:
                (ws_dir / "analyzer_notes.md").write_text(persistent_notes)
            except Exception:
                pass

        crit_path = self.output_dir / "last_critique.md"
        if self._critique_due_this_round() and crit_path.exists():
            try:
                crit_txt = sanitize_model_visible_text(
                    crit_path.read_text()
                ).strip()
                if crit_txt:
                    (ws_dir / "requires_critique_response.flag").write_text(
                        "Write critique_response.md before ending this synthesis turn.\n"
                    )
                    context += (
                        "\n\n# MANDATORY generalization critique response "
                        "(independent critic, previous round)\n"
                        "This is not just advisory. Before ending your synthesis "
                        "turn, decide which findings are valid, revise "
                        "`game_engine.py` accordingly, and write "
                        "`critique_response.md` with Applied / Rejected / "
                        "Remaining sections. If you reject a finding, give the "
                        "specific replay-buffer evidence.\n"
                        "```\n" + crit_txt + "\n```\n"
                    )
            except Exception:
                pass

        anim_path = self.output_dir / "animation_analysis.md"
        anim_txt = self._read_text_artifact(anim_path, limit=8000)
        if anim_txt:
            context += (
                "\n\n# Intermediate animation analysis (independent reviewer)\n"
                "Use this as evidence when revising transition dynamics. The "
                "final settled frame is still the verifier target, but the tick "
                "frames often reveal movement order, collision resolution, "
                "timing, or triggered state transitions.\n"
                "```\n" + anim_txt + "\n```\n"
            )

        planner_fb = self._recent_planner_feedback(n=10)
        planner_latest = self._planner_last_status or {}
        if planner_fb or planner_latest:
            context += (
                "\n\n# Planner feedback (engine-side, advisory)\n"
                "The engine runs your synthesized `planner` after a level clear, "
                "validates it on completed-level starts, and checks it step by "
                "step in the real env. Recent outcomes -- timeouts, no-plan, and "
                "env divergences -- are below.\n"
                "Quickly judge whether `planner` actually reaches reward within "
                "its wall-clock budget. If it has been timing out, finding no "
                "plan, or diverging, treat that as a defect to fix THIS round: "
                "make it an explicit budgeted goal-directed search (prune "
                "aggressively, plan over your object abstraction, return None "
                "fast), not a naive uninformed search.\n"
            )
            if planner_fb:
                context += (
                    "```\n"
                    + sanitize_model_visible_text(planner_fb)
                    + "\n```\n"
                )
            if planner_latest:
                context += (
                    "Latest planner status: "
                    + json.dumps(planner_latest, default=str)[:600] + "\n"
                )

        context = sanitize_model_visible_text(context)
        (ws_dir / "context.txt").write_text(context)

        will_escalate = (
            self._consecutive_failed_syntheses
            >= self.config.escalate_to_opus_after
        )
        prev_ws = (self.output_dir / "synthesis" / f"run_{self.synthesis_count - 1:03d}").resolve()
        prev_code = None
        if prev_ws.exists() and (prev_ws / "game_engine.py").exists():
            prev_code = (prev_ws / "game_engine.py").read_text()

        if self.config.frames_only:
            stub_text = self.adapter.format_code_stub_frames(
                structure=self.config.synth_mode,
            )
        else:
            stub_text = self.adapter.format_code_stub(
                structure=self.config.synth_mode,
            )

        if will_escalate and prev_code:
            (ws_dir / "previous_attempt.py").write_text(prev_code)
            (ws_dir / "game_engine.py").write_text(stub_text)
            self.logger.log(
                "SYNTHESIS",
                "ESCALATION: reset to fresh stub; previous code in previous_attempt.py"
            )
        elif prev_code and len(prev_code) > 200:
            (ws_dir / "game_engine.py").write_text(prev_code)
            self.logger.log("SYNTHESIS", "Carried forward previous code")
        else:
            (ws_dir / "game_engine.py").write_text(stub_text)
            self.logger.log(
                "SYNTHESIS",
                f"Fresh code stub (mode={self.config.synth_mode}, "
                f"frames_only={self.config.frames_only})",
            )

        if prev_ws.exists() and prev_ws != ws_dir:
            _carry_skip = {
                "game_engine.py", "previous_attempt.py", "replay_buffer.pkl",
                "context.txt", "test_runner.py", "test_runner_output.txt",
                "analyzer_notes.md", "critique.md", "last_critique.md",
                "critique_response.md", "requires_critique_response.flag",
                "claude_chat.jsonl", "claude_prompt.txt",
                "xi_updates.json", "xi_updates_applied.json",
            }
            carried: list[str] = []
            for src in prev_ws.iterdir():
                try:
                    if src.is_symlink() or not src.is_file():
                        continue
                    name = src.name
                    if (name in _carry_skip
                            or name.endswith(".stderr.txt")
                            or name.endswith("_initial.pkl")):
                        continue
                    dst = ws_dir / name
                    if dst.exists() or dst.is_symlink():
                        continue
                    shutil.copy2(src, dst)
                    carried.append(name)
                except Exception:
                    continue
            if carried:
                self.logger.log(
                    "SYNTHESIS",
                    f"carried forward synth data file(s) from "
                    f"run_{self.synthesis_count - 1:03d}: "
                    + ", ".join(sorted(carried)),
                )

        baseline = self._run_tests(ws_dir)
        self.logger.log("SYNTHESIS", f"Baseline: {baseline}")

        if self.config.frames_only:
            prompt = self.adapter.format_synthesis_prompt_frames(
                workspace_dir=str(ws_dir),
                test_runner_path=str(ws_dir / "test_runner.py"),
                project_root=str(Path(__file__).resolve().parents[3]),
                structure=self.config.synth_mode,
            )
        elif self.crystallised:
            scope_extra = sorted(self.crystallised_scope_extra.keys())
            prompt = self.adapter.format_synthesis_prompt_crystallised(
                workspace_dir=str(ws_dir),
                test_runner_path=str(ws_dir / "test_runner.py"),
                project_root=str(Path(__file__).resolve().parents[3]),
                partition=dict(self.crystallised_partition),
                scope_extra_tags=scope_extra,
                structure=self.config.synth_mode,
            )
        else:
            prompt = self.adapter.format_synthesis_prompt(
                workspace_dir=str(ws_dir),
                test_runner_path=str(ws_dir / "test_runner.py"),
                project_root=str(Path(__file__).resolve().parents[3]),
                structure=self.config.synth_mode,
            )

        interval = int(getattr(self.config, "synth_simplify_interval", 0) or 0)
        if (interval > 0 and self.synthesis_count > 1
                and self.synthesis_count % interval == 0):
            try:
                simp = (
                    Path(__file__).resolve().parent
                    / "prompts" / "synthesizer" / "shared" / "simplify.md"
                ).read_text(encoding="utf-8")
            except Exception:
                simp = ""
            if simp:
                prompt = simp + "\n" + prompt
                self.logger.log(
                    "SYNTHESIS",
                    f"simplification pass (round {self.synthesis_count})")

        prompt = sanitize_model_visible_text(prompt)
        escalating = (
            self._consecutive_failed_syntheses
            >= self.config.escalate_to_opus_after
        )
        if escalating:
            self.logger.log(
                "SYNTHESIS",
                f"ESCALATING to {self.config.escalation_synthesis_model} "
                f"(sonnet failed {self._consecutive_failed_syntheses} "
                "consecutive times)"
            )
        result = self._invoke_claude(ws_dir, prompt, escalating=escalating)
        if result.get("quota_limited"):
            accuracy = baseline
            self.logger.log(
                "SYNTHESIS",
                "skipping verifier/critique repair after Codex quota limit",
            )
        else:
            accuracy = self._run_tests(ws_dir)
            accuracy, result = self._run_critique_repair_loop(
                ws_dir=ws_dir,
                accuracy=accuracy,
                result=result,
                escalating=escalating,
            )

        self._consume_xi_updates(ws_dir)
        if self.config.frames_only:
            self._refresh_spriteless_diagnostics(
                step, force=True, reason="post_synthesis",
            )

        self.logger.log("SYNTHESIS", f"Result: {accuracy}")
        self.logger.log("SYNTHESIS", f"Duration: {result.get('duration_s', '?')}s")

        if "ALL TESTS PASSED" in accuracy:
            if self._consecutive_failed_syntheses > 0:
                self.logger.log(
                    "SYNTHESIS",
                    f"recovered after {self._consecutive_failed_syntheses} "
                    "failed attempts, escalation counter reset"
            )
            self._consecutive_failed_syntheses = 0
        else:
            self._consecutive_failed_syntheses += 1
            self.logger.log(
                "SYNTHESIS",
                f"did not reach 100% (consecutive_failed="
                f"{self._consecutive_failed_syntheses}/"
                f"{self.config.escalate_to_opus_after})"
            )

        self._parse_and_record_accuracy(accuracy)

        latest = self.output_dir / "synthesis" / "latest"
        if latest.exists():
            latest.unlink()
        latest.symlink_to(ws_dir.name)

        self._update_world_model_after_synthesis(
            accuracy,
            ws_dir,
            state,
            mission,
            shared_doc_snapshot=shared_doc_snapshot,
        )

        self.run_log.append_synthesis(
            step=step,
            run_idx=self.synthesis_count,
            accuracy=accuracy,
            reward_src=self.goal_hypothesis_code,
            duration_s=result.get("duration_s"),
            escalated=escalating,
        )

        try:
            self._emit_curve_row(
                step=step, ws_dir=ws_dir, accuracy=accuracy,
                duration_s=float(result.get("duration_s") or 0.0),
                escalating=escalating,
            )
        except Exception as exc:
            self.logger.log(
                "SYNTHESIS",
                f"curve-row emit failed: {type(exc).__name__}: {exc}",
            )

        if not (
            result.get("rate_limited")
            or result.get("quota_limited")
            or result.get("error")
        ):
            self._reset_model_error_debt()
            self._last_synthesis_level = int(self.current_level)

        self._write_synth_status(step)

    def _invoke_claude(
        self, ws_dir: Path, prompt: str, escalating: bool = False,
    ) -> dict:
        """Spawn the synth model. Dispatches to Codex when backend=='codex',
        else Claude Code. When escalating, switches model and prepends context."""
        if self.config.backend == "codex":
            return self._invoke_codex_synth(ws_dir, prompt)
        claude_path = shutil.which("claude")
        if not claude_path:
            self.logger.log("ERROR", "Claude CLI not found")
            return {"error": "claude not found", "duration_s": 0}
        from .sandbox import (
            claude_popen_kwargs,
            describe_claude_resource_limits,
            terminate_process_group,
            wait_with_resource_monitor,
        )

        if escalating:
            model_to_use = self.config.escalation_synthesis_model
            escalation_block = (
                "*** ESCALATION CONTEXT ***\n"
                f"You are {self.config.escalation_synthesis_model.upper()}, "
                "and you have been called because the default synthesis "
                f"model ({self.config.synthesis_model.upper()}) attempted "
                f"this exact synthesis problem "
                f"{self._consecutive_failed_syntheses} consecutive times "
                "and did NOT reach 100% accuracy on the test runner.\n\n"
                "CRITICAL: the previous model's REASONING WAS WRONG. The "
                f"previous attempt's code is in `previous_attempt.py` "
                "(NOT in `game_engine.py`, that has been reset to a stub "
                "so you can start clean). It is built on hypotheses that "
                "DO NOT match the real game; the previous model kept "
                "patching around its broken hypotheses instead of "
                "replacing them. Treat `previous_attempt.py` as a "
                "SUSPECT, not as a foundation.\n\n"
                "BEFORE attempting any edits:\n"
                "  1. READ `test_runner_output.txt` FIRST. It lists the "
                "exact failing transitions with predicted vs actual values. "
                "Look at the FAILURE PATTERN, are the failures clustered "
                "in one region of state space, one action, one object "
                "configuration? That clustering reveals the true mechanic.\n"
                "  2. READ `previous_attempt.py` and identify which class "
                "/ function / hypothesis was producing the wrong "
                "predictions. Quote the lines that were wrong. Note which "
                "of those hypotheses you want to KEEP and which to "
                "DISCARD.\n"
                "  3. READ `replay_buffer.pkl` (via Python in Bash) for "
                "full state context around the failing step indices. "
                "Compare the structured state at a failing step vs a "
                "similar PASSING step, what's different?\n\n"
                "DIAGNOSE first, then EDIT. Be willing to:\n"
                "  - DELETE entire chunks of the previous code that were "
                "modeling the wrong mechanic.\n"
                "  - REWRITE the transition_function with a different "
                "structural assumption (e.g., the freeze isn't a 'trap', "
                "it's a position-conditional block; or it's not "
                "object-triggered, it's coordinate-triggered; or the "
                "blocking condition involves an object the previous model "
                "ignored entirely).\n"
                "  - QUESTION the hypothesized roles of objects (the "
                "previous model may have mistaken a UI counter for a game "
                "object, or vice versa).\n\n"
                "The previous fixes have been exhausted. Doing 'one more "
                "patch' on the same broken hypothesis will fail again. The "
                "right move is to find the wrong premise and replace it.\n"
                "*** END ESCALATION CONTEXT ***\n\n"
            )
            prompt = escalation_block + prompt
        else:
            model_to_use = self.config.synthesis_model

        prompt = sanitize_model_visible_text(prompt)
        cmd = [
            claude_path, "-p", prompt,
            "--model", model_to_use,
            "--effort", self.config.synthesis_effort,
            "--max-turns", str(self.config.synthesis_max_turns),
            "--output-format", "stream-json", "--verbose",
            "--no-session-persistence",
            "--allowedTools",
            "Bash,Read,Write,Edit,Grep,Glob,TodoWrite,NotebookEdit,Task",
        ]

        try:
            prompt_name = (
                "claude_prompt_escalation.txt" if escalating
                else "claude_prompt.txt"
            )
            (ws_dir / prompt_name).write_text(prompt)
        except Exception:
            pass

        stdout_path = ws_dir / "claude_chat.jsonl"
        stderr_path = ws_dir / "claude_stderr.txt"

        timeout_val = (
            self.config.synthesis_timeout
            if self.config.synthesis_timeout and self.config.synthesis_timeout > 0
            else None
        )

        from .agentic_consumer import _is_rate_limited
        rate_limit_retry_s = 60
        rate_limit_max_wait_s = 12 * 3600
        t_overall = time.time()
        waited_s = 0.0
        attempts = 0
        timed_out = False
        rate_limited_giveup = False
        try:
            self.logger.log(
                "SYNTHESIS",
                "claude resource limits: "
                f"{describe_claude_resource_limits()}",
            )
            while True:
                attempts += 1
                t0 = time.time()
                run_cmd, popen_kwargs, _dname = self._wrap_claude_subprocess(
                    cmd, ws_dir
                )
                with open(stdout_path, "w") as out_f, open(stderr_path, "w") as err_f:
                    proc = subprocess.Popen(
                        run_cmd, stdout=out_f, stderr=err_f, text=True,
                        cwd=str(ws_dir),
                        **popen_kwargs,
                    )
                    try:
                        rc = wait_with_resource_monitor(
                            proc,
                            timeout_s=timeout_val,
                            log_fn=self.logger.log,
                        )
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        terminate_process_group(proc)
                        self._docker_rm(_dname)
                        rc = -1
                duration_s = time.time() - t0
                out_txt = ""
                err_txt = ""
                try:
                    out_txt = stdout_path.read_text()
                except Exception:
                    pass
                try:
                    err_txt = stderr_path.read_text()
                except Exception:
                    pass
                if _is_rate_limited(out_txt, err_txt, rc, duration_s):
                    if waited_s >= rate_limit_max_wait_s:
                        rate_limited_giveup = True
                        break
                    self.logger.log(
                        "SYNTHESIS",
                        f"rate-limited (attempt {attempts}, rc={rc}, "
                        f"dur={duration_s:.1f}s); sleeping "
                        f"{rate_limit_retry_s}s "
                        f"(cumulative wait {int(waited_s)}s)"
                    )
                    time.sleep(rate_limit_retry_s)
                    waited_s += rate_limit_retry_s
                    continue
                break
            duration = time.time() - t_overall
            if timed_out:
                self.logger.log(
                    "SYNTHESIS",
                    f"TIMED OUT after {timeout_val}s "
                    f"(partial output saved to {stdout_path.name})"
                )
            if rate_limited_giveup:
                self.logger.log(
                    "SYNTHESIS",
                    f"GAVE UP after {int(waited_s)}s of rate-limit waiting"
                )
            return {
                "duration_s": round(duration, 1),
                "timed_out": timed_out,
                "rate_limited": rate_limited_giveup,
                "rate_limit_wait_s": int(waited_s),
                "attempts": attempts,
            }
        except Exception as e:
            duration = time.time() - t0
            self.logger.log("SYNTHESIS", f"EXCEPTION {type(e).__name__}: {e}")
            return {"duration_s": round(duration, 1), "error": str(e)}

    def _invoke_codex_synth(self, ws_dir: Path, prompt: str) -> dict:
        """Run a synthesis turn in the locked-down Codex container.

        The synth edits game_engine.py in ws_dir (file-based contract, identical
        to the claude path). The engine re-runs test_runner.py afterwards. The
        run dir is mounted RO with ws_dir overlaid RW. Egress is API-only.
        """
        from . import codex_backend as cx
        prompt = sanitize_model_visible_text(prompt)
        ws_dir = ws_dir.resolve()
        run_dir = self.output_dir.resolve()
        try:
            rel = ws_dir.relative_to(run_dir)
        except ValueError:
            self.logger.log("SYNTHESIS", f"codex: ws_dir {ws_dir} not under run dir")
            return {"duration_s": 0.0, "error": "ws outside run dir"}
        try:
            (ws_dir / "claude_prompt.txt").write_text(prompt)
        except Exception:
            pass
        timeout_val = (
            self.config.synthesis_timeout
            if self.config.synthesis_timeout and self.config.synthesis_timeout > 0
            else None
        )
        use_resume = (
            bool(getattr(self.config, "synth_continue", False))
            and self._codex_synth_session_id is not None
        )
        stdout_path = ws_dir / "claude_chat.jsonl"
        stderr_path = ws_dir / "claude_stderr.txt"
        for stale_retry_log in (
            ws_dir / "codex_quota_wait.txt",
            ws_dir / "codex_retry_tracker.jsonl",
        ):
            try:
                stale_retry_log.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
        codex_home = cx.codex_home_path(self.config.codex_home)
        game_engine_path = ws_dir / "game_engine.py"
        before_game_engine_hash = (
            hashlib.sha256(game_engine_path.read_bytes()).hexdigest()
            if game_engine_path.exists()
            else None
        )
        attempts = 0
        quota_wait_s = 0
        rate_limit_retry_s = 60
        while True:
            attempts += 1
            res = cx.run_codex_turn(
                prompt=prompt,
                workspace_dir=ws_dir,
                run_dir=run_dir,
                container_cd=f"/run/{rel}",
                model=self.config.codex_model,
                effort=self.config.codex_effort,
                codex_home=codex_home,
                session_id=(self._codex_synth_session_id if use_resume else None),
                timeout_s=timeout_val,
                image_name=self.config.codex_image,
                network=self.config.codex_network,
                gateway=self.config.codex_gateway,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            retryable_infra = bool(res.get("retryable_infra_failure"))
            quota_limited = bool(res.get("quota_limited"))
            if retryable_infra and game_engine_path.exists():
                after_game_engine_hash = hashlib.sha256(
                    game_engine_path.read_bytes()
                ).hexdigest()
                if after_game_engine_hash != before_game_engine_hash:
                    res["retryable_infra_failure"] = False
                    res["reason"] = "ok_dirty_stream_with_game_engine"
                    retryable_infra = False
            if not (quota_limited or retryable_infra):
                break
            quota_wait_s += rate_limit_retry_s
            event = {
                "kind": "quota_exhausted" if quota_limited else "codex_infra_retry",
                "synthesis_run": self.synthesis_count,
                "attempt": attempts,
                "reason": res.get("reason"),
                "returncode": res.get("returncode"),
                "duration_s": res.get("duration_s"),
                "quota_limited": quota_limited,
                "retryable_infra_failure": retryable_infra,
                "remote_compact_failed": bool(res.get("remote_compact_failed")),
                "using_resume": use_resume,
                "wait_s": rate_limit_retry_s,
                "total_wait_s": quota_wait_s,
                "ts": time.time(),
            }
            msg = json.dumps(event, sort_keys=True)
            self.logger.log("SYNTHESIS", msg)
            try:
                with open(ws_dir / "codex_quota_wait.txt", "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
            try:
                with open(ws_dir / "codex_retry_tracker.jsonl", "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
            time.sleep(rate_limit_retry_s)
        if getattr(self.config, "synth_continue", False) and res.get("session_id"):
            self._codex_synth_session_id = res["session_id"]
            try:
                (ws_dir / "codex_synth_session_id.txt").write_text(
                    self._codex_synth_session_id
                )
            except Exception:
                pass
        self.logger.log(
            "SYNTHESIS",
            f"codex turn: resume={use_resume} "
            f"reason={res['reason']} rc={res['returncode']} "
            f"dur={res['duration_s']}s attempts={attempts} "
            f"quota_wait_s={quota_wait_s} "
            f"usage={res.get('usage')}"
        )
        quota_limited = bool(res.get("quota_limited"))
        return {
            "duration_s": res["duration_s"],
            "timed_out": res["reason"] == "timed_out",
            "rate_limited": quota_limited,
            "quota_limited": quota_limited,
            "rate_limit_wait_s": quota_wait_s,
            "attempts": attempts,
        }

    def _run_tests(self, ws_dir: Path) -> str:
        """Run test_runner.py and return result string. Full output saved to test_runner_output.txt."""
        ws_abs = ws_dir.resolve()
        try:
            from .sandbox import claude_popen_kwargs
            result = subprocess.run(
                ["python", str(ws_abs / "test_runner.py")],
                capture_output=True, text=True, timeout=30,
                cwd=str(ws_abs),
                **claude_popen_kwargs(),
            )
            full = result.stdout or ""
            try:
                (ws_abs / "test_runner_output.txt").write_text(full)
                if result.stderr:
                    (ws_abs / "test_runner_stderr.txt").write_text(result.stderr)
            except Exception:
                pass
            lines = [
                line for line in full.strip().split("\n")
                if line.startswith((
                    "TRANSITION:", "REWARD:", "RESULT:", "ALL ",
                    "OBJECTS:", "LOAD_ERROR",
                ))
            ]
            return " | ".join(lines) if lines else f"NO OUTPUT (stderr: {result.stderr[:200]})"
        except Exception as e:
            return f"ERROR: {e}"

    @staticmethod
    def _sum_round_tokens(chat_jsonl: Path) -> dict[str, int]:
        """Sum token counts across stream-json events for one synthesis round."""
        totals = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        if not chat_jsonl.exists():
            return totals
        try:
            for raw in chat_jsonl.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                usage = (
                    msg.get("message", {}).get("usage")
                    if isinstance(msg.get("message"), dict)
                    else None
                ) or msg.get("usage")
                if not isinstance(usage, dict):
                    continue
                for k in totals:
                    v = usage.get(k)
                    if isinstance(v, (int, float)):
                        totals[k] += int(v)
        except Exception:
            pass
        return totals

    @staticmethod
    def _parse_transition_accuracy(accuracy_str: str) -> tuple[int, int]:
        """Extract (passed, total) from the TRANSITION clause of an accuracy string."""
        import re as _re
        m = _re.search(
            r"TRANSITION:\s*(\d+)\s*/\s*(\d+)\s*passed",
            accuracy_str,
        )
        if m is None:
            return (0, 0)
        return (int(m.group(1)), int(m.group(2)))

    def _emit_curve_row(
        self,
        *,
        step: int,
        ws_dir: Path,
        accuracy: str,
        duration_s: float,
        escalating: bool,
    ) -> None:
        """Append one cost-to-soundness row to synthesis_curve.jsonl."""
        ge_path = ws_dir / "game_engine.py"
        try:
            loc = sum(
                1 for line in ge_path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            )
        except Exception:
            loc = -1
        passed, total = self._parse_transition_accuracy(accuracy)
        tokens = self._sum_round_tokens(ws_dir / "claude_chat.jsonl")
        row = {
            "round": int(self.synthesis_count),
            "step": int(step),
            "synth_mode": str(self.config.synth_mode),
            "n_transitions": len(self.replay_buffer),
            "game_engine_loc": int(loc),
            "passed": passed,
            "total": total,
            "accuracy_pct": (
                round(100 * passed / total, 2) if total > 0 else None
            ),
            "all_passed": "ALL TESTS PASSED" in accuracy,
            "escalated": bool(escalating),
            "wall_s_this_round": round(float(duration_s), 2),
            **tokens,
        }
        path = self.output_dir / "synthesis_curve.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def _serialize_transitions(self) -> list[dict]:
        """Convert replay buffer to plain dicts. Under frames_only, omits sprite keys."""
        out = []
        for t in self.replay_buffer:
            if self.config.frames_only:
                d = {
                    "action_id": t.action_id,
                    "action_name": t.action_name,
                    "before_frame": t.before_frame,
                    "after_frame": t.after_frame,
                    "diff_text": t.diff_text,
                    "reward": t.reward,
                    "done": t.done,
                    "timestep": t.timestep,
                    "level": t.level,
                }
            else:
                d = {
                    "before_state": t.before_state,
                    "action_id": t.action_id,
                    "action_name": t.action_name,
                    "after_state": t.after_state,
                    "diff_text": t.diff_text,
                    "reward": t.reward,
                    "done": t.done,
                    "timestep": t.timestep,
                    "level": t.level,
                }
            if hasattr(t, "click_x") and hasattr(t, "click_y"):
                d["click_x"] = t.click_x
                d["click_y"] = t.click_y
            out.append(d)
        return out

    def _parse_and_record_accuracy(self, accuracy_str: str):
        import re
        for part in accuracy_str.split("|"):
            part = part.strip()
            m = re.match(r"TRANSITION:\s*(\d+)/(\d+)", part)
            if m:
                acc = int(m.group(1)) / max(int(m.group(2)), 1)
                self.best_transition_accuracy = max(self.best_transition_accuracy, acc)
            m = re.match(r"REWARD:\s*(\d+)/(\d+)", part)
            if m:
                acc = int(m.group(1)) / max(int(m.group(2)), 1)
                self.best_reward_accuracy = max(self.best_reward_accuracy, acc)

    def _extract_goal_in_english(self, ws_dir: Path) -> str | None:
        """Return the text of the synth's "# GOAL: ..." comment from game_engine.py, or None."""
        code_path = ws_dir / "game_engine.py"
        if not code_path.exists():
            return None
        try:
            lines = code_path.read_text().splitlines()
        except Exception:
            return None
        for line in lines[:60]:
            stripped = line.strip()
            if stripped.startswith("# GOAL:"):
                txt = stripped[len("# GOAL:"):].strip()
                if txt:
                    return txt[:400]
        return None

    def _consume_synthesis_handoff_artifacts(
        self,
        ws_dir: Path,
        *,
        shared_doc_snapshot: dict[str, str] | None = None,
    ) -> None:
        """Persist synth-written text artifacts that feed later prompts."""
        artifacts = [
            ("synth_learnings.md", "synth_learnings", "SYNTH_LEARNINGS", 8000),
            ("critique_response.md", "critique_response", "CRITIQUE", 8000),
        ]
        for filename, attr, log_label, limit in artifacts:
            src = ws_dir / filename
            txt = self._read_text_artifact(src, limit=limit)
            if not txt:
                continue
            setattr(self, attr, txt)
            try:
                (self.output_dir / filename).write_text(txt)
            except Exception:
                pass
            self.logger.log(
                log_label,
                f"captured {filename} from synth run {self.synthesis_count}",
            )

        if (ws_dir / "requires_critique_response.flag").exists() and not (
            ws_dir / "critique_response.md"
        ).exists():
            self.logger.log(
                "CRITIQUE",
                "WARN: synth received critique but wrote no critique_response.md",
            )

        anim = self._read_text_artifact(
            self.output_dir / "animation_analysis.md", limit=8000
        )
        if anim:
            self.animation_findings = anim

        shared_summary = self._capture_shared_model_artifacts(
            ws_dir,
            source="synth",
            before_snapshot=shared_doc_snapshot,
        )
        if shared_summary:
            self.shared_model_updates = shared_summary
            try:
                synth_path = self.output_dir / "synth_learnings.md"
                existing = synth_path.read_text() if synth_path.exists() else ""
                marker = (
                    "## Shared world-model document updates "
                    f"(run {self.synthesis_count})"
                )
                synth_path.write_text(
                    sanitize_model_visible_text(existing).rstrip()
                    + f"\n\n{marker}\n"
                    + sanitize_model_visible_text(shared_summary)
                )
                self.synth_learnings = sanitize_model_visible_text(
                    synth_path.read_text()
                )
            except Exception:
                pass

    def _consume_xi_updates(self, ws_dir: Path) -> None:
        """Read xi_updates.json, verify each proposed feature against worst-K strata, and apply survivors.

        SKIPPED under frames_only. A missing file is a no-op.
        """
        if self.config.frames_only:
            return
        path = ws_dir / "xi_updates.json"
        if not path.exists():
            return
        try:
            updates = json.loads(path.read_text())
        except Exception as exc:
            self.logger.log(
                "XI",
                f"xi_updates.json parse failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return
        if not isinstance(updates, dict):
            self.logger.log(
                "XI", "xi_updates.json is not a dict; ignoring"
            )
            return
        proposals = updates.get("add", [])
        if not isinstance(proposals, list):
            self.logger.log(
                "XI", "xi_updates 'add' is not a list; ignoring"
            )
            return

        latest = self.ontology._latest or {}
        flat = latest.get("flat") or {}
        worst_strata = flat.get("worst_strata") or []
        E = max(2, int(flat.get("effect_alphabet_size", 2)))
        alpha_0 = self.ontology.alpha_0
        current_committed = self.ontology.committed_features()

        v_n_min = 3
        v_modal_frac_min = 0.95
        v_min_eta_reduction = 0.05
        v_top_k_strata = 3

        from .epistemic import _collect_stratum_transitions, _score_xi_candidate

        transitions = self._serialize_transitions()

        verified: list[dict] = []
        rejected: list[dict] = []
        verdicts_log: list[dict] = []

        for raw in proposals:
            norm = self.ontology._validate_feature(raw)
            if norm is None:
                rejected.append({
                    "raw": raw, "reason": "invalid schema",
                })
                continue
            if norm in current_committed:
                rejected.append({
                    "feature": norm, "reason": "already committed",
                })
                continue
            if norm in verified:
                rejected.append({
                    "feature": norm,
                    "reason": "duplicate within this update",
                })
                continue

            if not worst_strata:
                rejected.append({
                    "feature": norm,
                    "reason": ("no worst-K strata available "
                               "(no η measure yet)"),
                })
                continue

            stratum_verdicts: list[dict] = []
            accepted_on_any = False
            for stratum in worst_strata[:v_top_k_strata]:
                ty = stratum.get("type")
                aid = stratum.get("action_id")
                ctx = stratum.get("context")
                eta_old = float(stratum.get("eta_g", 1.0))
                stratum_trans = _collect_stratum_transitions(
                    transitions, ty, aid, ctx,
                    committed_features=current_committed or None,
                )
                if not stratum_trans:
                    continue
                scored = _score_xi_candidate(
                    stratum_trans, norm,
                    alpha_0=alpha_0,
                    effect_alphabet_size=E,
                    n_min=v_n_min,
                    modal_frac_min=v_modal_frac_min,
                )
                if scored is None:
                    continue
                reduction = eta_old - scored["eta_new"]
                accepted_here = (
                    reduction >= v_min_eta_reduction
                    and scored["identified_substrata"] >= 1
                )
                stratum_verdicts.append({
                    "type": ty, "action_id": aid,
                    "eta_old": round(eta_old, 6),
                    "eta_new": scored["eta_new"],
                    "eta_reduction": round(reduction, 6),
                    "n_substrata": scored["n_substrata"],
                    "identified_substrata": scored["identified_substrata"],
                    "accepted": accepted_here,
                })
                if accepted_here:
                    accepted_on_any = True

            verdicts_log.append({
                "feature": norm,
                "verdicts": stratum_verdicts,
                "accepted": accepted_on_any,
            })

            if accepted_on_any:
                verified.append(norm)
            else:
                rejected.append({
                    "feature": norm,
                    "reason": "verifier rejected (no worst-K stratum "
                              "met Δη≥0.05 with identification)",
                    "verdicts": stratum_verdicts,
                })

        summary = self.ontology.apply_xi_updates({"add": verified})

        rationale = str(updates.get("rationale", "") or "").strip()
        n_total = len(self.ontology.committed_features())
        self.logger.log(
            "XI",
            f"xi_updates: {len(proposals)} proposed, "
            f"+{summary['added']} verified+applied, "
            f"{len(rejected)} rejected; "
            f"operative ξ features now {n_total}"
            + (f"; rationale={rationale[:120]!r}" if rationale else "")
        )
        for r in rejected[:3]:
            feat = r.get("feature") or r.get("raw")
            self.logger.log(
                "XI", f"  rejected {feat!r}: {r.get('reason')}"
            )

        try:
            applied = {
                "step_consumed": int(getattr(self, "_cur_step", -1)),
                "synthesis_run": int(self.synthesis_count),
                "updates_in": updates,
                "verifier_params": {
                    "n_min": v_n_min,
                    "modal_frac_min": v_modal_frac_min,
                    "min_eta_reduction": v_min_eta_reduction,
                    "top_k_strata": v_top_k_strata,
                },
                "verdicts": verdicts_log,
                "verified_and_applied": verified,
                "rejected": rejected,
                "apply_summary": summary,
                "operative_features_after": (
                    self.ontology.committed_features()
                ),
            }
            (ws_dir / "xi_updates_applied.json").write_text(
                json.dumps(applied, indent=2, default=str)
            )
        except Exception:
            pass

    def _extract_reward_function(self, ws_dir: Path) -> str | None:
        """Return the source of reward_function from game_engine.py, or None if absent."""
        code_path = ws_dir / "game_engine.py"
        if not code_path.exists():
            return None
        try:
            code = code_path.read_text()
        except Exception:
            return None

        lines = code.split("\n")
        start = None
        for i, line in enumerate(lines):
            if line.startswith("def reward_function"):
                start = i
                break
        if start is None:
            return None

        end = len(lines)
        for i in range(start + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                continue
            if line[0] in (" ", "\t"):
                continue
            end = i
            break

        return "\n".join(lines[start:end]).rstrip()

    def _update_world_model_after_synthesis(
        self, accuracy_str: str, ws_dir: Path,
        state: list[dict], mission: str | None,
        *,
        shared_doc_snapshot: dict[str, str] | None = None,
    ):
        """Extract reward_function, rebuild the world model doc, append synthesis feedback."""
        reward_src = self._extract_reward_function(ws_dir)
        if reward_src:
            self.goal_hypothesis_code = reward_src
            self.goal_hypothesis_synthesis_run = self.synthesis_count
            self.logger.log("GOAL_HYPO",
                f"extracted reward_function from run {self.synthesis_count} "
                f"({len(reward_src)} chars)")
        else:
            self.logger.log("GOAL_HYPO",
                f"WARN: no reward_function found in run {self.synthesis_count}")

        prev_eng = self.goal_in_english
        self.goal_in_english = self._extract_goal_in_english(ws_dir)
        if self.goal_in_english and self.goal_in_english != prev_eng:
            self.logger.log(
                "GOAL_HYPO",
                f"# GOAL: {self.goal_in_english}",
            )

        self._consume_synthesis_handoff_artifacts(
            ws_dir,
            shared_doc_snapshot=shared_doc_snapshot,
        )
        self._update_world_model_doc(state, mission)

        self.world_model_doc += f"\n\n## Synthesis Feedback (run {self.synthesis_count})\n"
        self.world_model_doc += f"{accuracy_str}\n"

        if "ALL TESTS PASSED" in accuracy_str:
            self.world_model_doc += "The synthesized model is fully consistent with observations.\n"
        else:
            self.world_model_doc += (
                "The synthesized model has inconsistencies. "
                "Further exploration may reveal missing rules.\n"
            )

    SNAPSHOTS_DIRNAME = "snapshots"

    def _snapshot_due(self, step: int) -> bool:
        """True when a stop-and-snapshot target is set and this step is at/after it."""
        target = self.config.stop_and_snapshot_at_step
        return target is not None and step >= int(target)

    def _snapshot_root(self) -> Path:
        if self.config.snapshot_dir:
            return Path(self.config.snapshot_dir)
        return self.output_dir / self.SNAPSHOTS_DIRNAME

    def _take_snapshot(
        self, completed_step: int, *, reason: str = "",
    ) -> Path | None:
        """Write a self-contained, reloadable copy of the ENTIRE run directory.

        The snapshot captures completion through ``completed_step``: a fresh
        checkpoint.pkl plus every on-disk artifact (frames, syntheses, analyzer
        logs, run_log, ontology trace, consumer workspace). Resuming a run from
        the snapshot's checkpoint.pkl reproduces the run identically -- the env
        is rebuilt by replaying _actions_taken and the remaining state is fully
        restored. The copy dereferences symlinks so the snapshot never points
        back at the live dir (no future-state leak) and is portable.
        """
        self._save_checkpoint(completed_step)
        self._snapshot_completed_step = int(completed_step)

        snap_root = self._snapshot_root()
        try:
            snap_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.logger.log("SNAPSHOT", f"mkdir failed: {exc}")
            return None
        snap_dir = snap_root / f"snapshot_step_{int(completed_step):04d}"
        if snap_dir.exists():
            shutil.rmtree(snap_dir, ignore_errors=True)

        out_resolved = self.output_dir.resolve()
        snap_root_resolved = snap_root.resolve()

        def _ignore(dirpath, names):
            ignored: set[str] = set()
            dp = Path(dirpath).resolve()
            for n in names:
                child = (Path(dirpath) / n).resolve()
                if child == snap_root_resolved:
                    ignored.add(n)
                elif dp == out_resolved and n in ("replay.mp4", "replay.gif"):
                    ignored.add(n)
            return ignored

        try:
            shutil.copytree(
                self.output_dir, snap_dir,
                symlinks=False, ignore=_ignore,
                ignore_dangling_symlinks=True,
            )
        except Exception as exc:
            self.logger.log(
                "SNAPSHOT",
                f"copytree failed: {type(exc).__name__}: {exc}"
            )
            return None

        try:
            meta = {
                "completed_step": int(completed_step),
                "resume_step": int(completed_step) + 1,
                "n_actions": len(self._actions_taken),
                "n_transitions": len(self.replay_buffer),
                "synthesis_count": self.synthesis_count,
                "levels_completed": self.levels_completed,
                "current_level": self.current_level,
                "game_won": self.game_won,
                "crystallised": self.crystallised,
                "queued_plan_len": len(self._llm_plan),
                "queued_planner_plan_len": len(self._planner_queue),
                "model_error_count": self._model_error_count,
                "model_error_first_step": self._model_error_first_step,
                "model_error_action_plan_count":
                    self._model_error_action_plan_count,
                "frames_only": self.config.frames_only,
                "reason": reason,
                "source_output_dir": str(self.output_dir),
                "schema_version": 8,
            }
            (snap_dir / "snapshot_meta.json").write_text(
                json.dumps(meta, indent=2, default=str)
            )
        except Exception:
            pass

        self.logger.log(
            "SNAPSHOT",
            f"wrote {snap_dir} (completed_step={completed_step}, "
            f"resume_step={completed_step + 1}, "
            f"n_actions={len(self._actions_taken)}, reason={reason})",
        )
        return snap_dir

    CHECKPOINT_FILENAME = "checkpoint.pkl"

    def _save_checkpoint(self, step: int) -> None:
        """Atomically save engine state. Env state is not pickled. It is restored by action replay on resume."""
        try:
            ck = {
                "schema_version": 8,
                "step": step,
                "actions_taken": list(self._actions_taken),
                "agentic_consumer_call_count":
                    self.agentic_consumer.call_count,
                "agentic_consumer_needs_fresh_session":
                    self.agentic_consumer._needs_fresh_session,
                "agentic_consumer_codex_session_id": getattr(
                    self.agentic_consumer, "_codex_session_id", None
                ),
                "agentic_consumer_claude_session_id": getattr(
                    self.agentic_consumer, "_claude_session_id", None
                ),
                "agentic_consumer_continue_chain_len": getattr(
                    self.agentic_consumer, "_continue_chain_len", 0
                ),
                "agentic_consumer_cumulative_usage":
                    dict(self.agentic_consumer.cumulative_usage),
                "config": self.config,
                "replay_buffer": self.replay_buffer,
                "world_model_doc": self.world_model_doc,
                "goal_hypothesis_code": self.goal_hypothesis_code,
                "goal_hypothesis_synthesis_run":
                    self.goal_hypothesis_synthesis_run,
                "goal_in_english": self.goal_in_english,
                "synth_learnings": self.synth_learnings,
                "critique_findings": self.critique_findings,
                "critique_response": self.critique_response,
                "animation_findings": self.animation_findings,
                "shared_model_updates": self.shared_model_updates,
                "animation_analysis_count": self._animation_analysis_count,
                "synthesis_count": self.synthesis_count,
                "best_transition_accuracy": self.best_transition_accuracy,
                "best_reward_accuracy": self.best_reward_accuracy,
                "last_synthesis_step": self.last_synthesis_step,
                "pending_synthesis_step": self._pending_synthesis_step,
                "consecutive_failed_syntheses":
                    self._consecutive_failed_syntheses,
                "known_types": self.known_types,
                "type_aliases": self.type_aliases,
                "level_states": self.level_states,
                "level_frames": self.level_frames,
                "current_level": self.current_level,
                "total_reward": self.total_reward,
                "levels_completed": self.levels_completed,
                "game_won": self.game_won,
                "llm_plan": self._llm_plan,
                "llm_plan_origin_step": self._llm_plan_origin_step,
                "llm_plan_no_effect_streak": self._llm_plan_no_effect_streak,
                "planner_queue": self._planner_queue,
                "planner_trace": self._planner_trace,
                "planner_expectation": self._planner_expectation,
                "planner_plan_origin_step": self._planner_plan_origin_step,
                "planner_blocked_round": self._planner_blocked_round,
                "planner_retry_after_step": self._planner_retry_after_step,
                "planner_last_status": self._planner_last_status,
                "warmup_queue": self._warmup_queue,
                "game_over_streak": self._game_over_streak,
                "steps_since_analyzer": list(self._steps_since_analyzer),
                "model_error_first_step": self._model_error_first_step,
                "model_error_last_step": self._model_error_last_step,
                "model_error_count": self._model_error_count,
                "model_error_action_plan_count":
                    self._model_error_action_plan_count,
                "model_error_level_completed":
                    self._model_error_level_completed,
                "last_synthesis_level": self._last_synthesis_level,
                "rng_state": self.rng.bit_generator.state,
                "last_plan_hint": self._last_plan_hint,
                "ontology_phase": self.ontology.phase.value,
                "crystallised": self.crystallised,
                "crystallisation_step": self.crystallisation_step,
                "crystallised_partition": self.crystallised_partition,
                "crystallised_scope_extra": (
                    self.crystallised_scope_extra
                ),
                "committed_xi_features": (
                    self.ontology.committed_features()
                ),
            }
            path = self.output_dir / self.CHECKPOINT_FILENAME
            tmp = path.with_suffix(".pkl.tmp")
            with open(tmp, "wb") as f:
                pickle.dump(ck, f)
            tmp.replace(path)
        except Exception as e:
            try:
                self.logger.log(
                    "CHECKPOINT", f"save failed: {type(e).__name__}: {e}"
                )
            except Exception:
                pass

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore checkpoint state and replay actions to re-sync env. Call after __init__, before run()."""
        path = Path(path)
        with open(path, "rb") as f:
            ck = pickle.load(f)

        schema = ck.get("schema_version", 0)
        if schema not in (1, 2, 3, 4, 5, 6, 7, 8):
            raise ValueError(
                f"checkpoint schema {schema} not supported"
            )

        self.replay_buffer = ck["replay_buffer"]
        self.world_model_doc = ck["world_model_doc"]
        self.goal_hypothesis_code = ck["goal_hypothesis_code"]
        self.goal_hypothesis_synthesis_run = ck["goal_hypothesis_synthesis_run"]
        self.goal_in_english = ck.get("goal_in_english")
        self.synth_learnings = ck.get("synth_learnings", "")
        self.critique_findings = ck.get("critique_findings", "")
        self.critique_response = ck.get("critique_response", "")
        self.animation_findings = ck.get("animation_findings", "")
        self.shared_model_updates = ck.get("shared_model_updates", "")
        self._animation_analysis_count = int(ck.get("animation_analysis_count", 0))
        self.synthesis_count = ck["synthesis_count"]
        self.best_transition_accuracy = ck["best_transition_accuracy"]
        self.best_reward_accuracy = ck["best_reward_accuracy"]
        self.last_synthesis_step = ck["last_synthesis_step"]
        self._pending_synthesis_step = ck.get("pending_synthesis_step")
        self._consecutive_failed_syntheses = ck["consecutive_failed_syntheses"]
        self.known_types = ck["known_types"]
        self.type_aliases = ck.get("type_aliases", {})
        for tag in self.known_types:
            aliases_mod.ensure_seeded(self.type_aliases, tag)
        self.level_states = ck["level_states"]
        self.level_frames = ck.get("level_frames", {})
        self.current_level = ck["current_level"]
        self.total_reward = ck["total_reward"]
        self.levels_completed = ck["levels_completed"]
        self.game_won = ck["game_won"]
        self._llm_plan = ck["llm_plan"]
        self._llm_plan_origin_step = ck["llm_plan_origin_step"]
        self._llm_plan_no_effect_streak = ck["llm_plan_no_effect_streak"]
        self._planner_queue = list(ck.get("planner_queue", []))
        self._planner_trace = list(ck.get("planner_trace", []))
        self._planner_expectation = ck.get("planner_expectation")
        self._planner_plan_origin_step = int(
            ck.get("planner_plan_origin_step", -1)
        )
        self._planner_blocked_round = int(
            ck.get("planner_blocked_round", -1)
        )
        self._planner_retry_after_step = int(
            ck.get("planner_retry_after_step", 0)
        )
        self._planner_last_status = ck.get(
            "planner_last_status", {"ok": False, "reason": "restored"}
        )
        self._warmup_queue = list(ck.get("warmup_queue", []))
        self._game_over_streak = int(ck.get("game_over_streak", 0))
        self._steps_since_analyzer = list(ck.get("steps_since_analyzer", []))
        self._model_error_first_step = ck.get("model_error_first_step")
        self._model_error_last_step = ck.get("model_error_last_step")
        self._model_error_count = int(ck.get("model_error_count", 0) or 0)
        self._model_error_action_plan_count = int(
            ck.get("model_error_action_plan_count", 0) or 0
        )
        self._model_error_level_completed = bool(
            ck.get("model_error_level_completed", False)
        )
        self._last_synthesis_level = int(
            ck.get("last_synthesis_level", self.current_level)
        )
        self._div_model = None
        self._div_reward_model = None
        self._div_model_round = -1
        self._div_mask = frozenset()
        self._planner_model = None
        self._planner_model_round = -1
        self._planner_consistency_key = None
        self._planner_verification_key = None
        self._last_plan_hint = ck.get("last_plan_hint")
        try:
            from .ontology import Phase
            self.ontology.phase = Phase(ck.get("ontology_phase", "flat"))
        except Exception:
            pass
        self.crystallised = ck.get("crystallised", False)
        self.crystallisation_step = ck.get("crystallisation_step")
        self.crystallised_partition = ck.get(
            "crystallised_partition", {}
        )
        self.crystallised_scope_extra = ck.get(
            "crystallised_scope_extra", {}
        )
        prior_features = ck.get("committed_xi_features") or []
        if prior_features:
            self.ontology.apply_xi_updates({"add": prior_features})
        self.rng.bit_generator.state = ck["rng_state"]
        self._actions_taken = list(ck["actions_taken"])
        self.agentic_consumer.call_count = ck.get(
            "agentic_consumer_call_count", 0
        )
        if ck.get("agentic_consumer_codex_session_id"):
            self.agentic_consumer._codex_session_id = ck.get(
                "agentic_consumer_codex_session_id"
            )
        if ck.get("agentic_consumer_claude_session_id"):
            self.agentic_consumer._claude_session_id = ck.get(
                "agentic_consumer_claude_session_id"
            )
        self.agentic_consumer._continue_chain_len = int(
            ck.get("agentic_consumer_continue_chain_len", 0) or 0
        )
        try:
            self.agentic_consumer.cumulative_usage.update(
                ck.get("agentic_consumer_cumulative_usage", {}) or {}
            )
        except Exception:
            pass
        self.agentic_consumer._needs_fresh_session = bool(
            ck.get("agentic_consumer_needs_fresh_session", False)
        )

        self.env.reset()
        for action in self._actions_taken:
            try:
                self.env.step(action)
            except Exception as e:
                raise RuntimeError(
                    f"env replay failed at action {action!r} during "
                    f"checkpoint resume: {e}"
                )

        self._resume_step = ck["step"] + 1

        self._ensure_shared_model_artifacts(current_level=self.current_level)
        try:
            self._write_synth_status(self._resume_step - 1)
        except Exception as exc:
            self.logger.log(
                "RESUME", f"synth_status rewrite failed: {exc}"
            )
        if self.config.frames_only:
            self._refresh_spriteless_diagnostics(
                self._resume_step - 1, force=True, reason="resume",
            )
        else:
            try:
                dump_epistemic_matrix(
                    self._serialize_transitions(),
                    self.output_dir / "epistemic_matrix.json",
                    alpha_0=self.config.epistemic_alpha_0,
                    beta_0=self.config.epistemic_beta_0,
                    kappa=self.config.epistemic_kappa,
                    sort_by=self.config.epistemic_sort_by,
                    rng=np.random.default_rng(self.config.seed),
                )
            except Exception as exc:
                self.logger.log(
                    "RESUME", f"epistemic re-dump failed: {exc}"
                )

        self.logger.log(
            "RESUME",
            f"loaded checkpoint: step={ck['step']}, "
            f"transitions={len(self.replay_buffer)}, "
            f"synthesis_count={self.synthesis_count}, "
            f"next_step={self._resume_step}"
        )

    def _build_summary(self, final_step: int) -> dict:
        summary = {
            "total_steps": final_step + 1,
            "total_transitions": len(self.replay_buffer),
            "levels_completed": self.levels_completed,
            "game_won": self.game_won,
            "total_reward": self.total_reward,
            "synthesis_runs": self.synthesis_count,
            "best_transition_accuracy": self.best_transition_accuracy,
            "best_reward_accuracy": self.best_reward_accuracy,
            "planner_last_status": self._planner_last_status,
            "known_types": list(self.known_types.keys()),
            "transitions_per_level": {},
        }

        for t in self.replay_buffer:
            key = str(t.level)
            summary["transitions_per_level"][key] = \
                summary["transitions_per_level"].get(key, 0) + 1

        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.section("RUN COMPLETE")
        self.logger.log("SUMMARY", f"Steps: {summary['total_steps']}")
        self.logger.log("SUMMARY", f"Levels completed: {summary['levels_completed']}")
        self.logger.log("SUMMARY", f"Game won: {summary['game_won']}")
        self.logger.log("SUMMARY", f"Transition accuracy: {summary['best_transition_accuracy']:.0%}")
        self.logger.log("SUMMARY", f"Reward accuracy: {summary['best_reward_accuracy']:.0%}")
        self.logger.log("SUMMARY", f"Known types: {len(self.known_types)}")
        self.logger.log("SUMMARY", f"Results: {summary_path}")

        if not self._stopped_for_snapshot:
            try:
                replay = self._compose_replay()
                if replay is not None:
                    summary["replay_path"] = str(replay)
                    self.logger.log("SUMMARY", f"Replay: {replay}")
            except Exception as exc:
                self.logger.log(
                    "SUMMARY",
                    f"replay composition failed: {type(exc).__name__}: {exc}"
                )

        try:
            summary_path.write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

        return summary
