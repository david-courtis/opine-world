#!/usr/bin/env python3
"""
Run the core synthesis engine on any local ARC-AGI-3 game.

The game is selected by ``--game <name>`` where ``<name>`` is a directory
under ``environment_files/``. The runner auto-discovers the game class
inside that directory (the only subclass of ``arcengine.ARCBaseGame``).

Usage:
    python play.py --game pushbox
    python play.py --game complex_maze --max-actions 250
    python play.py --game ls20 --resume results/ls20_run
"""
import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent


def _find_game_dir(name: str) -> tuple[Path, Path]:
    """Return (game_dir, game_py_path). game_dir is a directory containing
    {name}.py. Some games nest under a version subdir, others do not."""
    root = _PROJECT / "environment_files" / name
    candidates = sorted(root.rglob(f"{name}.py"))
    if not candidates:
        raise FileNotFoundError(f"no {name}.py under {root}")
    py = candidates[0]
    return py.parent, py


def _load_game_class(name: str, py_path: Path):
    """Import the game module by path and return the ARCBaseGame subclass."""
    from arcengine import ARCBaseGame
    spec = importlib.util.spec_from_file_location(name, py_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if (isinstance(obj, type)
                and obj is not ARCBaseGame
                and issubclass(obj, ARCBaseGame)):
            return obj
    raise RuntimeError(f"no ARCBaseGame subclass in {py_path}")


def _register_pkg_stubs():
    packages = {
        "src": _PROJECT / "src",
        "src.opine_world": (
            _PROJECT / "src/opine_world"
        ),
        "src.opine_world.synth_loop": (
            _PROJECT / "src/opine_world/synth_loop"
        ),
        "src.opine_world.synth_loop.adapters": (
            _PROJECT / "src/opine_world/synth_loop/adapters"
        ),
    }
    for pkg, path in packages.items():
        mod = sys.modules.setdefault(pkg, types.ModuleType(pkg))
        mod.__path__ = [str(path)]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    parent, _, child = name.rpartition(".")
    if parent in sys.modules:
        setattr(sys.modules[parent], child, mod)
    return mod


def _load_prompt(name: str) -> str:
    """Read a prompt file (exact text, no stripping) from the synth_loop prompts dir."""
    return (
        _PROJECT
        / "src/opine_world/synth_loop/prompts"
        / name
    ).read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run synthesis engine on any ARC-AGI-3 game")
    parser.add_argument("--game", type=str, required=True,
                        help="Game name (matches environment_files/<game>/)")
    parser.add_argument("--max-actions", type=int, default=250)
    parser.add_argument("--synthesis-interval", type=int, default=30)
    parser.add_argument(
        "--synthesis-defer-min-moves-after-divergence",
        type=int,
        default=12,
        help=(
            "After a concrete world-model divergence, defer CEGIS until at "
            "least this many real moves have executed since the first "
            "divergence, unless another delayed-CEGIS trigger fires first."
        ),
    )
    parser.add_argument(
        "--synthesis-defer-max-errors",
        type=int,
        default=6,
        help=(
            "After a concrete world-model divergence, run CEGIS once this "
            "many model prediction errors have accumulated, unless another "
            "delayed-CEGIS trigger fires first."
        ),
    )
    parser.add_argument(
        "--synthesis-defer-min-action-plans-after-divergence",
        type=int,
        default=4,
        help=(
            "After a concrete world-model divergence, defer CEGIS until this "
            "many analyzer/planner action batches have drained, unless "
            "another delayed-CEGIS trigger fires first."
        ),
    )
    parser.add_argument("--model", type=str, default="claude-opus-4-7[1m]")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--effort", type=str, default="max",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="Claude Code thinking effort for synthesis "
                             "(default max, preserves prior behavior).")
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Defaults to results/baseline_oop_<game>")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--agentic-consumer-model", type=str,
                        default="claude-opus-4-7")
    parser.add_argument("--agentic-consumer-effort", type=str, default="max",
                        choices=["low", "medium", "high", "xhigh", "max"],
                        help="Claude Code thinking effort for the analyzer "
                             "(default max, preserves prior behavior).")
    parser.add_argument("--agentic-consumer-max-turns", type=int, default=30)
    parser.add_argument(
        "--agentic-consumer-timeout",
        type=int,
        default=int(os.environ.get("ARC_AGENTIC_CONSUMER_TIMEOUT_S", "1800")),
        help=(
            "Analyzer wall-clock timeout in seconds. Defaults to 1800 to "
            "prevent hung Codex/Claude turns from occupying a worker forever; "
            "set 0 to disable deliberately."
        ),
    )
    parser.add_argument("--agentic-consumer-max-retries", type=int, default=5)
    parser.add_argument("--no-sandbox", dest="subprocess_sandbox",
                        action="store_false", default=True)
    parser.add_argument("--debug", action="store_true", default=False,
                        help=("Keep the full operational run output (timestamped "
                              "logs, raw transcripts, checkpoints). By default the "
                              "run is finalized to its clean, shareable form."))
    parser.add_argument("--synth-mode", type=str, default="free",
                        choices=["free", "oop", "factored", "monolithic"],
                        help=("Synthesis template. 'free' (default, no "
                              "structural enforcement; the synth chooses "
                              "classes, free functions, lookup tables, "
                              "whatever fits) is the paper configuration. "
                              "'oop' is per-tag classes. 'factored' enforces "
                              "RDDL-style per-object-type update rules with a "
                              "TYPE_RULES registry and named pairwise "
                              "interaction rules. 'monolithic' enforces a "
                              "single transition_function with no classes and "
                              "no transition-logic helpers, AND strips the "
                              "whole factorization apparatus (object-ontology "
                              "prescriptions, xi audit, fluent contract) as "
                              "one bundle. 'factored' vs 'monolithic' is the "
                              "structural ablation pair."))
    parser.add_argument("--frames-only", dest="frames_only",
                        action="store_true", default=False,
                        help=("Hide all sprite-level perception from both "
                              "the synth and the analyzer. The world model "
                              "operates on raw 64x64 frames (palette indices "
                              "0..15); the synth invents its own object "
                              "structure inside game_engine.py. Leaderboard-"
                              "honest mode."))
    parser.add_argument("--backend", type=str, default="claude",
                        choices=["claude", "codex"],
                        help=("LLM backend for both analyzer and synthesizer. "
                              "Default: claude."))
    parser.add_argument("--codex-model", type=str, default="gpt-5.5",
                        help="Codex model when --backend codex is used.")
    parser.add_argument("--codex-effort", type=str, default="high",
                        help=("Codex model_reasoning_effort when --backend "
                              "codex is used."))
    parser.add_argument("--codex-home", type=str, default=None,
                        help=("Host CODEX_HOME to mount for Codex auth "
                              "(default ~/.codex-arc)."))
    parser.add_argument("--codex-image", type=str, default="codex-agent",
                        help="Docker image for the Codex agent container.")
    parser.add_argument("--codex-network", type=str, default="codex-filtered",
                        help=("Docker network for Codex agent turns; "
                              "codex_backend/gateway_up.sh creates "
                              "codex-filtered."))
    parser.add_argument("--codex-gateway", type=str, default=None,
                        help=("Gateway IP for Codex egress. Defaults to "
                              "codex_backend/gateway_internal_ip.txt."))
    parser.add_argument(
        "--claude-isolation", type=str,
        default=os.environ.get("ARC3_CLAUDE_ISOLATION", "docker"),
        choices=["bwrap", "docker"],
        help=("Filesystem/network isolation for the claude backend. 'docker' "
              "(default): locked-down claude-agent container on the "
              "claude-filtered network behind the egress gateway (run "
              "docker/gateway_up.sh + build.sh first). 'bwrap': local "
              "bubblewrap sandbox, open network. Env: "
              "ARC3_CLAUDE_ISOLATION."))
    parser.add_argument(
        "--claude-image", type=str,
        default=os.environ.get("ARC3_CLAUDE_IMAGE", "claude-agent"),
        help="Docker image for the claude agent container (docker isolation).")
    parser.add_argument(
        "--claude-network", type=str,
        default=os.environ.get("ARC3_CLAUDE_NETWORK", "claude-filtered"),
        help="Docker network for claude turns (docker isolation).")
    parser.add_argument(
        "--claude-gateway", type=str,
        default=os.environ.get("ARC3_CLAUDE_GATEWAY") or None,
        help=("Gateway IP for claude egress. Defaults to the live "
              "claude-gateway container."))
    parser.add_argument("--synth-continue", action="store_true", default=False,
                        help=("Codex-only: resume the same synthesizer Codex "
                              "session across synthesis calls in this live run. "
                              "Ignored by the Claude backend and not preserved "
                              "across snapshot/resume."))
    parser.add_argument("--crystallise", dest="crystallise",
                        action="store_true", default=False,
                        help=("C2: enable crystallisation-as-optimal-"
                              "stopping. No synthesis fires before the "
                              "analyzer's alias hypothesis space "
                              "concentrates on a confident goal-relevant "
                              "non-decorative partition AND reward has "
                              "been observed. At crystallisation, synth "
                              "fires once against the committed partition "
                              "+ observed-changing tags (the scope); "
                              "post-crystallisation synthesis runs in a "
                              "scoped verifier regime."))
    parser.add_argument("--crystallise-alias-min-score", type=int,
                        default=5,
                        help="Top alias score required to commit a tag.")
    parser.add_argument("--crystallise-alias-margin", type=int,
                        default=3,
                        help="Top-vs-next alias score margin required.")
    parser.add_argument("--crystallise-modal-frac", type=float,
                        default=0.95,
                        help=("Per-stratum modal_frac threshold for the "
                              "Prop. 5 identification gate. Set to 0 to "
                              "disable the stratum gate."))
    parser.add_argument("--crystallise-stratum-n-min", type=int,
                        default=3,
                        help="Per-stratum n_min for the identification gate.")
    parser.add_argument("--stop-at-step", type=int, default=None,
                        help=("Run until this absolute step is reached, then "
                              "write a self-contained snapshot at the first "
                              "clean analyzer boundary (post-analyzer decision, "
                              "pre-execution) and stop. Used by the round-robin "
                              "driver for its per-segment move cap."))
    parser.add_argument("--snapshot-dir", type=str, default=None,
                        help=("Directory to write reloadable run snapshots into "
                              "(default <output_dir>/snapshots)."))
    parser.add_argument("--synth-simplify-interval", type=int, default=4,
                        help=("Every N synthesis rounds, prepend a 'compress the "
                              "model' simplification directive (baseline1-style "
                              "anti-overfitting). 0 disables. Default 4."))
    parser.add_argument("--signal", type=str, default="legacy",
                        choices=["legacy", "sigma", "both", "none"],
                        help=("Which epistemic signal the analyzer sees "
                              "(the ablation-arm switch). legacy = the "
                              "epistemic matrix (reproduces prior runs), "
                              "sigma = the per-object substrate, both, or "
                              "none. Sigma is computed in every sprite-mode "
                              "run regardless of visibility."))
    parser.add_argument("--no-goal-grounding",
                        dest="sigma_goal_grounding_enabled",
                        action="store_false", default=True,
                        help=("Disable the goal-requirement grounding cycle "
                              "(one LLM call at level advances and goal "
                              "revisions)."))
    parser.add_argument("--no-fluent-harvest", dest="sigma_fluent_harvest",
                        action="store_false", default=True,
                        help=("Disable harvesting the model's declared "
                              "FLUENTS registry (on by default: absorbs "
                              "admitted refinements into sigma and adds the "
                              "declaration contract to the synthesis "
                              "prompt)."))
    parser.add_argument("--no-model-resolved", dest="sigma_model_resolved",
                        action="store_false", default=True,
                        help=("Disable three-state U_c (on by default: "
                              "enough correct forward predictions resolve a "
                              "mixed cell as a representational confound "
                              "instead of an epistemic unknown)."))
    parser.add_argument("--no-briefing", dest="sigma_briefing",
                        action="store_false", default=True,
                        help=("Disable the event-gated epistemic briefing "
                              "(on by default: injected on new level, "
                              "post-repair, and evidence stall)."))
    parser.add_argument("--ablate-epistemic", dest="ablate_epistemic",
                        action="store_true", default=False,
                        help=("Systems-level ablation: no epistemic machinery "
                              "at all. No sigma/matrix computation, no "
                              "ontology measure or xi ledger, no label audit, "
                              "no goal grounding, and no epistemic prompt "
                              "sections for either agent. Combine with "
                              "--signal none."))
    parser.add_argument("--visibility-only-ablation",
                        dest="visibility_only_ablation",
                        action="store_true", default=False,
                        help=("Allow --signal none without --ablate-epistemic "
                              "(hide the signal from the analyzer while still "
                              "computing everything). Refused otherwise, "
                              "because it silently produces a non-baseline "
                              "ablation arm."))
    parser.add_argument("--warmup-click-all", dest="warmup_click_all",
                        action="store_true", default=False,
                        help=("Re-enable the engine-scripted click-all warmup "
                              "at game start (sprite mode). Off by default so "
                              "action selection is analyzer-driven from step "
                              "0, matching frames-only runs."))
    parser.add_argument("--critique", dest="critique_enabled",
                        action="store_true", default=False,
                        help=("After passing synth rounds, run an adversarial "
                              "generalization critique of game_engine.py on the "
                              "--critique-interval cadence and feed findings "
                              "into later synth/analyzer prompts. Off by "
                              "default."))
    parser.add_argument("--critique-interval", type=int, default=5,
                        help=("With --critique, run the automatic critic every "
                              "N synthesis rounds. 1 restores every-round "
                              "behavior; 0 disables automatic critic calls. "
                              "Default 5."))
    parser.add_argument("--critique-repair",
                        dest="critique_repair_enabled",
                        action="store_true", default=False,
                        help=("With --critique, also run a bounded engine-forced "
                              "synth repair loop after the critic. Normally the "
                              "synth prompt itself decides how to use critique."))
    parser.add_argument("--critique-repair-rounds", type=int, default=1,
                        help=("Maximum in-round synth repair turns after "
                              "critique when --critique-repair is set. "
                              "0 disables forced repair."))
    parser.add_argument("--no-critique-recheck-after-repair",
                        dest="critique_recheck_after_repair",
                        action="store_false", default=True,
                        help=("Skip the final critic re-check after an in-round "
                              "critique repair."))
    parser.add_argument("--animation-analysis",
                        dest="animation_analysis_enabled",
                        action="store_true", default=False,
                        help=("When an action emits intermediate animation frames, "
                              "run a backend-matched animation reviewer and feed "
                              "its findings to synthesis/analyzer prompts. Costs "
                              "extra LLM calls; off by default."))
    parser.add_argument("--animation-analysis-max-events", type=int, default=8,
                        help=("Maximum animated steps to send to the animation "
                              "reviewer per run. 0 disables the cap."))
    parser.add_argument("--animation-analysis-timeout", type=int, default=600,
                        help="Animation reviewer timeout in seconds.")
    parser.add_argument("--no-planner", dest="planner_enabled",
                        action="store_false", default=True,
                        help=("Disable C3 planner/executor. By default the planner "
                              "is enabled but gated until after at least one level "
                              "transition, and until crystallisation when "
                              "--crystallise is used."))
    parser.add_argument("--planner-autonomous", dest="planner_autonomous",
                        action="store_true", default=False,
                        help=("Let the ENGINE autonomously generate and execute "
                              "world-model plans (WorldCoder-style), bypassing the "
                              "actions agent. OFF by default: the planner is a tool "
                              "the analyzer calls on demand via tools/plan.py, and "
                              "the analyzer decides every move."))
    parser.add_argument("--planner-after-levels-completed", type=int, default=1,
                        help="Minimum completed levels before C3 can run.")
    parser.add_argument("--planner-max-depth", type=int, default=0,
                        help="Planner search depth cap. 0 disables the cap.")
    parser.add_argument("--planner-max-nodes", type=int, default=0,
                        help="Generic BFS node cap. 0 disables the cap.")
    parser.add_argument("--planner-timeout", type=int, default=30,
                        help="Planner wall-clock timeout in seconds.")
    parser.add_argument("--planner-retry-interval", type=int, default=10,
                        help="Steps to wait after a no-plan result before retrying.")
    parser.add_argument("--planner-max-click-targets", type=int, default=0,
                        help="ACTION6 click target cap. 0 disables the cap.")
    parser.add_argument("--no-planner-completed-verification",
                        dest="planner_require_completed_verification",
                        action="store_false", default=True,
                        help=("Skip the completed-level planner verification gate. "
                              "Default keeps the baseline1-style gate on."))
    parser.add_argument("--planner-verify-max-levels", type=int, default=0,
                        help=("Completed level starts to verify before live C3 "
                              "use. 0 verifies all completed starts."))
    parser.add_argument("--ablate-natural-language",
                        dest="ablate_natural_language",
                        action="store_true", default=False,
                        help=("Natural-language ablation: every artifact the "
                              "agents persist between invocations must be "
                              "Python. game_engine.py comments and docstrings "
                              "are stripped before verification and before "
                              "carry-forward, and the shared handoff "
                              "documents must parse as Python or they are "
                              "discarded. The dual-agent architecture is left "
                              "intact, so the comparison isolates the NL "
                              "intermediate representation instead of "
                              "confounding it with removing an agent."))
    parser.add_argument("--action-script", type=str, default=None,
                        help=("Fixed-trajectory arm: replay this action "
                              "trajectory instead of running the acting "
                              "agent. Accepts a prior run's "
                              "frames/actions.jsonl (one JSON object per "
                              "line with action_id, plus x/y for ACTION6). "
                              "The acting agent and the planner are removed "
                              "from the loop entirely, so every arm sees an "
                              "identical replay buffer and only synthesis "
                              "differs. Used for the structural ablation."))
    parser.add_argument("--action-script-batch", type=int, default=6,
                        help=("Scripted actions handed to the engine per "
                              "batch. A drained batch is the plan boundary "
                              "the deferred-CEGIS gate counts, so this sets "
                              "the resolution of the synthesis-count "
                              "measurement. Default 6, the median analyzer "
                              "plan length in the g50t reference run."))
    parser.add_argument("--no-competition", dest="competition",
                        action="store_false", default=True,
                        help=("Disable competition mode. Competition mode is ON "
                              "by default (sets ONLY_RESET_LEVELS=true): RESET "
                              "becomes a level-reset, and a level whose per-attempt "
                              "step budget runs out enters game-over and must be "
                              "RESET to retry -- matching the ARC-AGI-3 leaderboard "
                              "scorecard semantics. Disable only for debugging."))
    args = parser.parse_args()

    if args.signal == "none" and not args.ablate_epistemic \
            and not args.visibility_only_ablation:
        parser.error(
            "--signal none only hides the epistemic signal from the analyzer; "
            "sigma, fluent harvest, goal grounding, label audit and the xi "
            "ledger all still run. A baseline arm needs --ablate-epistemic as "
            "well. Pass --visibility-only-ablation if hiding the signal while "
            "computing everything is intentional."
        )

    if args.synth_mode == "factored" and args.frames_only:
        parser.error(
            "--synth-mode factored is a sprite-mode arm: it enforces "
            "per-object-type rules over the extractor's object records, "
            "which frames-only does not provide."
        )
    if args.synth_mode == "monolithic":
        # The factorization apparatus is stripped as one bundle. The fluent
        # registry is the RDDL lifted-predicate layer, so leaving harvest on
        # would keep half the treatment inside the ablated arm.
        args.sigma_fluent_harvest = False

    if args.action_script:
        script_path = Path(args.action_script)
        if not script_path.exists():
            parser.error(f"--action-script not found: {script_path}")
        if args.action_script_batch < 1:
            parser.error("--action-script-batch must be >= 1")
        if args.planner_autonomous:
            parser.error(
                "--action-script with --planner-autonomous lets the planner "
                "inject actions the trajectory does not contain, which "
                "desyncs the arms. Drop one."
            )
        if args.warmup_click_all:
            parser.error(
                "--action-script with --warmup-click-all prepends engine "
                "clicks that are not in the trajectory, which desyncs the "
                "arms. Drop one."
            )
        if args.crystallise:
            parser.error(
                "--action-script with --crystallise gates synthesis on an "
                "analyzer alias partition, but the fixed-trajectory arm has "
                "no analyzer, so synthesis would never fire."
            )
        n_script = sum(
            1 for line in script_path.read_text().splitlines() if line.strip()
        )
        # The trajectory IS the budget. Without this the argparse default
        # silently truncates the arm partway and the run reports itself
        # complete, which is indistinguishable from a full replay in every
        # artifact except the engine's exhaustion marker.
        # Resuming carries the prior run's step counter, and the engine's loop
        # compares that absolute step against max_actions. A 292-action script
        # resumed at step 757 is therefore already "over budget" and replays
        # zero actions while reporting itself complete. Offset the budget by
        # what the resumed run already executed so the script gets its full
        # length either way.
        resume_offset = 0
        if args.resume:
            rp = Path(args.resume)
            acts = (rp.parent if rp.name == "checkpoint.pkl" else rp) / "frames" / "actions.jsonl"
            try:
                with acts.open() as fh:
                    resume_offset = sum(1 for _ in fh)
            except OSError:
                resume_offset = 0
            if resume_offset:
                print(
                    f"[action-script] resuming after {resume_offset} recorded "
                    f"action(s); budget offset accordingly",
                    file=sys.stderr,
                )
        n_script += resume_offset

        explicit_cap = "--max-actions" in sys.argv
        if not explicit_cap:
            if args.max_actions != n_script:
                print(
                    f"[action-script] --max-actions {args.max_actions} -> "
                    f"{n_script} (trajectory length)",
                    file=sys.stderr,
                )
            args.max_actions = n_script
        elif args.max_actions > n_script:
            print(
                f"[action-script] clamping --max-actions "
                f"{args.max_actions} -> {n_script} (trajectory length)",
                file=sys.stderr,
            )
            args.max_actions = n_script
        elif args.max_actions < n_script:
            print(
                f"[action-script] WARNING: --max-actions {args.max_actions} "
                f"is below the {n_script}-action trajectory. This arm will "
                f"be INCOMPLETE and not comparable to a full-trajectory arm.",
                file=sys.stderr,
            )

    if args.competition:
        os.environ["ONLY_RESET_LEVELS"] = "true"

    game_dir, game_py = _find_game_dir(args.game)
    sys.path.insert(0, str(game_dir))

    _register_pkg_stubs()
    _load("src.opine_world.synth_loop.domain_adapter",
          str(_PROJECT / "src/opine_world/synth_loop/domain_adapter.py"))
    _load("src.opine_world.synth_loop.runlog",
          str(_PROJECT / "src/opine_world/synth_loop/runlog.py"))
    _load("src.opine_world.synth_loop.epistemic",
          str(_PROJECT / "src/opine_world/synth_loop/epistemic.py"))
    _load("src.opine_world.synth_loop.ontology",
          str(_PROJECT / "src/opine_world/synth_loop/ontology.py"))
    _load("src.opine_world.synth_loop.sandbox",
          str(_PROJECT / "src/opine_world/synth_loop/sandbox.py"))
    _load("src.opine_world.synth_loop.vision",
          str(_PROJECT / "src/opine_world/synth_loop/vision.py"))
    _load("src.opine_world.synth_loop.click_utils",
          str(_PROJECT / "src/opine_world/synth_loop/click_utils.py"))
    _load("src.opine_world.synth_loop.aliases",
          str(_PROJECT / "src/opine_world/synth_loop/aliases.py"))
    _load("src.opine_world.synth_loop.codex_backend",
          str(_PROJECT / "src/opine_world/synth_loop/codex_backend.py"))
    _load("src.opine_world.synth_loop.agentic_consumer",
          str(_PROJECT / "src/opine_world/synth_loop/agentic_consumer.py"))
    _load("src.opine_world.synth_loop.spriteless_eta",
          str(_PROJECT / "src/opine_world/synth_loop/spriteless_eta.py"))
    _load("src.opine_world.synth_loop.planner",
          str(_PROJECT / "src/opine_world/synth_loop/planner.py"))

    engine_mod = _load(
        "src.opine_world.synth_loop.engine",
        str(_PROJECT / "src/opine_world/synth_loop/engine.py"),
    )
    adapter_mod = _load(
        "src.opine_world.synth_loop.adapters.arcengine_adapter",
        str(_PROJECT / "src/opine_world/synth_loop/adapters/arcengine_adapter.py"),
    )
    SynthesisEngine = engine_mod.SynthesisEngine
    EngineConfig = engine_mod.EngineConfig
    AnalyzerNoPlanError = getattr(engine_mod, "AnalyzerNoPlanError", None)
    ArcEngineEnv = adapter_mod.ArcEngineEnv
    ArcEngineAdapter = adapter_mod.ArcEngineAdapter

    game_cls = _load_game_class(args.game, game_py)
    game = game_cls()

    env = ArcEngineEnv(
        game,
        action_names={1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE",
                      6: "CLICK", 7: "UNDO"},
    )
    adapter = ArcEngineAdapter(
        action_names={1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "SPACE",
                      6: "CLICK", 7: "UNDO"},
    )

    if args.output_dir:
        output_dir = args.output_dir
    elif args.resume:
        rp = Path(args.resume)
        output_dir = str(rp.parent if rp.name == "checkpoint.pkl" else rp)
    else:
        output_dir = f"results/baseline_oop_{args.game}"

    config = EngineConfig(
        max_actions=args.max_actions,
        synthesis_interval=args.synthesis_interval,
        synthesis_defer_min_moves_after_divergence=(
            args.synthesis_defer_min_moves_after_divergence
        ),
        synthesis_defer_max_errors=args.synthesis_defer_max_errors,
        synthesis_defer_min_action_plans_after_divergence=(
            args.synthesis_defer_min_action_plans_after_divergence
        ),
        synthesis_model=args.model,
        synthesis_max_turns=args.max_turns,
        synthesis_effort=args.effort,
        synthesis_timeout=args.timeout,
        seed=args.seed,
        output_dir=output_dir,
        agentic_consumer_model=args.agentic_consumer_model,
        agentic_consumer_effort=args.agentic_consumer_effort,
        agentic_consumer_max_turns=args.agentic_consumer_max_turns,
        agentic_consumer_timeout_s=args.agentic_consumer_timeout,
        agentic_consumer_max_retries=args.agentic_consumer_max_retries,
        subprocess_sandbox=args.subprocess_sandbox,
        debug=args.debug,
        synth_mode=args.synth_mode,
        frames_only=args.frames_only,
        backend=args.backend,
        codex_model=args.codex_model,
        codex_effort=args.codex_effort,
        codex_home=args.codex_home,
        codex_image=args.codex_image,
        codex_network=args.codex_network,
        codex_gateway=args.codex_gateway,
        claude_isolation=args.claude_isolation,
        claude_image=args.claude_image,
        claude_network=args.claude_network,
        claude_gateway=args.claude_gateway,
        synth_continue=args.synth_continue,
        crystallisation_enabled=args.crystallise,
        crystallisation_alias_min_score=args.crystallise_alias_min_score,
        crystallisation_alias_margin=args.crystallise_alias_margin,
        crystallisation_modal_frac_threshold=args.crystallise_modal_frac,
        crystallisation_stratum_n_min=args.crystallise_stratum_n_min,
        stop_and_snapshot_at_step=args.stop_at_step,
        snapshot_dir=args.snapshot_dir,
        synth_simplify_interval=args.synth_simplify_interval,
        critique_enabled=args.critique_enabled,
        critique_interval=args.critique_interval,
        critique_repair_enabled=args.critique_repair_enabled,
        critique_repair_rounds=args.critique_repair_rounds,
        critique_recheck_after_repair=args.critique_recheck_after_repair,
        animation_analysis_enabled=args.animation_analysis_enabled,
        animation_analysis_max_events=args.animation_analysis_max_events,
        animation_analysis_timeout_s=args.animation_analysis_timeout,
        planner_enabled=args.planner_enabled,
        planner_autonomous=args.planner_autonomous,
        planner_after_levels_completed=args.planner_after_levels_completed,
        planner_max_depth=args.planner_max_depth,
        planner_max_nodes=args.planner_max_nodes,
        planner_timeout_s=args.planner_timeout,
        planner_retry_interval=args.planner_retry_interval,
        planner_max_click_targets=args.planner_max_click_targets,
        planner_require_completed_verification=(
            args.planner_require_completed_verification
        ),
        planner_verify_max_levels=args.planner_verify_max_levels,
        epistemic_visible_to_analyzer=args.signal in ("legacy", "both"),
        sigma_visible_to_analyzer=args.signal in ("sigma", "both"),
        sigma_goal_grounding_enabled=args.sigma_goal_grounding_enabled,
        sigma_fluent_harvest=args.sigma_fluent_harvest,
        sigma_model_resolved=args.sigma_model_resolved,
        sigma_briefing=args.sigma_briefing,
        ablate_epistemic=args.ablate_epistemic,
        warmup_click_all=args.warmup_click_all,
        action_script=args.action_script,
        action_script_batch=args.action_script_batch,
        ablate_natural_language=args.ablate_natural_language,
        goal_hint=_load_prompt("engine/goal_hint.md"),
    )

    engine = SynthesisEngine(env=env, adapter=adapter, config=config)
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.is_dir():
            resume_path = resume_path / "checkpoint.pkl"
        if not resume_path.exists():
            raise FileNotFoundError(f"checkpoint not found at {resume_path}")
        engine.load_checkpoint(resume_path)
        print(f"Resumed from checkpoint at {resume_path}")

    try:
        summary = engine.run()
    except (AnalyzerNoPlanError or ()) as exc:
        print(f"\nABORTED (analyzer no-plan): {exc}\n"
              "Run state preserved; resume the checkpoint to retry.",
              file=sys.stderr)
        sys.exit(3)
    print(f"\nFinal: {summary['levels_completed']} levels, "
          f"transition={summary['best_transition_accuracy']:.0%}, "
          f"reward={summary['best_reward_accuracy']:.0%}, "
          f"won={summary['game_won']}")


if __name__ == "__main__":
    main()
