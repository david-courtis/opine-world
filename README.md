# OPINE-World

**Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration**

OPINE-World is an LLM agent that learns an object-centric programmatic world model online from interaction alone. It couples two cooperating LLM agents in a loop of hypothesis and test. One agent acts in the environment and the other synthesizes the model as Python code with adversarial hypothesis nudges. The loop uses exact-replay verification and model-based planning, and it steers exploration with a Bayesian measure of object-type adequacy called the ontology error.

On ARC-AGI-3, a benchmark for skill-acquisition efficiency in which the object vocabulary, the goal, and the action semantics are withheld, OPINE-World solves 20 of 25 games and 160 of 183 levels with no per-game training. It reaches an action-efficiency score of 78.4 against the human baseline.

> **Paper:** <http://arxiv.org/abs/2607.01531>

> **Run artifacts archive:** <https://drive.google.com/drive/folders/1IiwYWG5jthunJGrM7-EpVmX_UcPty2LV>

This repository is the open source behind the OPINE-World entry on the ARC Prize Community Leaderboard.

## How it works

OPINE-World runs one loop over a growing replay buffer of observed transitions.

1. **Act.** A goal-directed agent plays the live game. It reads the interaction log with ordinary file tools and chooses the next action or short sequence.
2. **Synthesize.** A seperate, CEGIS style world-model agent rewrites a single Python file, `game_engine.py`, that stubs `transition_function(state, action)` and `reward_function(state, action, new_state)`. In the spriteless regime we used for ARC-3 it also exposes an `extract_objects(frame)` perception function. The model is factored by object type, so a wrong rule is repaired one type at a time. This agent is freshly initialized at every synthesis cycle, triggered by counterexamples above a configurable threshold, or manually, with instructions, by the goal-directed agent.
3. **Verify.** A candidate model is admitted only when it reproduces every recorded transition exactly and passes an adversarial analysis by a subagent. This is counterexample-guided synthesis. The adversarial subagent is used for hypothesis invalidation, and generalization against the observed buffer, with priority towards counterexample-firing transitions.
4. **Plan.** Once a model is admitted and a level has been cleared, a synthesized planner searches the verified model for a route to the goal. This planner is constantly refined by both the engine and the goal agent. If the planner is not capable of planning properly, as was observed in more complicated domains on the edge of markovianity, or in cases where a load-bearing hypothesis needed more observations before a repair could be initiated, the goal-directed agent may propose a custom plan. Each planned step is validated against the live game, and any mismatch becomes a raised counterexample in the next cycle.

A cheap Bayesian diagnostic, the ontology error `η`, runs alongside the loop. It scores how well the current object types explain the observed effects and steers the agents toward what they do not yet explain. Correctness is decided by the exact-replay verifier, not by `η`.

The full method and formalization are in the paper and its appendix.

## Evaluation integrity

Every result was produced from pure agent interaction. No domain-specific information from the ARC-AGI-3 source is exposed to the agents. 

The agents are given only the general structure of the benchmark and the task to solve: the raw rendered frames (64x64 grids of color indices), the set of available action ids for the current game, and the sparse level-advance reward signal. Although this codebase can run in a mode that exposes sprite level object information through game engine source access, that mode was not enabled for the reported results. Future work will rely on this mode for ablation and object-centric results testing, but this was not enabled in the official ARC-3 leaderboard submission. 

They are never given, and cannot read, any domain-specific knowledge: the game's source code, object or sprite identities and types, the goal, the action semantics, or any human-authored solution or hint. The object vocabulary, the goal, and each action's effect are all inferred by acting in the environment.

The action and synthesis agents run filesystem and network-confined within a docker env, so the directories holding the ground-truth game source are never mounted, and network egress is blocked or allowlisted.

## Repository layout

```
run.sh                           # run one game with the paper configuration
play.py                          # single-game runner (entry point)
src/opine_world/synth_loop/      # the engine, agents, verifier, planner, diagnostics
scripts/download_cloud_games.py  # fetch the ARC-AGI-3 public games into environment_files/
scripts/setup_agent_pkgs.sh      # build the read-only numpy toolbox the bwrap sandbox exposes
docker/                          # locked-down Docker and filtered-network sandbox for the agents
vendor/arc                       # submodule: official arcprize/ARC-AGI-3-Agents (reference)
```

The ARC-AGI-3 games are not bundled in this repo. They are fetched from ARC on demand into a local, git-ignored `environment_files/` directory, as described below. The agents never read game source. Dynamics are inferred from observation alone.

`vendor/arc` pins the official ARC-AGI-3 agents and environment harness for reference. The runtime dependency is the `arc-agi` package on PyPI, declared in `pyproject.toml`.

## Install

Requires Python 3.12 or newer and [`uv`](https://docs.astral.sh/uv/). The agents run the local `claude` CLI ([Claude Code](https://claude.com/claude-code)).

```bash
git clone --recurse-submodules https://github.com/david-courtis/opine-world.git
cd opine-world
uv sync
```

The commands below use `uv run` so they execute in this project's environment. A bare `python` would use whichever virtualenv happens to be active.

### Fetch the games

The 25 public ARC-AGI-3 games are fetched from ARC, not bundled. Set `ARC_API_KEY` (see `.env.example`), then download them into a local, git-ignored `environment_files/`:

```bash
uv run python scripts/download_cloud_games.py            # all 25
uv run python scripts/download_cloud_games.py --games ar25 ft09   # a subset
```

### Sandbox (recommended)

The action and synthesis agents are filesystem-confined so the ground-truth game source is never readable. Docker on a filtered network is the default; bubblewrap, a local filesystem sandbox, is also supported.

- **Docker with a filtered network (default)**: `docker/` builds a `claude-agent` container on a network whose only egress is an allowlisting gateway. Run `docker/gateway_up.sh` first. The reported results were run in this mode.

- **bubblewrap**: a local filesystem sandbox with open network; pass `--claude-isolation bwrap`. Build the read-only Python toolbox it exposes to the agents once: `bash scripts/setup_agent_pkgs.sh`.

## Run a game

The quickest path is `run.sh`. It launches a game with the configuration used for the paper: Claude Opus 4.8 for both agents, the critic, the deferred-CEGIS gate, and the planner settings. Any flag can be overridden by appending it.

```bash
./run.sh ar25                        # one game, paper configuration
./run.sh m0r0 --max-actions 3000     # override any flag
./run.sh ft09 --claude-isolation bwrap
```

Or call the runner directly:

```bash
uv run python play.py --game ar25
uv run python play.py --game ls20 --resume results/ls20_run
```

Run `uv run python play.py --help` for the full set of model, effort, sandbox, planner, and synthesis-cadence flags.

### Output

By default a run is finalized into a shareable form under `results/<game>/`.

Pass `--debug` to keep the full operational output (timestamped `engine.log`, raw transcripts, checkpoints, and snapshots), e.g. `./run.sh ft09 --debug`. This will use a few GB of space.

## Reproducing the paper results

The reported runs used Claude Opus 4.8 (`claude-opus-4-8[1m]`) at `high` reasoning effort for both agents. Each game was played once, online, in Competition Mode under the live action budget. The agents are general pretrained LLMs acting against a live environment, so runs are stochastic. Re-running a game does not reproduce an action count exactly, but it reproduces the method.

The complete per-game run artifacts (final synthesized world models, full transition traces, natural-language world models, frames, and replays) are published as a separate archive:

> **Run artifacts archive:** <https://drive.google.com/drive/folders/1IiwYWG5jthunJGrM7-EpVmX_UcPty2LV>

### Cost and quota

The paper's sweep ran across four Claude Max accounts at $200 per account per month, about $800 in total, to stay within per-account weekly quotas while playing all 25 games in parallel. As a rough guide, a single Max 20x account can complete on the order of 4 to 8 games within its weekly quota. Using the agents through the API directly is possible but more expensive.

## Citation

```
@misc{courtis2026opineworldprogrammaticworldmodeling,
      title={OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration}, 
      author={David Courtis and Wenhao Li and Scott Sanner},
      year={2026},
      eprint={2607.01531},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.01531}, 
}
```
