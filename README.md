# OPINE-World

**Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration**
Learning a useful world model from minimal interaction is central to building agents that adapt to unfamiliar tasks. Programmatic world modeling approaches such as WorldCoder quickly learn transition models when given pre-supplied symbolic representations; however, such approaches are insufficient when the symbolic representation and goal are unknown and continually changing, and are intractable under large action spaces due to rigid planning heuristics. Extending the programmatic world modeling approach, OPINE-World additionally maintains an abstracted representation in which provisional objects, incomplete causal explanations, and unresolved interactions can exist in a Bayesian, exploration-centric hypothesis space. This enables learning an unknown, dynamically evolving state, transition, and goal. Exploration and program revision share this evolving abstraction, allowing new evidence to revise both the hypothesized mechanics and their implementation in tandem at test time. ARC-AGI-3 presents this open-world problem as a general-intelligence learning benchmark. Through our OPINE-World model learning paradigm, we improve the Relative Human Action Efficiency score of Claude Opus 4.8 high from 1.5% to 78.40% on the ARC-AGI-3 benchmark.

> **Paper:** <http://arxiv.org/abs/2607.01531>

> **Run artifacts:** <https://drive.google.com/drive/folders/1IiwYWG5jthunJGrM7-EpVmX_UcPty2LV>

> **Blog:** https://david-courtis.github.io/opine-world/

This repository is the open source behind the OPINE-World entry on the ARC Prize Community Leaderboard.

### Setup and Models
This harness was run with Opus 4.8 (high), limited to a 500k context length, and 5 of the 20 games encountered some rate limiting issues that caused premature closure of their scorecards well below the 2000 action limit, while their trajectories were still very promising. Our true score lies somewhere between 92.1 (assuming those failed games scored the average of the others) and 78.4.

We believe that with a stronger model (GPT-5.6 Sol or newer) or a higher effort (xhigh or max effort), will deliver significant improvements on top of this.

Even so, we still consider this preliminary result an impressive score, and hope that our uniqueness in modeling epistemic knowledge, where other current harnesses seem to take a purely worldcoder-adjacent approach, will be useful to the community.

## Methodology

### Model representation

$$
\widehat{\mathcal W} _t=\big((\widehat{\mathcal S} _{\mathrm{NL},t},\widehat T _{\mathrm{NL},t},\widehat R _{\mathrm{NL},t}),\widehat{\mathcal S} _{\mathrm B,t},(\widehat T _{\mathrm P,t},\widehat R _{\mathrm P,t},\widehat P _t,\widehat f _t^{-1})\big). \tag{2}
$$

OPINE-World represents the unknown state, transition, and reward through a unified NL, Bayesian, and executable Python system. At a high level, the representation has five parts:

1. **Known action interface $\mathcal A$.** The agent receives the action space (but not their effects) (Section 2).
2. **Unknown state $\mathcal S$.** $\widehat f^{-1}$ extracts object instances, inferred types, properties, and relations from $\omega$; $\widehat{\mathcal S} _{\mathrm{NL}}$ describes this ontology and $\widehat{\mathcal S} _{\mathrm B}$ quantifies object uncertainty (Sections 3.1.1 and 3.1.2).
3. **Unknown transition $T$.** $\widehat T _{\mathrm{NL}}$ encodes qualitative and partially specified mechanics, while a parallel representation $\widehat T _{\mathrm P}$ encodes a programmatic transition model (Section 3.1.3).
4. **Unknown reward $R$.** $\widehat R _{\mathrm{NL}}$ encodes candidate goals and unresolved hypotheses, while $\widehat R _{\mathrm P}$ implements a programmatic reward model (Section 3.1.4).
5. **Planning $\widehat P$.** The planner rolls $(\widehat T _{\mathrm P},\widehat R _{\mathrm P})$ forward to pursue reward, or uses $\widehat T _{\mathrm P}$ with $\widehat{\mathcal S} _{\mathrm B}$ and the NL model for exploration (Section 3.2.2).

### Model learning loop

1. **Observe.** Receive $(\omega _{t+1},r _t)$ after executing $a _t$ (Section 3.2.2).
2. **Abstract and store.** Append the transition to replay buffer $\mathcal D _{t+1}$, apply $\widehat f _t^{-1}$, associate objects across $\omega _t$ and $\omega _{t+1}$, and update $\widehat{\mathcal S} _{\mathrm B,t+1}$ (Sections 3.1.2 and 3.2.3).
3. **Counterexample-guided inductive synthesis (CEGIS).** Compare the prediction with the observed truth. Treat each disagreement as a counterexample for the next candidate program and synthesize a revised $(\widehat T _{\mathrm P},\widehat R _{\mathrm P},\widehat P,\widehat f^{-1})$ under sufficient batches (Section 3.2.1) (Solar-Lezama et al., 2006; Jha et al., 2010; Alur et al., 2013).
4. **Choose exploitation or exploration.** For a consistent model, use $\widehat P(\mathcal S;\widehat T _{\mathrm P},\widehat R _{\mathrm P})$ to pursue a goal or $\widehat P(\mathcal S;\widehat T _{\mathrm P})$, informed by $(\widehat{\mathcal S} _{\mathrm{NL}},\widehat T _{\mathrm{NL}},\widehat R _{\mathrm{NL}})$ and $\widehat{\mathcal S} _{\mathrm B}$, to explore (Sections 3.2.2 and 3.2.3).
5. **Execute the next actions**, producing a new observation $\omega$ (Section 3.2.2).

**Algorithm 1** OPINE-World online learning loop (schematic)

> 1: **Input:** known action interface $\mathcal A$, action budget $B$<br>
> 2: $\omega\leftarrow\mathrm{Env.Reset}()$; &emsp; $\mathcal D\leftarrow\varnothing$<br>
> 3: $\mathrm{NL},\widehat{\mathcal S} _{\mathrm B},\widehat T _{\mathrm P},\widehat R _{\mathrm P},\widehat P,\widehat f^{-1}\leftarrow\mathrm{Initialize}(\omega,\mathcal A)$<br>
> 4: **while** budget remains and the game is unfinished **do**<br>
> 5: &emsp; **if** $\mathrm{RepairGate}(\mathcal D)$ and $\neg\Phi _1(\widehat f^{-1},\widehat T _{\mathrm P},\widehat R _{\mathrm P};\mathcal D)$ **then**<br>
> 6: &emsp;&emsp; **repeat**<br>
> 7: &emsp;&emsp;&emsp; $\mathrm{NL},\widehat T _{\mathrm P},\widehat R _{\mathrm P},\widehat P,\widehat f^{-1}\leftarrow\mathrm{Synthesize}(\mathcal D,\mathrm{NL},\widehat{\mathcal S} _{\mathrm B})$<br>
> 8: &emsp;&emsp; **until** $\Phi _1(\widehat f^{-1},\widehat T _{\mathrm P},\widehat R _{\mathrm P};\mathcal D)$<br>
> 9: &emsp; **end if**<br>
> 10: &emsp; Reconstruct $\widehat s$ from $\omega$ and recorded history using $\widehat f^{-1}$.<br>
> 11: &emsp; $c\leftarrow\Phi _1(\widehat f^{-1},\widehat T _{\mathrm P},\widehat R _{\mathrm P};\mathcal D)$<br>
> 12: &emsp; **if** $c$ and $\mathrm{ExploitGoal?}(\widehat s,\mathrm{NL},\widehat{\mathcal S} _{\mathrm B})$ **then**<br>
> 13: &emsp;&emsp; $\pi\leftarrow\widehat P(\widehat s;\widehat T _{\mathrm P},\widehat R _{\mathrm P})$<br>
> 14: &emsp; **else**<br>
> 15: &emsp;&emsp; $\pi\leftarrow\mathrm{EpistemicExperiment}(\widehat s,\mathrm{NL},\widehat{\mathcal S} _{\mathrm B})$<br>
> 16: &emsp; **end if**<br>
> 17: &emsp; **for** $a\in\mathrm{ShortPrefix}(\pi)$, within the remaining action budget **do**<br>
> 18: &emsp;&emsp; $(\omega',r)\leftarrow\mathrm{Env.Step}(a)$<br>
> 19: &emsp;&emsp; $\mathcal D\leftarrow\mathcal D\cup\lbrace(\omega,a,r,\omega')\rbrace$<br>
> 20: &emsp;&emsp; $\mathrm{NL},\widehat{\mathcal S} _{\mathrm B}\leftarrow\mathrm{Update}(\mathcal D,\mathrm{NL},\widehat{\mathcal S} _{\mathrm B})$<br>
> 21: &emsp;&emsp; $m\leftarrow\neg\mathrm{LiveCheck} _{\Phi _1}(\omega,a,r,\omega')$<br>
> 22: &emsp;&emsp; $\omega\leftarrow\omega'$<br>
> 23: &emsp;&emsp; **if** $m$ **then**<br>
> 24: &emsp;&emsp;&emsp; discard the unexecuted suffix of $\pi$; **break**<br>
> 25: &emsp;&emsp; **end if**<br>
> 26: &emsp; **end for**<br>
> 27: **end while**

The full method and formalization are in the paper and its appendix.

## Evaluation integrity

Results were produced solely from agent interaction. No domain-specific information from the ARC-AGI-3 source is exposed to the agents.

The agents are given only the general structure of the benchmark and the task to solve: the raw rendered frames (64x64 grids of color indices), the set of available action ids for the current game, and the level-advance reward signal. Although this codebase can run in a mode that exposes sprite level object information through game engine source access, that mode was not enabled for the reported results. Ablation studies rely on this mode for ablation and code structure testing, but this was not enabled in the official ARC-3 leaderboard submission.

The agents run filesystem and network-confined within a docker env.

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

The ARC-AGI-3 games are not bundled in this repo. They are fetched from ARC on demand into a local, git-ignored `environment_files/` directory, as described below. The agents never read game source.

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

- **bubblewrap**: a local filesystem sandbox with open network; pass `--claude-isolation bwrap`. Build the read-only Python toolbox it exposes to the agents once: `bash scripts/setup_agent_pkgs.sh`. Not reccomended.

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

Run `uv run python play.py --help` for the full set of model, effort, sandbox, planner, synthesis-cadence, and epistemic-layer flags.

### Output

By default a run is finalized into a shareable form under `results/<game>/`.

Pass `--debug` to keep the full operational output (timestamped `engine.log`, raw transcripts, checkpoints, and snapshots), e.g. `./run.sh ft09 --debug`. This will use a few GB of space.

## Reproducing the paper results

The reported runs used Claude Opus 4.8 (`claude-opus-4-8[1m]`) at `high` reasoning effort for both agents. Each game was played once, online, in Competition Mode under the live action budget. The agents are general pretrained LLMs acting against a live environment, so runs are stochastic. Re-running a game does not reproduce an action count exactly, but it reproduces the method.

The complete per-game run artifacts (final synthesized world models, full transition traces, natural-language world models, frames, and replays) are published as a separate archive:

> **Run artifacts archive:** <https://drive.google.com/drive/folders/1IiwYWG5jthunJGrM7-EpVmX_UcPty2LV>

### Cost and quota

The paper's sweep ran across four Claude Max accounts at $200 per account per month, about $800 in total, to stay within per-account weekly quotas while playing all 25 games in parallel. As a rough guide, a single Max 20x account can complete 4 to 8 games within its weekly quota. Using the agents through the API directly is possible but more expensive.

## Citation

```
@misc{courtis2026opineworldprogrammaticworldmodeling,
      title={OPINE-World: Programmatic World Modeling with Ontology-error-Prioritized Interactive Exploration for ARC-AGI-3}, 
      author={David Courtis and Wenhao Li and Scott Sanner},
      year={2026},
      eprint={2607.01531},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2607.01531}, 
}
```
