You are building a crystallised, scoped object-centric world model for an ARC-AGI-3 game. Workspace: %%WORKSPACE_DIR%%

FILES:
- context.txt: World-model doc + observed transitions. READ FIRST.
- game_engine.py: YOUR CODE. Edit this file.
- test_runner.py: Run: python %%TEST_RUNNER_PATH%%
- replay_buffer.pkl: Ground truth (don't modify).
- ontology_error.json: η matrix + ξ-refinement CANDIDATE LEDGER. Read it to
  decide whether to commit any new ContextFeatures (see below).
- animation_events.jsonl / animation_analysis.md: Optional intermediate
  animation evidence. Predict the final settled state, but use tick frames
  to infer movement order, interactions, and timing.
- last_critique.md: Optional independent critique from a prior passing
  model. The engine links/injects it only on configured critique-cadence
  rounds; respond in critique_response.md only when that mandatory critique
  section is present.
- synth_learnings.md: Write/update this concise handoff for the exploration
  agent.
- world_model.md: Shared textual world model, read and updated by both
  synthesis and exploration.
- level_N_reasoning_log.md: Shared per-level hypothesis / mismatch /
  correction log. Update the current level's file when your model changes.
- level_N_report.md: Shared per-level completion report. Fill it after a
  level is solved or when synthesis has a final explanation for that level.
- shared_model_updates.md: Engine-written summary of recent shared-doc edits.

The agent has crystallised: the analyzer has committed a partition
over goal-relevant non-decorative sprite types. The synthesised model
is SCOPED to that partition plus any sprite type observed changing in
the buffer (the dynamics safety net). Out-of-scope sprites are NOT
checked by the verifier; your transition_function should return them
unchanged.

COMMITTED PARTITION (analyzer-assigned goal-relevant roles):
%%LABELLED_LINES%%

OBSERVED-CHANGING UNLABELLED TAGS (you must still model them; assign
an internal role name as you see fit):
%%EXTRA_LINES%%

IN-SCOPE TAG COUNT: %%SCOPE_COUNT%%. The verifier ignores any sprite
whose tag is not in this set, so you do not need to model walls,
scenery, decorations, or other sprites the analyzer judged irrelevant.

TASK: Implement transition_function, reward_function, and a conservative
planner(state, available_actions=None, max_depth=None) -> list[action] | None.
Run tests, iterate until ALL TESTS PASSED.

ANTI-LOCK-IN / REPAIR DISCIPLINE:
You may inherit an existing `game_engine.py`, world-model notes, or critique
from a prior synthesis round. Treat that model as a suspect hypothesis, not as
authority. You were called because the current mechanics formalization is still
incomplete, overfit, or possibly wrong at the abstraction level. It may be
largely flawed or wrongly factored. Incrementally build on it only when doing so
keeps the model simple and general; prefer rewriting major components when that
de-janks the world model, removes special cases, or gives a cleaner object/state
factorization.

Actively compare competing hypotheses before locking in a mechanic. Read
`ontology_error.json` and the xi candidate ledger when present: use the
eta/confounding report to ask whether the model is missing a state variable,
object split, relation, layer, context feature, or action-conditioned effect.
Do not blindly add complexity; choose the factorization that best explains the
replay buffer and generalizes to unseen layouts. Do not launch your own
general-purpose critic or adversarial subagent during ordinary synthesis.
Independent critique is scheduled by the engine on its configured cadence; on
off-cadence rounds, repair from the replay buffer and model notes without
reading, answering, or launching critique unless the engine has injected a
mandatory critique section.

Log competing hypotheses, rejected alternatives, remaining uncertainties, and
high-value probes in `synth_learnings.md`. If the engine injects a mandatory
critique section, use `critique_response.md` to say what you fixed, what you
rejected with evidence, and what still needs new game observations.

SHARED WORLD-MODEL DOCUMENTS:
Maintain `world_model.md` as one evolving model across the game, not a parallel
model per level. It must include: Mechanics of the Game with an explicit visual
ontology; Target of the Game; How the player is expected to infer the target;
Ad Hoc Elements Inventory; and Newly Introduced But Unexplained Elements.
Whenever you see a new level, mismatch, death/reset-like event, unresolved pixel
detail, temporary cache/mask use, or level-specific branch, update these lists
concretely. Prefer entries that cite the level/step/region and the competing
hypotheses.

Maintain the current `level_N_reasoning_log.md` with tested hypotheses,
supporting/rejecting evidence, mismatches, and corrections. Maintain
`level_N_report.md` after a level is solved or when the model for that level is
stable enough to summarize. If you change any shared model document, summarize
the operational change in `synth_learnings.md` so the exploration agent sees it
explicitly in its handoff.

STATE RECONSTRUCTION PRINCIPLES:
Do not solve later states by writing arbitrary checkpoint reconstruction
functions, exact-state lookup tables, or per-step replay patches. A non-terminal
state should be represented by the committed object ontology plus the observed
changing safety-net tags: persistent geometry/structures, object families, and
current dynamic fields. The transition rule should advance that state by the
action. For the uncomputable level-entry / RESET cases, `l<N>_initial.pkl` caches
may seed the entry state inside `transition_function`, but they are not a
general state-reconstruction escape hatch and must never be used by
`reward_function`.

Keep one shared ontology across levels whenever the visual evidence supports it:
use the analyzer's committed roles as semantic guidance, reuse object families
for similar motifs, prefer one shared classification strategy over separate
per-level detectors, and express new level behavior as per-level parameters of
known rules before inventing new object families or latent variables. If a later
level adds partial visibility, sliding, layering, or hidden state, extend the
observation/state representation while preserving the underlying object family
unless the buffer forces a genuinely new mechanic.

Before introducing any new level-specific state variable, ask whether the
phenomenon is already a known object family, a known dynamic field, an
observation/visibility/layering effect, or a parameterization of an existing
rule. Only add a new latent variable or object family when the previous ontology
cannot explain the observations.

VERIFICATION-ONLY FALLBACKS AND MODELING DEBT:
The closest analogue to baseline1's temporary renderer override is our narrow
level-entry cache allowance: use `l<N>_initial.pkl` only for level-advance/RESET
transition states that cannot be derived from current state alone. Do not hide
game logic, planning logic, reward predicates, or ordinary in-level transitions
behind cache reads or per-level branches. Any frame-local special case, cache
dependence, unexplained state field, or duplicated object family is evidence that
the model is still missing a mechanic, object identity, latent state variable, or
observation rule. List these debts concretely in `synth_learnings.md` and remove
them once a clean mechanic explains the behavior.

Objects have: name, tags, x, y, w, h, display_x, display_y, display_w,
display_h, visible, collidable, layer, rotation, pixels. %%CLASSES_LINE%%

CRITICAL RULES:
1. transition_function returns a list of object dicts with the SAME
   SCHEMA as the input. Sprites whose tag is OUT of scope must pass
   through unchanged. Sprites whose tag is IN scope must have their
   correct post-action state predicted.
2. reward_function must hypothesise the goal condition over the
   in-scope sprites' configuration.
3. NO REPLAY-BUFFER LOOKUP. transition_function and reward_function
   must NEVER read replay_buffer.pkl. Per-level entry caches
   (l<N>_initial.pkl) MAY be read by transition_function to model the
   un-computable level-advance / RESET transitions. reward_function
   MUST NOT reference any l<N>_initial cache; that treats the env's
   level-advance signal as the goal predicate instead of synthesising
   one. The test runner statically rejects this pattern.
   If the replay buffer already contains a level-advance into level N
   but `l<N>_initial.pkl` is absent, treat that as an infrastructure
   defect to report in `synth_learnings.md`, not as a legitimate
   accuracy ceiling. Do not normalize or accept a failed observed
   level-entry transition because a cache file is absent.
4. EVERY TAG IN THE SCOPE LIST ABOVE must be modelled. Use the
   analyzer-assigned roles as semantic class names (e.g.
   `class Cursor:` for a tag whose role is "cursor"). For observed-
   changing tags without an assigned role, pick an internal name.
5. planner searches through your own transition_function +
   reward_function and returns a reward-reaching action list only when
   found; otherwise return None. The engine executes it one step at a
   time and aborts on any model/environment mismatch. Make this an explicit,
   budgeted, goal-directed search, not a naive uninformed BFS whose high
   branching factor will not finish in time on ARC-3.
6. Write/update `synth_learnings.md` with short bullets for the
   exploration agent: known mechanics, uncertain hypotheses, high-value
   probes, and avoid-repeat failures. Include an ad-hoc/debt inventory:
   unresolved visual details, temporary cache use, level-specific branches,
   duplicate object families, and competing hypotheses that need future
   probes. This text is injected into the analyzer prompt.
7. If context includes a mandatory critique section, revise the model
   where the critique is valid and write `critique_response.md` with
   Applied / Rejected / Remaining sections. Reject findings only with
   replay-buffer evidence.
8. If animation events or animation analysis are present, inspect them
   before editing dynamics. You do not need to reproduce intermediate
   frames, but their sequence can reveal the actual mechanic. For ANY
   transition you cannot yet explain, READ EVERY intermediate frame for that
   step -- the `tick_frames` paths in `animation_events.jsonl`
   (`step_<NNNN>_tick_<KK>.png`). Do NOT skip frames or discount any as
   unimportant: the mechanic (movement/collision-resolution order, a one-tick
   flash, a mid-animation trigger or counter) can live in a single frame.
   ESPECIALLY when a step's net diff is `Nothing changed` (a no-op before and
   after) yet it carries intermediate ticks: the mechanic lives ENTIRELY in
   those intermediate frames -- the action did something mid-animation that
   reverted by the settled frame, so that is exactly where the information-dense
   evidence is; read every tick. If a step has more than 20 intermediate frames,
   hand them to a Task subagent to read them all in order and report back,
   rather than skipping any.

The reward condition observed in the buffer involves only in-scope
sprites, which is exactly why the partition was committed. Use the
analyzer's role labels as semantic guidance; verify exact mechanics
from the replay buffer.

================================================================
ξ-REFINEMENT AUDIT (optional but recommended this round)
================================================================
The engine measures ontology error η = how confounded the current
factorisation is. The matrix and the worst strata live in
`ontology_error.json`. Under `latest.xi_candidate_ledger` you will
find a mechanically-enumerated list of ContextFeature candidates
whose acceptance would reduce η on the top-K worst strata. Each
candidate carries: feature descriptor, η_old, η_new, η_reduction,
n_substrata, identified_substrata, accepted.

Acceptance criterion (verifier-applied):
  η_reduction ≥ 0.05  AND  identified_substrata ≥ 1

If you read the ledger and decide that one or more of the ACCEPTED
candidates names a genuine conditioning feature (i.e., the feature
captures something your transition_function would NEED to read to
predict the dynamics correctly), commit those features by writing
``xi_updates.json`` to this workspace. Schema:

  {
    "add": [
      {"kind": "target_field", "field": "rotation"},
      {"kind": "neighbour_at_offset", "dx": 1, "dy": 0}
    ],
    "rationale": "<one sentence on why these features matter>"
  }

Known ``kind`` values and required parameters:
  target_field            field: str  (sprite-own attribute, e.g.
                                       rotation, visible, scale,
                                       mirror_ud, mirror_lr,
                                       pixels_hash)
  neighbour_at_offset     dx: int, dy: int  (relative grid offset
                                              within {-1,0,1}^2)
  neighbourhood_radius    r: int  (widen the base 3×3 neighbour
                                    window to (2r+1)×(2r+1))
  click_offset            (no params; for action 6 only)

Features are MONOTONICALLY ACCUMULATED; once committed, a feature
remains in the operative ξ for the rest of the run. Do not add a
feature unless you would actually use it in your code; spurious
commitments make subsequent η measurements noisier without
benefit. If no candidate looks meaningful, do not write the file.

ENGINE-SIDE VERIFIER: every feature you write to xi_updates.json is
RE-SCORED by the engine after you exit. A proposal is committed iff
it produces Δη ≥ 0.05 with ≥ 1 identified sub-stratum on at least
one of the current top-3 worst strata, the same rule the ledger
uses for its `accepted: true/false` verdicts. Features that look
plausible to you but don't survive verification will be silently
rejected and logged. Practical implication: pick features that
appear with `accepted: true` in the ledger, AND that you can
articulate a mechanic-level reason for. Reading
``xi_updates_applied.json`` AFTER you finish (or on the next round
via the prior workspace) shows the per-feature verdict.

Your judgement is what selects WHICH accepted candidates to commit;
the verifier guarantees no commitment can REGRESS η. When in doubt,
do not commit; the next CEGIS round will give you another ledger.

START: read context.txt, run tests, implement transition_function +
reward_function + planner, iterate.
