You are building a world model for an ARC-AGI-3 game. Workspace: %%WORKSPACE_DIR%%

FILES:
- context.txt: Frame-level transition observations. READ FIRST.
- game_engine.py: YOUR CODE. Edit this file. 
- test_runner.py: Run: python %%TEST_RUNNER_PATH%%
- replay_buffer.pkl: Ground truth (don't modify).
- ontology_error.json: Optional previous spriteless ETA report, when
  available. Read it if present to see which induced object types/actions
  remain confounded.
- spriteless_object_abstraction.json: Optional summary of your previous
  `extract_objects(frame)` output, when available.
- animation_events.jsonl / animation_analysis.md: Optional intermediate
  animation evidence. The final settled frame is the verifier target, but
  tick frames often reveal the rule ordering.
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

TASK: Implement
    transition_function(frame, action_id) -> next_frame
    reward_function(frame, action_id, next_frame) -> (reward: float, done: bool)
    planner(frame, available_actions=None, max_depth=None) -> list[action] | None
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
`ontology_error.json` and `spriteless_object_abstraction.json` when present:
use the eta/confounding report to ask whether the model is missing a state
variable, object split, relation, layer, context feature, or action-conditioned
effect. Do not blindly add complexity; choose the factorization that best
explains the replay buffer and generalizes to unseen layouts. Do not launch
your own general-purpose critic or adversarial subagent during ordinary
synthesis. Independent critique is scheduled by the engine on its configured
cadence; on off-cadence rounds, repair from the replay buffer and model notes
without reading, answering, or launching critique unless the engine has injected
a mandatory critique section.

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

STATE / RENDERING RECONSTRUCTION PRINCIPLES:
Do not solve later states by writing arbitrary checkpoint reconstructors, frame
lookup tables, or per-step replay patches. In this verifier each
`transition_function` call receives the current observed frame; your internal
state should be reconstructed from that frame through one shared visual ontology
(`extract_objects`, Sprite hypotheses, persistent geometry, dynamic fields), and
then advanced by the action rule. For the uncomputable level-entry / RESET
cases, `l<N>_initial.pkl` caches may seed the next level's entry frame inside
`transition_function`, but they are not a general state-reconstruction escape
hatch and must never be used by `reward_function`.

Keep one shared ontology across levels whenever the visual evidence supports it:
use the same object families for similar motifs, prefer one shared
extraction/classification strategy over separate per-level detectors, and express
new level behavior as per-level parameters of known rules before inventing new
object families or latent variables. If a later level adds partial visibility,
sliding, layering, or hidden state, extend the observation/state representation
while preserving the underlying object family unless the buffer forces a genuinely
new mechanic.

Before introducing any new level-specific state variable, ask whether the
phenomenon is already a known object family, a known dynamic field, an
observation/visibility/layering effect, or a parameterization of an existing
rule. Only add a new latent variable or object family when the previous ontology
cannot explain the observations.

VERIFICATION-ONLY FALLBACKS AND MODELING DEBT:
This framework does not provide a general `apply_render_overrides` hook. The
narrow fallbacks are `l<N>_initial.pkl` for level-entry/RESET transitions and
`move_counter_mask()` for one thin HUD timer/counter strip. Use them only in
their allowed roles. Any other frame-local special case, cache dependence,
counter mask, or unresolved pixel correction is evidence that the model is still
missing a real mechanic, object identity, latent state variable, or observation
rule. List these debts concretely in `synth_learnings.md` and remove them once a
clean mechanic explains the pixels.

DELIVERABLE CHECKLIST:
- Keep `transition_function` and `reward_function` passing the verifier.
- Keep `planner` conservative. It should search through your own
  transition/reward model and return a reward-reaching action list only
  when one is found; otherwise return None. The engine will not execute
  it until after a real level completion and completed-level verification.
  Make this an explicit, budgeted, goal-directed search over your
  `extract_objects` abstraction, not a naive uninformed BFS over raw frames
  whose high branching factor will not finish in time on ARC-3.
- Maintain `extract_objects(frame) -> list[dict]` as a lightweight summary
  of the visual objects your model recognizes. This is not another
  verifier target and should not trigger extra complexity; reuse the
  object parsing / Sprite hypotheses you already need for the transition
  model. The engine uses it after your turn to build the spriteless ETA
  matrix.
- Each object dict should include stable `name`, semantic `type`, `x`,
  `y`, `w`, `h`, optional `pixels`, optional `layer`, optional `mask`
  or `alpha_mask`, and optional `tags`. Names should stay stable for
  the same hypothesized object across adjacent frames when possible;
  `type` should name the object's mechanical role or visual family
  (`player`, `button`, `gate`, `counter`, `target`, `wall`, etc.).
  For transparent or layered sprites, do not rely on the flattened
  rendered crop: return sprite-local `pixels` with `-1` transparent
  cells and a meaningful `layer`/mask when your model has inferred one.
- Write/update `synth_learnings.md` with short bullets for the exploration
  agent: known mechanics, uncertain hypotheses, high-value probes, and
  avoid-repeat failures. Include an ad-hoc/debt inventory: unresolved pixels,
  temporary cache/mask use, level-specific branches, duplicate object families,
  and competing hypotheses that need future probes. This is injected into the
  analyzer prompt, so make it operational rather than reflective.
- If context includes a mandatory critique section, address it in code and
  write `critique_response.md` with Applied / Rejected / Remaining sections.
  Do not leave critique as a passive note.
- If animation events or animation analysis are present, inspect them before
  changing dynamics. You do not need to reproduce intermediate frames (the
  verifier target is the settled after-frame), but their order can reveal
  transition rules. For ANY transition you cannot yet explain, READ EVERY
  intermediate frame for that step -- the `tick_frames` paths in
  `animation_events.jsonl` (`step_<NNNN>_tick_<KK>.png`). Do NOT skip frames or
  discount any as unimportant: the mechanic (movement/collision-resolution
  order, a one-tick flash, a mid-animation trigger or counter) can live in a
  single frame. ESPECIALLY when a step's net diff is `Nothing changed` (a no-op
  before and after) yet it carries intermediate ticks: the mechanic lives
  ENTIRELY in those intermediate frames -- the action did something mid-animation
  that reverted by the settled frame, so that is exactly where the
  information-dense evidence is; read every tick. If a step has more than 20
  intermediate frames, hand them to a Task subagent to read them all in order
  and report back, rather than skipping any.

`frame` is a 2D list of palette indices 0..15 (typically 64x64). It is
the ONLY observation. There is no structured sprite list, no tag-keyed
ontology, no role annotations. Objects exist only as visual patterns
you induce from the frames in the buffer.

Objects (as you induce them) can be partially obscured, change shape,
swap colour, become multi-coloured, move under a higher-layer sprite,
or leave the canvas entirely. Multiple sprites can share the same role
or tag. For example, a wall is typically dozens of separate small sprites each
with their own (x, y), not one sprite with funky dimensions. A sprite's
pixels, rotation, position, and visibility may all change in response
to an action. Directly editing the grid cell-by-cell each step gets
brittle fast under these effects. The more tractable pattern is to
maintain an internal list of object/sprite hypotheses with (pixels,
x, y, layer, visible) and re-render it to a frame each step, exactly
how the env itself composites the display. A minimal `Sprite` class
+ `render_sprites` helper is already in game_engine.py for this. Use
it (or replace it) if your hypothesis benefits from a structured
intermediate; operate directly on the grid only when the mechanic
genuinely is per-cell.

Actions: integers in `available_actions`. The action ids listed in context.txt
are the only legal ids for this run; absent ids do not exist here and must not
be hypothesized as goal alternatives. ARC-AGI-3 universal convention for ids
that are present (treat as hints, verify from the buffer):
  0 = RESET (global reset to L1; ACTION0)
  1, 2, 3, 4 = up, down, left, right (in order); per-game effect -- discover from buffer
  5 = space / interact / no-op
  6 = single-point click. Carries (x, y) in DISPLAY space (0..63).
       Synth gets the action as {"action_id": 6, "x": int, "y": int}.
       Two consecutive ACTION6 calls are two independent clicks, never
       a drag or swipe.
  7 = UNDO (often a no-op on local games)

CRITICAL RULES:
1. %%CLASSES_LINE%%
2. transition_function returns a frame with the SAME SHAPE as the input.
3. reward_function MUST hypothesise a goal condition. (0.0, False) for
   every transition is rejected.
4. NO REPLAY-BUFFER LOOKUP. transition_function and reward_function must
   NEVER read ``replay_buffer.pkl``; it is the verifier's input and
   reading it is the canonical lookup-table cheat. The test runner
   statically rejects any model that references that filename. In-level
   transitions must be COMPUTED from (frame, action).
5. LEVEL-ENTRY CACHES are provided for the un-computable cases. The
   engine writes ``l<N>_initial.pkl`` files into this workspace, one
   per level the agent has observed. Each file is a pickled 2D list
   (the level's hand-designed entry frame, same shape as
   before_frame / after_frame). Use them for the level-advance and
   RESET transitions whose after_frame cannot be derived from rules:
   ```
   import os, pickle
   def _load_level_initial(fname):
       here = os.path.dirname(os.path.abspath(__file__))
       try:
           with open(os.path.join(here, fname), 'rb') as f:
               return pickle.load(f)
       except Exception:
           return None  # cache absent; fall through to in-level logic
   ```
   If a cache for an unobserved future level is missing, the loader returns
   None and your code must fall through gracefully. If the replay buffer
   already contains a level-advance into level N but `l<N>_initial.pkl` is
   absent, treat that as an infrastructure defect to report in
   `synth_learnings.md`, not as a legitimate accuracy ceiling. Do not
   normalize or accept a failed observed level-entry transition because a
   cache file is absent.

   reward_function MUST NOT reference these caches: using a cached
   level-initial as the goal predicate treats the env's level-advance as
   the goal instead of synthesising one -- a cheat the test runner rejects;
   the goal must be derivable from observable state.

GOAL CONDITIONS HAVE PRECONDITIONS. Reaching the goal is rarely a
single trivial predicate. Encode the joint precondition + configuration
that distinguishes the moment reward was earned from the moments it
was not. If your reward_function is too permissive (predicts reward
where none was observed) or too restrictive (misses an observed reward),
revise the precondition pattern.

EVERY VISIBLE PATTERN HAS A PURPOSE. Levels are hand-designed; cells
that change in response to your actions are mechanic-bearing. Cells
that NEVER change during gameplay are not necessarily scenery; they
may encode rules, targets, legends, or constants that the mechanics
resolve against. A reward_function that ignores visible objects or
patterns whose role you haven't pinned down should be treated as
incomplete, not finished. Any pixel change an action produces -- a
colour swap, rotation, shape or appearance change -- signals a real
state change and is mechanically meaningful; model it, never dismiss it
as a decorative or cosmetic highlight.

MOVE-COUNTER MASK. If a thin HUD strip encodes a
per-move step/timer counter whose per-level quantization you cannot predict
cleanly, you MAY define `def move_counter_mask(): return [(r, c), ...]`
returning ONE continuous line of cells at most 2 pixels wide along that
counter region. Cells in it are excluded from transition verification, so
counter ticks neither fail tests nor force resynthesis.

If you are very confident in your reward_function, you may consider
suggesting the goal in English to the action agent (e.g. as a
`# GOAL: <one sentence>` comment at the top of game_engine.py) if
you aren't doing this already.

Consider sanity-checking that your reward_function is reachable under
your transition_function on the observed buffer. If no state
reachable from the buffer satisfies your goal predicate, your goal
hypothesis is likely wrong and needs revising, even if every per-
step transition and reward prediction matches the buffer.

START: read context.txt, run tests, implement transition_function +
reward_function + planner, iterate.
