You are building a world model for an ARC-AGI-3 game. Workspace: %%WORKSPACE_DIR%%

FILES:
- context.txt: World model + observed transitions. READ FIRST.
- game_engine.py: YOUR CODE. Edit this file.
- test_runner.py: Run: python %%TEST_RUNNER_PATH%%
- replay_buffer.pkl: Ground truth (don't modify).
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

Python packages available: numpy, scipy, PIL, cv2, imageio, networkx,
matplotlib, sympy, pandas, sklearn, skimage, shapely, z3, yaml.

TASK: Implement transition_function, reward_function, and a conservative
planner(state, available_actions=None, max_depth=None) -> list[action] | None.
Run tests, iterate until ALL TESTS PASSED.

ANTI-LOCK-IN / REPAIR DISCIPLINE:
You may inherit an existing `game_engine.py`, world-model notes, or critique
from a prior synthesis round. Treat that model as a suspect hypothesis, not as
authority. You were called because the current mechanics formalization is still
incomplete, overfit, or possibly wrong at the abstraction level. It may be
largely flawed. Incrementally build on it only when doing so keeps the model
simple and general; prefer rewriting major components when that de-janks the
world model, removes special cases, or gives a cleaner state representation.

Actively compare competing hypotheses before locking in a mechanic.
Do not blindly add complexity; choose the rule set that best explains the
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
inventory; Target of the Game; How the player is expected to infer the target;
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
state should be represented by the records already present in the state list:
persistent geometry, the records themselves, and current dynamic fields. The
transition rule should advance that state by the action. For the
uncomputable level-entry / RESET cases, `l<N>_initial.pkl` caches may seed the
entry state inside `transition_function`, but they are not a general
state-reconstruction escape hatch and must never be used by `reward_function`.

Keep one rule set across levels whenever the visual evidence supports it:
use the same rule for similar motifs, prefer one shared rule over separate
per-level detectors, and express new level behavior as per-level parameters
of known rules before inventing new branches or latent variables. If a later
level adds partial visibility, sliding, layering, or hidden state, extend the
state representation while preserving the existing rules unless the buffer
forces a genuinely new mechanic.

Before introducing any new level-specific state variable, ask whether the
phenomenon is already covered by an existing rule, a known dynamic field, an
observation/visibility/layering effect, or a parameterization of an existing
rule. Only add a new latent variable when the previous rule set cannot explain
the observations.

GENERALISATION, AND MECHANICS THE CURRENT STATE CANNOT EXPLAIN:
Your model is graded on states it has never seen. Passing the replay is the
floor, not the goal.

NEVER HARDCODE OBSERVED STATE. Do not write a table mapping observed
positions, configurations, or state signatures to their observed successors,
and do not special-case a step by its index or exact board layout. A table
transcribed from the buffer passes the verifier perfectly and is worthless:
it answers only the questions already answered, and it is the same cheat as
reading the buffer directly, just copied in by hand. Static level geometry a
rule reads (a wall map, a sprite's cell pattern) is data. A map from a state
to its next state is not; it is the rule you failed to find.

SOME MECHANICS ARE GENUINELY NOT A FUNCTION OF THE CURRENT STATE: a patrol
route, a replayed recording, a spawn order, a counter with no visible
readout. The information exists in the game and is absent from the records
you are given. Do not conclude such a mechanic is unmodelable and tabulate
it. It is recoverable, because THE REPLAY BUFFER IS ONE CONTINUOUS,
STRICTLY SEQUENTIAL PLAYTHROUGH: the verifier walks it in order and each
step's before_state is the previous step's after_state. Your module may
ACCUMULATE hidden state across calls and carry it forward, exactly as the
real game does.

GATE THAT ON CONTINUITY OR IT WILL BE WRONG. You are also called on states
that do NOT continue your own last output: the planner explores hypothetical
branches, and a level can reset. Keep a signature of the state you last
returned; when the incoming state matches it your accumulated state applies,
and when it does not, fall back to a conservative default (freeze the thing
you were tracking) rather than applying stale history. Say so in a comment.

An accumulator that reconstructs a hidden mechanic from the sequence
generalises to any trajectory. A table of observed transitions does not. If
you find yourself enumerating cases, you have stopped modelling.

VERIFICATION-ONLY FALLBACKS AND MODELING DEBT:
The closest analogue to baseline1's temporary renderer override is our narrow
level-entry cache allowance: use `l<N>_initial.pkl` only for level-advance/RESET
transition states that cannot be derived from current state alone. Do not hide
game logic, planning logic, reward predicates, or ordinary in-level transitions
behind cache reads or per-level branches. Any frame-local special case, cache
dependence, unexplained state field, or duplicated branch is evidence that
the model is still missing a mechanic, a latent state variable, or an
observation rule. List these debts concretely in `synth_learnings.md` and remove
them once a clean mechanic explains the behavior.

State records have: name, tags, x, y, w, h, display_x, display_y, display_w,
display_h, visible, collidable, layer, rotation, pixels.

`x, y, w, h` are the sprite's CAMERA-GRID rectangle (the level's logical
coord space (what game rules operate on)). `display_x, display_y,
display_w, display_h` are the same rectangle in DISPLAY-SPACE (0..63 in
both axes, the canonical 64×64 ARC-AGI-3 canvas: the camera grid scaled
uniformly and letterboxed/centred, possibly with a camera origin
offset). The two spaces generally DIFFER by more than a scale factor;
do not derive one from the other yourself, use the precomputed fields.
ACTION6 clicks take coordinates in DISPLAY space; click transitions in
the replay buffer store both the raw click (`click_x`, `click_y`,
display space) and `click_grid` (the same click mapped into camera-grid
space, `null` when the click landed outside the play area, e.g. in the
letterbox). Compare `click_grid` against `(x, y, w, h)` when modelling
click mechanics. A click only registers on a solid (non-transparent)
pixel of a sprite; transparent cells inside the bbox miss. Game rules
(movement, collision, sprite logic) operate on camera-grid coords
`(x, y, w, h)`.

`pixels` is the sprite's CURRENT INTERNAL PATTERN, a list[list[int]]
of shape (h, w) in CAMERA-GRID resolution (NOT pre-scaled). `rotation`
is one of {0, 90, 180, 270} degrees clockwise. Both can change in
response to actions: e.g. clicking a switch may rotate a HUD sprite's
pixels by 90°, leaving (x, y, w, h) unchanged. **Your
transition_function MUST predict pixel and rotation changes when they
occur**, since phi_1 requires predicting EVERY observable change per step,
not just position/visibility. The verifier compares predicted vs
actual at pixel fidelity.
Actions are integers from `available_actions`. The action ids listed in
context.txt are the only legal ids for this run; absent ids do not exist here
and must not be hypothesized as goal alternatives. Across ARC-AGI-3 the
universal action-semantic CONVENTION for ids that are present is:
  ACTION1 = UP arrow key
  ACTION2 = DOWN arrow key
  ACTION3 = LEFT arrow key
  ACTION4 = RIGHT arrow key
  ACTION5  = "space" / interact / no-op (per-game variable)
  ACTION6  = single-point click. The action carries one (x, y); the
             replay buffer's click transitions store click_x / click_y
             alongside action_id. ARC-3 has NO drag, swipe, or
             source→destination semantics; do not encode multi-step
             click protocols. Two consecutive ACTION6 transitions are
             two unrelated single clicks.
  ACTION7  = UNDO. Cloud-implemented; on local games typically a no-op
             (your replay buffer will reveal which).
Treat these as HINTS, not as guaranteed mechanics. What an "arrow key" does
on this specific level is up to the level: it might move something, rotate
something, switch what is controlled, or do nothing in some
contexts. Verify via the replay buffer's observed transitions before
encoding it.

%%RULES_BLOCK%%

PLANNER DELIVERABLE:
Implement `planner` as a model-side search over your own
transition_function + reward_function. Return a real action sequence only
when it reaches reward under the model; return None when no plan is found
within budget. The engine uses this only after a real level completion
(and after crystallisation when that mode is enabled), then validates each
step against the environment. Make this an explicit, budgeted, goal-directed
search, not a naive uninformed BFS whose high branching factor will not finish
in time on ARC-3.

HANDOFF DELIVERABLES:
- Write/update `synth_learnings.md` with short bullets for the exploration
  agent: known mechanics, uncertain hypotheses, high-value probes, and
  avoid-repeat failures. Include an ad-hoc/debt inventory: unresolved visual
  details, temporary cache use, level-specific branches, duplicated branches,
  and competing hypotheses that need future probes. This text is
  injected into the analyzer prompt, so make it operational.
- If context includes a mandatory critique section, revise the model where
  the critique is valid and write `critique_response.md` with Applied /
  Rejected / Remaining sections. Reject findings only with replay-buffer
  evidence.
- If animation events or animation analysis are present, inspect them before
  editing transition dynamics. You do not need to reproduce intermediate
  frames (the verifier target is the settled after-state), but their sequence
  can reveal the actual mechanic. For ANY transition you cannot yet explain,
  READ EVERY intermediate frame for that step -- the `tick_frames` paths in
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

ARC-AGI-3 GOAL CONTINUITY: the goal never fundamentally changes between
levels of a game. A correct goal hypothesis is level-agnostic: it must
account for every reward observed on every completed level. When revising
the goal hypothesis you may only GENERALIZE it. The revised statement must
still explain all previous levels' rewards while accommodating the new
evidence. Never propose a goal that contradicts a previous level's observed
reward, and never propose one specific to the current level. Prefer the most
elegant, simple, level-agnostic statement. Do not overcomplicate or reach.
Level-specific detail belongs in the instantiation of the goal (what fills
which role this level), not in the goal itself.
Revisions are ADDITIONS and GENERALIZATIONS only: the core goal
mechanic established by the first observed reward must remain in
every later hypothesis. New evidence may widen its conditions or add
clauses beside it, never replace or delete it.

GROUNDING CROSS-CHECK: when `goal_requirements.json` is present in this
workspace, it is the disposal record of YOUR previous goal hypothesis --
an independent pass decomposed `goal_in_english` into executable
requirement predicates and the replay buffer tested each one at every
observed reward moment. Read it before revising `reward_function`:
- `rejected_necessity` entries are DISPROVEN readings of the goal (they
  failed at a real reward). Your revised reward_function must not encode
  them.
- `accepted` entries held at every reward observed so far. Preserve what
  they express under revision (additions and generalizations only).
- `injections` are accepted requirements never yet satisfied on the
  current level: the concrete gap between the goal hypothesis and every
  state seen so far.
If your current reward_function disagrees with this record, reconcile
them: either your code encodes a disproven reading, or the grounded
hypothesis lags evidence your code already uses. The buffer decides.
If you conclude the record itself is wrong, do not edit or fight it
in place -- goal_requirements.json is read-only engine output and is
wholesale re-derived from YOUR hypothesis: revise goal_in_english
(regrounding re-runs automatically on any revision, and your revised
reward_function must still replay every observed reward), and record
the disputed verdict concretely in world_model.md so the next
grounding cycle and the exploration agent both see it. A rejected
predicate killed one implementation, not the idea: propose a
corrected formulation instead of re-encoding the disproven one.

REWARD FUNCTION REQUIREMENT (phi_2, optimism under uncertainty):
The reward_function must NOT always return (0.0, False). Even if no reward has
been observed in the training data, you MUST hypothesize a goal condition and
implement it.

DO NOT BAKE IN DOMAIN ASSUMPTIONS. Don't assume this is a gridworld, that
there's a single 'player', that one specific tag is the goal, that
actions move an actor, or that the reward is a tile-touch. The level may be
any of: navigation, matching/sorting, sequencing, construction, rotation,
elimination, timing, multi-actor coordination, or a combination. Discover
which from the replay buffer.

GOAL CONDITIONS ARE USUALLY A CONJUNCTION, NOT A SINGLE PREDICATE. Reward
typically needs one or more PRECONDITIONS (collecting/moving things in
order, toggling state, matching configurations, unlocking passages,
visiting cells in sequence) together with a trigger. Canonical joint
patterns: a precondition AND a specific cell/region being occupied; a
precondition that UNLOCKS a region with reward firing only on ENTERING it;
or a precondition holding at the same time as a positional trigger. Encode
the FULL precondition-plus-completion pattern, not a naive single check.
Revise if the hypothesis is too permissive (predicts reward where none was
observed) or too restrictive (misses an observed reward); and if a
world-state predicate seemed satisfied yet no reward fired, the missing
piece is almost certainly a position/region/access requirement. Look for
what is structurally distinct about the moment(s) reward was earned versus
those it was not.

%%OBJECTS_CLAUSE%%

NOTHING IN A LEVEL IS INERT. Sprites in a hand-designed level are almost never
no-op. If your model ignores part of the state, you're likely missing a
mechanic. Look for evidence in the replay buffer for every record you have not
yet accounted for. Any change an action produces -- colour, rotation, shape,
appearance, position, visibility -- signals a real state change and is
mechanically meaningful; model it, never dismiss it as a decorative or cosmetic
highlight.

Discover all rules from context.txt and the replay buffer: action effects,
guarded interactions, position-conditional blocks, hidden state
changes, and the precondition pattern that gates the reward.

MOVE-COUNTER MASK. If a thin HUD strip sprite encodes a per-move step/timer
counter whose per-level quantization you cannot predict cleanly, you MAY
define `def move_counter_mask(): return [(r, c), ...]` returning ONE
continuous line of DISPLAY-space cells at most 2 pixels wide along that
counter region. Sprites whose display rectangle lies entirely inside it are
excluded from transition verification, so counter ticks neither fail tests
nor force resynthesis. The verifier rejects any wider mask; it may NOT
cover real mechanics.

START: read context.txt, run tests, implement transition_function +
reward_function + planner, iterate.
