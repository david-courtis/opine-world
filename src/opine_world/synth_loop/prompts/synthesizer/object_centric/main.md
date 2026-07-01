You are building an object-centric world model for an ARC-AGI-3 game. Workspace: %%WORKSPACE_DIR%%

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
state should be represented by the shared object ontology already present in the
state list: persistent geometry/structures, object families, and current dynamic
fields. The transition rule should advance that state by the action. For the
uncomputable level-entry / RESET cases, `l<N>_initial.pkl` caches may seed the
entry state inside `transition_function`, but they are not a general
state-reconstruction escape hatch and must never be used by `reward_function`.

Keep one shared ontology across levels whenever the visual evidence supports it:
use the same object families for similar motifs, prefer one shared
classification strategy over separate per-level detectors, and express new level
behavior as per-level parameters of known rules before inventing new object
families or latent variables. If a later level adds partial visibility, sliding,
layering, or hidden state, extend the observation/state representation while
preserving the underlying object family unless the buffer forces a genuinely new
mechanic.

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
display_h, visible, collidable, layer, rotation, pixels.

`x, y, w, h` are the sprite's CAMERA-GRID rectangle (the level's logical
coord space (what game rules operate on)). `display_x, display_y,
display_w, display_h` are the same rectangle in DISPLAY-SPACE (0..63 in
both axes, the canonical 64×64 ARC-AGI-3 canvas, which the camera grid
scales uniformly to fill). The ratio is `display_x = x * scale`, where
`scale = 64 // camera_width` (typically 1, 2, 3, or 4). ACTION6 clicks
take coordinates in DISPLAY space, so to click a sprite you target
`(display_x .. display_x + display_w - 1, display_y .. display_y +
display_h - 1)`. Game rules (movement, collision, sprite logic) operate
on camera-grid coords `(x, y, w, h)`.

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
on this specific level is up to the level: it might move an object, rotate
something, switch which actor is controlled, or do nothing in some
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
  details, temporary cache use, level-specific branches, duplicate object
  families, and competing hypotheses that need future probes. This text is
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

REWARD FUNCTION REQUIREMENT (phi_2, optimism under uncertainty):
The reward_function must NOT always return (0.0, False). Even if no reward has
been observed in the training data, you MUST hypothesize a goal condition and
implement it.

DO NOT BAKE IN DOMAIN ASSUMPTIONS. Don't assume this is a gridworld, that
there's a single 'player' object, that one specific tag is the goal, that
actions move an actor, or that the reward is a tile-touch. The level may be
any of: navigation, matching/sorting, sequencing, construction, rotation,
elimination, timing, multi-actor coordination, or a combination. Discover
which from the replay buffer.

GOAL CONDITIONS ARE USUALLY A CONJUNCTION, NOT A SINGLE PREDICATE. Reward
typically needs one or more PRECONDITIONS (collecting/moving objects in
order, toggling state, matching configurations, unlocking passages,
visiting cells in sequence) together with a trigger. Canonical joint
patterns: a precondition AND the actor in a specific cell/region; a
precondition that UNLOCKS a region with reward firing only on ENTERING it;
or a precondition holding at the same time as a positional trigger. Encode
the FULL precondition-+-completion pattern, not a naive single-object check.
Revise if the hypothesis is too permissive (predicts reward where none was
observed) or too restrictive (misses an observed reward); and if a
world-state predicate seemed satisfied yet no reward fired, the missing
piece is almost certainly a position/region/access requirement. Look for
what is structurally distinct about the moment(s) reward was earned versus
those it was not.

%%OBJECTS_CLAUSE%%

EVERY OBJECT HAS A PURPOSE. Sprites in a hand-designed level are almost never
no-op. If your model treats some object as inert, you're likely missing its
role. Look for evidence in the replay buffer of every object you have not yet
modeled. Any change an action produces -- colour, rotation, shape, appearance,
position, visibility -- signals a real state change and is mechanically
meaningful; model it, never dismiss it as a decorative or cosmetic highlight.

Discover all rules from context.txt and the replay buffer: action effects,
guarded inter-object interactions, position-conditional blocks, hidden state
changes, and the precondition pattern that gates the reward.

================================================================
ξ-REFINEMENT AUDIT (optional, available once η measure has run)
================================================================
The engine measures ontology error η = how confounded the current
object factorisation is. Per-(type, action, context) strata land in
`ontology_error.json` (symlinked into this workspace), and under
`latest.xi_candidate_ledger` you will find a mechanically-enumerated
list of ContextFeature candidates whose acceptance would reduce η on
the top-K worst strata. Each candidate carries: feature descriptor,
η_old, η_new, η_reduction, n_substrata, identified_substrata, accepted.

Acceptance criterion (verifier-applied):
  η_reduction ≥ 0.05  AND  identified_substrata ≥ 1

If you find one or more ACCEPTED candidates that name a genuine
conditioning feature your transition_function would need to read,
commit them by writing ``xi_updates.json`` to this workspace:

  {
    "add": [
      {"kind": "target_field", "field": "rotation"},
      {"kind": "neighbour_at_offset", "dx": 1, "dy": 0}
    ],
    "rationale": "<one sentence on why these features matter>"
  }

Known ``kind`` values and required parameters:
  target_field            field: str  (e.g. rotation, visible, scale,
                                       mirror_ud, mirror_lr, pixels_hash)
  neighbour_at_offset     dx: int, dy: int  (within {-1,0,1}^2)
  neighbourhood_radius    r: int   (widen the base 3×3 window)
  click_offset            (no params; for action 6 only)

ENGINE-SIDE VERIFIER: every feature you write is RE-SCORED by the
engine after you exit. A proposal is committed iff it produces
Δη ≥ 0.05 with ≥ 1 identified sub-stratum on at least one of the
current top-3 worst strata, same rule as the ledger. Features that
look plausible to you but don't survive verification are silently
rejected and logged. Pick features that appear with ``accepted: true``
in the ledger AND that you can articulate a mechanic-level reason for.

Features are MONOTONICALLY ACCUMULATED. The verifier guarantees no
commitment can REGRESS η. If no candidate looks meaningful, do not
write the file. The pre-ledger case (no η measure yet) is a no-op:
xi_updates.json is ignored without an η measurement to verify against.

START: read context.txt, run tests, implement transition_function +
reward_function + planner, iterate.
