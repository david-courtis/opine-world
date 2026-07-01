# Workspace artifacts (read me first -- frames-only mode)

You can `Read`, `Grep`, `Bash`, and `Task` over everything in this directory.

**run_log.txt** -- monotonic structured log. Section markers:
  `[STEP N]` ... `[/STEP N]`        per env step
  `[SYNTHESIS step=N run=K]`         synthesis events
  `[LEVEL_ADVANCE step=N from=I to=J]` level transitions
  `[NOTE step=N source=X]`           notes (yours land here under source=consumer)

  Per-step blocks include the action, the level, the reward, a pixel-
  level [DIFF] summary, and an [ASCII_FRAME] block (64 lines of 64 chars,
  digits 0-9 then A-F for palette values 10-15).

  To extract step N's post-action ASCII frame:
    ```
    python -c "
    import re
    log = open('run_log.txt').read()
    N = 17
    pat = r'\[STEP ' + str(N) + r'\][\s\S]+?\[/STEP ' + str(N) + r'\]'
    block = re.search(pat, log).group()
    m = re.search(r'\[ASCII_FRAME\]\s*(.*?)\s*\[/ASCII_FRAME\]', block, re.DOTALL)
    print(m.group(1))
    "
    ```

**current_state.json** -- `{ "frame": [[..64..], ..64..],
  "available_actions": [...], "moves_remaining": <int|null>,
  "step": N, "level": L }`. The pre-action 64x64 palette grid only.
  On the first call for level 1 and on every later new level, parse/read this
  full 64x64 frame before choosing actions; do not rely on partial slices.

**replay_buffer.pkl** -- list[dict] of every transition observed. Each
  dict has before_frame, action_id, after_frame, diff_text, reward,
  done, level (and click_x/click_y on click steps). Pickle, not JSON.

**epistemic_matrix.json** -- when the synthesizer has exported
  `extract_objects(frame)`, this is an ETA-style per-(induced object
  type, action) priority matrix built from those recognized visual
  objects. Treat it as advisory: the object abstraction is synthesized
  from frames and can be wrong, but high-priority rows are good probes.

**ontology_error.json** -- ETA / eta* diagnostic over the same induced
  objects. High eta means the current object abstraction or context
  features are still mixing incompatible effects. Low eta means the
  synthesized abstraction is becoming Markov-like enough to trust more.

**spriteless_object_abstraction.json** -- small summary of the latest
  synth-provided `extract_objects(frame)` pass: whether it ran, object
  types seen, object-count ranges, and any extractor errors. Check this
  before trusting the matrix.

%%WM_SECTION%%%%ESCAPE_SECTION%%
**synth_status.json** -- engine-side snapshot of the CEGIS synthesizer's
  state. Read to decide whether to trust the current world model.
  Fields: synthesis_count, model_present, best_transition_accuracy,
  consecutive_failed_syntheses, goal_hypothesis_run,
  goal_hypothesis_snippet, goal_in_english, n_transitions, engine_step,
  levels_completed, current_level, last_synthesis_step, synthesis_gate, planner,
  synth_learnings, critique_findings, critique_response,
  animation_findings, shared_model_updates.
  `synthesis_gate` reports delayed CEGIS state: whether repair debt is active,
  whether synthesis is ready, how many moves/errors/completed action plans
  remain before auto-repair, and the ready reasons. If active but not ready,
  keep exploring and write concrete hypotheses/questions for the synthesizer;
  auto-repair opens on level completion, after the first action sequence on a
  brand new level, after the move countdown reaches zero, after the error
  countdown reaches zero, or after the completed-action-plan countdown reaches
  zero.
  `planner` reports whether engine-side C3 is enabled, queued, blocked,
  cooling down, or verified. If C3 is blocked/no-plan, keep exploring.

  `goal_in_english` is a plain-English win-condition sentence written
  by the synth ONLY when it considers its reward_function stable across
  rounds. Empty string when absent. When present, it is the synth's
  current best guess at "what does winning look like" -- still a
  hypothesis (unproven until reward fires), but a strong prior to
  drive toward.

  The engine also injects a compact SYNTHESIZER HANDOFF block directly into
  your prompt when synth_learnings / critique / animation findings exist.
  Use it explicitly as a prior, and contradict it in synth_control.json when
  real observations prove it wrong.

**World-model divergence frames (auto-injected)** -- when the world model
  mispredicted your last plan, the ACTUAL frames are attached directly to
  your prompt: before+after of the first diverging action, then the after
  frame of each later action in that plan. They are the highest-signal
  events on the board -- study them for the mechanic the model missed.
  Per-tick intermediate frames (if an action animated) remain on disk at
  ../frames/step_<NNNN>_tick_<KK>.png if you want them. Use Task as a
  second-opinion animation-analysis subagent when timing/order is unclear.

**synth_learnings.md, last_critique.md, critique_response.md,
animation_analysis.md, animation_events.jsonl** -- optional symlinks to the
  latest synthesis-side handoff/review files. These summarize mechanics the
  synthesizer thinks it learned, independent critique, the synth's response,
  and intermediate animation readings.

**world_model.md, level_N_reasoning_log.md, level_N_report.md,
shared_model_updates.md** -- shared Markdown artifacts used by both analyzer
  and synthesizer. `world_model.md` should maintain the mechanics, target
  hypothesis, how the target is visually inferred, ad-hoc inventory, and newly
  introduced unexplained elements. The current level's reasoning log should
  record hypotheses tested, evidence, mismatches, and corrections. Reports are
  filled after level completion or when a stable explanation exists. Update
  these files with Python append/write commands when your observations change
  the model or expose an unresolved visual debt.

**../frames/step_<NNNN>.png** -- pre-rendered 64x64 PNG of each step,
  palette-coloured with grid + coordinate labels. `Read ../frames/step_NNNN.png`.
  A first-class visual for interpretation and discovering mechanics. Use it to
  see what is happening, then read the ASCII frame for the exact values. Reach
  for it whenever colour or shape gestalt matters or when stuck.

**synth_control.json** -- YOU WRITE this. Steers synthesis focus and
  deferral. CEGIS itself only fires after the engine observes that the
  executable model mispredicted a newly executed action sequence
  (transition pixels or reward/done), so `force_now` is not a license to
  synthesize on speculation. If unrepaired model divergence is blocking C3 and
  you need planner help now, use `force_now` with a precise focus note;
  otherwise let the delayed CEGIS gate collect more evidence first.
  Fields (all optional):
    defer_until_step  int  -- skip synthesis until step reaches this.
    force_now         bool -- request synthesis at the next plan boundary
                              after a real model mismatch; auto-cleared.
    focus             str  -- free-text hint surfaced to synthesis.

  Example:
    ```
    python -c "
    import json
    json.dump({'force_now': True,
               'focus': 'the last executed gate click changed pixels the model did not predict; repair that transition'},
              open('synth_control.json','w'))
    "
    ```

**notes.md** -- your own scratchpad. Carries across calls. Append to it.

## Available actions

Action ids: %%ACTIONS_STR%%

These are the only legal actions in the current game/state. Any id absent from
this list does not exist here; do not output it or treat it as an unexplored
probe.

## Output

Write your decision to `next_actions.json`:

    {"plan": [<step>, ...], "reasoning": "<one sentence>"}

Each step is either a bare int (e.g. 3 = ACTION3), or
``{"action": "ACTION6", "x": <col>, "y": <row>}`` for a click.
Plan length: shorter (3-5 steps) is preferred when exploring, but up to 10 steps is fine when you are more certain.

## Project root (for absolute path debugging)

%%PROJECT_ROOT%%
