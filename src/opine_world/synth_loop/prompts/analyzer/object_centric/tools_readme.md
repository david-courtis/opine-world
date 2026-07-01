# Workspace artifacts (read me first)

You can `Read`, `Grep`, `Bash`, and `Task` over everything in this directory.

**run_log.txt** -- monotonic structured log. Section markers:
  `[STEP N]` ... `[/STEP N]`        per env step
  `[SYNTHESIS step=N run=K]`         synthesis events with reward fn
  `[LEVEL_ADVANCE step=N from=I to=J]` level transitions
  `[NOTE step=N source=X]`           notes (yours land here under
                                     source=consumer)

  Useful greps:
    grep -c '\[STEP ' run_log.txt           total steps so far
    grep -A2 '\[DIFF\]' run_log.txt | head  recent diffs
    grep '\[REWARD\]' run_log.txt | grep -v 0.0  reward events

  Per-step blocks include action, level, reward, diff, state_desc, the
  full structured-state JSON (single line, parseable with python), and
  optionally an ASCII frame.

  IMPORTANT -- segment by level before reasoning over past transitions.
  `run_log.txt` contains ALL levels' steps. The engine ALSO writes per-
  level files at `../levels/level_<N>.log` (where N is the 0-indexed
  level). Each level's file contains only that level's [STEP] blocks
  plus [LEVEL_START] / [LEVEL_END] boundary markers. Use these for
  level-scoped queries -- analyzing L2 reachability with L0's transition
  history mixed in is a known failure mode (L0-reachable cells may be
  walled off in L2, and you'll plan paths that don't actually work).

  Current level's log: parse the LATEST [STEP] in run_log.txt, read its
  [LEVEL] line, then grep `../levels/level_<that_level>.log`.

  To extract step N's after-state:
    ```
    python -c "
    import re, json
    log = open('run_log.txt').read()
    N = 17  # the step you want
    pat = r'\[STEP ' + str(N) + r'\][\s\S]+?\[/STEP ' + str(N) + r'\]'
    block = re.search(pat, log).group()
    state_json = re.search(r'\[STATE_JSON\] (.+)', block).group(1)
    print(json.loads(state_json))
    "
    ```

**epistemic_matrix.json** -- per-(type, action) cells with frequentist
  AND Bayesian fields. Cells are pre-sorted by `m['sort_by']` (default
  `"thompson"`); use directly or re-sort with your own rule.
  When goal progress has stalled (regime (b) of *Your objective* in the
  system prompt) THIS FILE selects your plan: the top cells under the
  active priority are the least-understood `(type, action)` dynamics and
  your actions must drive them. In that regime it is the objective, not
  an advisory hint.

  Frequentist fields (formalism §5.2):
    n        observation count
    d        distinct context fingerprints seen
    c        effect consistency in [0,1] (majority-effect fraction)
    cond     1 if the (type, action) pair shows conditional effects
    priority heuristic ranking, w1/(1+n) + w2/(1+d) + w3*(1-c)

  Bayesian fields (formalism §5.3, Beta posterior on consistency):
    alpha, beta        posterior parameters (alpha_0+s, beta_0+(n-s))
    mu, sigma          posterior mean / std of consistency probability
    priority_ucb       (1 - mu) + kappa * sigma         (Beta-UCB)
    priority_thompson  1 - p_tilde, p_tilde ~ Beta(alpha, beta)
                        -- fresh sample each engine step. HIGHER = more
                        worth exploring under Thompson sampling.

  Top cells by the active priority ARE your exploration targets when
  goal progress has stalled:
    ```
    python -c "
    import json
    m = json.load(open('epistemic_matrix.json'))
    print('sorted by:', m['sort_by'])
    for c in m['cells'][:5]:
        print(c['type'], 'A'+str(c['action_id']),
              'n='+str(c['n']), 'c='+str(c['c']),
              'mu='+str(c['mu']), 'sigma='+str(c['sigma']),
              'pi_th='+str(c['priority_thompson']))
    "
    ```

  To draw your OWN Thompson samples (e.g., for tie-breaking or to
  diversify across multiple action proposals):
    ```
    python -c "
    import json, random
    m = json.load(open('epistemic_matrix.json'))
    rng = random.Random(0xC0FFEE)
    scored = []
    for c in m['cells']:
        # 1 - Beta(alpha, beta) sample
        p_tilde = rng.betavariate(c['alpha'], c['beta'])
        scored.append((1 - p_tilde, c['type'], c['action_id']))
    scored.sort(reverse=True)
    for s, ty, a in scored[:5]:
        print(f'{s:.3f}  {ty} A{a}')
    "
    ```

**current_state.json** -- `{ "state": [...], "describe": "...",
  "available_actions": [...], "moves_remaining": <int|null>,
  "step": N, "level": L }`. The current pre-action state.
  On the first call for level 1 and on every later new level, also read the
  latest complete 64x64 `[ASCII_FRAME]` block in `run_log.txt` before choosing
  actions; do not rely on partial slices.

**replay_buffer.pkl** -- list[dict] of every transition observed. Each
  dict has before_state, action_id, after_state, diff_text, reward,
  done, level. Pickle, not JSON -- open with python.

%%WM_SECTION%%
%%ESCAPE_SECTION%%
**synth_status.json** -- engine-side snapshot of the CEGIS synthesizer's
  state. Read to decide whether to trust the current world model.
  Fields:
    synthesis_count                how many synthesis runs have completed
    model_present                  bool -- is game_engine.py available?
    best_transition_accuracy       in [0,1] -- last verified accuracy
    consecutive_failed_syntheses   if >=3, the synth is struggling on
                                   something; the model may be unreliable
    goal_hypothesis_run            which synth run produced the current
                                   reward_function (older = staler)
    goal_hypothesis_snippet        first 800 chars of reward_function
    synthesis_gate                 delayed CEGIS gate: active, ready,
                                   moves/errors/action-plans left before
                                   auto-repair, and ready_reasons
    synth_learnings                concise mechanics/probes handoff from synth
    critique_findings              independent critique of the synth model
    critique_response              synth's applied/rejected response
    animation_findings             independent reading of intermediate frames
    shared_model_updates           summary of recent shared-doc edits by synth
    n_transitions, engine_step, levels_completed, current_level
    last_synthesis_step
    planner                        C3 status: enabled, queued_actions,
                                   blocked_round, retry_after_step,
                                   last_status. If C3 is blocked/no-plan,
                                   continue real-environment exploration.

  Quick read:
    ```
    python -c "
    import json
    s = json.load(open('synth_status.json'))
    print('model_present:', s['model_present'])
    print('accuracy:', s['best_transition_accuracy'])
    print('consecutive_failed:', s['consecutive_failed_syntheses'])
    print('goal_run:', s['goal_hypothesis_run'])
    "
    ```

  The engine also injects a compact SYNTHESIZER HANDOFF block directly into
  your prompt when these fields exist. Use it explicitly as a prior, but trust
  real observations over it and write synth_control.json when you find a
  contradiction.

  If `synthesis_gate.active` is true and `ready` is false, CEGIS is being held
  back to gather more evidence. Keep exploring, inspect the divergence frames,
  and leave concrete questions/interpretations for the synthesizer. Automatic
  repair opens when any one happens: the current level is completed, the first
  action sequence on a brand new level has executed, the move countdown reaches
  zero, the error countdown reaches zero, or the completed-action-plan countdown
  reaches zero. If you need the planner now, write `force_now: true` with a
  focused repair note; otherwise avoid relying on C3.

**World-model divergence frames (auto-injected)** -- when the world model
  mispredicted your last plan, the ACTUAL frames are attached directly to
  your prompt: before+after of the first diverging action, then the after
  frame of each later action in that plan. They are the highest-signal
  events on the board -- study them for the mechanic the model missed. If the
  step animated, inspect `animation_events.jsonl` and
  `../frames/step_<NNNN>_tick_<KK>.png`; use Task as a second-opinion
  animation-analysis subagent when timing/order is unclear.

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

**synth_control.json** -- YOU WRITE this. It steers synthesis focus and
  deferral. CEGIS itself only fires after the engine observes that the
  executable model mispredicted a newly executed action sequence
  (transition/state or reward/done), so `force_now` is not a license to
  synthesize on speculation. Fields (all optional):
    defer_until_step    int     -- engine will skip synthesis until step
                                   reaches this value. Use when you're
                                   mid-probe and don't want the model
                                   changing under your feet.
    force_now           bool    -- request synthesis at the next plan
                                   boundary after a real model mismatch
                                   (auto-cleared after one use). Use
                                   when you've observed a concrete
                                   transition/reward the current model
                                   gets wrong.
    focus               string  -- free-text hint that synthesis sees
                                   in its context.txt under "Synthesis
                                   focus requested by analyzer". Tell
                                   it which transition or mechanic to
                                   pay attention to.

  Examples (write via Bash):
    ```
    # Defer synthesis for the next 10 steps while you finish a probe.
    # Read your current step from synth_status.json or run_log.txt.
    python -c "
    import json
    cur = 5  # replace with the actual current step you read
    json.dump({'defer_until_step': cur + 10},
              open('synth_control.json','w'))
    "

    # Request CEGIS after a concrete model mismatch, with a focus hint
    python -c "
    import json
    json.dump({'force_now': True,
               'focus': 'the last push changed pixels the model did not predict; repair the push rule around the sprite obstacle.'},
              open('synth_control.json','w'))
    "
    ```

**notes.md** -- your own scratchpad. Carries across calls. Append to
  it (`python -c "open('notes.md','a').write('observation: ...\n')"`)
  when you discover something you'd want next call. Do NOT visually
  re-read it from previous calls -- grep it.

**type_aliases.json** -- current role-hypothesis space per object type
  tag. Maintained across calls and visible to the synthesis subprocess.
  Each type maps to a ranked list of candidates (≥4) with scores:
    ```
    {
      "0001sruqbuvukh": [
         {"alias": "target",      "score": 12},
         {"alias": "goal_marker", "score":  4},
         {"alias": "unknown_1",   "score":  0},
         {"alias": "unknown_2",   "score":  0}
      ],
      ...
    }
    ```
  Read with:
    ```
    python -c "import json; [print(t, a[:2]) for t,a in json.load(open('type_aliases.json')).items()]"
    ```

**alias_updates.json** -- YOU WRITE this. Engine consumes it after
  THIS call and clears it. Use to evolve the role hypotheses:
    add     introduce a new candidate at score 0
    upvote  increment a candidate's score (default by=1)
    remove  drop a candidate (auto-padded back to ≥4 with unknown_N)
  Example:
    ```
    python -c "
    import json
    json.dump({
      'add':    [{'type':'0001sruqbuvukh','alias':'target'}],
      'upvote': [{'type':'0007arvfmhagbj','alias':'cursor','by':2}],
      'remove': [{'type':'0001sruqbuvukh','alias':'unknown_1'}]
    }, open('alias_updates.json','w'))
    "
    ```
  Upvote a candidate **on each new piece of evidence that it's the
  right role** (RGB convention: +1 per consistent observation, e.g.
  one observed transition that fits the role). Add new candidates as
  hypotheses come to you. The top candidate per type rises with use
  and is shown to synthesis as the hypothesised role.

## Available actions

Action ids: %%ACTIONS_STR%%

These are the only legal actions in the current game/state. Any id absent from
this list does not exist here; do not output it or treat it as an unexplored
probe.

## Output

Write your decision to `next_actions.json`:

    {"plan": [<step>, ...], "reasoning": "<one sentence>"}

Each step is either a bare int (e.g. 3 = ACTION3), or
``{"action": "ACTION6", "x": <col>, "y": <row>}`` for a click
(standard image convention; x = column, y = row).
ACTION6 is only valid when the available action set above includes 6.
**ACTION6 is a single-point click. ARC-3 has no drag, swipe, or
source→destination semantics.** Two consecutive ACTION6 calls at
different coordinates are two unrelated clicks, not a drag.
Plan length: shorter (3-5 steps) is preferred when exploring, but up to 10 steps is fine when you are more certain.
Only commit to >5 steps when you have very
high confidence.

## Project root (for absolute path debugging)

%%PROJECT_ROOT%%
