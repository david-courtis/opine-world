**tools/escape_sequence.py** -- Lévy-flight self-avoiding walk
  generator. Call when you diagnose yourself as stuck (e.g., the
  last 10 steps are oscillating between two positions, or every
  recent action produced 'Nothing changed'):
    ```
    python tools/escape_sequence.py --actions %%ACTIONS_STR_SPACES%% --length 50
    ```
  Outputs JSON: {"plan": [...]}. You can adopt the plan
  wholesale, slice off a prefix, or use it as a hint.

**tools/plan.py** -- YOUR planner. Searches the synthesised world model
  (`game_engine.py`) for an action sequence that the MODEL predicts reaches
  reward from the CURRENT board, and prints it as JSON
  (`{"ok", "plan", "source", "reason", "nodes"}`). Only present once synthesis
  has produced a model.
    ```
    python tools/plan.py --max-depth 12 --timeout 30
    ```
  The planner does NOT run on its own and never executes moves -- YOU decide
  whether to adopt the returned plan, and you remain responsible for every move.
  The model can be wrong in states it hasn't seen, so sanity-check the route and
  treat it as a suggestion. If the model is stale or mispredicts, FORCE A CEGIS
  REPAIR FIRST -- write `synth_control.json` with `{"force_now": true, "focus":
  "<concrete mismatch>"}`, let the engine re-synthesise, then re-run
  `tools/plan.py` against the fresh model to plan a route.

**tools/view_sprite.py** -- ASCII slice of any sprite's pixel
  content. The structured state lists sprite bounding boxes; this
  tool extracts what's INSIDE the box from the latest [ASCII_FRAME]
  in run_log.txt. FIRST thing to do for any sprite whose role is
  unclear (HUD/pattern display/indicator boxes) -- the bounding
  box is a placeholder; the INTERNAL PATTERN is the meaning.
    ```
    python tools/view_sprite.py --name <sprite_name>
    python tools/view_sprite.py --tag <tag> --pad 1
    python tools/view_sprite.py --xywh 12 38 9 9
    ```
  Output: ASCII rows of the slice -- same encoding as the
  [ASCII_FRAME] block (digits 0-9, then A-F for 10-15).

**../frames/step_NNNN.png** -- pre-rendered 64x64 PNG of each step,
  palette-coloured with grid lines and coordinate labels. Read
  via `Read ../frames/step_NNNN.png` (Claude Code Read ingests
  images). A first-class visual for interpretation: use it to see
  what is happening, then slice the state/ASCII with Python for the
  exact values. Reach for it whenever shape or colour gestalt matters:
    - You can't tell from the ASCII whether two patterns match
      by symmetry or rotation (HUD encoding puzzles).
    - You want to verify a hypothesis about visual structure
      (corridor, barrier between rooms).
    - You suspect a UI element you've ignored (move-budget bar at
      bottom, score indicator).
  tools/view_sprite.py complements it for the exact pixel content
  inside a single sprite's box.
