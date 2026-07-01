**tools/escape_sequence.py** -- Lévy-flight self-avoiding walk
  generator. Call when you diagnose yourself as stuck:
    ```
    python tools/escape_sequence.py --actions %%ACTIONS_STR_SPACES%% --length 50
    ```

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
