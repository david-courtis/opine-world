**game_engine.py** -- synthesised world model. Importable.
  Functions: `transition_function(state, action_id) -> state'`,
             `reward_function(state, action_id, new_state) -> (r, done)`,
             optional `planner(state, available_actions=None, max_depth=None)`.
  A fallible hint fit to observed transitions only; the real env is
  ground truth. Consult/simulate, but verify against what happens:
    ```
    python -c "
    import json, pickle, copy
    from game_engine import transition_function, reward_function
    s = json.load(open('current_state.json'))['state']
    s2 = transition_function(copy.deepcopy(s), 3)
    print(reward_function(s, 3, s2))
    "
    ```
  When C3 is gated on, the engine runs/validates this planner itself and
  falls back to you on no-plan or mismatch; see synth_status.json.planner.
