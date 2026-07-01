**game_engine.py** -- synthesised world model. Importable.
  `transition_function(frame, action_id) -> next_frame`,
  `reward_function(frame, action_id, next_frame) -> (r, done)`,
  optional `planner(frame, available_actions=None, max_depth=None)`.
  A fallible hint fit to observed transitions only; the real env is
  ground truth. Consult it, but verify against what actually happens.
  When C3 is gated on, the engine runs/validates this planner itself and
  falls back to you on no-plan or mismatch; see synth_status.json.planner.
