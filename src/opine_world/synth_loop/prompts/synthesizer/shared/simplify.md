*** SIMPLIFICATION PASS (this round) ***
The carried-forward model is likely explaining the observations with machinery
that is too specific and too close to the exact seen trajectories. Assume by
default that the real mechanics are simpler than the current code.

This round, prioritise compressing the model over extending it:
- find where the model fits observations too literally instead of a small
  underlying rule, and replace case-by-case behaviour with shared parameterised
  rules;
- remove object types, state fields, and distinctions not forced by evidence;
- merge equivalent classes; push any genuinely level-specific facts into compact
  data, not engine logic.
- if this is a frames-only model, keep `extract_objects(frame)` aligned with
  the simplified internal object representation.
- remove temporary cache/mask/render exceptions whenever a shared object,
  dynamic state variable, visibility effect, or per-level parameter can explain
  them mechanistically.
- keep `synth_learnings.md` honest: list any remaining ad-hoc visual detail,
  level-specific branch, duplicated object family, cache dependence, or
  unresolved competing hypothesis as modeling debt for future rounds.

The core mechanics should be expressible in a few short sentences. Keep the
simplest model that still passes the verifier -- run the tests after each change
and do not regress accuracy.
*** END SIMPLIFICATION PASS ***
