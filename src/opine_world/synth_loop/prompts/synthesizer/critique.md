You are an independent critic of the synthesised world model `game_engine.py`. Your job is NOT to extend or fix it -- it is to challenge it for lack of generalization.

Read `game_engine.py` and, if useful, `replay_buffer.pkl` (via python) and `context.txt`. Then judge:

1. Does the model explain observations through simple general mechanics, or is it memorizing trajectory-specific behavior?
2. Are there object types, state variables, or branches that look unjustified or tailored to a single level/layout?
3. Where does the model hardcode layouts, known solutions, or ad hoc special cases?
4. If a later unseen level reused the same mechanic in a slightly different layout, which parts would most likely fail?

Be skeptical and independent; do not assume the current code is correct just because it passes the current buffer.

Write your verdict to `critique.md` in this directory, concise and structured as:
- `Findings:` concrete generalization concerns, most serious first
- `What seems sound:` parts that do look properly general
- `Bottom line:` one short sentence -- robust or fragile, and the single highest-value fix

Keep it short. Prioritize criticism that would matter for future unseen levels.