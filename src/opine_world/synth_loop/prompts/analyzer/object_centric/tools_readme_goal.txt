**goal_requirements.json** -- the engine's grounded goal hypothesis,
  refreshed after each level advance and each new reward. Fields:
    goal_hypothesis    one-sentence level-agnostic statement of the win
                       condition, distilled from EVERY observed reward
                       (including mid-animation reward-tick states)
    requirements       executable predicates over the object state; each
                       `accepted` one held at every reward moment observed
                       so far (necessity-checked); `rejected_necessity`
                       ones failed at some reward and are disproven
    injections         accepted requirements not yet satisfied on the
                       current level -- open coverage holes; these goal
                       modes keep eta high for their objects in sigma.json
  A compact version is injected into your prompt as GROUNDED GOAL
  HYPOTHESIS. It is a necessity-checked prior, not ground truth:
  cross-check your own goal reasoning against it before inventing a new
  rule, run the predicates yourself against candidate board states, and
  record any real-environment contradiction in world_model.md so the next
  grounding round revises it.
