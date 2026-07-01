ARC-AGI-3 LEVELS ARE NON-TRIVIAL HAND-DESIGNED CHALLENGES. Each level rewards completion via the env returning a positive reward signal at some point. The condition that triggers that reward is UNKNOWN; you must hypothesize and refine it from observation.

DO NOT ASSUME ANYTHING ABOUT THE DOMAIN. You don't know if this is a gridworld. You don't know if there's a single controllable 'player'. You don't know which objects (if any) the actions move. You don't know what shape the goal takes. Even the action semantics are not guaranteed. Treat ALL of this as discoverable.

ACTION VOCABULARY (universal): action_id 0 = RESET, 1-4 = up, down, left, right (in order), 5 = space, 6 = click(x, y). Per-game semantics vary; discover by probing. RESET (0) restarts the current level, USE IT when you've made an irreversible mistake or concluded the current attempt is unsolvable, rather than wasting your action budget.

EVERY OBJECT HAS A PURPOSE. The level is hand-designed; objects in it are almost never decorative. If you haven't established what an object does, that is an UNRESOLVED HYPOTHESIS, probe it.

OBJECTS INFLUENCE ONE ANOTHER, AND INFLUENCE HAS PRECONDITIONS. The reward predicate usually requires multiple preconditions to be jointly satisfied. The synthesised reward_function MUST encode the joint precondition-+-configuration that constitutes 'goal reached', not a single object's position.