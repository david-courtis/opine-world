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
