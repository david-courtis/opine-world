**model_fluents.json** -- (when present) the synthesized model's declared
  predicates and relations, evaluated over the whole buffer. Per fluent:
  status (`admitted` = its value bins deterministically explain a
  previously-mixed (type, action) cell; `rejected`/`invalid` = not
  supported by the buffer), the discovered applies-to set, and the
  per-tag value codomain. Each admitted cell lists its `bins`: value, n,
  modal_outcome, and U = 1 - modal_count/(n+1), with evidence pooled
  across all objects of the same tag. U near 0 means that value of the
  fluent is settled; a mixed cell that an admitted fluent splits into
  low-U bins is understood, not unknown. `holes` and `untested` list
  values the fluent produces on observed states that were never
  exercised under an action: each one tests a guard branch of the
  model's own code, so probing it yields confirmation or a
  counterexample. Rejected entries are the model's UNVERIFIED
  self-description: their `untested` values are cheap probe hints but
  carry no evidential weight.
