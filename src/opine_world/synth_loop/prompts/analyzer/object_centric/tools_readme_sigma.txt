**sigma.json** -- per-object epistemic substrate Σ(o); one row per sprite,
  sorted by eta descending. Row fields:
    eta   [0,1] knowledge-uncertainty over the object's behavior classes.
          1 = nothing established. Unexercised classes sit at the prior
          (U=1), so an untouched object reads UNKNOWN even if it has been
          idly present for hundreds of steps.
    C     prequential forward accuracy of the synthesized model on THIS
          object (recency-weighted; null until the model has scored an
          engaged prediction). C=1 fully predicted; C=0 chronically wrong.
    C_n   effective sample weight behind C (small = weak evidence).
    C_fields  C split per record family (x, y, visible, rotation, pixels,
          existence): WHICH update rule is failing, not just which object.
          Low C with only C_fields.pixels low means the model's pixel
          rule for this object is the broken piece.
    LP    |change in C| over the object's recent engaged encounters:
          learning progress. >0 means still being learned or recently
          broken by a model revision; 0 with high C means mastered.
    cov   fraction of classes mastered.   nu: novelty floor (1+n)^-1/2.
    CAI   action-influence in [0,1]; ~0 means the object's changes do not
          depend on your action -- inert OR untested-in-the-right-context.
    rel   goal-relevance prior in [0,1] (reward adjacency + role votes).
    Pi    mechanical priority (rel*LP + omega*nu)/cost -- a fallback
          ordering, not a substitute for your own judgment.
    classes  per-class table: key (an action cell like `a3` / `click_on`,
          a `mode|...` configuration, or a model-fluent bin like
          `a6|model:governors=('exact', 8)`), provenance (backbone /
          observed / pooled / goal / model), n engaged evidence,
          n_pred/k_pred forward predictions scored/correct, k_att failed
          probes (the action fired adjacent to this object with zero
          board-wide effect; decays the untried cell's weight,
          reversibly), U class uncertainty in [0,1]. Some rows carry:
            resolved_by: "model"  effects in this cell are mixed but the
              model forward-predicts it exactly: a representational
              confound (the cell key cannot express the model's own
              fingerprint), NOT an unknown. Do not spend probes here.
            refined_by + w=0  the cell was split by a declared model
              fluent; the `model` bins beside it carry its accounting.
            tag_level: true  on `model` bins: evidence pooled across all
              same-tag instances.
    eta_structural  (when present) share of eta resolved only by the
          model's own fingerprint: representational debt, not ignorance.
  Reading discipline:
  - When goal progress has stalled, unexercised classes (n=0) of high-rel
    objects are your probe targets.
  - A `model` bin with seen=false is the synthesized model's OWN guard
    branch that no observation has ever tested (a forbid-requirement
    never exercised under a click): the most information-dense probe
    available -- it either confirms the model's rule or yields a
    counterexample.
  - A `pooled` mode with seen=false means ANOTHER instance of this tag
    exhibited a configuration this object never has -- strong hint that a
    reachable state of this object remains unvisited (a door that has
    never opened here, but opens elsewhere).
  - A `goal` mode names a configuration the goal hypothesis requires but
    no observed state has ever satisfied -- a localized coverage hole.
  - High cov with CAI~0 means inert-or-untested: worth ONE deliberate
    probe, not a conclusion.
  - Trust C where C_n is nontrivial. eta and C answer different
    questions: what the data has established vs what the model predicts.

  Value and limits, so you neither ignore nor over-trust this file: the
  tracking is mechanical, so its evidence counts are facts (an n=0 here
  cannot be hallucinated and is hard to reconstruct from the log
  yourself). But it only expresses situations its vocabulary covers, so
  real mechanics can be invisible to it, and its classes deliberately
  over-approximate, so some listed unknowns are spurious. Consult it
  when struggling or when falsifying a hypothesis (did I neglect an
  important state?); challenge it when a hypothesis built on it keeps
  failing.

**model_fluents.json** -- (fluent-harvest runs, when present) the
  synthesized model's declared predicates and relations, evaluated over
  the whole buffer. Per fluent: status (`admitted` = its value bins
  deterministically explain a previously-mixed cell and appear as
  `model` classes in sigma.json; `rejected`/`invalid` = kept OUT of the
  accounting), the discovered applies-to set, per-tag value codomain,
  and `untested` -- values the fluent produces on observed states that
  were never exercised under any action. Rejected entries are the
  model's UNVERIFIED self-description: their `untested` values are
  cheap, high-yield probe hints (each one tests a guard branch of the
  model's own code, yielding confirmation or a counterexample), but
  they carry no weight in eta.
    ```
    python -c "
    import json
    s = json.load(open('sigma.json'))
    for r in s['objects'][:8]:
        print(r['name'], r['tag'], 'eta=', r['eta'], 'C=', r['C'],
              'LP=', r['LP'], 'CAI=', r['CAI'], 'Pi=', r['Pi'])
    "
    ```
