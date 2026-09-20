**label_audit.json** -- mechanical audit of the role labels against the
  buffer, refreshed on a cadence. Graded by evidence strength:
    refuted             a decorative label whose inertness commitment was
                        falsified by a cited transition. Already retired
                        automatically. Do not re-propose it without
                        explaining the witness (see label_constraints.jsonl
                        in the run dir for history).
    contradictions      two tags sharing a label behave differently in the
                        same (action, context) stratum, both deterministic
                        at exact-delta granularity. Proves EITHER a mislabel
                        OR a missing context feature. Dispose by relabeling
                        one tag via alias_updates.json or by describing the
                        distinguishing condition for the synthesizer.
    anomalies           correlational flags (role posterior disagreement,
                        controllability mismatch). Investigate before
                        trusting.
    unlabeled_relevant  tags with reward-adjacent or action-influence
                        evidence and no committed role. Propose labels.
    merge_compatible    differently-labeled tag pairs indistinguishable so
                        far, with sample support. The weakest grade.

