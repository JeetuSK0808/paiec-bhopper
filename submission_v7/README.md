Team BHopper, submission v7 = v6 + labeling.py. Prediction code (model.py, textprior.py, subjprior.py, subject_prior.json) is
byte-identical to v6: hierarchical pooled adaptation per benchmark (logit p = benchmark + system + item + item_features groups),
MAP-fitted on all revealed labels of the item's benchmark; item prior from a hashed TF-IDF ridge, system prior from name-matched public
strength. Budget 0: 0.5. New in v7: a stateless acquisition rule. A candidate item that another system of the same benchmark has already
revealed is skipped (it lies in that system's acquisition pool, so its label can never inform that system's scored items); otherwise
the evaluator's own random rule is reproduced exactly, and every candidate is taken when the 31-label allowance needs it. The rule
never looks at outcomes. numpy + standard library only, deterministic, never raises.
