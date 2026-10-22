Team BHopper, submission v6. Hierarchical pooled adaptation per benchmark (logit p = benchmark + system + item + item_features
groups), MAP-fitted on all revealed labels of the item's benchmark; item prior from a hashed TF-IDF ridge, system prior from
name-matched public strength (subject_prior.json). Budget 0: 0.5. v6 re-calibrates the benchmark-level prior (TAU_B 1.4) to the
hidden benchmarks' accuracy spread, adds item_features group effects, accepts soft labels, and never raises (falls back to a
smoothed pair mean, then 0.5). numpy + standard library only, deterministic.
