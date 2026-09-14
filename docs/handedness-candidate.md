# V1 handedness candidate

Corrects pooled handedness and unregularized pitch rates. Uses only history strictly before each target date. Infers batting side from prior appearances against the opposing pitcher's throwing hand. Unknown hand history stays unavailable. Pitch mix is conditioned on batter side and smoothed toward overall pitcher usage. Rates shrink toward same-hand/pitch league rates with 200 prior pitches; contact rates use 30 prior measured contacts. These starting constants have not been tuned or shown to improve results.

V1 and V2 composite weights remain unchanged. Saved boards are untouched. This branch is a candidate, not a validated promotion. September 13's 6/50 alone does not identify optimal weights. Park-factor provenance and roof-aware weather remain unresolved; this change does not claim to fix them. Full chronological multi-date performance validation is still required before replacing production.
