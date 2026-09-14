# Pregame data integrity

Earlier models included same-game contact statistics during training. Their
reported test metrics are invalid for pregame prediction. Existing boards remain
historical artifacts and must not be relabeled as validated pregame forecasts.

Training now uses an explicit 29-feature allowlist of prior-day rolling statistics
and derived changes. Scoring uses the same construction on a target-day placeholder,
including all available earlier games. Doubleheader games share prior-day features.
Provider responses are filtered to each requested chunk; scoring independently
filters out the target day and later data. Old feature manifests are rejected.
Retraining is mandatory.

The default board includes only games not yet started, with both complete announced
lineups and identified probable starters. It is an as-of snapshot, not a guarantee
against later scratches or starter changes. Coverage metadata records omitted games.
Use --allow-projected explicitly for projected or reconstructed boards; this does
not establish that historical lineup information was available pregame.

Historical observed weather is no longer substituted for a pregame forecast.
Unavailable historical forecasts use neutral weather. Park factors remain the
existing static table; their historical publication dates have not been validated.
Historical reconstruction is therefore not a fully point-in-time backtest yet.

V1 and V2 retain their agreed weights and shared scorer. Venom Analytics inspires
the transparent pitch-usage-weighted matchup and contact-quality breakdown:
https://venomanalytics.io/
Its proprietary formulas are not replicated, and no performance claims are adopted.

The validation workflow runs regression tests and retrains an isolated model
using data through September 3. It does not overwrite production boards.
