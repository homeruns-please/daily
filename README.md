# Daily Home Run Prediction Engine

A leakage-safe MLB home-run probability pipeline. The project turns historical Statcast, schedule, pitcher, lineup, park, and weather data into daily batter-level probabilities for hitting at least one home run.

## Design goals

- Build every feature using information available before the target game.
- Prefer pitch-level Statcast data for contact quality and pitch-mix features.
- Preserve raw data separately from feature tables.
- Use rolling windows (including last 20 games) without future leakage.
- Validate chronologically, never with a random split.
- Calibrate probabilities so a 15% prediction means roughly 15 HRs per 100 comparable opportunities.
- Keep a simple baseline alongside the production model so model improvements can be measured.

## Implemented pipeline

`raw Statcast -> normalized pitch data -> batter-game table -> prior 5/10/20-game features -> time-based training/calibration/test`

The repository also contains MLB schedule/probable-pitcher ingestion, pitcher Statcast aggregations, pitch-mix calculations, batter-vs-pitch-type matchup features, and a Markdown Daily Home Run Board renderer.

### Build a historical dataset

```bash
daily-hr-build data/raw/statcast.csv data/processed/hr_training.parquet
```

The builder creates the HR target (`1+ HR in the game`) and leakage-safe prior-game features including barrel%, hard-hit%, fly-ball%, launch angle, exit velocity, plate-appearance volume, HR totals, and HR/PA over the prior 20 games.

### Train and evaluate

```bash
daily-hr-train data/processed/hr_training.parquet
```

Training uses chronological train/calibration/test blocks and evaluates ROC AUC, log loss, and Brier score. The production probability path uses isotonic calibration fitted only on a later time block.

## Feature families

### Batter
- Barrel%, Hard-Hit%, fly-ball%, pull%, launch angle, exit velocity
- HR/PA, ISO, xwOBA, xSLG, sweet-spot rate
- Rolling 7/14/20-game and season-to-date versions
- Platoon splits
- Recent plate-appearance volume

### Pitcher
- HR/PA, HR/9, barrel% allowed, hard-hit% allowed, fly-ball% allowed
- xwOBA/xSLG allowed and strikeout/walk indicators
- Pitch mix, velocity, spin, movement, release/extension
- Pitch-type-specific HR/contact damage allowed
- Rolling recent form and season-to-date versions

### Batter/pitcher interaction
- Batter performance versus each relevant pitch type
- Pitcher usage of those pitches against the batter's handedness
- Handedness matchup
- Expected pitch-type exposure
- Interaction features such as batter barrel rate against a pitch family multiplied by pitcher usage

### Environment/context
- Park and handedness-specific HR factors
- Temperature, wind speed/direction, humidity/air-density proxies
- Roof status where available
- Expected lineup slot and playing-time/PA opportunity
- Opposing bullpen quality after the starter exits

## Data sources

Baseball Savant Statcast is the primary source for pitch-level contact and pitch characteristics. MLB's schedule/probable-pitcher data supplies game context and starting-pitcher information. Source adapters remain isolated so providers can be replaced without changing model code.

## Status

**Working foundation:** ingestion adapters, leakage-safe historical feature construction, chronological calibration, pitch-type matchup engine, reporting, CLI commands, and automated tests are in the repo.

**Next production step:** add the point-in-time daily scorer that joins confirmed lineups, probable starters, weather, park context, and pitcher/batter matchup features, then emits the ranked Daily Home Run Board.
