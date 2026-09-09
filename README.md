# Daily Home Run Prediction Engine

A leakage-safe MLB home-run probability pipeline. The project is designed to turn historical Statcast, schedule, pitcher, lineup, park, and weather data into daily batter-level probabilities for hitting at least one home run.

## Design goals

- Build every feature using information available before the target game.
- Prefer pitch-level Statcast data for contact quality and pitch-mix features.
- Preserve raw data separately from feature tables.
- Use rolling windows (including last 20 games) without future leakage.
- Validate chronologically, never with a random split.
- Calibrate probabilities so a 15% prediction means roughly 15 HRs per 100 comparable opportunities.
- Keep a simple baseline alongside the production model so model improvements can be measured.

## Planned pipeline

`raw data -> normalized tables -> point-in-time features -> training set -> time-series validation -> calibrated model -> daily predictions`

## Initial feature families

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

Baseball Savant Statcast is the primary source for pitch-level contact and pitch characteristics. MLB's schedule/probable-pitcher data supplies game context and starting-pitcher information. Source adapters should remain isolated so providers can be replaced without changing model code.

## Status

Phase 1: repository scaffold and leakage-safe feature architecture.

Next: implement data ingestion adapters and build the first historical training dataset.
