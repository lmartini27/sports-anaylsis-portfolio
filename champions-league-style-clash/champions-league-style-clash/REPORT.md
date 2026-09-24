# Champions League Clash of Styles — Methodology & Results

> Fill in the [Results](#results) and [Discussion](#discussion) sections
> with your actual numbers once you've run the pipeline — everything
> below is scaffolded to match how the fantasy football and March
> Madness reports are structured, so keep the tone the same: plain
> about what the model does and doesn't show.

## 1. Motivation

Most match-prediction models reduce a team to a single strength number
(Elo, xG differential, etc.) and predict outcomes from the gap between
two strength numbers. That's a reasonable baseline, but it throws away
something coaches and pundits talk about constantly: *stylistic*
matchups. A possession-heavy side that dominates the ball domestically
can come unstuck against a team that presses aggressively and forces
turnovers in dangerous areas — regardless of the "quality gap" between
them. The Champions League is the best natural experiment for this,
because it's the one competition where clubs shaped by five different
domestic tactical cultures are guaranteed to meet.

This project asks a narrower, more answerable question than "who wins":
**given two teams' playing styles, is there a systematic edge for one
style over another, independent of which specific clubs are involved?**

## 2. Data

- **Source:** FBref advanced squad stats, pulled via the `soccerdata`
  Python package (no API key required, mirrors public FBref tables).
- **Domestic leagues:** Premier League, La Liga, Serie A, Bundesliga,
  Ligue 1 — the leagues that supply the large majority of CL
  participants.
- **Seasons:** `config.SEASONS` (three seasons by default — widen this
  once the pipeline runs cleanly, since more seasons = more matchup
  data per style pair).
- **Match results:** FBref's Champions League schedule/results table
  for the same seasons.

**Sample size caveat, stated up front:** even across three seasons,
the number of CL matches that land in any *specific* style-pair cell
(e.g. "High Press home vs. Possession Control away") can be small —
sometimes single digits. Every percentage in the win-rate matrix should
be read next to its sample size, not on its own. This is flagged again
in [Limitations](#5-limitations).

## 3. Style Features & Clustering

Each team-season is described by a small set of features chosen to
capture *how* a team plays rather than *how well*:

| Feature | What it captures |
|---|---|
| `possession_pct` | ball-dominance tendency |
| `pass_completion_pct` | control vs. risk in possession |
| `progressive_passes_per90` | how much ground is gained via passing |
| `passes_into_final_third_per90` | penetration through passing |
| `press_height_pct` | proxy for how high up the pitch defensive actions happen (see note below) |
| `shots_per90` | attacking output/volume |
| `touches_att_pen_per90` | direct penetration into the box |

Features are standardized (z-scored) and clustered with **k-means**
(`config.N_STYLE_CLUSTERS`, default 4) into style archetypes. The
number of clusters isn't derived from first principles — it's set to
match commonly-discussed tactical buckets (e.g. high press,
possession-control, direct/counter, low-block) and should be checked
against the elbow-method output that `03_style_clustering.py` prints
before trusting it.

**On `press_height_pct`:** true pressing intensity is usually measured
as PPDA (opponent passes allowed per defensive action) computed from
match-level event data. The season-aggregate squad tables used here
don't expose opponent pass counts cleanly, so this project uses a proxy
— the share of a team's tackles+interceptions made in the middle/
attacking third rather than their own defensive third. It captures
*where* a team defends, which is most of what "high press" means for
style-clustering purposes, but it is not the same statistic as PPDA and
shouldn't be reported as one.

## 4. Matchup Dataset & Model

Each CL match is joined to both teams' style cluster from their most
recent domestic season, producing a table of
`home_style, away_style, result`. From this:

1. **Win-rate matrix** — for every (home style, away style) pair, the
   empirical share of home wins / draws / away wins. This is the
   project's core descriptive result.
2. **Classifier check** — a multinomial logistic regression predicting
   match result from `home_style` + `away_style` (one-hot encoded),
   compared against a "always predict the most common result" baseline.
   This is a deliberately modest model: it has no team-strength signal
   at all, so it isolates how much *style alone* (divorced from quality)
   predicts outcomes. Expect a small lift over baseline, not a large
   one — see [Discussion](#discussion).

## 5. Results

*(run `05_train_model.py` and `06_visualize_results.py`, then fill in)*

- Named style archetypes and their defining features:
  - Style 0 — `___`
  - Style 1 — `___`
  - Style 2 — `___`
  - Style 3 — `___`
- Style radar chart: `outputs/figures/style_radar_chart.png`
- Matchup heatmap: `outputs/figures/matchup_heatmap.png`
- Headline matchup finding(s): `___`
- Baseline accuracy vs. style-matchup model accuracy: `___` vs `___`
- Sample sizes behind the most interesting cells — call these out
  explicitly wherever a win rate looks extreme, since extreme
  percentages on tiny samples are the most likely thing to mislead a
  reader here.

## Discussion

*(fill in once you have results — some prompts to answer honestly:)*

- Which style matchup(s) showed the clearest edge, and does the sample
  size behind that edge actually support the claim?
- Did the classifier beat the baseline by a meaningful margin, or was
  the lift small? Either answer is a legitimate finding — a small lift
  says style matters at the margins, not that it dominates outcomes.
- Do the style-cluster labels look tactically sensible when you
  spot-check a few known clubs, or does the clustering need more/fewer
  clusters or different features?

## 5. Limitations

- **Small per-cell samples.** Some style-pair cells will have very few
  matches; treat any single extreme percentage with real skepticism.
- **No team-strength control.** This project deliberately isolates
  style from quality, but that also means the win-rate matrix partly
  reflects *which clubs happen to fall into which style bucket*, not
  a pure style effect. A team that's simply the best team in the
  competition will look good in whatever style cluster it lands in.
  A natural extension: control for strength (e.g. add an Elo or
  goal-difference rating) and check whether the style effect survives.
- **`press_height_pct` is a proxy, not real PPDA** — see Section 3.
- **Style is not static.** A club's style can change mid-season (new
  manager, injuries) or evolve year to year; using one seasonal
  average per team-season smooths over that.
- **Squad-level stats, not CL-specific stats.** Style features are
  drawn from each team's *domestic* league performance, on the
  assumption that broad style is stable across competitions — teams
  don't usually play a completely different way in Europe, but some do
  adjust tactically for tougher opposition, and that adjustment isn't
  captured here.
- **Name-matching between competitions.** FBref occasionally spells a
  club differently on its CL page vs. its domestic league page (see
  `TEAM_NAME_FIXES` in `04_build_matchup_dataset.py`); unmatched teams
  get dropped rather than silently mismatched, which shrinks the
  usable sample somewhat.
