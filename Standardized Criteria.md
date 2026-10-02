# Standardized Behavioral Classification and Reaction-Time Criteria in Crickets

This document specifies the response classification rules and reaction-time (RT) measurement framework in Cercus.

The criteria below reflect the operational analysis threshold of **50 mm/s** (the package default is 98 mm/s; any configured burst threshold automatically updates these criteria). For trials containing airflow stimuli, the evaluation window spans the first **250 ms** post-wind onset. Visual-only baseline trials are evaluated up to theoretical collision (Time to Collision, TTC).

---

## 1. Classification Hierarchy and Precedence

To eliminate classification ambiguity, trials are evaluated through a single-direction priority cascade:

$$\text{NoResponse} \longrightarrow \text{PreEscape} \longrightarrow \text{PreWalk} \longrightarrow \text{Escape}$$

```
                          ┌── [No qualifying burst] ──────────────────────────► NoResponse
                          │
[Response-window burst?]  ├── [Burst qualified ∧ onset < stimulus arrival] ───► PreEscape
                          │
                          ├── [Burst qualified ∧ active walking before wind] ──► PreWalk
                          │
                          └── [Remaining qualified trials] ────────────────────► Escape
```

1. **NoResponse**: Fails to meet the burst threshold within the post-stimulus window.
2. **PreEscape**: Qualifies for a burst, but movement onset occurred prior to stimulus arrival.
3. **PreWalk**: Qualifies for a burst, did not start early, and was actively locomoting prior to stimulus arrival.
4. **Escape**: Default response category for crickets that initiated escape from an initial stationary state after stimulus onset.

---

## 2. Category Definitions

### 2.1 Escape

#### Airflow-containing trials (Pure-wind & Multimodal)
- **Burst Criterion**: Peak walking speed exceeds 50 mm/s within the 250-ms post-wind window.
- **Exclusion**: Does not meet either PreEscape or PreWalk criteria.
- **Interpretation**: Represents escapes initiated from immobility triggered by the stimulus. Trials with unobserved pre-stimulus history but an unambiguous post-stimulus burst and onset also fall into this category.

#### Visual-only baseline trials (`baseline_visual`)
- **Burst Criterion**: Peak walking speed exceeds 50 mm/s between trial onset and theoretical collision ($t \le \text{TTC}$).
- **Exclusion**: Walking speed exceeds 10 mm/s in no more than 15% of valid frames in the window from 1000 ms to 50 ms before escape onset.
- **Window**: The 250-ms airflow window and PreEscape category do not apply to visual baselines.

---

### 2.2 PreEscape

- **Scope**: Applied to all wind-containing trials when enabled by `classification.use_preescape: true`.
- **Timing Logic**: The qualifying burst peak must remain within the post-stimulus window ($[t_w, t_w + 250\text{ ms}]$). However, the backward onset search is permitted to cross before wind onset. If the detected onset occurs before airflow arrival ($t_e < t_w - \text{buffer}$, default buffer = 0 ms), the trial is classified as PreEscape.
- **Biological Context**: Demonstrates that escape acceleration began before airflow arrived. In multisensory trials, this reflects escapes triggered by the preceding visual looming stimulus. In pure-wind trials, it reflects spontaneous pre-stimulus acceleration.
- **Priority**: When a trial meets both PreEscape and PreWalk requirements, PreEscape takes precedence.

---

### 2.3 PreWalk

#### Airflow-containing trials
- **Burst Criterion**: Peak speed exceeds 50 mm/s within 250 ms after wind onset.
- **Exclusion**: Does not qualify as PreEscape.
- **Locomotor Eligibility (`pause_moving_eligible`)**:
  1. The 1000-ms pre-stimulus period on the hardware acquisition clock (`ard_time`) is fully observable without unbridgeable data loss.
  2. The duration-weighted occupancy of causal 20-ms displacement-averaged speed exceeding 10 mm/s is at least 15% (`prewalk.min_moving_fraction`, default 0.15).
  3. The causal averaged speed at the instant of stimulus arrival strictly exceeds 10 mm/s.
- **Interpretation**: Identifies crickets already in motion when the stimulus arrives. It does not require a successfully detected stopping endpoint; qualifying movers remain PreWalk even if subsequent stopping cannot be resolved.

#### Visual-only baseline trials
- Evaluated on the legacy window (1000 ms to 50 ms before escape onset), requiring >15% of frames above 10 mm/s.

---

### 2.4 NoResponse

- **Criterion**: Maximum walking speed remains $\le 50\text{ mm/s}$ throughout the evaluation window (250 ms post-wind, or up to TTC for visual baselines).
- **Veto**: The absence of a burst acts as an absolute veto. Pre-stimulus movement cannot override a lack of post-stimulus burst.

---

## 3. Reaction Time and Kinematic Endpoints

The pipeline strictly separates **escape movement onset** from **stimulus-induced stopping**.

### 3.1 Escape Latency (`escape_reaction_time_ms`)

- **Definition**: The interval from the calibrated airflow reference to escape movement onset: $t_e - t_a$.
- **Onset Detection**: Backward search from the qualifying burst peak to the last speed sample $\le 10\text{ mm/s}$; the subsequent frame marks $t_e$.
- **Values**:
  - Positive for post-stimulus `Escape`.
  - Negative for `PreEscape`, quantifying pre-stimulus lead time without artificial truncation.
  - Set to `NaN` if speed never drops below 10 mm/s prior to the burst (unresolved onset).

### 3.2 Causal Stopping and Re-escape (T1, T2, RTm)

For PreWalk trials, stopping and subsequent re-escape are evaluated using causal 20-ms displacement averages on the source acquisition clock (`ard_time`):

1. **Stopping Latency T1 (`pause_stopping_time_ms`)**:
   - The latency from airflow arrival to the first drop below the 10 mm/s quiet threshold.
   - Guarded by optical count-resolution margins (requiring initial speed $>13.33\text{ mm/s}$ and confirming the drop below $<6.67\text{ mm/s}$) to filter out sensor quantization noise.
2. **Pause-to-Escape Interval T2 (`pause_to_escape_time_ms`)**:
   - The interval from confirmed stopping to the subsequent escape onset.
3. **Causal Composite RTm (`pause_reaction_time_ms`)**:
   - Calculated as $\text{RTm} = \text{T1} + \text{T2}$.
   - **Handling Missing Values**: If stopping or subsequent escape cannot be resolved, T2 and RTm remain `NaN`. The pipeline never imputes missing causal endpoints with centered trajectory estimates.

### 3.3 Cohort and Reporting Principles

- **Classifier Authority**: Population summaries and cross-paradigm comparisons adhere strictly to the final classifier output (`response_type`).
- **Sample Integrity**: Missing reaction-time endpoints do not alter cohort sizes; trials with unresolved RT remain in their respective response class, and counts are reported alongside coverage statistics.
- **Escape Distance**: Distance integration (`distance_mm` and `distance_500ms_mm`) begins at movement onset ($t_e$), not at the stopping point.
