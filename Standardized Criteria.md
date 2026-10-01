# Standardized Criteria for Escape, PreEscape, PreWalk, and No Response in Crickets

The criteria below use the current analysis threshold of 50 mm/s. The package default is 98 mm/s; when a different burst threshold is selected, it replaces 50 mm/s throughout these criteria. The response window for airflow-containing trials is 250 ms after airflow onset. Visual-only baseline trials are evaluated up to theoretical collision instead.

## Escape

### Airflow-containing trials

- The maximum walking speed was > 50 mm/s during the 250-ms period after airflow stimulus onset.
- The trial did not meet the PreEscape or PreWalk criteria. Trials with a qualifying burst were classified as Escape only after both categories had been excluded.
- In particular, walking speed exceeded 10 mm/s in no more than 15% of valid frames during the period from 1000 ms before to 50 ms before airflow onset, when this period contained valid observations.

### Visual-only baseline trials

- The maximum walking speed was > 50 mm/s during the recorded period up to and including theoretical collision.
- The trial did not meet the PreWalk criteria: walking speed exceeded 10 mm/s in no more than 15% of valid frames during the period from 1000 ms before to 50 ms before the measured escape onset, when this period contained valid observations.
- The 250-ms post-airflow window and the PreEscape category did not apply.

## PreEscape

- This category applied only to trials combining looming and airflow stimulation.
- The maximum walking speed was > 50 mm/s during the 250-ms period after airflow stimulus onset.
- The measured escape response began before airflow was applied. With the current zero timing buffer, an onset exactly at airflow onset was not considered PreEscape.
- These trials were excluded from strict Escape because the measured escape movement had already begun before airflow arrived. This timing criterion alone did not establish that the visual stimulus caused the movement.
- Trials meeting both PreEscape and PreWalk criteria were classified as PreEscape.

## PreWalk

### Airflow-containing trials

- The maximum walking speed was > 50 mm/s during the 250-ms period after airflow stimulus onset.
- Walking speed exceeded 10 mm/s in more than 15% of valid frames during the period from 1000 ms before to 50 ms before airflow onset.
- The trial did not meet the PreEscape criteria.
- Trials in which a qualifying burst followed this pre-stimulus walking were excluded from strict Escape.

### Visual-only baseline trials

- The maximum walking speed was > 50 mm/s during the recorded period up to and including theoretical collision.
- Walking speed exceeded 10 mm/s in more than 15% of valid frames during the period from 1000 ms before to 50 ms before the measured escape onset.
- These trials were excluded from strict Escape; no 250-ms post-airflow requirement applied.

## No Response

### Airflow-containing trials

- The maximum walking speed was ≤ 50 mm/s during the 250-ms period after airflow stimulus onset.
- Trials failing to exceed this threshold in that period were classified as No Response, regardless of their pre-stimulus movement. A burst occurring only outside the response window did not satisfy the response criterion.

### Visual-only baseline trials

- The maximum walking speed was ≤ 50 mm/s during the recorded period up to and including theoretical collision.
- Trials failing to exceed this threshold were classified as No Response, regardless of preceding movement. Movement occurring only after theoretical collision did not satisfy the response criterion.

## Response measurement and interpretation

- A qualifying burst was established first. Trials were then classified in the order No Response, PreEscape, PreWalk, and Escape; PreEscape was considered only for eligible multisensory trials.
- For pure-airflow trials, the measured escape onset was the sample following the last speed below 10 mm/s before a qualifying post-airflow burst, without reusing a pre-airflow walking onset. Ongoing movement was skipped when a later qualifying burst followed a low-speed interval. Multisensory trials allowed pre-airflow onset measurement for PreEscape. The 250-ms window limited burst detection, not the duration of the entire movement. A qualifying burst with an unresolved onset retained its response class but had missing onset metrics.
- For visual-only baseline trials, onset measurement was referenced to the maximum-speed peak up to theoretical collision rather than to the first post-airflow threshold crossing.
- A separate early movement that subsided below 10 mm/s before a later burst was not necessarily selected as the measured escape onset. Such movement could remain visible in a full-trial heatmap without meeting the current PreEscape check.
- Missing observations were not evidence of immobility. The current analysis assigned No Response when no valid response-window speeds were available, and did not assign PreWalk on the pre-walking check alone when that period lacked valid frames.

### Stopping latency versus escape latency

- Escape latency measured the interval from airflow arrival to escape **movement** onset and was exported as `escape_reaction_time_ms`.
- In airflow PreWalk trials, the first below-threshold causal 20-ms displacement-average crossing was measured only when the cricket was demonstrably moving at the airflow reference. The configured primary-axis displacement quantum was about 0.047 mm per count, observed in the audited recordings and consistent with the local recorder calibration, not a universal sensor datasheet value. Its physical calibration provenance and the recording versions were not independently established. The average at airflow onset had to exceed 10 mm/s by a diagonal-count resolution margin. A detected below-threshold excursion was accepted only when it moved below that margin before returning to at least 10 mm/s; an unresolved dip was not replaced by a later crossing. At the defaults, motion qualification required >13.33 mm/s and confirmation required <6.67 mm/s, while the endpoint remained the first <10 mm/s crossing. These local gates were stricter than the paper's threshold alone. This safeguard was specific to the optical measurement and was not a calibrated speed-error bound or a biological response-time floor.
- The full preceding 1000-ms history was separately labeled `continuous_moving` when every valid causal averaged-speed observation was at least 10 mm/s, or `intermittent_moving` otherwise. A measurable endpoint in an intermittent trial was a local stopping estimate, not a verified paper-cohort T1 measurement. These flags described observed averaged speed; near-threshold quantization could break a genuinely continuous walk, and the paper did not specify how its >1-s criterion handled averaging or intermittent counts. Neither flag proved paper-cohort membership. Neither broad PreWalk activity nor earlier intermittent walking alone established motion at the stimulus reference.
- Acquisition time (`ard_time`, firmware milliseconds), rather than serial receipt time (`sys_time`), determined interval velocities and stopping timestamps. The full 20-ms averaging interval had to be wholly post-reference, limiting eligible endpoints to at least one averaging window after the reference rather than establishing a biological minimum. A boundary-straddling first crossing was unresolved and not replaced by a later crossing. When escape onset was observed, its row was mapped to the acquisition clock and stopping had to precede it; an escape onset at or before the reference yielded `escape_first`. Unresolved escape onset did not itself prevent stopping measurement.
- Finite, ordered acquisition intervals without excessive gaps were required throughout the preceding-state and stopping history. Invalid or insufficient observations yielded `NaN` with `stillness_status`; missing later observations did not erase an already measured event. Missing source timestamps were not replaced with centered-smoothed speed. Response classification and movement-distance integration were unchanged.
- `stillness_reaction_time_ms` and compatibility `reaction_time_ms` represented the measured averaged-speed crossing endpoint minus the stimulus reference, not an exact causal physiological reaction instant. Averaging imposed speed-dependent detection lag; no fixed delay subtraction was justified. The preceding 20-ms measurement support was not a confidence interval or a validated bound on true reaction time. An `observed` status established measurement eligibility, not stimulus causation.
- `stillness_reaction_time_ms` is T1-like. The separate `stop_to_escape_interval_ms` is T2-like: mapped escape movement onset minus the stopping crossing, with both endpoints on `t_acquisition_rel`; it remains `NaN` when either endpoint is unresolved or the intervening source timestamps are invalid. It is not a claim that the animal remained continuously immobile and does not reproduce the paper's later escape window.
- `stillness_presence` separately records whether a causal averaged low-speed observation was present after the reference and before escape (or the 250-ms limit): `low_speed`, `no_low_speed`, or `unobserved`. Already-stopped animals can have `low_speed` without a stopping latency. Missing or unresolved RT must not be counted as absence of stillness. The plot reports all three categories, separately from RT coverage.
- The paper's Figure 1D pause incidence of 88.75% applied to its moving cohort under 1.00 m/s air currents, including trials without subsequent escape. It is not the fraction of broad PreWalk trials with measurable stopping latency. Its T2 value, 87.16 ± 65.31 ms, describes stopping-to-response time, not T1 or pause incidence.
- The stopping threshold, causal velocity window, earlier period, gap tolerance, count-resolution margin, and measured hardware-to-airflow delay were configurable. A zero arrival-delay setting remained stimulus-flag-relative, not calibrated airflow-arrival-relative. This did not reproduce the paper's full preprocessing, trial selection, post-stopping escape window, or T1/T2/RTm analysis.
- Escape latency below 50 ms remained marked `short_rt` without clamping. Its existing centered-speed onset measurement was unchanged and required separate timing validation; a stopping correction did not resolve short Escape values. Negative PreEscape timing remained a lead, not a negative reaction.
- A separate `pause_*` measurement runs on every airflow-containing trial without changing these classifications or distance intervals. It exports causal source-clock `pause_stopping_time_ms` (T1-like), `pause_to_escape_time_ms` (T2-like), and `pause_reaction_time_ms` (RTm-like). The first stopping excursion is sought within the stimulus-relative 250-ms earlier period, bounded before any observed escape or earlier qualifying burst to exclude escape deceleration. The later escape is sought within 250 ms of the stopping endpoint using the same causal speed basis and configured 10/50-mm/s criteria. Earlier boundary, count-resolution, and moving-history safeguards still apply; no fixed filter-delay correction is justified. This reproduces the two observation-window anchors, not the paper's undocumented preprocessing or verified moving cohort.
- Missing subsequent escape leaves causal T2 and RTm as `NaN` with `pause_escape_status`, retaining an already observed T1. Missing or corrupt source intervals cannot be bridged; later corruption cannot erase an already confirmed pair. Pauses without subsequent escape remain separately observable even in legacy No Response trials. For `pause_baseline_status`, the continuous-history proxy requires every valid causal observation over the preceding configured 1-s history, including the reference, to be strictly above 10 mm/s (roundoff-safe). Threshold equality or earlier low-speed interruptions are intermittent; local transitions remain measured but excluded from strict-moving summaries. Legacy `stillness_baseline_status` is unchanged. This is not validated paper cohort membership.
- The population RT figure shows stimulus-to-escape latency for PreWalk, not compatibility stopping RT. It prefers the separate causal RTm-like value for observed pauses, leaving unresolved endpoints missing rather than substituting centered-speed escape timing; RT coverage is annotated. Other trials retain their existing escape endpoints. Its T1/T2 panel defaults to the strict-moving proxy across all airflow response classes when causal fields exist; absent history cannot establish eligibility. A separately labeled `pause_cohort="all_local"` diagnostic view retains intermittent transitions. Both use complete trial pairs only, split by pure airflow and multisensory stimulation: boxes/dots summarize each subject, while faint lines with coincident endpoint markers connect actual trial pairs. Counts and legends stay outside the data rectangle. Paired coverage is not pause incidence, and a low-speed level in an already-stopped animal is not a stopping transition. Legacy tables retain an explicitly labeled PreWalk fallback. This remains in the existing worker-controlled population rendering task.
- Legacy `stillness_*` and new `pause_*` stopping use the same raw acquisition basis but different escape caps (mapped centered onset versus causal onset/earlier burst). Their status can differ; absence of a pre-escape stop is scoped to that method's cap, not evidence of no low-speed state throughout the trial.
- Escape distance continued to be integrated from movement onset, not from the start of stillness. Visual-only reaction-time definitions were unchanged by this airflow-specific correction.
