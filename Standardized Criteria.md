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
- For airflow trials, the measured escape onset was the sample following the last speed below 10 mm/s before the first speed above 50 mm/s in the post-airflow response window. The measured onset could therefore precede airflow. The 250-ms window limited burst detection, not the duration of the entire movement.
- For visual-only baseline trials, onset measurement was referenced to the maximum-speed peak up to theoretical collision rather than to the first post-airflow threshold crossing.
- A separate early movement that subsided below 10 mm/s before a later burst was not necessarily selected as the measured escape onset. Such movement could remain visible in a full-trial heatmap without meeting the current PreEscape check.
- Missing observations were not evidence of immobility. The current analysis assigned No Response when no valid response-window speeds were available, and did not assign PreWalk on the pre-walking check alone when that period lacked valid frames.
