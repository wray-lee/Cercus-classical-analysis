# Decision: defer PreEscape subdivision

## Status

Deferred on 2026-09-29. This decision supersedes proposals to split PreEscape or add auxiliary PreEscape/PreWalk subtype flags. It does not authorize changes to the classifier.

## Research question and rationale

The multisensory-integration analysis is anchored to **wind onset**. Relative to that reference, subdividing PreEscape into responses preceded by walking versus responses without preceding walking adds little value to the current research question.

A measured escape onset after wind arrival does not establish an airflow-only trigger. Visual processing may already be underway, and airflow may contribute to rapid multisensory integration. The present onset-timing classification cannot disentangle those causal contributions and must not be presented as doing so.

## Decision

- Retain the four primary categories: NoResponse, PreEscape, PreWalk, and Escape.
- Do not introduce PreEscape-PreWalk or PreEscape-Escape subcategories, auxiliary subtype fields, or corresponding plots and statistical groups.
- Preserve the current wind-anchored burst qualification: a speed above the configured burst threshold must occur within 250 ms after wind arrival.
- Preserve onset measurement by backward search from the selected post-wind burst to the preceding low-speed boundary.
- For eligible looming + wind trials with PreEscape enabled, compare the measured interval onset directly with wind arrival. The current PreEscape buffer is zero: onset before wind qualifies; onset exactly at wind does not.
- Preserve classification priority: NoResponse first, then PreEscape, then PreWalk, otherwise Escape.
- The 50-ms exclusion at the end of the PreWalk assessment window is a separate criterion and remains unchanged.
- Preserve the existing visual-baseline exception and all current classification outputs.

## Known limitation retained

An independent earlier movement may subside below 10 mm/s before a later post-wind burst. In that case the later burst can supply the measured escape onset; the earlier movement remains visible in a full-trial heatmap without automatically making the trial PreEscape. Heatmap sorting does not change this behavior.

Searching the entire trial for the earliest burst or independently classifying all movement episodes is not part of this decision. These alternatives are deferred along with PreEscape subdivision.

## Revisit only if

A future research question specifically concerns pre-wind locomotor state, independent movement episodes, or causal sensory contributions, and a prespecified analysis or experimental design makes those distinctions useful. No implementation work is currently scheduled.
