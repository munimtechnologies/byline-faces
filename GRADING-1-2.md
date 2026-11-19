# Grading rules for surveys 1 and 2 (written 19 Nov 2026, before either job returned data)

`grade_surveys12.py` turns the output folders of survey 1 (`job.py`) and survey 2
(`eras.py`) into one verdict per pre-registered prediction. Both jobs were still queued when
this file and the grader were published, so these rules were fixed before any result was
seen. They fill only the points PREREGISTRATION.md and PREREGISTRATION-2.md leave open; where
the pre-registration is explicit, the grader follows it.

Verdicts: `true`, `false`, `no_data`, `invalid` (needs Community Forensics but E1 failed),
`partial` (a prediction with several parts: none false, at least one true, at least one
without data), `pending_review` (needs a person's by-eye review), `undetermined` (a
majority rule ended in a tie). Thresholds are inclusive where the text says "at least" or
"at most" and strict where it says "fewer than" or "under".

## Survey 1 (P1-P8, 1b)

- **P1, P1b, P2** use the job's own rates (`calibration.json` holdout and real_ffhq;
  `negatives.json`). P2 needs both tests to hold.
- **P3** counts the unique single-face images taken from we-news.com's staff page (image
  kinds starting `seed_`), as amendment 1 specifies. Article-page images on that site are
  not counted.
- **P4** "reachable": a population site whose home page was read, i.e. status `ok`, or any
  articles, author pages or images reached before the time budget ran out. Sites the crawl
  never reached (status `deadline` with nothing read) are not in the denominator. If
  `sites.jsonl` is missing, the denominator falls back to status `ok`.
- **P5** has two parts. The Test-A share of unique population headshots is graded
  mechanically. The site-level part stays `pending_review` until a person has looked at
  every flagged population face; the review file (private) has one line per face,
  `{"sha256": ..., "judged_gan": true|false}`, and the share is distinct hosts with a face
  judged GAN-made over population sites with any single-face headshot.
- **P6, P7** pool the persona-page headshots of every documented case (unique within each
  case). **P8** is `no_data` if no Sports Illustrated/AdVon persona headshot was recovered.

## Survey 2 (E1-E9)

- **E1** is read from `calibration_cf.json` (`repro.ok`). If it is false, every prediction
  that uses Community Forensics (E2-E9) is graded `invalid`, as pre-registered.
- **E2, E3** use the job's per-set rates at 0.5. A set that failed to download counts as
  `no_data` for its part, so the prediction can end `partial`; it is never counted as passed.
- **E4** is the era-A rate at 0.5 among primary headshots.
- **E5** is `true` only if the B_new-minus-A difference at T95 is under 0.02 and the outlet
  bootstrap 95% interval includes zero. The grader also gives an outcome: `null` (interval
  includes zero), `rise` (lower bound above zero), `fall` (upper bound below zero). A rise is
  relabelled `unresolved` if it vanishes once image formats that make up under 5% of era A
  are left out: the difference at 0.5, recomputed without those formats, is zero or less.
  (The job reports per-format counts only at 0.5, so this check uses 0.5.) No T95 (fewer
  than 40 era-A images) means `no_data`.
- **E6** uses unique staff-page avatars (`seed_` images) from `records.jsonl`; without that
  file it falls back to all control images in `summary.json` and says so.
- **E7** pools persona-page headshots over every `documented:` case; both parts must hold.
- **E8** uses `records.jsonl`: unique single-face images on Elle Belgium persona pages
  whose path names "sophie" or "justine". It is `true` if more than half of them are
  flagged by an eye test and by Community Forensics, `undetermined` on a tie and `no_data`
  if none was recovered.
- **E9** uses every single-face image read on the three live Watermark sites (the
  `documented-live` group).

## What the grades are for

Mechanical grades come first and are published as they are. Flags are leads: any face
that would be named needs a person's review, and outlets outside the already documented
cases are not named before they have been asked for comment.

Text: CC BY 4.0. Code: Apache-2.0. Made by MV2 of Munim, Inc. with AI agents.
