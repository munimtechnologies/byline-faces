# Grading rules for survey 3 (fixed before any data)

Written and published on 17 November 2026 with `grade_atlas.py`, while the survey 3 job
(`byline-faces-atlas1-v2`) was still waiting in the queue. No result of surveys 1, 2 or 3
existed when these rules were written. They settle the points that PREREGISTRATION-3.md leaves
open, so that the predictions are scored by a published program rather than by judgement
after the fact. Made with AI agents working for MV2 of Munim, Inc. Text CC BY 4.0, code
Apache-2.0.

## What is graded

`python grade_atlas.py atlas_summary.json atlas_faces.jsonl --out grades.json` grades the
eleven predictions A1-A11 from the job's two public output files. The verdicts printed by
this program are the results that will be reported, whatever they are.

## Rules

- **Units.** A "persona face" is a line of `atlas_faces.jsonl`: a single-face image on an
  archived persona page, already de-duplicated by SHA-256 within its case by the job.
- **Flags.** Eye test: `eye_aligned` or `eye_centred_level`. Community Forensics: score at
  least 0.5; an image with no score (`cf` null) counts as not flagged. "Either test": eye
  test or Community Forensics.
- **Validity (A1).** A1 is true only if the job reports `cf_repro_ok: true`. If it is false
  or missing, A5, A7 and A11 are graded `invalid`, and A6, A9 and A10 are graded on the eye
  test alone, marked "eye test only".
- **Pooling (A3, A9, A10).** Faces of the listed cases, de-duplicated by SHA-256 across
  those cases. A3 uses Oliver Taylor, Joseph Labba, PeaceData and NAEBC. A9 uses every case
  labelled `stylegan`, `diffusion` or `ai-unspecified` (not `mixed`, not `stolen`). A10 uses
  the cases labelled `stolen`.
- **Periods (A4, A5).** As in the job: all cases in order with one global SHA-256 set, each
  face counted in the period of the first case it appears in. A4 is true only if both the
  pooled difference and the difference of case means (each case with faces counting once)
  are at most -0.20. When no review file is used, the grader also checks its pooled
  difference against the job's own `primary.diff_late_minus_early` and reports any mismatch.
- **A5 denominator.** All 2025-2026 persona faces, including any without a score.
- **Single-persona predictions (A6, A7, A8).** An archived page can hold more than one face
  (a sidebar, an advert, another writer). A6 says the Badani photos are flagged by neither
  test, so it is true only if none of that case's persona faces is flagged. A7 (Irish Times)
  and A8 (Drew Ortiz: pages whose URL contains `drew-ortiz`) are true if more than half of the
  persona's faces meet the condition, false if fewer than half do, and `undetermined` on a
  tie.
- **A11.** For each of the two suspected 2026 cases, the share of all its persona faces
  flagged by Community Forensics must be at least one half. Overall: true if both are true;
  false if either is false; `partial` if one is true and the other has no faces; `no_data`
  if neither has faces.
- **No data.** A prediction with nothing to grade is `no_data`, never true or false. A2 is
  false if the job found persona faces in fewer than 10 cases, including none.
- **Thresholds** are compared with a tolerance of 1e-9, so that 0.5 counts as at least one
  half despite floating-point rounding.

## Human review (a sensitivity analysis, not a replacement)

Every persona face is also checked by eye, privately, before any case-level claim is made.
The reviewer writes `review.jsonl`, one line per image: `{"sha256": ..., "case": ...,
"is_persona_headshot": true/false, "reason": ...}`, marking whether the image is the
persona's own headshot (not an advert, logo, another writer or a page element).
`grade_atlas.py ... --review review.jsonl` then grades again on the marked headshots only.
Both sets of verdicts are published side by side. The mechanical verdicts stay the
pre-registered result; the reviewed ones show whether page noise changed any of them.
