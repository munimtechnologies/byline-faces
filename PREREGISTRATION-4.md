# Pre-registration 4: persona timelines (byline-faces survey 4)

Written on 2026-11-19, before any survey 4 data was collected, by AI agents working for MV2 of
Munim, Inc. Released with byline-faces v0.5.0 while the job was still queued. Licence: CC BY 4.0.
Grader: `grade()` in `timeline.py`, published with this document; its tests are in
`test_timeline.py`.

## Why

Survey 3 (the atlas) asks where invented journalists got their faces, but it reads each
persona's **earliest** archived writer page only. Some suspected fake writers lasted for years.
Press Gazette (Rob Waugh, 12 May 2026, "Prolific finance journalists facing questions over
identities") reported four freelance finance and crypto "journalists" with more than 1,000
articles in outlets such as Forbes, Investing.com, HuffPost, CoinTelegraph, VentureBeat and
TheStreet, none of whom the outlets could show were real. Press Gazette's checks found
the faces changed over time:

- **Nikolai Kuznetsov** (bylines 2015-2025): current profile picture "100% AI-generated
  according to Identifai"; "early byline pictures ... appear to show an entirely different
  person".
- **Reuben Jackson**: X picture "fully AI-generated, according to Identifai"; "previous images
  of Jackson do not carry the hallmarks of AI".
- **Luis Aureliano**: Investing.com picture scans 0% AI and resembles a real chef.
- **Joe Liebkind** (mostly 2016-2019): picture appears to be a free-use Flickr photo also used
  on Wikipedia.

A firm named in the story denied operating the profiles. These personas are **suspected**, not
proven, fakes, and this survey measures their pictures only. Two more groups are followed the
same way: the seven "Bogus Bylines" personas (OpenAI, Oct 2026; Marc Owen Jones) and "Daniel
Robson" (CBC, Canadaland, 2026). Pages per persona are listed in `timeline.py`.

No published study we found follows fake journalists' faces through time. Fake-face studies
(Yang, Singh and Menczer 2024; Ricker et al. 2024; Graphika; DFRLab 2025) measure social media
accounts at one moment. The question here is new: **when a long-running fake persona changes
its face, what does it change to, and when?** If the answer is "from real or stock photos to
AI faces, and since 2023 to faces the StyleGAN eye test cannot see", that would be direct
evidence that the people behind fake bylines have moved to newer face generators.

## Method (fixed)

1. For every listed writer page, the Wayback Machine CDX index is asked for every capture with
   status 200 of exactly that URL (with and without a trailing slash), at most one per month.
   Up to 10 captures per page are read, spread evenly over the page's archived life, always
   including the first and the last.
2. From each archived page, candidate headshots are found as in surveys 1-3, ordered: image URL
   names the persona, then JSON-LD Person image, then `<img>` headshots, then og/twitter image.
   The first of up to three candidates that contains exactly one face is kept: one picture per
   capture.
3. Each picture is scored by the StyleGAN eye test (byline_faces.py, calibrated as in survey 1
   when the calibration sample is available) and by Community Forensics (commfor.py), whose
   validity check (`commfor.reproduce`, the authors' published scores) runs first.
4. A persona's pictures are grouped into **distinct headshots**: the same SHA-256, or a 64-bit
   difference hash of the whole image (8 x 9 grey grid) within 10 bits of the group's first
   picture. A headshot's date is its first capture. Its Community Forensics score is the median
   over its distinct image files. It is **CF-flagged** if that median is at least 0.5. It is
   **eye-flagged** if any of its files passes either eye test.
5. **Control:** a distinct image file whose page capture and image capture both precede
   2018-12-01 cannot be a StyleGAN or diffusion face (StyleGAN's preprint: 12 Dec 2018; This
   Person Does Not Exist: Feb 2019; public diffusion models: 2022). CF flags on these are false
   alarms.

## Predictions

The Press Gazette personas are Kuznetsov, Jackson, Aureliano and Liebkind.

- **T1 (coverage):** at least 3 of the 4 Press Gazette personas yield at least one archived
  single-face headshot.
- **T2 (face swaps exist):** at least 2 of the 4 show two or more distinct headshots over
  time. (No data if fewer than 2 personas yield any headshot.)
- **T3 (direction, all personas):** among personas with two or more distinct headshots whose
  earliest and latest headshots both have scores and differ, the latest headshot scores
  higher than the earliest in more than half. Needs at least 3 such personas, else no data.
  One-sided sign-test p is reported but does not decide the verdict.
- **T4 (Kuznetsov):** among his scored distinct headshots, the earliest is not CF-flagged
  and the latest is (agreeing with Identifai's result and with Press Gazette).
  Needs two scored headshots.
- **T5 (control):** at most 10% of control images are CF-flagged. Needs at least 5 control
  images. If T5 is false, CF flags on later pictures cannot be read as evidence of AI
  generation, and the write-up must say so first.
- **T6 (newer generators):** of the distinct headshots first seen on or after 2023-01-01 that
  are CF-flagged, fewer than half are eye-flagged. In other words, the AI-looking faces
  adopted since 2023 are mostly not StyleGAN-aligned. Needs at least one such headshot.
- **T7 (Liebkind, real-photo control):** none of Liebkind's scored headshots is CF-flagged.
- **T8 (Aureliano):** none of Aureliano's scored headshots is CF-flagged.

## Grading rules

- Verdicts are `true`, `false`, `no_data` or `invalid`. T3-T8 depend on Community Forensics.
  If its validity check does not pass (`cf_repro_ok` is not true), they are `invalid`, and no
  conclusion about AI faces is drawn from this survey.
- `grade()` in `timeline.py` is the grader. Its output (`timeline_grades.json`) is reported
  as is, next to a by-eye review: a person (or, until one is available, the agents, saying
  so) looks at every distinct headshot and records whether it is the persona's own headshot
  and whether the difference-hash grouping joined or split pictures wrongly. The review is
  reported as a sensitivity analysis and never replaces the mechanical verdicts.
- A persona's pages are read in the order listed. Personas not reached before the time
  budget ends are reported as `not_reached` or `deadline`, and predictions that need them
  fall to `no_data` by the rules above.
- Public output: `timeline_faces.jsonl`, one line per kept picture, with the archived page,
  the image's SHA-256, difference hash and both scores. All personas were named in published
  reporting. Flagged images themselves are kept for review and are not published.

## Limits

- Ten personas from three cases, chosen because reporting documented them. This is a set of
  case histories, not a prevalence estimate.
- Wayback coverage of writer pages is uneven, and some listed pages may never have existed.
  Those return no captures.
- The difference hash groups identical pictures at different sizes. Different crops of the
  same photo can split into two groups. The by-eye review checks this.
- Community Forensics' error rate on small, re-compressed headshots is measured here only on
  the control images and in surveys 2 and 3.
