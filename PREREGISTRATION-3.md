# Pre-registration 3: the fake-journalist face atlas, 2020-2026

Written on 17 November 2026, before any survey 3 data was collected. Survey 1
(`byline-faces-survey1-v2`) and survey 2 (`byline-faces-eras1-v1`) had not returned any
result when this was written either, so nothing here was tuned on data.
Made with AI agents working for MV2 of Munim, Inc. Apache-2.0 (code); text CC BY 4.0.

## Question

Where do the people who invent journalists get their faces, and has that changed since
diffusion models became public in 2022?

Reporters have exposed invented journalists many times since 2020: the Reuters "Oliver
Taylor" story, the Daily Beast's 19-persona op-ed network, the Russian PeaceData and NAEBC
sites, Sports Illustrated's AdVon writers, Hoodline, the Irish Times hoax, Elle Belgium, the
South Florida Standard, OpenAI's "Bogus Bylines" and others. Each story is told case by case.
We found no study that puts the faces from all of them side by side and measures them with
the same tests. The closest work measures GAN faces among Twitter profile pictures (Yang et
al. 2024; Ricker et al. 2024, arXiv 2404.14244) or describes the GAN-to-diffusion shift in
influence operations' social media accounts (DFRLab, May 2025). This survey asks the
question for fake journalists' faces on the news pages themselves.

## What is measured

For each of the 14 cases in `atlas.py` (`CASES`, every one with its published source), the
Wayback Machine's CDX index is searched under the listed URL prefixes; archived writer pages
of the documented personas (pages whose path holds a persona's name, or the staff pages of
sites whose listed staff were all fake, or any writer page of a site that was fake as a
whole) are read at the capture the index returns first, and up to two candidate headshots
per page are fetched from the archive. Each image with exactly one face is scored by:

- **the eye test** (survey 1): eyes at the fixed positions StyleGAN-family generators use
  ("aligned" or "centred and level", thresholds from the 140k StyleGAN/FFHQ calibration);
- **Community Forensics** (survey 2): probability of being generated, flagged at >= 0.5.

Images are de-duplicated by SHA-256 within each case and again across cases when pooled.
Cases are grouped by the year they were exposed: 2019-2021 (before public diffusion
models), 2022-2024, 2025-2026. The reporters' own description of each case's faces is kept
as a label: StyleGAN family (GAN per experts, This Person Does Not Exist, Generated Photos),
a named diffusion model (DALL-E), "AI-generated" without a method, stolen real photos, mixed.

## Predictions (fixed before data)

- **A1 (validity gate).** Community Forensics reproduces its authors' five published scores
  within 0.02 (as E1 in pre-registration 2). If it does not, every Community Forensics
  number below is reported as invalid and only the eye test is read.
- **A2 (coverage).** At least 8 of the 14 cases yield at least one persona face from the
  archive.
- **A3 (GAN era).** Among persona faces of the 2019-2021 cases that the reporting calls GAN
  (Oliver Taylor, Joseph Labba, PeaceData, NAEBC), at least half carry the eye-test
  signature.
- **A4 (the shift, main).** The share of persona faces with the eye-test signature is at
  least 20 percentage points lower in the 2025-2026 cases than in the 2019-2021 cases, both
  pooled over images and as a mean over cases.
- **A5 (new faces).** At least 25% of 2025-2026 persona faces are flagged by Community
  Forensics but not by the eye test.
- **A6 (stolen photos).** The "Raphael Badani" photos, which the Daily Beast says were taken
  from a real person, are flagged by neither test.
- **A7 (a named diffusion face).** The Irish Times "Adriana Acosta-Cortez" byline photo
  (made with DALL-E, according to the hoaxer) is flagged by Community Forensics and not by
  the eye test.
- **A8 (Generated Photos).** The Sports Illustrated "Drew Ortiz" headshot, which Futurism
  found for sale on a site selling AI-generated headshots, carries the eye-test signature.
- **A9 (sensitivity in the wild).** Pooling every persona face from cases labelled StyleGAN
  family, DALL-E or "AI-generated", at least 50% are flagged by at least one of the two
  tests.

## How the result will be read

- A4 true: the people inventing journalists have largely moved past StyleGAN faces, and an
  eye-placement check (the cheapest test an editor can run) no longer catches most of them.
- A4 false with a high 2025-2026 eye-test share: StyleGAN faces from 2019-era tools are still
  the norm for fake journalists in 2026, which is itself news (Elle Belgium's 2025 persona
  used a face from This Person Does Not Exist's database).
- The counts are small (a few cases per period, tens of faces at most), so every share is
  reported with its Wilson 95% interval and with case-level means, and nothing is called
  significant. This is an atlas of documented cases, not a prevalence estimate: cases with
  obviously AI-looking faces may be more likely to be exposed in the first place.
- Persona pages are found by name; a page of a real writer who shares a persona's name on
  the same outlet would be miscounted. Persona-specific URL prefixes on real outlets keep
  this small; the private review checks it.

## What is published

`atlas_summary.json` (counts per case, period and origin label) and `atlas_faces.jsonl` (one
line per persona face: case, archived page URL on the Wayback Machine, image SHA-256 and
both scores). All cases are already public in published reporting. Image files are not
republished; flagged images are kept privately for review by eye, and every case-level claim
is checked by eye before it is reported.

## Amendment 1 (17 November 2026, still before any survey 3 data)

Written before the survey 3 job had run and before any result of surveys 1-3 existed.

**Why.** The first version of the job could not run: one URL prefix in the Bogus Bylines case
began with "www.", which the case-table test forbids (the Wayback CDX index folds "www."
anyway), so the test step failed before any data was read. The job is re-filed as
`byline-faces-atlas1-v2` with that prefix removed. Since the case table was being changed
anyway, three newly documented cases and one more persona were added. None was chosen by
looking at its faces with either test; each is added because its published reporting says
where the headshots came from.

**Added cases (17 in total now):**

- `dc-weekly-2023` (stolen): Clemson University's Media Forensics Hub (2023) and NewsGuard
  (29 May 2024) found DC Weekly's journalists invented, with photos taken from real people
  (the writer Judy Batalion for "Jessica Devlin", a Spain-based model for "Lucas Turner") or
  stock art. A second stolen-photo case, in the 2022-2024 period.
- `daniel-robson-2026` (AI, method not stated): CBC News (30 Sep 2026) and Canadaland
  (7 Oct 2026); the Ottawa Citizen's editor-in-chief said the "Daniel Robson" headshot
  "appeared to be AI generated" by online analysis tools.
- `paradise-media-pa-2026` (AI, method not stated, suspected): Billy Penn at WHYY
  (11 Nov 2026) on the Northeast Times and Lower Bucks Times; Hive Detect rated the "Ellen
  Kowalczyk" photo 99.9% likely AI-generated.
- In `si-advon-2023`, TheStreet's "Denise McNamara", whose profile picture Futurism found on
  the same AI-headshot marketplace as "Drew Ortiz".

**Deliberately left out:** Paradise Media's centraljersey.com, where real reporters' names were
shown with photos that were not of them. The job reads the earliest archived capture of each
writer page, which for real reporters may predate the fake photo and show the real person.

**Predictions changed or added (the rest stand unchanged):**

- **A2 (coverage), amended.** At least 10 of the 17 cases yield at least one persona face
  (the same share as 8 of 14).
- **A10 (stolen photos, pooled).** Pooling every persona face from the two cases labelled
  stolen (Raphael Badani; DC Weekly), at most 20% are flagged by either test. This is the
  false-positive check in the wild: real people's photos used by fake journalists.
- **A11 (suspected 2026 cases).** In each of `watermark-b2b-2026` and
  `paradise-media-pa-2026`, at least half of the persona faces found are flagged by
  Community Forensics. Both are labelled "suspected" in the reporting, so a miss here would
  be reported as a miss, not as evidence that the faces are real.
