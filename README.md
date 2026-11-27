# byline-faces

How often do news sites present AI-generated faces as their writers? And where do the people
who invent journalists get their faces?

Fake writer profiles with AI-made headshots have been exposed one outlet at a time:
Sports Illustrated (AdVon, 2023), Elle Belgium (Ventures Media, 2025), the South Florida
Standard (2026), VideoGamer and other Clickout Media sites (2026), Brown Brothers Media's
sites (2026), We-News.com's "AI correspondents" (Canada, 2026), the "Daniel Robson" columns
removed by Canadian outlets (2026), Paradise Media's Philadelphia-area sites (Billy Penn, 11
Nov 2026) and three B2B trade sites (Press Gazette, 17 Nov 2026). In October 2026 Google's
guidance for site owners began calling invented creator profiles with AI-generated headshots
"a form of deception". Nobody has measured how common they are across outlets. This project
does, with an open and reproducible method.

## Survey 1: the StyleGAN eye test

Faces from StyleGAN-family generators (thispersondoesnotexist.com, Generated Photos and
similar) are made on an aligned grid, so the eyes land on nearly the same pixels in every
image. Yang, Singh and Menczer (2024, arXiv:2401.02627) used that to measure AI faces on
Twitter/X. byline-faces applies the same test to writer headshots on news sites. It adds a
second test that survives scaling and centred cropping: the eyes are centred, level and at
the generator's height. Both tests are calibrated on a fixed sample of StyleGAN faces and
checked against real faces (Labeled Faces in the Wild, and FFHQ photos that share the
generator's crop). Plan and predictions: [PREREGISTRATION.md](PREREGISTRATION.md).

Survey 1 has four parts: calibration and negatives; a positive control (we-news.com, whose
staff page labels all its writers AI agents); documented cases (Wayback Machine copies of
writer pages at outlets where reporters documented invented writers); and a population of
about 2,500 English-language news outlets listed in Wikidata.

## Survey 2: then and now, with a detector that also sees diffusion faces

The eye test is blind to diffusion-model faces. Survey 2 adds Community Forensics (Park and
Owens, CVPR 2025), a small open detector trained on images from 4,803 generators, and gives
it a baseline measured on the same outlets:

- For each outlet, writer headshots are read from the Wayback Machine twice: as archived in
  2019-2021, before diffusion models were public, and as archived since July 2025. Both
  eras go through the same archive, so compression and format are handled alike.
- Pre-2022 headshots cannot be diffusion-made, so their flag rate is the detector's
  false-positive rate on these outlets. The question is whether headshots that are new
  since 2025 are flagged more often.
- The detector is first checked against its authors' published scores, then measured on
  known faces: StyleGAN, Stable Diffusion 1.5 (DeepFakeFace), and real FFHQ, IMDB-WIKI and
  LFW photos.

Plan and predictions, fixed before any data: [PREREGISTRATION-2.md](PREREGISTRATION-2.md).

**Graded by a program published before the data.** `grade_surveys12.py` turns the output of
survey 1 and of survey 2 into a verdict for each of their seventeen predictions (P1-P8 with
1b, E1-E9). [GRADING-1-2.md](GRADING-1-2.md) fixes the rules the pre-registrations left
open: which sites count as reachable, what happens when a calibration set fails to download,
and when a rise in flagged headshots is called "unresolved" because it comes from an image
format that was rare before 2022. Both were released while the two jobs were still queued.

## Survey 3: the fake-journalist face atlas, 2020-2026

Seventeen cases of invented journalists exposed in published reporting since 2020, from the
Reuters "Oliver Taylor" story (2020), the Daily Beast's op-ed network (2020) and the Russian
PeaceData and NAEBC sites (2020), through Sports Illustrated's and TheStreet's AdVon writers,
Hoodline, the Irish Times hoax and DC Weekly (2023), to Elle Belgium (2025), the South
Florida Standard, we-news.com, OpenAI's "Bogus Bylines", "Daniel Robson", Paradise Media's
Northeast Times and Lower Bucks Times, and Watermark's B2B sites (2026). Each case is listed
in `atlas.py` with its source. The archived writer pages of the personas are read from the
Wayback Machine and every headshot is scored by both tests.

- **The question:** did the people inventing journalists move from StyleGAN faces (the eye
  test sees them) to diffusion faces (only the general detector can) once diffusion models
  became public in 2022?
- **A labelled test in the wild:** the reporting says how many of these faces were made (GAN
  per experts, This Person Does Not Exist, Generated Photos, DALL-E) or that they were
  stolen from real people (Raphael Badani, DC Weekly). Those labels give the first
  measurement of both detectors on fake-journalist headshots found in the wild, including
  their false-positive rate on real people's photos used by fake journalists.
- **Open data:** `atlas_faces.jsonl` lists every persona face found, with its archived page
  on the Wayback Machine, the image's SHA-256 and both scores. All cases are already public.
- **Graded by a program published before the data:** `grade_atlas.py` turns the job's
  output into a verdict for each of the eleven predictions, and
  [GRADING-3.md](GRADING-3.md) fixes the rules the pre-registration left open (pooling,
  ties, what happens if the detector's validity check fails). Both were released while the
  survey 3 job was still queued.

Plan and predictions, fixed before any data: [PREREGISTRATION-3.md](PREREGISTRATION-3.md)
(nine predictions, plus amendment 1, also written before any data, which adds three cases
and two predictions).

## Survey 4: persona timelines

Survey 3 reads each persona's earliest archived page. Some suspected fake writers lasted
for years, and their faces changed. Press Gazette (12 May 2026) reported that the current
picture of one long-running finance "contributor" scans as 100% AI-generated, while his
early byline pictures appear to show a different person. Survey 4 (`timeline.py`) follows
each persona's writer pages through the Wayback Machine, with up to ten captures per page
spread over its archived life. It keeps the persona's headshot from each capture, groups
identical pictures (SHA-256 and a difference hash) and scores every distinct headshot with
both tests.

- **The question:** when a long-running fake persona changes its face, what does it change
  to, and when? Ten personas from three documented cases: the four finance writers
  questioned by Press Gazette (suspected, not proven), OpenAI's seven "Bogus Bylines"
  personas and "Daniel Robson".
- **A built-in control:** a picture captured before December 2018 cannot come from StyleGAN
  or a diffusion model, so the detector's flags on those pictures are false alarms.
- **Graded by code published before the data:** `grade()` in `timeline.py` turns the output
  into verdicts for predictions T1-T8.

Plan, predictions and grading rules, fixed before any data:
[PREREGISTRATION-4.md](PREREGISTRATION-4.md).

## Survey 5: one site, before and after

Surveys 1-4 compare pictures from different sites and years. A detector could separate two
such sets by their compression or cropping rather than by how the faces were made. Survey 5
removes that confound with one documented case. After its sale to Clickout Media, the gaming
site VideoGamer replaced its staff bylines with invented writers whose profile pictures
looked AI-made. One of those image files was named "ChatGPT-Image-Oct-20-2025-...png"
([Kotaku, 26 Feb 2026](https://kotaku.com/resident-evil-requiem-metacritic-ai-review-videogamer-2000674042);
[Press Gazette, 27 Feb 2026](https://pressgazette.co.uk/news/staff-journalists-sacked-and-misleadingly-replaced-with-ai-writers/)).

- **The test:** `survey5.py` scores the writer headshots the same site showed on the same
  page template, before the change (archived before September 2025) and after it (archived
  from 19 February 2026 on).
- **The questions:** does Community Forensics separate the two periods (V1, V2)? Does it flag
  the face whose own file names its generator (V3)? Do the new faces escape the StyleGAN eye
  test (V4)?
- **Graded by code published before the data:** `grade()` in `survey5.py`.
- **Privacy:** real former staff are never named, and their pictures appear only as counts
  and scores.

Plan and predictions, fixed before any data: [PREREGISTRATION-5.md](PREREGISTRATION-5.md).
Its amendment 1, written with the code and also before any data, fixes how captures are
chosen. Survey 5 runs after surveys 1-4 have shown that the pipeline works on real data.

## Limits, stated up front

- A flag is a lead, not a verdict. Real photos can trip either test, and every flagged face
  has to be looked at by a person.
- Survey 1's eye test cannot see diffusion faces. Survey 2's detector can in principle, but
  its error rate on small, re-compressed headshots is only known once it has been measured.
- Surveys 3 and 4 study exposed cases, not prevalence, and their counts are small. Survey 5
  is a single site.
- Results are published as counts. An outlet is named only after a person has reviewed the
  face and the outlet has been asked for comment (documented cases, already public, are
  the exception). Flagged images are kept for review and are not published.

## Use

```
pip install -r requirements.txt
python byline_faces.py headshot.jpg            # one image: eye geometry and both flags
python job.py --out-dir out --data-dir data --max-sites 50 --skip-documented   # survey 1, small
python -m unittest -v test_byline_faces test_eras test_atlas test_grade_atlas test_timeline test_grade_surveys12 test_survey5   # standard-library tests, no network

# surveys 2-5 also need a CPU build of PyTorch:
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu
pip install -r requirements-eras.txt
python eras.py --out-dir out2 --data-dir data --max-sites 50 --time-budget-min 30
python atlas.py --out-dir out3 --data-dir data --time-budget-min 60
python timeline.py --out-dir out4 --data-dir data --time-budget-min 60
python survey5.py --out-dir out5 --data-dir data --time-budget-min 60

# grade surveys 1 and 2 (standard library only)
python grade_surveys12.py survey1 out --out grades1.json
python grade_surveys12.py survey2 out2 --out grades2.json
# grade survey 3 (standard library only); --review adds the by-eye sensitivity analysis
python grade_atlas.py out3/atlas_summary.json out3/atlas_faces.jsonl --out grades.json
# surveys 4 and 5 grade themselves: out4/timeline_grades.json, out5/survey5_summary.json
```

job.py honours robots.txt, sends a few requests to each site, one at a time, and identifies
itself as `byline-faces/0.2` with a contact address. eras.py, atlas.py, timeline.py and
survey5.py read pages only through the Wayback Machine, paced, and back off on "too many
requests" answers.

## Status

- v0.5.2: survey 5 code (survey5.py, test_survey5.py) and PREREGISTRATION-5.md amendment 1,
  published before any survey 5 data.
- v0.5.1: grade_surveys12.py, test_grade_surveys12.py and GRADING-1-2.md, the grader for
  surveys 1 and 2, published before any survey 1 or survey 2 data.
- v0.5.0: survey 4, persona timelines (timeline.py, test_timeline.py,
  PREREGISTRATION-4.md), with its grader, published before any survey 4 data.
- v0.4.1: grade_atlas.py, test_grade_atlas.py and GRADING-3.md, the grader for survey 3's
  eleven predictions, published before any survey 3 data.
- v0.4.0: survey 3, the fake-journalist face atlas (atlas.py, test_atlas.py,
  PREREGISTRATION-3.md), 17 sourced cases from 2020 to 2026.
- v0.3.0: survey 2 (eras.py, commfor.py, test_eras.py, PREREGISTRATION-2.md); two newly
  documented cases (Elle Belgium, 2025; Watermark Publishing Group's B2B sites, 2026).
- v0.2.0: calibration moved to a fixed StyleGAN sample (thispersondoesnotexist.com is now a
  parked domain), new documented-cases group, article-page headshots, we-news.com control
  read from its staff page. Pre-registration amendment 1 added before any data was seen.
- v0.1.0: method, tests and pre-registration.
- Surveys 1-4 are waiting for compute; results and their review will be added here.

Made by MV2 of Munim, Inc. (mv2@munimtech.com), with AI agents. Companion to
[byline-check](https://github.com/munimtechnologies/byline-check), a tool editors can use to
vet freelance pitches.

Licence: Apache-2.0 (code); pre-registration and grading text CC BY 4.0. Prior work: Yang,
Singh & Menczer, "Characteristics and prevalence of fake social media profiles with
AI-generated faces" (2024); Ricker et al., "AI-Generated Faces in the Real World" (2024); Park
& Owens, "Community Forensics: Using Thousands of Generators to Train Fake Image Detectors"
(CVPR 2025); DFRLab, "The evolving role of AI-generated media in shaping disinformation
campaigns" (May 2025). Reporting used for survey 4: Press Gazette, "Prolific finance
journalists facing questions over identities" (Rob Waugh, 12 May 2026). Reporting used for
survey 5: Kotaku (Lewis Parker with Ethan Gach, 26 Feb 2026) and Press Gazette (Rob Waugh,
27 Feb 2026, and Nov 2026). Data and models used at run time, not bundled: the YuNet face
detector (OpenCV Zoo, MIT); Community Forensics weights (OwensLab/commfor-model-384, MIT);
TheKernel01/140k-Real-and-Fake-Faces on Hugging Face (a mirror of xhlulu's Kaggle set: FFHQ
photos and Bojan Tunguz's "1 Million Fake Faces"); OpenRL/DeepFakeFace (Stable Diffusion
faces and IMDB-WIKI photos); Labeled Faces in the Wild; the Internet Archive's Wayback
Machine.
