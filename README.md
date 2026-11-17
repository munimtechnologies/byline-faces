# byline-faces

How often do news sites present AI-generated faces as their writers?

Fake writer profiles with AI-made headshots have been exposed one outlet at a time:
Sports Illustrated (AdVon, 2023), Elle Belgium (Ventures Media, 2025), the South Florida
Standard (2026), Brown Brothers Media's sites (2026), We-News.com's "AI correspondents"
(Canada, 2026) and, this week, three B2B trade sites (Press Gazette, 17 Nov 2026). In 2026
Google's guidelines began naming AI-generated author photos as an example of fake profiles.
Nobody has measured how common they are across outlets. This project does, with an open and
reproducible method.

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
- Every headshot of the documented fake journalists is scored by both tests, which asks
  whether persona makers have moved on from StyleGAN.

Plan and predictions, fixed before any data: [PREREGISTRATION-2.md](PREREGISTRATION-2.md).

## Limits, stated up front

- A flag is a lead, not a verdict. Real photos can trip either test, and every flagged face
  has to be looked at by a person.
- Survey 1's eye test cannot see diffusion faces. Survey 2's detector can in principle, but
  its error rate on small, re-compressed headshots is only known once it has been measured.
- Results are published as counts. An outlet is named only after a person has reviewed the
  face and the outlet has been asked for comment (documented cases, already public, are
  the exception). Flagged images are kept for review and are not published.

## Use

```
pip install -r requirements.txt
python byline_faces.py headshot.jpg            # one image: eye geometry and both flags
python job.py --out-dir out --data-dir data --max-sites 50 --skip-documented   # survey 1, small
python -m unittest -v test_byline_faces test_eras   # standard-library tests, no network

# survey 2 also needs a CPU build of PyTorch:
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1+cpu torchvision==0.20.1+cpu
pip install -r requirements-eras.txt
python eras.py --out-dir out2 --data-dir data --max-sites 50 --time-budget-min 30
```

job.py honours robots.txt, sends a few requests to each site, one at a time, and identifies
itself as `byline-faces/0.2` with a contact address. eras.py reads outlets only through the
Wayback Machine, paced at about 1.5 requests per second in total, and stops after repeated
"too many requests" answers.

## Status

- v0.3.0: survey 2 (eras.py, commfor.py, test_eras.py, PREREGISTRATION-2.md); two newly
  documented cases (Elle Belgium, 2025; Watermark Publishing Group's B2B sites, 2026).
- v0.2.0: calibration moved to a fixed StyleGAN sample (thispersondoesnotexist.com is now a
  parked domain), new documented-cases group, article-page headshots, we-news.com control
  read from its staff page. Pre-registration amendment 1 added before any data was seen.
- v0.1.0: method, tests and pre-registration.
- Surveys 1 and 2 are queued; results and their review will be added here.

Made by MV2 of Munim, Inc. (mv2@munimtech.com), with AI agents. Companion to
[byline-check](https://github.com/munimtechnologies/byline-check), a tool editors can use to
vet freelance pitches.

Licence: Apache-2.0. Prior work: Yang, Singh & Menczer, "Characteristics and prevalence of
fake social media profiles with AI-generated faces" (2024); Ricker et al., "AI-Generated
Faces in the Real World" (2024); Park & Owens, "Community Forensics: Using Thousands of
Generators to Train Fake Image Detectors" (CVPR 2025). Data and models used at run time, not
bundled: the YuNet face detector (OpenCV Zoo, MIT); Community Forensics weights
(OwensLab/commfor-model-384, MIT); TheKernel01/140k-Real-and-Fake-Faces on Hugging Face (a
mirror of xhlulu's Kaggle set: FFHQ photos and Bojan Tunguz's "1 Million Fake Faces");
OpenRL/DeepFakeFace (Stable Diffusion faces and IMDB-WIKI photos); Labeled Faces in the
Wild; the Internet Archive's Wayback Machine.
