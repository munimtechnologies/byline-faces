# byline-faces

How often do news sites present AI-generated faces as their writers?

Fake writer profiles with AI-made headshots have been exposed one outlet at a time:
Sports Illustrated (AdVon, 2023), the South Florida Standard (2026), Brown Brothers Media's
sites (2026) and We-News.com's "AI correspondents" (Canada, 2026). In 2026 Google's
guidelines began naming AI-generated author photos as an example of fake profiles. Nobody
has measured how common they are across outlets. This project does, with an open and
reproducible method.

**Method.** Faces from StyleGAN-family generators (thispersondoesnotexist.com, Generated
Photos and similar) are made on an aligned grid, so the eyes land on nearly the same pixels
in every image. Yang, Singh and Menczer (2024, arXiv:2401.02627) used that to measure AI
faces on Twitter/X. byline-faces applies the same test to writer headshots on news sites. It
adds a second test that survives scaling and centred cropping: the eyes are centred, level
and at the generator's height. Both tests are calibrated on a fixed sample of StyleGAN faces
and checked against real faces (Labeled Faces in the Wild, and FFHQ photos that share the
generator's crop). The full plan and the predictions were fixed before any data came in: see
[PREREGISTRATION.md](PREREGISTRATION.md).

The survey has four parts:
- **Calibration and negatives**: StyleGAN faces, real FFHQ photos and LFW.
- **Positive control**: we-news.com, whose staff page labels all its writers AI agents.
- **Documented cases**: Wayback Machine copies of writer pages on outlets where reporters
  documented invented writers. This asks a question of its own: do the fake personas found
  since 2023 still carry the StyleGAN signature, or have they moved on to diffusion models?
- **Population**: about 2,500 English-language news outlets listed in Wikidata, reading the
  author pages and the byline headshots on article pages.

**Limits, stated up front.**
- A flag is a lead, not a verdict. A real photo that happens to be centred, or one cropped
  with the FFHQ recipe, can trip the test, and every flagged face has to be looked at by a
  person.
- Faces made by diffusion models, which have been common since 2023, do not have the
  fixed-eye artifact, so this test cannot see them.
- Results are published as counts. An outlet is named only after a person has reviewed the
  face and the outlet has been asked for comment. Flagged images are kept for review and are
  not published.

## Use

```
pip install -r requirements.txt
python byline_faces.py headshot.jpg            # one image: eye geometry and both flags
python job.py --out-dir out --data-dir data --max-sites 50 --skip-documented   # small run
python -m unittest -v test_byline_faces        # standard-library tests, no network
```

job.py honours robots.txt, sends a few requests to each site, one at a time, paces Wayback
Machine requests at one per second, and identifies itself as `byline-faces/0.2` with a
contact address.

## Status

- v0.2.0: calibration moved to a fixed StyleGAN sample (thispersondoesnotexist.com is now a
  parked domain), new documented-cases group, article-page headshots, we-news.com control
  read from its staff page. Pre-registration amendment 1 added before any data was seen.
- v0.1.0: method, tests and pre-registration.
- Survey 1 is queued; results and their review will be added here.

Made by MV2 of Munim, Inc. (mv2@munimtech.com), with AI agents. Companion to
[byline-check](https://github.com/munimtechnologies/byline-check), a tool editors can use to
vet freelance pitches.

Licence: Apache-2.0. Prior work: Yang, Singh & Menczer, "Characteristics and prevalence of
fake social media profiles with AI-generated faces" (2024); Ricker et al., "AI-Generated
Faces in the Real World" (2024). Data used at run time, not bundled: the YuNet face detector
(OpenCV Zoo, MIT), TheKernel01/140k-Real-and-Fake-Faces on Hugging Face (a mirror of
xhlulu's Kaggle set: FFHQ photos and Bojan Tunguz's "1 Million Fake Faces"), and Labeled
Faces in the Wild.
