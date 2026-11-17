# byline-faces

How often do news sites present AI-generated faces as their writers?

Fake writer profiles with AI-made headshots have been exposed one outlet at a time:
Sports Illustrated (AdVon, 2023), Ventures Media's magazines (2025), and We-News.com's
121 "AI correspondents" (Canada, 2026). Nobody has measured it across outlets. This
project does, with an open and reproducible method.

**Method.** Faces from StyleGAN-family generators (thispersondoesnotexist.com, Generated
Photos and similar) are made on an aligned grid, so the eyes land on nearly the same pixels
in every image. Yang, Singh and Menczer (2024, arXiv:2401.02627) used that to measure AI
faces on Twitter/X. byline-faces applies the same test to writer headshots on news sites. It
adds a second test that survives scaling and centred cropping: the eyes are centred, level
and at the generator's height. Both tests are calibrated on fresh generated faces and
checked against real faces (Labeled Faces in the Wild). The full plan and the predictions
were fixed before any data came in: see [PREREGISTRATION.md](PREREGISTRATION.md).

**Limits, stated up front.**
- A flag is a lead, not a verdict. A real photo that happens to be centred can trip the test,
  and every flagged face has to be looked at by a person.
- Faces made by diffusion models, which have been common since 2023, do not have the
  fixed-eye artifact, so this test cannot see them.
- Results are published as counts. An outlet is named only after a person has reviewed the
  face and the outlet has been asked for comment.

## Use

```
pip install -r requirements.txt
python byline_faces.py headshot.jpg            # one image: eye geometry and both flags
python job.py --out-dir out --max-sites 50     # small survey run
python -m unittest -v test_byline_faces        # standard-library tests, no network
```

job.py honours robots.txt, sends at most a few requests to each site, one at a time, and
identifies itself as `byline-faces/0.1` with a contact address.

## Status

- v0.1.0: method, tests and pre-registration. Survey 1 is queued; results and their review
  will be added here.

Made by MV2 of Munim, Inc. (mv2@munimtech.com), with AI agents. Companion to
[byline-check](https://github.com/munimtechnologies/byline-check), a tool editors can use to
vet freelance pitches.

Licence: Apache-2.0. Prior work: Yang, Singh & Menczer, "Characteristics and prevalence of
fake social media profiles with AI-generated faces" (2024); Ricker et al., "AI-Generated
Faces in the Real World" (2024). The YuNet face detector is from OpenCV Zoo (MIT); it is
downloaded at run time, not bundled here.
