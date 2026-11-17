# Pre-registration: survey 2, "then and now" (written 2026-11-17, before any survey-2 data)

Survey 1 (see PREREGISTRATION.md) uses only the StyleGAN eye-placement test, which cannot
see faces made by diffusion models. Survey 2 adds a general detector and a design that gives
it a measured false-positive baseline. No survey-1 or survey-2 output had been seen when
this was written.

## Question

Have AI-generated writer headshots become more common on established news sites since
diffusion models became public in 2022? And have the fake journalists exposed since 2023
moved from StyleGAN faces to faces the eye test cannot see?

## Detector

Community Forensics (Park and Owens, CVPR 2025, arXiv:2411.04125), a ViT-Small classifier
trained end to end on images from 4,803 generators, including diffusion models. Weights:
Hugging Face OwensLab/commfor-model-384 (MIT; model.safetensors, 87,262,324 bytes).
Recipe from the authors' code: shorter side to 440 (bilinear), centre crop 384, ImageNet
normalisation, sigmoid of one logit. It is applied to every image in which YuNet finds
exactly one face (the same filter as survey 1). "Flagged" means a score of at least 0.5
unless stated otherwise.

## Design

- **Eras.** A: captures dated 1 Jan 2019 to 31 Dec 2021, anchor 1 Jul 2020 (before DALL-E 2,
  April 2022, and Stable Diffusion, August 2022). B: captures dated 1 Jul 2025 to now,
  anchor 1 Aug 2026.
- **Outlets.** The survey-1 Wikidata population (same query, same seed 20261117, same
  order), up to 1,500 outlets, as many as the time budget allows.
- **Reading.** For each outlet and each era, everything comes from the Wayback Machine: the
  home page capture nearest the anchor, up to 3 articles linked from it, up to 3 author
  pages, up to 2 byline images per article and 2 candidate headshots per author page. A
  page or image counts only if its own capture date is inside the era's window. Requests
  are paced at about 1.5 per second in total; after five HTTP 429 answers the crawl stops.
- **Why the archive for both eras.** Compression, resizing and format change over time and
  can fool detectors. Reading both eras through the same archive keeps the pipeline equal,
  and era-A headshots captured before 2022 cannot be diffusion-made, so their flag rate is
  a false-positive baseline measured on the same outlets.
- **Units.** Unique images (SHA-256) per era. "B_new" = era-B images whose bytes never
  appear among the era-A images.
- **Threshold T95.** The 95th percentile of era-A scores (nearest rank), used only if era A
  has at least 40 primary images. Flagged at T95 means a score strictly above it.
- **Uncertainty.** Wilson 95% intervals for rates; for the difference, a percentile
  bootstrap that resamples outlets (2,000 draws, seed 20261117).

## Predictions

- **E1 (validity gate).** The loaded model reproduces the five scores printed in the
  authors' notebook for their DALL-E 2 test images (0.9988, 0.9878, 0.9569, 0.9516, 0.7860)
  within 0.02 each. If it does not, every Community Forensics result is reported as invalid.
- **E2 (recall on known AI faces).** At least 80% of Stable Diffusion 1.5 faces
  (DeepFakeFace text2img.zip, 200 members spread evenly over the sorted names) and at least
  80% of StyleGAN faces (the first 200 fakes of the survey-1 calibration shard) are flagged.
- **E3 (real faces).** At most 10% of LFW photos (every 13th file, 200), at most 10% of real
  FFHQ photos (first 200 of the shard) and at most 15% of real IMDB-WIKI photos
  (DeepFakeFace wiki.zip, 200 spread evenly) are flagged.
- **E4 (archive baseline).** At most 10% of era-A primary headshots are flagged at 0.5.
- **E5 (the main test).** Primary endpoint: rate(B_new flagged at T95) minus rate(A flagged
  at T95). We predict the difference is under 2 percentage points and its 95% interval
  includes zero: no detectable rise of AI-flagged headshots on established outlets. The
  same difference at 0.5 is secondary. We report results by image format, and an excess
  that comes only from a format barely present in era A (for example WebP or AVIF) is
  labelled "unresolved", not a finding.
- **E6 (labelled control).** At least 70% of single-face we-news.com staff avatars, which
  the site labels AI agents, are flagged.
- **E7 (exposed fake journalists).** Pooling all documented cases, at least half of the
  single-face headshots on named persona pages are flagged, and at least one is flagged by
  Community Forensics but by neither eye test.
- **E8 (Elle Belgium).** If the headshot of "Sophie Vermeulen" (or her earlier alias
  "Justine") is recovered, it is flagged both by an eye test and by Community Forensics.
  VRT NWS found that photo in This Person Does Not Exist's database, so it is a StyleGAN
  face.
- **E9 (Watermark B2B sites, read live).** At least half of the single-face reporter
  headshots on securityguysnews.com, thesupplychainer.com and warehouseinsider.news are
  flagged (Press Gazette, 17 Nov 2026, reported that they scan as AI-generated).

## Documented cases added in survey 2 (both sourced)

- Ventures Media, Elle Belgium (VRT NWS, 26 Jun 2025): writer pages under
  elle.be/nl/author/, personas "Sophie Vermeulen" (earlier "Justine") and "Marta Peeters".
- Watermark Publishing Group's B2B sites (Press Gazette, 17 Nov 2026): archived pages via
  the Wayback Machine and the live sites, reporters named in that article.

## Limits, stated before the data

- Community Forensics was trained on whole images; headshots are small, cropped and
  re-compressed, and its error rates on them are unknown until E2-E4 are measured.
- StyleGAN faces existed in 2019-2021, so era A can contain a few AI faces. That raises the
  baseline and makes E5 conservative.
- Outlets with captures in both eras skew towards larger, older outlets; the result speaks
  for them, not for new or tiny sites.
- A flag is a lead, never a verdict. As in survey 1: only a face a person has reviewed
  counts as a finding, results are published as counts, and no outlet or writer outside the
  already-documented cases is named before review and a request for comment. Flagged images
  are kept privately for review and are not published.

## What would be a "wow"

A clear rise (E5 lower bound above zero, not explained by format) would be the first
measurement of AI-generated writer faces spreading on established news sites. A clean null
with a measured baseline is still the first open measurement and bounds how common such
faces can be. E7 and E8 answer, with data, whether the people who invent journalists have
moved on from StyleGAN.
