# Pre-registration: survey 1 (written 2026-11-17, before any data was collected)

**Question.** How many established English-language news outlets show writers whose
headshots were made by a StyleGAN-family face generator? Fake writer profiles with AI faces
have been documented one outlet at a time (Sports Illustrated/AdVon 2023, Ventures Media
2025, We-News.com 2026). Large measurements of AI faces exist for Twitter/X profiles
(Yang, Singh and Menczer 2024, arXiv:2401.02627; Ricker et al. 2024, arXiv:2404.14244).
We found no published measurement across news outlets' own writer pages.

**Method (fixed before data).**
- Detector: OpenCV YuNet (2023mar), keeping images with exactly one face (score >= 0.8).
- Calibration: about 60 fresh faces from thispersondoesnotexist.com (StyleGAN2). Reference
  eye positions are the mean normalised eye centres. Held-out recall is measured on alternate
  images. (Changed in amendment 1: that site is now parked.)
- Test A ("aligned"): Yang et al.'s GANEyeDistance < 0.02 against that reference.
- Test B ("centred_level"): eye midpoint centred (within max(3 sd, 0.03) eye distances),
  level (within max(3 sd, 1 degree)), and eye height above centre within max(3 sd, 0.03)
  eye distances of the GAN mean. This test is unchanged by scaling and by centred crops.
- Real-face negatives: about 1,000 images from Labeled Faces in the Wild (every 13th file).
  LFW images are face-detector crops, so they are harder negatives than ordinary portraits.
- Positive control: we-news.com, whose writer profiles the site itself labels AI-generated.
- Population: Wikidata items that are newspapers, daily or online newspapers, news websites
  or news agencies, in the US, UK, Canada, Australia, Ireland, New Zealand or India, with an
  official website and no dissolution date. Shuffled with seed 20261117; up to 2,500 sites.
  Per site: robots.txt is honoured, then the home page, up to 3 articles, up to 3 author
  pages and up to 3 candidate headshots per author page.
- Images that appear on 3 or more author pages (default avatars) are counted separately.

**Predictions.**
1. Held-out recall of Test A on fresh StyleGAN faces is at least 90%.
2. On LFW, the false-flag rate is at most 10% for Test A and at most 15% for Test B.
3. On we-news.com, fewer than 50% of single-face headshots are flagged by either test. We
   expect its 2026 personas to come from diffusion models, which lack the fixed-eye artifact.
   If this holds, it is the method's main blind spot, and we will say so.
4. At least 25% of reachable population sites yield at least one single-face headshot.
5. Among unique single-face population headshots, fewer than 2% are flagged by Test A.
   After visual review, fewer than 1% of sites with headshots show a GAN face.

**What counts as a finding.** Only a flagged face that a person has looked at and judged
GAN-made (background artifacts, mismatched earrings or glasses, the aligned eyes), on a
named outlet's own writer page. Raw flags are not findings. No outlet or writer is named in
public until that review is done and the outlet has been asked for comment.

## Amendment 1 (2026-11-17, v0.2.0, written before any survey data was seen)

Survey 1 had not produced any output when these changes were made; the v0.1.0 job is
replaced by v0.2.0 before it runs.

- **Calibration source.** thispersondoesnotexist.com is now a parked "domain for sale"
  page, so v0.1.0 could not have collected any GAN faces. v0.2.0 calibrates on a fixed,
  citable sample instead: the first 300 StyleGAN faces ("1 Million Fake Faces") and the first
  300 real FFHQ photos, in file order, of `data/test-00000-of-00002.parquet` in the Hugging
  Face dataset TheKernel01/140k-Real-and-Fake-Faces (256 x 256 copies). Reference positions
  come from all fakes; held-out recall (prediction 1) uses alternate images, as before.
- **New prediction 1b.** Test A flags at least 80% of the real FFHQ photos, because FFHQ
  photos are cropped with the same alignment recipe that StyleGAN learned. This shows what
  the test really detects (the FFHQ crop, not synthesis), and is why every flag needs a
  person's review.
- **Positive control.** we-news.com now runs in Italian with 120 writers its own staff page
  (/la-redazione) calls AI editorial agents ("agenti IA editoriali"); its writer pages have
  no /author/ path. The control now reads the avatars on that staff page (up to 60).
  Prediction 3 is unchanged.
- **Article-page headshots.** On every site, byline headshots shown on the article pages
  themselves are now scored too (up to 2 per article; og:image is skipped there). Results
  are reported by image kind.
- **New group: documented cases.** Wayback Machine copies of writer pages on outlets where
  reporters documented invented writers: Sports Illustrated/AdVon (Futurism, Nov 2023,
  "AI-generated headshots"), South Florida Standard (The Florida Trib/WLRN, May 2026),
  We-News.com's Canadian personas (Canadian Press, Oct 2026) and four Brown Brothers Media
  sites (Futurism, Sep 2026; headshot origin mixed, at least one was a stock photo). Up to 15
  archived pages per case, the named personas' pages first, 1 request per second.
- **Predictions 6-8.**
  6. At least one single-face headshot is recovered from a named persona's archived page.
  7. Among single-face headshots on named persona pages, fewer than half are flagged by
     either test (we expect most personas since 2023 to use diffusion or stock images).
  8. If any Sports Illustrated/AdVon persona headshot is recovered, at least one of them is
     flagged by Test A or Test B (those were reported as bought from an AI-headshot
     marketplace, where StyleGAN-type faces were the norm in 2023).

**What would be a "wow".** The first open measurement of synthetic writer faces across news
outlets, with any confirmed cases newsworthy in their own right. A clean negative ("none of
N outlets with headshots shows a StyleGAN face") is also worth reporting, with the blind
spot of prediction 3 stated next to it. The documented-cases group adds a second result:
whether the fake journalists exposed since 2023 still carry the StyleGAN signature at all.
