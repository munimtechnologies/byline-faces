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
  images.
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

**What would be a "wow".** The first open measurement of synthetic writer faces across news
outlets, with any confirmed cases newsworthy in their own right. A clean negative ("none of
N outlets with headshots shows a StyleGAN face") is also worth reporting, with the blind
spot of prediction 3 stated next to it.
