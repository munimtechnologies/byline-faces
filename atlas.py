#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""byline-faces survey 3: the fake-journalist face atlas, 2020-2026.

Where did the people who invent journalists get their faces, and has that changed since
diffusion models became public in 2022? Every case below was exposed in published reporting
(source given). For each case the archived writer pages of the fake personas are read from
the Wayback Machine (the capture nearest the time they were live), and every single-face
headshot is scored by two tests:

* the StyleGAN eye-placement test (byline_faces.py; Yang et al. 2024), which only sees faces
  from the StyleGAN family (This Person Does Not Exist, Generated Photos and the like), and
* Community Forensics (commfor.py; Park and Owens, CVPR 2025), a general detector trained
  on images from 4,803 generators, diffusion models included.

The reporters' own statements about each case's faces (GAN, DALL-E, stolen real photos,
"AI-generated" without a method) are kept as labels, so the atlas is also the first test of
both detectors on fake-journalist headshots found in the wild.

Pre-registered predictions: PREREGISTRATION-3.md (written before any data was collected;
amendment 1, also before any data, added three cases). Pure functions (case table, persona
rules, the analysis) need only the standard library; tests: test_atlas.py.
"""
import argparse
import json
import math
import os
import re
import time
from urllib.parse import urlparse

__version__ = "0.4.0"

PERIODS = [("2019-2021", 2019, 2021), ("2022-2024", 2022, 2024), ("2025-2026", 2025, 2026)]

# Origin classes, from what the reporting itself says about the faces.
ORIGIN_CLASSES = {
    "stylegan": "GAN faces (StyleGAN family: GAN per experts, This Person Does Not Exist, "
                "Generated Photos)",
    "diffusion": "a named diffusion model (DALL-E)",
    "ai-unspecified": "called AI-generated, method not stated",
    "stolen": "real person's photo taken without consent",
    "mixed": "the reporting says some faces were stolen and some AI-generated",
    "unknown": "origin not stated",
}

# Every case is sourced. "prefixes" are Wayback CDX URL prefixes (archived pages under them
# are candidates); a page is a persona page when its path contains one of "slugs", or
# matches "persona_path_re" (staff/about pages of sites whose listed staff were all fake),
# or the whole site was fake ("whole_site_fake"). Named personas are given persona-specific
# prefixes on real outlets, so a common name cannot pick up a real writer's page elsewhere.
# The CDX index folds "www.", so prefixes never start with it.
CASES = [
    # ---- 2019-2021: before public diffusion models ----
    {"id": "oliver-taylor-2020", "year": 2020, "origin": "stylegan",
     "source": "Reuters (Raphael Satter), 15 Jul 2020, 'Deepfake used to attack activist couple "
               "shows new disinformation frontier': six experts said the photo of the "
               "invented freelance writer 'Oliver Taylor' has the characteristics of a "
               "deepfake; his pieces ran in the Times of Israel, the Jerusalem Post, The "
               "Algemeiner and Arutz Sheva",
     "prefixes": ["blogs.timesofisrael.com/author/oliver-taylor",
                  "algemeiner.com/author/oliver-taylor", "jpost.com/author/oliver-taylor"],
     "slugs": ["oliver-taylor"]},
    {"id": "uae-network-labba-2020", "year": 2020, "origin": "stylegan",
     "source": "The Daily Beast (Adam Rawnsley), 6 Jul 2020, 'Right-Wing Media Outlets Duped "
               "by a Middle East Propaganda Campaign': 19 fake personas, 90+ op-eds in 46 "
               "outlets; the 'Joseph Labba' photo posted with a Post Millennial article "
               "shows glitches typical of AI-generated faces",
     "prefixes": ["thepostmillennial.com/author/joseph-labba",
                  "thepostmillennial.com/author/josephlabba"],
     "slugs": ["joseph-labba", "josephlabba"]},
    {"id": "uae-network-badani-2020", "year": 2020, "origin": "stolen",
     "source": "The Daily Beast, 6 Jul 2020: 'Raphael Badani' photos were stolen from the blog "
               "of a real San Diego startup founder; Newsmax listed him as an Insider "
               "columnist and later removed the page",
     "prefixes": ["newsmax.com/insiders/raphaelbadani",
                  "washingtonexaminer.com/author/raphael-badani"],
     "slugs": ["raphaelbadani", "raphael-badani"]},
    {"id": "peace-data-2020", "year": 2020, "origin": "stylegan",
     "source": "Graphika, Sep 2020, 'IRA Again: Unlucky Thirteen' (via PBS NewsHour and The "
               "Next Web): the editor and editorial staff of PeaceData were fake personas "
               "whose profile pictures Graphika judged GAN-generated with high confidence "
               "(six of six); real freelancers also wrote for the site",
     "prefixes": ["peacedata.net/"], "slugs": ["lacusta"],
     "persona_path_re": r"about|team|staff|editor"},
    {"id": "naebc-2020", "year": 2020, "origin": "stylegan",
     "source": "Reuters, 1 Oct 2020, and Graphika 'Step into My Parler' (via Axios): NAEBC's "
               "editors were fake personas with GAN-generated profile photos; real "
               "Americans were paid to write for it",
     "prefixes": ["naebc.com/"], "slugs": ["holst", "krupka"],
     "persona_path_re": r"about|team|staff|editor"},
    # ---- 2022-2024 ----
    {"id": "si-advon-2023", "year": 2023, "origin": "stylegan",
     "source": "Futurism, 27 Nov 2023: Sports Illustrated product reviews under nonexistent "
               "writers (AdVon Commerce); the 'Drew Ortiz' headshot was for sale on a site "
               "that sells AI-generated headshots (Generated Photos), and the profile picture "
               "of TheStreet's 'Denise McNamara' was listed on the same marketplace",
     "prefixes": ["si.com/review/author/drew-ortiz", "si.com/review/author/sora-tanaka",
                  "thestreet.com/author/denise-mcnamara",
                  "thestreet.com/authors/denise-mcnamara"],
     "slugs": ["drew-ortiz", "sora-tanaka", "denise-mcnamara"]},
    {"id": "hoodline-2023", "year": 2023, "origin": "ai-unspecified",
     "source": "CNN (via WRAL), May 2024, and Nieman Lab, Jun 2024: Internet Archive captures "
               "show Hoodline's AI author personas with what appeared to be AI-generated "
               "headshots and fake bios in 2023, later replaced by an 'AI' badge",
     "prefixes": ["hoodline.com/author/eileen-vargas", "hoodline.com/author/leticia-ruiz",
                  "hoodline.com/author/nina-singh-hudson", "hoodline.com/author/eric-tanaka",
                  "hoodline.com/author/tony-ng", "hoodline.com/author/will-o",
                  "hoodline.com/author/sam-cavanaugh"],
     "slugs": ["eileen-vargas", "leticia-ruiz", "nina-singh-hudson", "eric-tanaka", "tony-ng",
               "will-obrien", "will-o-brien", "sam-cavanaugh"]},
    {"id": "irish-times-2023", "year": 2023, "origin": "diffusion",
     "source": "The Irish Times, 14 May 2023 (editor Ruadhan Mac Cormaic) and Irish "
               "Independent: the 'Adriana Acosta-Cortez' op-ed byline photo 'may have been "
               "produced, at least in part, using generative AI'; the hoaxer said it was "
               "made with DALL-E",
     "prefixes": ["irishtimes.com/opinion/2023/05/1"],
     "slugs": ["fake-tan"]},
    # Amendment 1 (before any data): DC Weekly, a second "stolen photos" case.
    {"id": "dc-weekly-2023", "year": 2023, "origin": "stolen",
     "source": "Clemson University Media Forensics Hub (Linvill and Warren), 2023, and "
               "NewsGuard, 29 May 2024: DC Weekly's journalists were invented; 'Jessica "
               "Devlin' used a photo of Canadian writer Judy Batalion, 'Lucas Turner' one of a "
               "Spain-based model, and other author profiles used photos taken from stock art "
               "or real people",
     "prefixes": ["dcweekly.org/author/", "dcweekly.org/"],
     "slugs": ["jessica-devlin", "lucas-turner", "tina-morris", "mandy-lewis"],
     "persona_path_re": r"^/author/"},
    # ---- 2025-2026 ----
    {"id": "elle-belgium-2025", "year": 2025, "origin": "mixed",
     "source": "VRT NWS, 26 Jun 2025: Elle Belgium's invented writers; 'Sophie Vermeulen' "
               "has a photo found in This Person Does Not Exist's database, 'Marta Peeters' a "
               "photo Sightengine rated 99% AI",
     "prefixes": ["elle.be/nl/author/"],
     "slugs": ["sophie", "marta", "justine"]},
    {"id": "south-florida-standard-2026", "year": 2026, "origin": "ai-unspecified",
     "source": "The Florida Trib with KCRW, May 2026: every staff member of the South Florida "
               "Standard was invented, each with an AI-generated headshot; the operators "
               "said all names were randomly generated by AI",
     "prefixes": ["southfloridastandard.com/author/", "southfloridastandard.com/"],
     "slugs": ["sofia-delgado"], "whole_site_fake": True},
    {"id": "brown-brothers-media-2026", "year": 2026, "origin": "mixed",
     "source": "Futurism, 17 Sep 2026: invented writer personas on Brown Brothers Media sites; "
               "headshot origin mixed, at least one a stock photo",
     "prefixes": ["spacedaily.com/author/", "dmnews.com/author/", "vegoutmag.com/author/",
                  "siliconcanals.com/author/"],
     "slugs": ["katherine-chen", "james-whitfield", "rachel-vaughn", "jordan-cooper",
               "marlene-martin", "adam-kelton", "sarah-mitchell"]},
    {"id": "we-news-2026", "year": 2026, "origin": "ai-unspecified",
     "source": "The Canadian Press, 3 Oct 2026: we-news.com's 121 'AI correspondents'; the "
               "site itself labels its writers AI",
     "prefixes": ["we-news.com/author/", "we-news.com/la-redazione", "we-news.com/"],
     "slugs": ["leah-qammaniq", "kaylee-johns", "warren-lafferty", "gordon-halloway",
               "genevieve-tremblay"], "whole_site_fake": True},
    {"id": "bogus-bylines-2026", "year": 2026, "origin": "mixed",
     "source": "OpenAI threat report, Oct 2026, and Marc Owen Jones (Dysinfluence, 9 Oct 2026, "
               "'Operation Bogus Bylines'): seven fabricated journalist personas of a pro-Iran "
               "operation used stolen and AI-generated profile images; outlets per persona "
               "as listed by Jones",
     "prefixes": ["fairobserver.com/author/alice-johnson",
                  "middleeastmonitor.com/author/alice-johnson",
                  "middleeastmonitor.com/author/michael-harrison",
                  "middleeastmonitor.com/author/jenny-williams",
                  "middleeastmonitor.com/author/ervin-b-hoskins",
                  "middleeastmonitor.com/author/ervin-hoskins",
                  "asiatimes.com/author/noah-lamington",
                  "intpolicydigest.org/author/noah-lamington",
                  "eurasiareview.com/author/ericka-feusier",
                  "fpif.org/authors/ericka-feusier", "fpif.org/author/ericka-feusier",
                  "counterpunch.org/author/sophia-gonzalez",
                  "laprogressive.com/author/sophia-gonzalez",
                  "znetwork.org/author/sophia-gonzalez",
                  "peacevoice.info/author/sophia-gonzalez",
                  "peacevoice.info/author/ervin-b-hoskins",
                  "nationofchange.org/author/jenny-williams",
                  "libertarianinstitute.org/author/alice-johnson",
                  "blackstarnews.com/author/ervin-b-hoskins"],
     "slugs": ["alice-johnson", "michael-harrison", "jenny-williams", "ervin-b-hoskins",
               "ervin-hoskins", "noah-lamington", "ericka-feusier", "sophia-gonzalez"]},
    {"id": "watermark-b2b-2026", "year": 2026, "origin": "ai-unspecified",
     "source": "Press Gazette (Rob Waugh), 17 Nov 2026: Watermark Publishing Group's B2B "
               "sites; reporters' tiny headshots scan as AI-generated (Made By Humans "
               "detector); the company says some are pen names, so these are suspected",
     "prefixes": ["securityguysnews.com/author/", "thesupplychainer.com/author/",
                  "warehouseinsider.news/author/", "securityguysnews.com/",
                  "thesupplychainer.com/", "warehouseinsider.news/"],
     "slugs": ["paul-epstein", "sarah-oneill", "sarah-o-neill", "paula-vettori",
               "ellie-goldman", "freddie-bolton", "robin-pattinson", "beth-mcmillan",
               "angela-lee", "amy-cohen", "david-donovan", "alex-badmington", "hannah-kohr",
               "evan-porter"]},
    # Amendment 1 (before any data): two 2026 cases.
    {"id": "daniel-robson-2026", "year": 2026, "origin": "ai-unspecified",
     "source": "CBC News, 30 Sep 2026, and Canadaland, 7 Oct 2026: the Ottawa Citizen, the "
               "Montreal Gazette, Policy Options and other Canadian outlets removed work by "
               "'Daniel Robson'; the Citizen's editor-in-chief said his headshot, 'based on "
               "online analysis tools, appeared to be AI generated'",
     "prefixes": ["ottawacitizen.com/author/daniel-robson",
                  "ottawacitizen.com/author/danielrobson",
                  "montrealgazette.com/author/daniel-robson",
                  "montrealgazette.com/author/danielrobson",
                  "thehub.ca/author/daniel-robson", "thehub.ca/author/danielrobson",
                  "westernstandard.news/author/daniel-robson",
                  "policyoptions.irpp.org/authors/daniel-robson",
                  "opencanada.org/authors/daniel-robson",
                  "opencanada.org/author/daniel-robson"],
     "slugs": ["daniel-robson", "danielrobson"]},
    {"id": "paradise-media-pa-2026", "year": 2026, "origin": "ai-unspecified",
     "source": "Billy Penn at WHYY (Julia Binswanger and Violet Comber-Wilen), 11 Nov 2026: "
               "suspicious author photos on the Paradise Media-run Northeast Times and Lower "
               "Bucks Times; Hive Detect rated the 'Ellen Kowalczyk' photo 99.9% likely "
               "AI-generated and the reporters could not rule out AI for the others, so these "
               "are suspected",
     "prefixes": ["northeasttimes.com/authors/", "northeasttimes.com/author/",
                  "lowerbuckstimes.com/author/", "lowerbuckstimes.com/authors/"],
     "slugs": ["ellen-kowalczyk", "maya-brooks", "ray-doyle", "pat-nowak", "tyler-dunphey"]},
]

CASE_BY_ID = {c["id"]: c for c in CASES}


def group_of(case):
    return "documented:" + case["id"]


def case_of_group(group):
    head, _, rest = (group or "").partition(":")
    return CASE_BY_ID.get(rest) if head == "documented" else None


def period_of(year):
    for name, lo, hi in PERIODS:
        if lo <= year <= hi:
            return name
    return None


def page_path(url):
    u = url or ""
    return (urlparse(u if "://" in u else "http://" + u).path or "/").lower()


def is_persona(rec, case):
    """Is this image from a page of a documented fake persona (not a real writer)?"""
    if case.get("whole_site_fake"):
        return True
    if rec.get("persona_page"):
        return True
    path = page_path(rec.get("author_page"))
    if any(s.lower() in path for s in case.get("slugs", ())):
        return True
    pat = case.get("persona_path_re")
    return bool(pat and re.search(pat, path))


def eye_flag(rec):
    return bool(rec.get("aligned") or rec.get("centred_level"))


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _counts(items, t):
    n = len(items)
    eye = sum(1 for r in items if eye_flag(r))
    cf_n = sum(1 for r in items if r.get("cf") is not None)
    cfh = sum(1 for r in items if r.get("cf") is not None and r["cf"] >= t)
    both = sum(1 for r in items if eye_flag(r) and r.get("cf") is not None and r["cf"] >= t)
    out = {"n": n, "eye": eye, "eye_share": eye / n if n else None, "eye_wilson95": wilson(eye, n),
           "cf_scored": cf_n, "cf_flagged": cfh,
           "cf_share": cfh / cf_n if cf_n else None, "cf_wilson95": wilson(cfh, cf_n),
           "both": both, "eye_only": eye - both, "cf_only": cfh - both,
           "neither": sum(1 for r in items if not eye_flag(r)
                          and not (r.get("cf") is not None and r["cf"] >= t))}
    out["cf_only_share"] = out["cf_only"] / n if n else None
    return out


def faces_by_case(records):
    """{case id: {"persona": [rec], "other": [rec]}}: single-face images, de-duplicated by
    SHA-256 within each case (one face reused on several outlets counts once)."""
    out = {}
    seen = {}
    for r in records:
        case = case_of_group(r.get("group"))
        if case is None or "geom" not in r or not r.get("sha256"):
            continue
        key = (case["id"], r["sha256"])
        persona = is_persona(r, case)
        if key in seen:
            if persona and not seen[key][1]:
                d = out[case["id"]]
                d["other"].remove(seen[key][0])
                d["persona"].append(seen[key][0])
                seen[key] = (seen[key][0], True)
            continue
        d = out.setdefault(case["id"], {"persona": [], "other": []})
        d["persona" if persona else "other"].append(r)
        seen[key] = (r, persona)
    return out


def analyse_atlas(records, t=0.5):
    """The pre-registered summary (PREREGISTRATION-3.md)."""
    by_case = faces_by_case(records)
    cases = {}
    pooled = {}
    origin = {}
    global_seen = set()
    for c in CASES:
        d = by_case.get(c["id"], {"persona": [], "other": []})
        per = d["persona"]
        entry = {"year": c["year"], "period": period_of(c["year"]), "origin": c["origin"],
                 "source": c["source"], "persona": _counts(per, t),
                 "other_writers_same_pages": _counts(d["other"], t),
                 "persona_cf_scores": sorted(round(r["cf"], 4) for r in per
                                             if r.get("cf") is not None)}
        cases[c["id"]] = entry
        for r in per:
            if r["sha256"] in global_seen:
                continue
            global_seen.add(r["sha256"])
            pooled.setdefault(entry["period"], []).append(r)
            origin.setdefault(c["origin"], []).append(r)
    periods = {}
    for name, _lo, _hi in PERIODS:
        items = pooled.get(name, [])
        p = _counts(items, t)
        with_faces = [cid for cid, e in cases.items()
                      if e["period"] == name and e["persona"]["n"] > 0]
        p["cases"] = sum(1 for e in cases.values() if e["period"] == name)
        p["cases_with_persona_faces"] = len(with_faces)
        # Case-level means: each case counts once, whatever its number of faces.
        shares = [cases[cid]["persona"]["eye_share"] for cid in with_faces]
        p["case_mean_eye_share"] = sum(shares) / len(shares) if shares else None
        cfs = [cases[cid]["persona"]["cf_share"] for cid in with_faces
               if cases[cid]["persona"]["cf_share"] is not None]
        p["case_mean_cf_share"] = sum(cfs) / len(cfs) if cfs else None
        periods[name] = p
    early, late = periods["2019-2021"], periods["2025-2026"]
    primary = {"eye_share_2019_2021": early["eye_share"], "eye_share_2025_2026": late["eye_share"],
               "diff_late_minus_early": (late["eye_share"] - early["eye_share"])
               if early["eye_share"] is not None and late["eye_share"] is not None else None,
               "case_mean_diff": (late["case_mean_eye_share"] - early["case_mean_eye_share"])
               if early["case_mean_eye_share"] is not None
               and late["case_mean_eye_share"] is not None else None,
               "cf_only_share_2025_2026": late["cf_only_share"]}
    return {"version": __version__, "threshold_cf": t, "periods": periods, "primary": primary,
            "by_origin": {k: _counts(v, t) for k, v in origin.items()},
            "origin_classes": ORIGIN_CLASSES, "cases": cases}


def public_faces(records):
    """One public line per persona face: the case, the archived page (Wayback URL), the image
    hash and both scores. All cases are already public in published reporting."""
    out = []
    for cid, d in sorted(faces_by_case(records).items()):
        for r in d["persona"]:
            cap = r.get("capture") or ""
            page = r.get("author_page") or ""
            out.append({"case": cid, "archived_page": "https://web.archive.org/web/%s/%s"
                        % (cap, page) if cap and page else "",
                        "wayback_image": r.get("wayback_img", ""), "sha256": r["sha256"],
                        "eye_aligned": bool(r.get("aligned")),
                        "eye_centred_level": bool(r.get("centred_level")),
                        "cf": None if r.get("cf") is None else round(r["cf"], 4)})
    return out


# ---------------------------------------------------------------------------
# The job (needs the survey 1-2 modules, torch and opencv; imported lazily)
# ---------------------------------------------------------------------------

def documented_specs():
    return [{"group": group_of(c), "source": c["source"], "prefixes": c["prefixes"],
             "slugs": c.get("slugs", [])} for c in CASES]


def main(argv=None):
    ap = argparse.ArgumentParser(description="byline-faces survey 3 (fake-journalist face atlas)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--time-budget-min", type=float, default=120)
    ap.add_argument("--calib-n", type=int, default=100)
    ap.add_argument("--pages-per-case", type=int, default=30)
    args = ap.parse_args(argv)

    import byline_faces as bf
    import commfor
    import eras
    import job

    t0 = time.time()
    end = t0 + args.time_budget_min * 60
    run = job.Run(args.out_dir)
    run.log("byline-faces atlas", __version__, "cases", len(CASES))
    data_dir = args.data_dir if job.writable_dir(args.data_dir) else "/tmp"
    faces = bf.FaceDetector(model_path=os.path.join(data_dir, "yunet_2023mar.onnx"))
    cf = None
    try:
        wpath = os.path.join(data_dir, "commfor-model-384.safetensors")
        if commfor.ensure_weights(wpath, lambda u, p, s: job.download(run, u, p, s, timeout=600)):
            cf = commfor.CommFor(wpath)
    except Exception as e:
        run.log("commfor load failed", type(e).__name__, str(e)[:300])
    scorer = eras.Scorer(faces, cf)
    cal = {"model": getattr(cf, "name", None)}
    if cf is not None:
        cal["repro"] = commfor.reproduce(
            cf, lambda u: bf.fetch(u, max_bytes=20_000_000, timeout=60))
        run.log("commfor repro", json.dumps(cal["repro"]))
    eye_calib = None
    try:
        eye_calib = job.stage_calibration(run, faces, data_dir, 300, 300,
                                          min(end, time.time() + 12 * 60))
    except Exception as e:
        run.log("eye calibration failed", type(e).__name__, str(e)[:200])
    if cf is not None and args.calib_n > 0:
        try:
            cal.update(eras.stage_cf_calibration(run, cf, data_dir, args.calib_n,
                                                 min(end, time.time() + 20 * 60)))
        except Exception as e:
            run.log("cf calibration failed", type(e).__name__, str(e)[:200])
    run.write_json("calibration_cf.json", cal)

    job.DOCUMENTED = documented_specs()
    try:
        job.stage_documented(run, scorer, eye_calib, end - 4 * 60,
                             pages_per_group=args.pages_per_case)
    except Exception as e:
        run.log("documented failed", type(e).__name__, str(e)[:200])

    kept_dir = os.path.join(args.out_dir, "flagged_cf")
    os.makedirs(kept_dir, exist_ok=True)
    for sha, data in list(scorer.kept.items()):
        with open(os.path.join(kept_dir, sha[:16] + "." + eras.image_format(data)), "wb") as fh:
            fh.write(data)
    summary = analyse_atlas(run.records)
    summary.update({"cf_repro_ok": (cal.get("repro") or {}).get("ok"), "detector": scorer.name,
                    "sites": run.sites, "minutes": round((time.time() - t0) / 60, 1)})
    run.write_json("atlas_summary.json", summary)
    with open(os.path.join(args.out_dir, "atlas_faces.jsonl"), "w", encoding="utf-8") as fh:
        for line in public_faces(run.records):
            fh.write(json.dumps(line) + "\n")
    run.log("atlas summary", json.dumps({k: summary[k] for k in ("primary", "periods")})[:4000])


if __name__ == "__main__":
    main()
