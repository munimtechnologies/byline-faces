#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Grader for surveys 1 and 2 (PREREGISTRATION.md: P1-P8 with 1b; PREREGISTRATION-2.md: E1-E9).

Written and published on 19 November 2026, while both survey jobs were still queued and
before either had returned any data, so the way each prediction is scored is fixed before
anyone sees a result. Rules for the points the pre-registrations leave open are in
GRADING-1-2.md.

Survey 1 inputs (job.py output folder):
  calibration.json, negatives.json, summary.json   (required)
  sites.jsonl                                      (optional; used for P4)
  records.jsonl                                    (optional; used for the P5 review)
Survey 2 inputs (eras.py output folder):
  calibration_cf.json, eras_summary.json, summary.json   (required)
  records.jsonl                                          (optional; used for E6 and E8)

Verdicts: true, false, no_data (nothing to grade), invalid (needs Community Forensics but
E1 failed), partial (a two-part prediction with one part true and the other without data),
pending_review (needs a person's by-eye review that has not been supplied), undetermined
(a majority rule ended in a tie). E5 also gets an "outcome": null, rise, fall or unresolved.

Usage:
  python grade_surveys12.py survey1 OUT_DIR [--review review.jsonl] [--out grades1.json]
  python grade_surveys12.py survey2 OUT_DIR [--out grades2.json]
Standard library only.
"""
import argparse
import json
import math
import os
import sys

__version__ = "0.5.1"

EPS = 1e-9
CF_T = 0.5
CONTROL_PREFIX = "control:we-news"
SEED_KIND = "seed_"
SI_ADVON = "documented:si-advon-2023"
ELLE = "documented:ventures-media-elle-be-2025"
ELLE_SOPHIE_SLUGS = ("sophie", "justine")
LIVE_WATERMARK = "documented-live"
RARE_FORMAT_SHARE = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def verdict(cond):
    if cond is None:
        return "no_data"
    return "true" if cond else "false"


def combine(parts):
    """Verdict of an 'all parts must hold' prediction from its parts' verdicts."""
    vs = list(parts)
    if not vs:
        return "no_data"
    if "invalid" in vs:
        return "invalid"
    if "false" in vs:
        return "false"
    if all(v == "true" for v in vs):
        return "true"
    if all(v == "no_data" for v in vs):
        return "no_data"
    return "partial"


def rate_part(k, n, ok):
    """{'k','n','share','wilson95','verdict'} where ok(share) decides the verdict."""
    share = (k / n) if n else None
    return {"k": k, "n": n, "share": share, "wilson95": wilson(k, n),
            "verdict": verdict(ok(share) if share is not None else None)}


def stat_part(d, key, ok):
    """A rate already computed by the job (d[key]), with its n."""
    if not isinstance(d, dict) or d.get(key) is None:
        return {"n": (d or {}).get("n") if isinstance(d, dict) else None, "share": None,
                "verdict": "no_data",
                "error": (d or {}).get("error") if isinstance(d, dict) else None}
    share = d[key]
    return {"n": d.get("n"), "share": share, "verdict": verdict(ok(share))}


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def group_key(group):
    """Same rule as byline_faces.group_key."""
    head = (group or "").split(":")[0]
    return group if head in ("control", "documented") else head


def find_group(groups, prefix):
    for name, g in (groups or {}).items():
        if name.startswith(prefix):
            return name, g
    return None, None


def eye(r):
    return bool(r.get("aligned") or r.get("centred_level"))


def cf_hit(r):
    return r.get("cf") is not None and r["cf"] >= CF_T


def unique_single_face(records, pred):
    """Records with one face that satisfy pred, first record per SHA-256."""
    seen, out = set(), []
    for r in records:
        if "geom" not in r or not r.get("sha256") or r["sha256"] in seen or not pred(r):
            continue
        seen.add(r["sha256"])
        out.append(r)
    return out


def majority(items, pred):
    n = len(items)
    k = sum(1 for r in items if pred(r))
    if not n:
        v = "no_data"
    elif 2 * k > n:
        v = "true"
    elif 2 * k == n:
        v = "undetermined"
    else:
        v = "false"
    return {"n": n, "k": k, "verdict": v}


# ---------------------------------------------------------------------------
# Survey 1
# ---------------------------------------------------------------------------

def reachable(site):
    """A population site counts as reachable if its home page was read: the crawl finished
    ('ok') or it got as far as articles, author pages or images before stopping."""
    return (site.get("status") == "ok" or site.get("articles", 0) > 0
            or site.get("author_pages", 0) > 0 or site.get("images_tried", 0) > 0)


def grade_survey1(calibration, negatives, summary, sites=None, records=None, review=None):
    g = {}
    cal = calibration or {}
    ho = cal.get("holdout")
    p = stat_part(ho, "rate_aligned", lambda s: s >= 0.90 - EPS)
    g["P1"] = dict(p, needed="held-out StyleGAN recall of Test A >= 0.90")

    p = stat_part(cal.get("real_ffhq"), "rate_aligned", lambda s: s >= 0.80 - EPS)
    g["P1b"] = dict(p, needed="Test A flags >= 0.80 of real FFHQ photos")

    neg = negatives or {}
    pa = stat_part(neg, "rate_aligned", lambda s: s <= 0.10 + EPS)
    pb = stat_part(neg, "rate_centred_level", lambda s: s <= 0.15 + EPS)
    g["P2"] = {"verdict": combine([pa["verdict"], pb["verdict"]]), "n": neg.get("n"),
               "test_a": pa, "test_b": pb, "needed": "LFW: Test A <= 0.10 and Test B <= 0.15"}

    groups = (summary or {}).get("groups") or {}
    name, ctrl = find_group(groups, CONTROL_PREFIX)
    k = n = 0
    for kind, c in ((ctrl or {}).get("by_kind") or {}).items():
        if kind.startswith(SEED_KIND):
            n += c.get("n", 0)
            k += c.get("either", 0)
    g["P3"] = dict(rate_part(k, n, lambda s: s < 0.50 - EPS), group=name,
                   needed="< 0.50 of we-news.com staff-page avatars flagged by either test")

    pop = groups.get("population") or {}
    if sites is not None:
        ps = [s for s in sites if group_key(s.get("group", "")) == "population"]
        reach = [s for s in ps if reachable(s)]
        k4 = sum(1 for s in reach if s.get("single_face", 0) > 0)
        g["P4"] = dict(rate_part(k4, len(reach), lambda s: s >= 0.25 - EPS),
                       basis="sites.jsonl", sites=len(ps))
    else:
        n4 = (pop.get("status") or {}).get("ok", 0)
        k4 = pop.get("sites_with_single_face", 0)
        g["P4"] = dict(rate_part(min(k4, n4), n4, lambda s: s >= 0.25 - EPS),
                       basis="summary.json (status ok only)", sites=pop.get("sites"))
    g["P4"]["needed"] = ">= 0.25 of reachable population sites yield a single-face headshot"

    a = rate_part(pop.get("unique_aligned", 0), pop.get("unique_single_face_images", 0),
                  lambda s: s < 0.02 - EPS)
    if review is None or records is None:
        b = {"verdict": "pending_review"}
    else:
        host_of = {r["sha256"]: r.get("host") for r in records
                   if r.get("sha256") and group_key(r.get("group", "")) == "population"}
        gan_hosts = {host_of[s] for s, judged in review.items()
                     if judged and s in host_of and host_of[s]}
        b = rate_part(len(gan_hosts), pop.get("sites_with_single_face", 0),
                      lambda s: s < 0.01 - EPS)
    vs = [a["verdict"], b["verdict"]]
    v5 = "pending_review" if b["verdict"] == "pending_review" and a["verdict"] != "false" \
        else combine(vs)
    g["P5"] = {"verdict": v5, "test_a_unique": a, "reviewed_sites": b,
               "needed": "Test A < 0.02 of unique headshots; after review < 0.01 of sites"}

    doc = {k_: v for k_, v in groups.items() if k_.startswith("documented:")}
    pn = sum(d.get("persona_unique_single_face", 0) for d in doc.values())
    pe = sum(d.get("persona_either", 0) for d in doc.values())
    g["P6"] = {"verdict": verdict(pn >= 1) if doc else "no_data", "persona_single_face": pn,
               "groups": sorted(doc)}
    g["P7"] = dict(rate_part(pe, pn, lambda s: s < 0.50 - EPS),
                   needed="< 0.50 of persona headshots flagged by either test")
    si = doc.get(SI_ADVON) or {}
    sn, se = si.get("persona_unique_single_face", 0), si.get("persona_either", 0)
    g["P8"] = {"verdict": verdict(se >= 1) if sn else "no_data", "n": sn, "k": se,
               "needed": "if any SI/AdVon persona headshot: at least one flagged"}
    return {"grader": "grade_surveys12.py " + __version__, "preregistration": "PREREGISTRATION.md",
            "job_version": (summary or {}).get("version"), "review_applied": review is not None,
            "predictions": g}


# ---------------------------------------------------------------------------
# Survey 2
# ---------------------------------------------------------------------------

def cf_validity(calibration_cf, eras_summary, summary):
    rep = (calibration_cf or {}).get("repro")
    if isinstance(rep, dict) and "ok" in rep:
        return rep.get("ok") is True, rep.get("max_abs_diff")
    for d in (eras_summary, summary):
        if isinstance(d, dict) and "cf_repro_ok" in d:
            return d.get("cf_repro_ok") is True, None
    return False, None


def rare_formats(by_format):
    total_a = sum((f.get("A") or {}).get("n", 0) for f in by_format.values())
    if not total_a:
        return sorted(by_format)
    return sorted(name for name, f in by_format.items()
                  if (f.get("A") or {}).get("n", 0) / total_a < RARE_FORMAT_SHARE)


def diff_without(by_format, drop):
    ka = na = kb = nb = 0
    for name, f in by_format.items():
        if name in drop:
            continue
        a, b = f.get("A") or {}, f.get("B_new") or {}
        ka += a.get("cf_0.5", 0)
        na += a.get("n", 0)
        kb += b.get("cf_0.5", 0)
        nb += b.get("n", 0)
    if not na or not nb:
        return None
    return kb / nb - ka / na


def e5_outcome(res):
    if not res or res.get("diff_Bnew_minus_A") is None:
        return None, None
    d = res["diff_Bnew_minus_A"]
    ci = res.get("boot95_outlets")
    if not ci:
        return d, None
    lo, hi = ci
    if lo > 0:
        return d, "rise"
    if hi < 0:
        return d, "fall"
    return d, "null"


def grade_survey2(calibration_cf, eras_summary, summary, records=None):
    g = {}
    cal = calibration_cf or {}
    es = eras_summary or {}
    sm = summary or {}
    cf_valid, max_diff = cf_validity(cal, es, sm)
    g["E1"] = {"verdict": "true" if cf_valid else "false", "max_abs_diff": max_diff,
               "needed": "the five reference scores within 0.02 each"}

    def gated(part):
        if not cf_valid:
            part = dict(part, verdict="invalid")
        return part

    sd = gated(stat_part(cal.get("sd15_faces_dff"), "rate_0.5", lambda s: s >= 0.80 - EPS))
    sg = gated(stat_part(cal.get("stylegan_140k"), "rate_0.5", lambda s: s >= 0.80 - EPS))
    g["E2"] = {"verdict": combine([sd["verdict"], sg["verdict"]]), "sd15": sd, "stylegan": sg,
               "needed": ">= 0.80 of SD 1.5 and of StyleGAN faces flagged"}

    lf = gated(stat_part(cal.get("lfw_real"), "rate_0.5", lambda s: s <= 0.10 + EPS))
    ff = gated(stat_part(cal.get("ffhq_real_140k"), "rate_0.5", lambda s: s <= 0.10 + EPS))
    iw = gated(stat_part(cal.get("imdbwiki_real_dff"), "rate_0.5", lambda s: s <= 0.15 + EPS))
    g["E3"] = {"verdict": combine([lf["verdict"], ff["verdict"], iw["verdict"]]),
               "lfw": lf, "ffhq": ff, "imdbwiki": iw,
               "needed": "LFW <= 0.10, FFHQ <= 0.10, IMDB-WIKI <= 0.15 flagged"}

    th = es.get("thresholds") or {}
    a05 = (th.get("0.5") or {}).get("A") or {}
    e4 = rate_part(a05.get("k", 0), a05.get("n", 0), lambda s: s <= 0.10 + EPS)
    g["E4"] = gated(dict(e4, needed="<= 0.10 of era-A primary headshots flagged at 0.5"))

    res = th.get("t95_A")
    d, outcome = e5_outcome(res)
    ci = (res or {}).get("boot95_outlets")
    if d is None or ci is None:
        v5 = "no_data"
    else:
        v5 = verdict(d < 0.02 - EPS and ci[0] <= 0 <= ci[1])
    fmt = es.get("by_format") or {}
    rare = rare_formats(fmt) if fmt else []
    d_common = diff_without(fmt, set(rare)) if fmt else None
    if outcome == "rise" and (d_common is None or d_common <= 0):
        outcome = "unresolved"
    d05, out05 = e5_outcome(th.get("0.5"))
    g["E5"] = gated({"verdict": v5, "outcome": outcome, "diff_t95": d, "boot95_t95": ci,
                     "t95_A": es.get("t95_A"), "n_primary": es.get("n_primary"),
                     "hosts_with_primary": es.get("hosts_with_primary"),
                     "secondary_0.5": {"diff": d05, "boot95": (th.get("0.5") or {}).get(
                         "boot95_outlets"), "outcome": out05},
                     "rare_formats_in_A": rare, "diff_0.5_common_formats": d_common,
                     "needed": "diff(B_new - A at T95) < 0.02 and its 95% interval includes 0"})
    if not cf_valid:
        g["E5"]["outcome"] = "invalid"

    cfg = sm.get("cf_groups") or {}
    if records is not None:
        items = unique_single_face(records, lambda r: (r.get("group") or "").startswith(
            CONTROL_PREFIX) and (r.get("img_kind") or "").startswith(SEED_KIND)
            and r.get("cf") is not None)
        e6 = dict(rate_part(sum(1 for r in items if cf_hit(r)), len(items),
                            lambda s: s >= 0.70 - EPS), basis="records.jsonl, staff-page avatars")
    else:
        _n, c = find_group(cfg, CONTROL_PREFIX)
        c = c or {}
        e6 = dict(rate_part(c.get("cf_0.5", 0), c.get("cf_scored", 0),
                            lambda s: s >= 0.70 - EPS), basis="summary.json, all control images")
    g["E6"] = gated(dict(e6, needed=">= 0.70 of we-news.com staff avatars flagged"))

    doc = {k: v for k, v in cfg.items() if k.startswith("documented:")}
    pn = sum(v.get("persona_n", 0) for v in doc.values())
    pk = sum(v.get("persona_cf_0.5", 0) for v in doc.values())
    po = sum(v.get("persona_cf_only", 0) for v in doc.values())
    share = rate_part(pk, pn, lambda s: s >= 0.50 - EPS)
    only = verdict(po >= 1) if pn else "no_data"
    g["E7"] = gated({"verdict": combine([share["verdict"], only]), "persona": share,
                     "cf_only": po, "groups": sorted(doc),
                     "needed": ">= 0.50 of persona headshots flagged, and >= 1 CF-only"})

    if records is None:
        g["E8"] = {"verdict": "no_data", "note": "needs records.jsonl"}
    else:
        items = unique_single_face(records, lambda r: r.get("group") == ELLE and r.get(
            "persona_page") and any(s in (r.get("author_page") or "").lower()
                                    for s in ELLE_SOPHIE_SLUGS))
        m = majority(items, lambda r: eye(r) and cf_hit(r))
        g["E8"] = gated(dict(m, needed="Sophie/Justine headshot flagged by an eye test and CF "
                                       "(majority of unique images)"))

    live = cfg.get(LIVE_WATERMARK) or {}
    g["E9"] = gated(dict(rate_part(live.get("cf_0.5", 0), live.get("cf_scored", 0),
                                   lambda s: s >= 0.50 - EPS),
                         needed=">= 0.50 of single-face images on the live Watermark sites"))
    return {"grader": "grade_surveys12.py " + __version__,
            "preregistration": "PREREGISTRATION-2.md", "job_version": es.get("version"),
            "cf_valid": cf_valid, "predictions": g}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt(x):
    if x is None:
        return "-"
    if isinstance(x, float):
        return "%.4f" % x
    return str(x)


def to_markdown(result):
    lines = ["| Prediction | Verdict | Numbers |", "|---|---|---|"]
    for key, p in result["predictions"].items():
        nums = ["%s=%s" % (k, _fmt(p[k])) for k in
                ("n", "k", "share", "outcome", "diff_t95", "max_abs_diff", "cf_only",
                 "persona_single_face") if k in p]
        lines.append("| %s | %s | %s |" % (key, p["verdict"], ", ".join(nums)))
    return "\n".join(lines)


def load_review(path):
    return {r["sha256"]: r.get("judged_gan") is True for r in load_jsonl(path)
            if r.get("sha256")}


def _opt(path, loader):
    return loader(path) if os.path.exists(path) else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("survey", choices=("survey1", "survey2"))
    ap.add_argument("out_dir")
    ap.add_argument("--review", help="survey 1 only: private by-eye review (sha256, judged_gan)")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    j = lambda name: os.path.join(args.out_dir, name)  # noqa: E731
    records = _opt(j("records.jsonl"), load_jsonl)
    if args.survey == "survey1":
        result = grade_survey1(load_json(j("calibration.json")), load_json(j("negatives.json")),
                               load_json(j("summary.json")), _opt(j("sites.jsonl"), load_jsonl),
                               records, load_review(args.review) if args.review else None)
    else:
        result = grade_survey2(load_json(j("calibration_cf.json")),
                               load_json(j("eras_summary.json")), load_json(j("summary.json")),
                               records)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
    sys.stdout.write(to_markdown(result) + "\n")
    return result


if __name__ == "__main__":
    main()
