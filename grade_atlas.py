#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Grader for survey 3, the fake-journalist face atlas (PREREGISTRATION-3.md, A1-A11).

Written and published on 17 November 2026, before the survey 3 job had returned any data,
so the way each prediction is scored is fixed before anyone sees a result. The rules for
the points the pre-registration leaves open are in GRADING-3.md.

Inputs are the job's two output files:
  atlas_summary.json   (cases in atlas.CASES order, with year, period, origin; cf_repro_ok;
                        the job's own "primary" numbers, used here as a consistency check)
  atlas_faces.jsonl    (one line per persona face: case, archived_page, sha256,
                        eye_aligned, eye_centred_level, cf)
and, optionally, a private review file (one JSON object per line with "sha256" and
"is_persona_headshot": true/false). With --review, only faces a person marked as a
persona's headshot are graded; that is a sensitivity analysis, reported next to the
mechanical grades, never instead of them.

Verdicts: true, false, no_data (nothing to grade), invalid (needs Community Forensics but
A1 failed), undetermined (a majority rule ended in a tie), partial (A11: one case true,
the other without data).

Usage:
  python grade_atlas.py atlas_summary.json atlas_faces.jsonl [--review review.jsonl]
                        [--out grades.json]
Standard library only.
"""
import argparse
import json
import math
import sys

__version__ = "0.4.1"

CF_T = 0.5
EPS = 1e-9
EARLY, LATE = "2019-2021", "2025-2026"
GAN_ERA_CASES = ("oliver-taylor-2020", "uae-network-labba-2020", "peace-data-2020",
                 "naebc-2020")
AI_ORIGINS = ("stylegan", "diffusion", "ai-unspecified")
BADANI = "uae-network-badani-2020"
IRISH_TIMES = "irish-times-2023"
SI_ADVON, DREW_ORTIZ = "si-advon-2023", "drew-ortiz"
SUSPECTED_2026 = ("watermark-b2b-2026", "paradise-media-pa-2026")


def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def eye(f):
    return bool(f.get("eye_aligned") or f.get("eye_centred_level"))


def cf_hit(f):
    return f.get("cf") is not None and f["cf"] >= CF_T


def either(f, cf_valid):
    return eye(f) or (cf_valid and cf_hit(f))


def share(items, pred):
    n = len(items)
    k = sum(1 for f in items if pred(f))
    return {"n": n, "k": k, "share": (k / n) if n else None, "wilson95": wilson(k, n)}


def verdict(cond):
    if cond is None:
        return "no_data"
    return "true" if cond else "false"


def majority(items, pred):
    """Verdict for a single-persona prediction: more than half of the persona's faces."""
    n = len(items)
    k = sum(1 for f in items if pred(f))
    if not n:
        v = "no_data"
    elif 2 * k > n:
        v = "true"
    elif 2 * k == n:
        v = "undetermined"
    else:
        v = "false"
    return {"n": n, "k": k, "verdict": v}


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_review(path):
    return {r["sha256"]: r.get("is_persona_headshot") is True
            for r in load_jsonl(path) if r.get("sha256")}


def apply_review(faces, review):
    if review is None:
        return list(faces)
    return [f for f in faces if review.get(f.get("sha256")) is True]


def by_case(faces, case_ids):
    out = {cid: [] for cid in case_ids}
    for f in faces:
        if f.get("case") in out and f.get("sha256"):
            out[f["case"]].append(f)
    return out


def pool(fbc, case_ids):
    """Faces of the given cases, de-duplicated by SHA-256 (first case in order wins)."""
    seen, out = set(), []
    for cid in case_ids:
        for f in fbc.get(cid, []):
            if f["sha256"] in seen:
                continue
            seen.add(f["sha256"])
            out.append(f)
    return out


def pool_by_period(cases, fbc):
    """As atlas.analyse_atlas: every case in order, one global SHA-256 set, each face
    counted in the period of the first case it appears in."""
    seen, out = set(), {}
    for cid, meta in cases.items():
        for f in fbc.get(cid, []):
            if f["sha256"] in seen:
                continue
            seen.add(f["sha256"])
            out.setdefault(meta.get("period"), []).append(f)
    return out


def case_mean_eye(cases, fbc, period):
    vals = [share(fbc[cid], eye)["share"] for cid, meta in cases.items()
            if meta.get("period") == period and fbc.get(cid)]
    return sum(vals) / len(vals) if vals else None


def grade(summary, faces, review=None):
    cases = summary.get("cases") or {}
    case_ids = list(cases)
    cf_valid = summary.get("cf_repro_ok") is True
    faces = apply_review(faces, review)
    fbc = by_case(faces, case_ids)
    basis = "both tests" if cf_valid else "eye test only (A1 failed)"
    g = {}

    g["A1"] = {"verdict": "true" if cf_valid else "false",
               "cf_repro_ok": summary.get("cf_repro_ok")}

    with_faces = [cid for cid in case_ids if fbc[cid]]
    g["A2"] = {"verdict": verdict(len(with_faces) >= 10) if case_ids else "no_data",
               "cases": len(case_ids), "cases_with_persona_faces": len(with_faces),
               "needed": 10}

    s = share(pool(fbc, GAN_ERA_CASES), eye)
    g["A3"] = dict(s, verdict=verdict(s["share"] >= 0.5 - EPS if s["n"] else None))

    periods = pool_by_period(cases, fbc)
    early = share(periods.get(EARLY, []), eye)
    late = share(periods.get(LATE, []), eye)
    diff = (late["share"] - early["share"]
            if early["share"] is not None and late["share"] is not None else None)
    cm_e, cm_l = case_mean_eye(cases, fbc, EARLY), case_mean_eye(cases, fbc, LATE)
    cdiff = cm_l - cm_e if cm_e is not None and cm_l is not None else None
    cond = None if diff is None or cdiff is None else (diff <= -0.20 + EPS and
                                                       cdiff <= -0.20 + EPS)
    a4 = {"verdict": verdict(cond), "eye_2019_2021": early, "eye_2025_2026": late,
          "diff_pooled": diff, "case_mean_2019_2021": cm_e, "case_mean_2025_2026": cm_l,
          "diff_case_mean": cdiff, "needed": "both differences <= -0.20"}
    job_primary = summary.get("primary") or {}
    if review is None and diff is not None and \
            job_primary.get("diff_late_minus_early") is not None:
        a4["matches_job_summary"] = abs(job_primary["diff_late_minus_early"] - diff) < 1e-6
    g["A4"] = a4

    late_items = periods.get(LATE, [])
    s = share(late_items, lambda f: cf_hit(f) and not eye(f))
    g["A5"] = dict(s, verdict=("invalid" if not cf_valid else
                               verdict(s["share"] >= 0.25 - EPS if s["n"] else None)))

    items = fbc.get(BADANI, [])
    flagged = sum(1 for f in items if either(f, cf_valid))
    g["A6"] = {"n": len(items), "flagged": flagged, "basis": basis,
               "verdict": verdict(flagged == 0 if items else None)}

    m = majority(fbc.get(IRISH_TIMES, []), lambda f: cf_hit(f) and not eye(f))
    if not cf_valid:
        m["verdict"] = "invalid"
    g["A7"] = m

    drew = [f for f in fbc.get(SI_ADVON, [])
            if DREW_ORTIZ in (f.get("archived_page") or "").lower()]
    g["A8"] = majority(drew, eye)

    ai_ids = [cid for cid, meta in cases.items() if meta.get("origin") in AI_ORIGINS]
    s = share(pool(fbc, ai_ids), lambda f: either(f, cf_valid))
    g["A9"] = dict(s, basis=basis, cases=ai_ids,
                   verdict=verdict(s["share"] >= 0.5 - EPS if s["n"] else None))

    stolen_ids = [cid for cid, meta in cases.items() if meta.get("origin") == "stolen"]
    s = share(pool(fbc, stolen_ids), lambda f: either(f, cf_valid))
    g["A10"] = dict(s, basis=basis, cases=stolen_ids,
                    verdict=verdict(s["share"] <= 0.20 + EPS if s["n"] else None))

    per = {}
    for cid in SUSPECTED_2026:
        s = share(fbc.get(cid, []), cf_hit)
        v = "invalid" if not cf_valid else verdict(s["share"] >= 0.5 - EPS if s["n"] else None)
        per[cid] = dict(s, verdict=v)
    vs = [p["verdict"] for p in per.values()]
    if not cf_valid:
        overall = "invalid"
    elif "false" in vs:
        overall = "false"
    elif all(v == "true" for v in vs):
        overall = "true"
    elif all(v == "no_data" for v in vs):
        overall = "no_data"
    else:
        overall = "partial"
    g["A11"] = {"verdict": overall, "cases": per}

    return {"grader": "grade_atlas.py " + __version__, "preregistration": "PREREGISTRATION-3.md",
            "job_version": summary.get("version"), "review_applied": review is not None,
            "faces_graded": len(faces), "cf_valid": cf_valid, "predictions": g}


def _fmt(x):
    return "-" if x is None else ("%.3f" % x if isinstance(x, float) else str(x))


def to_markdown(result):
    lines = ["| Prediction | Verdict | Numbers |", "|---|---|---|"]
    for key, p in result["predictions"].items():
        nums = []
        for k in ("n", "k", "share", "flagged", "cases_with_persona_faces", "diff_pooled",
                  "diff_case_mean", "cf_repro_ok"):
            if k in p:
                nums.append("%s=%s" % (k, _fmt(p[k])))
        if key == "A11":
            nums = ["%s: %s (%s/%s)" % (cid, c["verdict"], c["k"], c["n"])
                    for cid, c in p["cases"].items()]
        lines.append("| %s | %s | %s |" % (key, p["verdict"], ", ".join(nums)))
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("summary")
    ap.add_argument("faces")
    ap.add_argument("--review")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    with open(args.summary, encoding="utf-8") as fh:
        summary = json.load(fh)
    faces = load_jsonl(args.faces)
    result = {"mechanical": grade(summary, faces)}
    if args.review:
        result["reviewed"] = grade(summary, faces, load_review(args.review))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
    for name, r in result.items():
        sys.stdout.write("## %s\n\n%s\n\n" % (name, to_markdown(r)))
    return result


if __name__ == "__main__":
    main()
