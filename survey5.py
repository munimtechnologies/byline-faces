#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""byline-faces survey 5: one site, real staff faces replaced by invented ones.

VideoGamer, after its sale to Clickout Media, replaced its staff bylines with invented
writers whose profile pictures looked AI-made; one image file was named
"ChatGPT-Image-Oct-20-2025-...png" (Kotaku, 26 Feb 2026). Survey 5 scores the writer
headshots the same site showed on the same page template before the change (captures before
2025-09-01) and after it (captures from 2026-02-19), so a detector that separates the two
periods cannot be doing it because of cross-site differences in compression or cropping.

Pre-registered plan and predictions V1-V4: PREREGISTRATION-5.md (written 2026-11-24, before
any data). This code was written on 2026-11-27, also before any data for this survey and
before any result of surveys 1-4. Implementation rules the pre-registration leaves open,
fixed here before data:

* Writer pages are paths of the form /author/<slug>/ (no pagination, feeds or index).
* Before uses captures from 2023-01-01 to 2025-08-31 (the template era nearest the change),
  and for each writer page the LATEST capture in that window; After uses the EARLIEST
  capture of each writer page from 2026-02-19 on (nearest the reported replacement).
* Up to two person images per archived page (survey 3's rule); units are distinct images
  (SHA-256) per period with exactly one detected face.
* V3 is graded on the largest file (bytes) among single-face images whose URL contains
  "ChatGPT-Image"; every such score is reported.

Privacy: real former staff are never named. records.jsonl, sites.jsonl, log.txt and the
flagged folders are private. The public file survey5_faces.jsonl lists After faces by hash
and score only, naming a page only for the personas named in the reporting, plus the
secondary persona pages. Before faces appear only as counts and scores.

Pure functions need only the standard library; tests: test_survey5.py.
"""
import argparse
import io
import json
import math
import os
import re
import time
from urllib.parse import urlencode, urlparse

__version__ = "0.1.0"

WAYBACK = "https://web.archive.org"
PREFIX = "videogamer.com/author/"          # the CDX index folds "www."
BEFORE_END = "20250901000000"              # Before: captures strictly earlier than this
AFTER_START = "20260219000000"             # After: captures at or after this
BEFORE_FROM = "20230101"
THRESHOLD = 0.5
MIN_N = 5
NAMED_GENERATOR_MARK = "ChatGPT-Image"
KNOWN_PERSONA_SLUGS = ("brian-merrygold", "shooter-orson", "steven-danielson")

# Secondary persona pages (not in V1, V2 or V4), all named in published reporting.
SECONDARY = [
    {"persona": "Brian Merrygold", "page": "videogamer.com/author/brian-merrygold/",
     "ts": "20260301", "source": "Kotaku, 26 Feb 2026; Press Gazette, 27 Feb 2026"},
    {"persona": "Shooter Orson", "page": "videogamer.com/author/shooter-orson/",
     "ts": "20260301", "source": "Kotaku, 26 Feb 2026"},
    {"persona": "Steven Danielson", "page": "videogamer.com/author/steven-danielson/",
     "ts": "20260301", "source": "Kotaku, 26 Feb 2026"},
    {"persona": "Callum Mercer", "page": "esports-news.co.uk/author/callummercer/",
     "ts": "20260407", "source": "Press Gazette, 27 Feb 2026"},
    {"persona": "Patrick Johnson", "page": "icobench.com/author/patrickjohnson/",
     "ts": "20261120", "source": "Press Gazette, Nov 2026"},
]

PROFILE_RE = re.compile(r"^/author/([^/]+)/?$")
SIZE_SUFFIX = re.compile(r"-\d{2,4}x\d{2,4}(?=\.(?:jpe?g|png|webp|gif)(?:$|\?))", re.I)


def pad_ts(ts):
    return (str(ts or "") + "00000000000000")[:14]


def period_of_ts(ts):
    """'before', 'after' or 'between' for a Wayback timestamp; None if there is none."""
    if not ts:
        return None
    t = pad_ts(ts)
    if t < BEFORE_END:
        return "before"
    if t >= AFTER_START:
        return "after"
    return "between"


def author_slug(url):
    """The writer slug of a /author/<slug>/ page (lower case), else None."""
    u = url or ""
    path = (urlparse(u if "://" in u else "http://" + u).path or "/").lower()
    m = PROFILE_RE.match(path)
    return m.group(1) if m else None


def cdx_url(prefix, frm=None, to=None, limit=50000):
    """CDX query for every HTML capture with status 200 under prefix (no collapsing, so
    each page's captures over time are all listed), optionally limited to [frm, to]."""
    params = [("url", prefix), ("matchType", "prefix"), ("output", "json"),
              ("fl", "timestamp,original,statuscode,mimetype"),
              ("filter", "statuscode:200"), ("filter", "mimetype:text/html")]
    if frm:
        params.append(("from", frm))
    if to:
        params.append(("to", to))
    params.append(("limit", str(limit)))
    return WAYBACK + "/cdx/search/cdx?" + urlencode(params)


def cdx_rows(cdx_json):
    """[(timestamp, original)] from CDX output=json (the first row is the header)."""
    if not isinstance(cdx_json, list) or not cdx_json:
        return []
    header = cdx_json[0]
    try:
        ti, oi = header.index("timestamp"), header.index("original")
    except (ValueError, AttributeError):
        return []
    return [(row[ti], row[oi]) for row in cdx_json[1:]
            if isinstance(row, list) and len(row) > max(ti, oi)]


def select_pages(rows, cap=80):
    """One capture per writer page and period: the latest Before capture and the earliest
    After capture. Each period is sorted by slug and capped; the two are interleaved so a
    time limit cannot starve one period."""
    best = {}
    for ts, orig in rows:
        per = period_of_ts(ts)
        if per not in ("before", "after"):
            continue
        slug = author_slug(orig)
        if not slug:
            continue
        key = (per, slug)
        t = pad_ts(ts)
        cur = best.get(key)
        if (cur is None or (per == "before" and t > pad_ts(cur[0]))
                or (per == "after" and t < pad_ts(cur[0]))):
            best[key] = (ts, orig)
    lists = {}
    for per in ("before", "after"):
        items = sorted((slug, ts, orig) for (p, slug), (ts, orig) in best.items() if p == per)
        lists[per] = [{"period": per, "slug": s, "ts": ts, "orig": o} for s, ts, o in items[:cap]]
    out = []
    b, a = lists["before"], lists["after"]
    for i in range(max(len(b), len(a))):
        if i < len(b):
            out.append(b[i])
        if i < len(a):
            out.append(a[i])
    return out


def full_size_variant(url):
    """WordPress thumbnail URL without its -WxH size suffix (the original upload)."""
    return SIZE_SUFFIX.sub("", url or "")


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


def auc(pos, neg):
    """Mann-Whitney AUC: chance a pos score beats a neg score (ties count half)."""
    if not pos or not neg:
        return None
    wins = 0.0
    for p in pos:
        for q in neg:
            wins += 1.0 if p > q else 0.5 if p == q else 0.0
    return wins / (len(pos) * len(neg))


def units(records):
    """{"before": [rec], "after": [rec]}: main-stage single-face images, one per SHA-256
    within each period."""
    out = {"before": {}, "after": {}}
    for r in records:
        if r.get("survey5_role") != "main":
            continue
        per = r.get("period")
        if per not in out or "geom" not in r or not r.get("sha256"):
            continue
        out[per].setdefault(r["sha256"], r)
    return {k: list(v.values()) for k, v in out.items()}


def period_stats(items, t=THRESHOLD):
    scores = [r["cf"] for r in items if r.get("cf") is not None]
    flagged = sum(1 for s in scores if s >= t)
    return {"n": len(items), "cf_scored": len(scores), "cf_flagged": flagged,
            "cf_share": flagged / len(scores) if scores else None,
            "cf_wilson95": wilson(flagged, len(scores)),
            "eye_flagged": sum(1 for r in items if eye_flag(r)),
            "cf_scores": sorted(round(s, 4) for s in scores)}


def named_generator_records(records):
    return [r for r in records if "geom" in r and NAMED_GENERATOR_MARK in
            ((r.get("img_url") or "") + " " + (r.get("wayback_img") or ""))]


def grade(records, repro_ok, t=THRESHOLD, min_n=MIN_N):
    """Verdicts for V1-V4 (PREREGISTRATION-5.md): pass, fail, invalid (the Community
    Forensics reproduction check did not pass) or unresolved (too few pictures)."""
    u = units(records)
    b, a = period_stats(u["before"], t), period_stats(u["after"], t)
    cf_ok = repro_ok is True
    enough_n = b["n"] >= min_n and a["n"] >= min_n
    enough_cf = b["cf_scored"] >= min_n and a["cf_scored"] >= min_n
    few = ("unresolved: too few archived pictures (before %d, after %d; %d needed in each)"
           % (b["n"], a["n"], min_n))
    grades = {}

    if not cf_ok:
        v1 = "invalid"
    elif not enough_cf:
        v1 = "unresolved"
    else:
        v1 = "pass" if a["cf_share"] >= 0.5 and b["cf_share"] <= 0.15 else "fail"
    grades["V1"] = {"verdict": v1, "after_share": a["cf_share"], "before_share": b["cf_share"],
                    "rule": "after share >= 0.50 and before share <= 0.15"}

    a_auc = auc([r["cf"] for r in u["after"] if r.get("cf") is not None],
                [r["cf"] for r in u["before"] if r.get("cf") is not None])
    if not cf_ok:
        v2 = "invalid"
    elif not enough_cf:
        v2 = "unresolved"
    else:
        v2 = "pass" if a_auc >= 0.80 else "fail"
    grades["V2"] = {"verdict": v2, "auc": a_auc, "rule": "AUC(after vs before) >= 0.80"}

    ng = [r for r in named_generator_records(records) if r.get("cf") is not None]
    chosen = max(ng, key=lambda r: r.get("bytes") or 0) if ng else None
    if not cf_ok:
        v3 = "invalid"
    elif chosen is None:
        v3 = "unresolved"
    else:
        v3 = "pass" if chosen["cf"] >= t else "fail"
    grades["V3"] = {"verdict": v3, "chosen_cf": None if chosen is None else chosen["cf"],
                    "all_scores": [round(r["cf"], 4) for r in ng],
                    "rule": "CF score of the ChatGPT-named headshot >= 0.50"}

    if not enough_n:
        v4 = "unresolved"
    else:
        v4 = "pass" if a["eye_flagged"] <= 1 else "fail"
    grades["V4"] = {"verdict": v4, "after_eye_flagged": a["eye_flagged"],
                    "rule": "at most one After headshot flagged by the StyleGAN eye test"}

    for g in grades.values():
        if g["verdict"] == "unresolved" and g is not grades["V3"]:
            g["note"] = few
    if v3 == "unresolved":
        grades["V3"]["note"] = "unresolved: no single-face ChatGPT-named image was scored"

    missed = []
    if v3 == "pass" and "fail" in (v1, v2):
        missed = sorted(({"sha256": r["sha256"], "cf": round(r["cf"], 4)} for r in u["after"]
                         if r.get("cf") is not None and r["cf"] < t),
                        key=lambda d: d["sha256"])
    shared = len({r["sha256"] for r in u["before"]} & {r["sha256"] for r in u["after"]})
    return {"version": __version__, "threshold_cf": t, "min_n": min_n, "cf_repro_ok": repro_ok,
            "periods": {"before": b, "after": a}, "auc_after_vs_before": a_auc,
            "grades": grades, "missed_after_faces": missed,
            "same_picture_in_both_periods": shared}


def _wb_page(ts, page):
    return "%s/web/%s/%s" % (WAYBACK, ts, page) if ts and page else ""


def public_faces(records):
    """Public lines: After faces by hash and scores (a page only for personas named in the
    reporting), then the secondary persona pages. No Before face is listed."""
    out = []
    for r in sorted(units(records)["after"], key=lambda r: r["sha256"]):
        e = {"role": "after", "sha256": r["sha256"],
             "cf": None if r.get("cf") is None else round(r["cf"], 4),
             "eye_aligned": bool(r.get("aligned")),
             "eye_centred_level": bool(r.get("centred_level")),
             "capture": r.get("capture", "")}
        if r.get("slug") in KNOWN_PERSONA_SLUGS:
            e["persona_slug"] = r["slug"]
            e["archived_page"] = _wb_page(r.get("capture"), r.get("author_page"))
        out.append(e)
    seen = set()
    for r in records:
        if r.get("survey5_role") != "secondary" or "geom" not in r:
            continue
        key = (r.get("persona"), r.get("sha256"))
        if key in seen:
            continue
        seen.add(key)
        out.append({"role": "secondary", "persona": r.get("persona"),
                    "archived_page": _wb_page(r.get("capture"), r.get("author_page")),
                    "wayback_image": r.get("wayback_img", ""), "sha256": r.get("sha256"),
                    "cf": None if r.get("cf") is None else round(r["cf"], 4),
                    "eye_aligned": bool(r.get("aligned")), "pixels": r.get("pixels")})
    return out


def pixel_size(data):
    try:
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return list(im.size)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# The job (needs the survey 1-3 modules, torch and opencv; imported lazily)
# ---------------------------------------------------------------------------

def wayback_getter(run, bf, job, deadline, pause=1.0, max_429=8):
    state = {"n429": 0}

    def get(url, accept, max_bytes=3_000_000, timeout=45):
        if time.time() > deadline:
            raise job.Deadline()
        time.sleep(pause)
        for attempt in (0, 1):
            try:
                st, f, ct, body = bf.fetch(url, max_bytes=max_bytes, timeout=timeout,
                                           accept=accept)
            except Exception as e:
                run.log("fetch error", type(e).__name__, str(e)[:120])
                return 0, url, "", b""
            if st != 429:
                return st, f, ct, body
            state["n429"] += 1
            run.log("wayback 429 number", state["n429"], "; waiting 60 s")
            if state["n429"] >= max_429:
                raise job.Deadline()
            time.sleep(60)
        return 429, url, "", b""
    return get


def stage_secondary(run, bf, eras, job, scorer, calib, wb_get):
    site = job.new_site("secondary", "", "survey5:secondary", "")
    log = []
    try:
        for s in SECONDARY:
            entry = {"persona": s["persona"], "page": s["page"], "source": s["source"]}
            log.append(entry)
            st, final, ct, body = wb_get(bf.wayback_url(s["ts"], s["page"], "id_"), "text/html")
            entry["page_status"] = st
            if st != 200 or not body:
                continue
            parsed = eras.parse_wayback_url(final)
            cap = parsed[0] if parsed else s["ts"]
            entry["capture"] = cap
            page = "https://" + s["page"]
            entry["images"] = []
            for im in bf.find_person_images(bf.decode_text(body, ct), page)[:2]:
                tries = [full_size_variant(im["url"]), im["url"]]
                for u in dict.fromkeys(tries):
                    st4, f4, ct4, b4 = wb_get(bf.wayback_url(cap, u, "im_"), "image/*",
                                              max_bytes=8_000_000)
                    if st4 != 200 or not bf.looks_like_image(b4):
                        continue
                    rec = job.score_into(run, scorer, calib, site, b4, ct4,
                                         {"author_page": s["page"], "capture": cap,
                                          "img_url": u, "wayback_img": f4,
                                          "img_kind": im["kind"], "survey5_role": "secondary",
                                          "persona": s["persona"], "pixels": pixel_size(b4)})
                    entry["images"].append({"url": u, "pixels": rec.get("pixels"),
                                            "single_face": "geom" in rec, "cf": rec.get("cf")})
                    break
    except job.Deadline:
        site["status"] = "deadline"
    except Exception as e:
        site["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:100])
    run.add_site(site)
    return log


def stage_main(run, bf, job, scorer, calib, pages, wb_get, images_per_page=2):
    site = job.new_site("videogamer.com", "", "survey5:videogamer", "")
    fetched = {"before": 0, "after": 0}
    seen = set()
    try:
        for p in pages:
            st, _f, ct, body = wb_get(bf.wayback_url(p["ts"], p["orig"], "id_"), "text/html")
            if st != 200 or not body:
                continue
            fetched[p["period"]] += 1
            site["author_pages"] += 1
            page = p["orig"] if "://" in p["orig"] else "http://" + p["orig"]
            n = 0
            for im in bf.find_person_images(bf.decode_text(body, ct), page):
                if n >= images_per_page:
                    break
                key = (p["period"], im["url"])
                if key in seen:
                    continue
                seen.add(key)
                st4, f4, ct4, b4 = wb_get(bf.wayback_url(p["ts"], im["url"], "im_"), "image/*",
                                          max_bytes=4_000_000)
                if st4 != 200 or not bf.looks_like_image(b4):
                    continue
                n += 1
                job.score_into(run, scorer, calib, site, b4, ct4,
                               {"author_page": p["orig"], "capture": p["ts"],
                                "img_url": im["url"], "wayback_img": f4,
                                "img_kind": im["kind"], "survey5_role": "main",
                                "period": p["period"], "slug": p["slug"]})
    except job.Deadline:
        site["status"] = "deadline"
    except Exception as e:
        site["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:100])
    run.add_site(site)
    return {"status": site["status"], "pages_fetched": fetched}


def main(argv=None):
    ap = argparse.ArgumentParser(description="byline-faces survey 5 (one site, before/after)")
    ap.add_argument("--out-dir", default="out5")
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--time-budget-min", type=float, default=100)
    ap.add_argument("--pages-per-period", type=int, default=80)
    ap.add_argument("--images-per-page", type=int, default=2)
    ap.add_argument("--pause", type=float, default=1.0)
    args = ap.parse_args(argv)

    import byline_faces as bf
    import commfor
    import eras
    import job

    t0 = time.time()
    end = t0 + args.time_budget_min * 60
    run = job.Run(args.out_dir)
    run.log("byline-faces survey 5", __version__)
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
    run.write_json("calibration_cf.json", cal)

    wb_get = wayback_getter(run, bf, job, end - 3 * 60, pause=args.pause)
    info = {"cdx": {}}
    rows = []
    try:
        for label, frm, to in (("before", BEFORE_FROM, "20250831"), ("after", "20260219", None)):
            st, _f, _ct, body = wb_get(cdx_url(PREFIX, frm, to), "application/json",
                                       max_bytes=60_000_000, timeout=120)
            got = []
            if st == 200 and body.strip():
                try:
                    got = cdx_rows(json.loads(body.decode("utf-8", "replace")))
                except ValueError:
                    got = []
            info["cdx"][label] = {"status": st, "rows": len(got)}
            run.log("cdx", label, st, len(got))
            rows.extend(got)
    except job.Deadline:
        info["cdx"]["stopped"] = "deadline"
    pages = select_pages(rows, args.pages_per_period)
    info["pages_selected"] = {per: sum(1 for p in pages if p["period"] == per)
                              for per in ("before", "after")}
    info["secondary"] = stage_secondary(run, bf, eras, job, scorer, eye_calib, wb_get)
    info["main"] = stage_main(run, bf, job, scorer, eye_calib, pages, wb_get,
                              args.images_per_page)

    kept_dir = os.path.join(args.out_dir, "flagged_cf")
    os.makedirs(kept_dir, exist_ok=True)
    for sha, data in list(scorer.kept.items()):
        with open(os.path.join(kept_dir, sha[:16] + "." + eras.image_format(data)), "wb") as fh:
            fh.write(data)
    summary = grade(run.records, (cal.get("repro") or {}).get("ok"))
    summary.update({"detector": scorer.name, "info": info,
                    "minutes": round((time.time() - t0) / 60, 1)})
    run.write_json("survey5_summary.json", summary)
    with open(os.path.join(args.out_dir, "survey5_faces.jsonl"), "w", encoding="utf-8") as fh:
        for line in public_faces(run.records):
            fh.write(json.dumps(line) + "\n")
    run.log("survey5 grades", json.dumps(summary["grades"])[:3000])


if __name__ == "__main__":
    main()
