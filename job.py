#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Survey run for byline-faces (see PREREGISTRATION.md).

Stages, each written to --out-dir as soon as it finishes:
  1. calibration.json  eye positions of known StyleGAN faces (the "1 Million Fake Faces"
                       sample in Hugging Face's 140k-Real-and-Fake-Faces test shard), held-out
                       recall, and the flag rate on that shard's real FFHQ photos (which are
                       cropped with the same alignment recipe)
  2. negatives.json    false-flag rate on real faces (Labeled Faces in the Wild)
  3. control           we-news.com, whose staff page labels all its writers AI agents
  4. documented        Wayback Machine copies of writer pages on outlets where fake personas
                       were documented by reporters (sources in DOCUMENTED below)
  5. population        English-language news outlets listed in Wikidata
  sites.jsonl / records.jsonl / summary.json, and flagged/ (copies of flagged headshots,
  kept for human review and NOT for publication).
"""
import argparse
import json
import os
import random
import re
import tarfile
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import byline_faces as bf

LFW_URLS = [
    "https://ndownloader.figshare.com/files/5976018",
    "http://vis-www.cs.umass.edu/lfw/lfw.tgz",
]
# thispersondoesnotexist.com (used by v0.1.0) is now a parked domain, so calibration uses a
# fixed, citable StyleGAN sample instead.
HF_PARQUET = ("https://huggingface.co/datasets/TheKernel01/140k-Real-and-Fake-Faces/resolve/main/"
              "data/test-00000-of-00002.parquet")
HF_SIZE = 287361642
CONTROL_SITES = [("we-news.com", "https://we-news.com/",
                  "control:we-news (site labels its writers AI agents)",
                  ["https://we-news.com/la-redazione"])]

DOCUMENTED = [
    {"group": "documented:si-advon-2023",
     "source": "Futurism, 27 Nov 2023: Sports Illustrated product reviews under nonexistent "
               "writers with AI-generated headshots (AdVon Commerce)",
     "prefixes": ["si.com/review/author/"], "slugs": ["drew-ortiz", "sora-tanaka"]},
    {"group": "documented:south-florida-standard-2026",
     "source": "The Florida Trib / WLRN, May 2026: fabricated reporters with fake headshots",
     "prefixes": ["southfloridastandard.com/"], "slugs": ["sofia-delgado"]},
    {"group": "documented:we-news-ca-2026",
     "source": "Canadian Press, 3 Oct 2026: 121 'AI correspondents'; the site labels them AI",
     "prefixes": ["we-news.com/"],
     "slugs": ["leah-qammaniq", "kaylee-johns", "warren-lafferty", "gordon-halloway",
               "genevieve-tremblay"]},
    {"group": "documented:brown-brothers-media-2026",
     "source": "Futurism, 17 Sep 2026: invented writer personas on Brown Brothers Media sites "
               "(headshot origin mixed; at least one was a stock photo)",
     "prefixes": ["spacedaily.com/author/", "dmnews.com/author/", "vegoutmag.com/author/",
                  "siliconcanals.com/author/"],
     "slugs": ["katherine-chen", "james-whitfield", "rachel-vaughn", "jordan-cooper",
               "marlene-martin", "adam-kelton", "sarah-mitchell"]},
]

SPARQL = """
SELECT DISTINCT ?item ?site WHERE {
  VALUES ?type { wd:Q11032 wd:Q1110794 wd:Q1153191 wd:Q17232649 wd:Q192283 }
  VALUES ?country { wd:Q30 wd:Q145 wd:Q16 wd:Q408 wd:Q27 wd:Q664 wd:Q668 }
  ?item wdt:P31 ?type ; wdt:P17 ?country ; wdt:P856 ?site .
  FILTER NOT EXISTS { ?item wdt:P576 ?end }
}
LIMIT 8000
"""

G_EDGES = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5]


class Deadline(Exception):
    pass


class Run:
    def __init__(self, out_dir):
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.lock = threading.Lock()
        self.logf = open(os.path.join(out_dir, "log.txt"), "a", encoding="utf-8")
        self.recf = open(os.path.join(out_dir, "records.jsonl"), "a", encoding="utf-8")
        self.sitef = open(os.path.join(out_dir, "sites.jsonl"), "a", encoding="utf-8")
        self.records, self.sites = [], []
        self.flag_saved = 0

    def log(self, *parts):
        line = time.strftime("%H:%M:%S ") + " ".join(str(p) for p in parts)
        print(line, flush=True)
        with self.lock:
            self.logf.write(line + "\n")
            self.logf.flush()

    def add_record(self, rec):
        with self.lock:
            self.records.append(rec)
            self.recf.write(json.dumps(rec) + "\n")
            self.recf.flush()

    def add_site(self, site):
        with self.lock:
            self.sites.append(site)
            self.sitef.write(json.dumps(site) + "\n")
            self.sitef.flush()

    def save_flagged(self, group, sha, data, ctype, limit=800):
        with self.lock:
            if self.flag_saved >= limit:
                return None
            self.flag_saved += 1
        ext = "png" if "png" in ctype else "webp" if "webp" in ctype else "jpg"
        sub = os.path.join(self.out, "flagged", re.sub(r"[^A-Za-z0-9_.-]+", "_", bf.group_key(group))[:60])
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, sha[:16] + "." + ext)
        with open(path, "wb") as fh:
            fh.write(data)
        return os.path.relpath(path, self.out)

    def write_json(self, name, obj):
        with open(os.path.join(self.out, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)


def new_site(host, home, group, qid):
    return {"host": host, "home": home, "group": group, "qid": qid, "status": "ok",
            "articles": 0, "author_urls": 0, "author_pages": 0, "images_tried": 0,
            "single_face": 0, "aligned": 0, "centred_level": 0}


def score_into(run, det, calib, site, data, ctype, extra):
    """Score one image, update the site's counters, save it if flagged, record it."""
    site["images_tried"] += 1
    sha = bf.sha256(data)
    rec = det.score(data, calib)
    rec.update(extra)
    rec.update({"host": site["host"], "group": site["group"], "sha256": sha, "bytes": len(data)})
    if "geom" in rec:
        site["single_face"] += 1
        site["aligned"] += int(rec["aligned"])
        site["centred_level"] += int(rec["centred_level"])
        if rec["aligned"] or rec["centred_level"]:
            rec["saved"] = run.save_flagged(site["group"], sha, data, ctype or "")
    run.add_record(rec)
    return rec


# ---------------------------------------------------------------------------

def writable_dir(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".probe")
        with open(probe, "w") as fh:
            fh.write("x")
        os.remove(probe)
        return True
    except OSError:
        return False


def download(run, url, path, expect_size=None, timeout=120):
    if os.path.exists(path) and (expect_size is None or os.path.getsize(path) == expect_size):
        run.log("cached", path, os.path.getsize(path))
        return True
    tmp = path + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": bf.USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    size = os.path.getsize(tmp)
    run.log("downloaded", url, size, "expected", expect_size)
    if size < 1_000_000:
        os.remove(tmp)
        return False
    os.replace(tmp, path)
    return True


def iter_parquet_faces(path):
    """(label, jpeg bytes) for each row of the 140k-Real-and-Fake-Faces shard
    (label 1 = StyleGAN fake, 0 = real FFHQ)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    lab_col = "label" if "label" in names else "generator"
    for batch in pf.iter_batches(batch_size=64, columns=["image", lab_col]):
        imgs = batch.column(batch.schema.get_field_index("image"))
        labels = batch.column(batch.schema.get_field_index(lab_col)).to_pylist()
        for i, lab in enumerate(labels):
            cell = imgs[i].as_py()
            data = cell.get("bytes") if isinstance(cell, dict) else cell
            if isinstance(lab, str):
                lab = 1 if "fake" in lab.lower() or "gan" in lab.lower() else 0
            if data and lab is not None:
                yield int(lab), data


def stage_calibration(run, det, data_dir, n_fake, n_real, deadline):
    sample_dir = os.path.join(run.out, "calibration_samples")
    os.makedirs(sample_dir, exist_ok=True)
    path = os.path.join(data_dir, "hf140k-test-00000-of-00002.parquet")
    result = {"source": HF_PARQUET, "detector": det.name}
    if not download(run, HF_PARQUET, path, HF_SIZE):
        result.update({"calibration": None, "note": "download failed; scoring uses FALLBACK_CALIB"})
        run.write_json("calibration.json", result)
        return None
    fake_geoms, real_geoms, seen = [], [], {0: 0, 1: 0}
    saved = 0
    for lab, data in iter_parquet_faces(path):
        if time.time() > deadline:
            break
        if seen[1] >= n_fake and seen[0] >= n_real:
            break
        if lab not in (0, 1) or (lab == 1 and seen[1] >= n_fake) or (lab == 0 and seen[0] >= n_real):
            continue
        seen[lab] += 1
        g = det.geometry(data)
        if lab == 1:
            if saved < 3:
                with open(os.path.join(sample_dir, "stylegan_%d.jpg" % saved), "wb") as fh:
                    fh.write(data)
                saved += 1
            if g:
                fake_geoms.append(g)
        elif g:
            real_geoms.append(g)
    result.update({"fake_images": seen.get(1, 0), "fake_single_face": len(fake_geoms),
                   "real_ffhq_images": seen.get(0, 0), "real_ffhq_single_face": len(real_geoms)})
    if len(fake_geoms) >= 10:
        train, test = fake_geoms[0::2], fake_geoms[1::2]
        c_train = bf.calibrate(train)
        flags = [bf.classify(g, c_train) for g in test]
        result["holdout"] = bf.rate_summary(flags)
        result["holdout"]["n_train"] = len(train)
        result["holdout"]["g_hist_edges"] = G_EDGES
        result["holdout"]["g_hist"] = bf.histogram([f["gan_eye_distance"] for f in flags], G_EDGES)
        rflags = [bf.classify(g, c_train) for g in real_geoms]
        result["real_ffhq"] = bf.rate_summary(rflags)
        result["real_ffhq"]["g_hist"] = bf.histogram([f["gan_eye_distance"] for f in rflags], G_EDGES)
        result["calibration"] = bf.calibrate(fake_geoms)
    else:
        result["calibration"] = None
        result["note"] = "fewer than 10 single-face StyleGAN images; scoring uses FALLBACK_CALIB"
    run.write_json("calibration.json", result)
    run.log("calibration", json.dumps({k: v for k, v in result.items() if k != "calibration"}))
    return result["calibration"]


def stage_negatives(run, det, calib, n, deadline, stride=13):
    out = {"source": None, "scored": 0, "single_face": 0}
    for url in LFW_URLS:
        if time.time() > deadline:
            break
        recs = []
        try:
            req = urllib.request.Request(url, headers={"User-Agent": bf.USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                tf = tarfile.open(fileobj=r, mode="r|gz")
                idx = 0
                for m in tf:
                    if not m.isfile() or not m.name.lower().endswith(".jpg"):
                        continue
                    idx += 1
                    if idx % stride:
                        continue
                    data = tf.extractfile(m).read()
                    recs.append(det.score(data, calib))
                    if len(recs) >= n or time.time() > deadline:
                        break
        except Exception as e:
            run.log("lfw error", url, type(e).__name__, str(e)[:160])
            if not recs:
                continue
        single = [r for r in recs if "geom" in r]
        out = {"source": url, "scored": len(recs), "single_face": len(single),
               "g_hist_edges": G_EDGES,
               "g_hist": bf.histogram([r["gan_eye_distance"] for r in single], G_EDGES)}
        out.update(bf.rate_summary(single))
        break
    run.write_json("negatives.json", out)
    run.log("negatives", json.dumps(out))
    return out


def crawl_site(run, det, calib, host, home, group, qid, opts, deadline, seed_pages=()):
    site = new_site(host, home, group, qid)
    tried = set()

    def get(url, accept, max_bytes=2_000_000):
        if time.time() > deadline:
            raise Deadline()
        return bf.fetch(url, max_bytes=max_bytes, accept=accept)

    def score_images(cands, page_url, limit, kind_prefix=""):
        n = 0
        for im in cands:
            if n >= limit:
                break
            if im["url"] in tried or not allowed(im["url"]):
                continue
            tried.add(im["url"])
            st, f4, ct4, b4 = get(im["url"], "image/avif;q=0,image/*", max_bytes=4_000_000)
            if st != 200 or not b4:
                continue
            n += 1
            score_into(run, det, calib, site, b4, ct4,
                       {"author_page": page_url, "img_url": f4, "img_kind": kind_prefix + im["kind"]})

    try:
        st, _f, ct, body = get(urljoin(home, "/robots.txt"), "text/plain")
        if st in (401, 403):
            site["status"] = "robots_forbidden"
            return site
        robots = bf.decode_text(body, ct) if st == 200 else ""

        def allowed(u):
            return (not bf.same_site(u, home)) or bf.robots_allows(robots, u)

        if not allowed(home):
            site["status"] = "robots_disallow"
            return site
        for sp in seed_pages:
            if not allowed(sp):
                continue
            st, fs, cts, bs = get(sp, "text/html")
            if st == 200:
                site["author_pages"] += 1
                score_images(bf.find_person_images(bf.decode_text(bs, cts), fs, meta=False),
                             fs, opts.get("max_seed_images", 60), "seed_")
        st, final, ct, body = get(home, "text/html")
        if st != 200 or "html" not in (ct or "").lower():
            site["status"] = "home_%s" % st
            return site
        html = bf.decode_text(body, ct)
        authors = bf.find_author_urls(html, final)
        arts = bf.find_article_urls(html, final, opts["max_articles"])
        site["articles"] = len(arts)
        for a in arts:
            if not allowed(a):
                continue
            st, f2, ct2, b2 = get(a, "text/html")
            if st != 200:
                continue
            text2 = bf.decode_text(b2, ct2)
            if len(authors) < opts["max_authors"]:
                for u in bf.find_author_urls(text2, f2):
                    if u not in authors:
                        authors.append(u)
            # byline headshots shown on the article page itself
            score_images(bf.find_person_images(text2, f2, meta=False), f2,
                         opts.get("max_article_images", 2), "article_")
        authors = authors[:opts["max_authors"]]
        site["author_urls"] = len(authors)
        for au in authors:
            if not allowed(au):
                continue
            st, f3, ct3, b3 = get(au, "text/html")
            if st != 200 or "html" not in (ct3 or "").lower():
                continue
            site["author_pages"] += 1
            score_images(bf.find_person_images(bf.decode_text(b3, ct3), f3), f3, opts["max_images"])
    except Deadline:
        site["status"] = "deadline"
    except Exception as e:
        site["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:100])
    return site


def crawl(run, det, calib, sites, opts, deadline, threads):
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(crawl_site, run, det, calib, h, home, group, qid, opts, deadline, seeds)
                for (h, home, group, qid, seeds) in sites]
        for fut in as_completed(futs):
            run.add_site(fut.result())
            done += 1
            if done % 100 == 0:
                run.log("sites done", done, "of", len(sites))


def stage_documented(run, det, calib, deadline, pages_per_group=15, images_per_page=2, pause=1.0):
    """Archived writer pages of outlets where reporters documented fake personas."""
    def wb_get(url, accept, max_bytes=3_000_000, timeout=45):
        if time.time() > deadline:
            raise Deadline()
        time.sleep(pause)
        st, f, ct, body = bf.fetch(url, max_bytes=max_bytes, timeout=timeout, accept=accept)
        if st == 429:
            run.log("wayback 429; waiting 60 s")
            time.sleep(60)
            st, f, ct, body = bf.fetch(url, max_bytes=max_bytes, timeout=timeout, accept=accept)
        return st, f, ct, body

    for spec in DOCUMENTED:
        site = new_site(spec["group"].split(":", 1)[1], "", spec["group"], "")
        site["source"] = spec["source"]
        site["persona_pages"] = 0
        site["persona_single_face"] = 0
        try:
            rows = []
            for prefix in spec["prefixes"]:
                st, _f, _ct, body = wb_get(bf.cdx_query_url(prefix), "application/json",
                                           max_bytes=20_000_000, timeout=90)
                if st == 200 and body.strip():
                    try:
                        got = bf.cdx_rows(json.loads(body.decode("utf-8", "replace")))
                    except ValueError:
                        got = []
                    rows.extend(got)
                    run.log("cdx", prefix, "rows", len(got))
                else:
                    run.log("cdx", prefix, "status", st)
            cands = bf.profile_candidates(rows, spec["slugs"], pages_per_group)
            site["author_urls"] = len(cands)
            seen_imgs = set()
            for ts, orig, slug_hit in cands:
                st, _f, ct, body = wb_get(bf.wayback_url(ts, orig, "id_"), "text/html")
                if st != 200 or not body:
                    continue
                site["author_pages"] += 1
                site["persona_pages"] += int(slug_hit)
                page = orig if "://" in orig else "http://" + orig
                n = 0
                for im in bf.find_person_images(bf.decode_text(body, ct), page):
                    if n >= images_per_page:
                        break
                    if im["url"] in seen_imgs:
                        continue
                    seen_imgs.add(im["url"])
                    st4, f4, ct4, b4 = wb_get(bf.wayback_url(ts, im["url"], "im_"), "image/*",
                                              max_bytes=4_000_000)
                    if st4 != 200 or not bf.looks_like_image(b4):
                        continue
                    n += 1
                    rec = score_into(run, det, calib, site, b4, ct4,
                                     {"author_page": orig, "capture": ts, "img_url": im["url"],
                                      "wayback_img": f4, "img_kind": im["kind"],
                                      "persona_page": slug_hit})
                    if slug_hit and "geom" in rec:
                        site["persona_single_face"] += 1
        except Deadline:
            site["status"] = "deadline"
        except Exception as e:
            site["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:100])
        run.add_site(site)
        run.log("documented", json.dumps(site))


def population_sites(run, max_sites, seed):
    url = "https://query.wikidata.org/sparql?format=json&query=" + urllib.parse.quote(SPARQL)
    try:
        st, _f, _ct, body = bf.fetch(url, max_bytes=50_000_000, timeout=120,
                                     accept="application/sparql-results+json")
        hosts = bf.wikidata_hosts(json.loads(body.decode("utf-8"))) if st == 200 else []
        run.log("wikidata status", st, "hosts", len(hosts))
    except Exception as e:
        run.log("wikidata error", type(e).__name__, str(e)[:160])
        hosts = []
    control = {c[0] for c in CONTROL_SITES}
    hosts = [h for h in hosts if h[0] not in control]
    random.Random(seed).shuffle(hosts)
    hosts = hosts[:max_sites]
    run.write_json("population.json", {"query": SPARQL, "seed": seed, "n": len(hosts),
                                       "hosts": [[h, q] for (h, _home, q) in hosts]})
    return [(h, home, "population", q, ()) for (h, home, q) in hosts]


def summarise(run, calib, negatives):
    groups = {}
    page_count = {}
    for r in run.records:
        if "sha256" in r:
            page_count.setdefault(r["sha256"], set()).add(r.get("author_page"))
    for s in run.sites:
        g = groups.setdefault(bf.group_key(s["group"]), {
            "sites": 0, "status": {}, "sites_with_author_page": 0,
            "sites_with_single_face": 0, "sites_flagged_aligned": 0, "sites_flagged_either": 0})
        g["sites"] += 1
        key = s["status"] if not s["status"].startswith("error") else "error"
        g["status"][key] = g["status"].get(key, 0) + 1
        g["sites_with_author_page"] += int(s["author_pages"] > 0)
        g["sites_with_single_face"] += int(s["single_face"] > 0)
        g["sites_flagged_aligned"] += int(s["aligned"] > 0)
        g["sites_flagged_either"] += int(s["aligned"] > 0 or s["centred_level"] > 0)
        if "source" in s:
            g["source"] = s["source"]
    for name, g in groups.items():
        recs = [r for r in run.records if bf.group_key(r.get("group", "")) == name and "geom" in r]
        uniq = {}
        for r in recs:
            uniq.setdefault(r["sha256"], r)
        shared = {h for h in uniq if len(page_count.get(h, ())) >= 3}
        u = list(uniq.values())
        g.update({
            "single_face_images": len(recs),
            "unique_single_face_images": len(u),
            "shared_images_3plus_pages": len(shared),
            "unique_aligned": sum(1 for r in u if r["aligned"]),
            "unique_centred_level": sum(1 for r in u if r["centred_level"]),
            "unique_either": sum(1 for r in u if r["aligned"] or r["centred_level"]),
            "unique_either_not_shared": sum(1 for r in u if (r["aligned"] or r["centred_level"])
                                            and r["sha256"] not in shared),
            "by_kind": {},
            "g_hist_edges": G_EDGES,
            "g_hist": bf.histogram([r["gan_eye_distance"] for r in u], G_EDGES),
        })
        for r in u:
            k = g["by_kind"].setdefault(r.get("img_kind", "?"), {"n": 0, "either": 0})
            k["n"] += 1
            k["either"] += int(r["aligned"] or r["centred_level"])
        persona = [r for r in u if r.get("persona_page")]
        if persona:
            g["persona_unique_single_face"] = len(persona)
            g["persona_either"] = sum(1 for r in persona if r["aligned"] or r["centred_level"])
    summary = {"version": bf.__version__, "detector": None, "calibration_n": (calib or {}).get("n", 0),
               "negatives": negatives, "groups": groups,
               "note": "Flags are leads for human review, not findings. flagged/ is not for publication."}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--time-budget-min", type=float, default=160)
    ap.add_argument("--calib-fake", type=int, default=300)
    ap.add_argument("--calib-real", type=int, default=300)
    ap.add_argument("--neg-n", type=int, default=1000)
    ap.add_argument("--max-sites", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--skip-documented", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    end = t0 + args.time_budget_min * 60
    run = Run(args.out_dir)
    run.log("byline-faces", bf.__version__, "budget min", args.time_budget_min)
    data_dir = args.data_dir if writable_dir(args.data_dir) else os.path.join(args.out_dir, "..", "tmp-data")
    if not writable_dir(data_dir):
        data_dir = "/tmp"
    run.log("data dir", data_dir)
    det = bf.FaceDetector(model_path=os.path.join(data_dir, "yunet_2023mar.onnx"))
    run.log("detector", det.name)

    calib = None
    try:
        calib = stage_calibration(run, det, data_dir, args.calib_fake, args.calib_real,
                                  min(end, time.time() + 15 * 60))
    except Exception as e:
        run.log("calibration failed", type(e).__name__, str(e)[:200])
    negatives = {}
    try:
        negatives = stage_negatives(run, det, calib, args.neg_n, min(end, time.time() + 15 * 60))
    except Exception as e:
        run.log("negatives failed", type(e).__name__, str(e)[:200])

    control_opts = {"max_articles": 6, "max_authors": 40, "max_images": 2,
                    "max_seed_images": 60, "max_article_images": 2}
    pop_opts = {"max_articles": 3, "max_authors": 3, "max_images": 3, "max_article_images": 2}
    try:
        crawl(run, det, calib, [(h, home, g, "", seeds) for (h, home, g, seeds) in CONTROL_SITES],
              control_opts, min(end, time.time() + 10 * 60), threads=1)
    except Exception as e:
        run.log("control failed", type(e).__name__, str(e)[:200])
    if not args.skip_documented:
        try:
            stage_documented(run, det, calib, min(end, time.time() + 30 * 60))
        except Exception as e:
            run.log("documented failed", type(e).__name__, str(e)[:200])
    try:
        sites = population_sites(run, args.max_sites, args.seed)
        crawl(run, det, calib, sites, pop_opts, end - 5 * 60, threads=args.threads)
    except Exception as e:
        run.log("population failed", type(e).__name__, str(e)[:200])

    summary = summarise(run, calib, negatives)
    summary["detector"] = det.name
    summary["minutes"] = round((time.time() - t0) / 60, 1)
    run.write_json("summary.json", summary)
    run.log("summary", json.dumps(summary)[:4000])


if __name__ == "__main__":
    main()
