#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Survey run for byline-faces (see PREREGISTRATION.md).

Stages, each written to --out-dir as soon as it finishes:
  1. calibration.json  eye positions of freshly generated StyleGAN faces
                       (thispersondoesnotexist.com), with a held-out recall check
  2. negatives.json    false-flag rate on real faces (Labeled Faces in the Wild)
  3. control           we-news.com, whose writer profiles the site itself labels AI-generated
  4. population        English-language news outlets listed in Wikidata
  sites.jsonl / records.jsonl / summary.json, and flagged/ (copies of flagged headshots,
  kept for human review and NOT for publication).
"""
import argparse
import json
import os
import random
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
TPDNE_URL = "https://thispersondoesnotexist.com/"
CONTROL_SITES = [("we-news.com", "https://we-news.com/", "control:we-news (site labels its writers AI-generated)")]

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

    def save_flagged(self, group, sha, data, ctype, limit=600):
        with self.lock:
            if self.flag_saved >= limit:
                return None
            self.flag_saved += 1
        ext = "png" if "png" in ctype else "webp" if "webp" in ctype else "jpg"
        sub = os.path.join(self.out, "flagged", group.split(":")[0])
        os.makedirs(sub, exist_ok=True)
        path = os.path.join(sub, sha[:16] + "." + ext)
        with open(path, "wb") as fh:
            fh.write(data)
        return os.path.relpath(path, self.out)

    def write_json(self, name, obj):
        with open(os.path.join(self.out, name), "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)


# ---------------------------------------------------------------------------

def stage_calibration(run, det, n, deadline):
    geoms, hashes, fetched = [], set(), 0
    sample_dir = os.path.join(run.out, "calibration_samples")
    os.makedirs(sample_dir, exist_ok=True)
    for i in range(n):
        if time.time() > deadline:
            break
        try:
            st, _f, ct, body = bf.fetch(TPDNE_URL, accept="image/*", timeout=30)
        except Exception as e:
            run.log("tpdne error", type(e).__name__, str(e)[:120])
            time.sleep(2)
            continue
        fetched += 1
        if st != 200 or not body:
            run.log("tpdne status", st)
            time.sleep(2)
            continue
        h = bf.sha256(body)
        if h in hashes:
            time.sleep(2)
            continue
        hashes.add(h)
        if len(hashes) <= 3:
            with open(os.path.join(sample_dir, h[:16] + ".jpg"), "wb") as fh:
                fh.write(body)
        info, err = det.detect(body)
        if not err and len(info["faces"]) == 1:
            f = info["faces"][0]
            g = bf.eye_geometry(f["eyes"][0], f["eyes"][1], info["w"], info["h"])
            if g:
                geoms.append(g)
        time.sleep(1.5)
    result = {"source": TPDNE_URL, "requests": fetched, "unique_images": len(hashes),
              "single_face": len(geoms), "detector": det.name}
    if len(geoms) >= 10:
        train, test = geoms[0::2], geoms[1::2]
        c_train = bf.calibrate(train)
        flags = [bf.classify(g, c_train) for g in test]
        result["holdout"] = {
            "n_train": len(train), "n_test": len(test),
            "recall_aligned": sum(f["aligned"] for f in flags) / len(test),
            "recall_centred_level": sum(f["centred_level"] for f in flags) / len(test),
            "recall_either": sum(f["aligned"] or f["centred_level"] for f in flags) / len(test),
            "g_hist_edges": G_EDGES,
            "g_hist": bf.histogram([f["gan_eye_distance"] for f in flags], G_EDGES),
        }
        result["calibration"] = bf.calibrate(geoms)
    else:
        result["calibration"] = None
        result["note"] = "fewer than 10 single-face GAN images; scoring uses FALLBACK_CALIB"
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
        out = {
            "source": url, "scored": len(recs), "single_face": len(single),
            "aligned": sum(1 for r in single if r["aligned"]),
            "centred_level": sum(1 for r in single if r["centred_level"]),
            "either": sum(1 for r in single if r["aligned"] or r["centred_level"]),
            "g_hist_edges": G_EDGES,
            "g_hist": bf.histogram([r["gan_eye_distance"] for r in single], G_EDGES),
        }
        if single:
            out["fpr_aligned"] = out["aligned"] / len(single)
            out["fpr_centred_level"] = out["centred_level"] / len(single)
            out["fpr_either"] = out["either"] / len(single)
        break
    run.write_json("negatives.json", out)
    run.log("negatives", json.dumps(out))
    return out


def crawl_site(run, det, calib, host, home, group, qid, opts, deadline):
    site = {"host": host, "home": home, "group": group, "qid": qid, "status": "ok",
            "articles": 0, "author_urls": 0, "author_pages": 0, "images_tried": 0,
            "single_face": 0, "aligned": 0, "centred_level": 0}

    def get(url, accept, max_bytes=2_000_000):
        if time.time() > deadline:
            raise Deadline()
        return bf.fetch(url, max_bytes=max_bytes, accept=accept)

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
        st, final, ct, body = get(home, "text/html")
        if st != 200 or "html" not in (ct or "").lower():
            site["status"] = "home_%s" % st
            return site
        html = bf.decode_text(body, ct)
        authors = bf.find_author_urls(html, final)
        arts = bf.find_article_urls(html, final, opts["max_articles"])
        site["articles"] = len(arts)
        for a in arts:
            if len(authors) >= opts["max_authors"]:
                break
            if not allowed(a):
                continue
            st, f2, ct2, b2 = get(a, "text/html")
            if st != 200:
                continue
            for u in bf.find_author_urls(bf.decode_text(b2, ct2), f2):
                if u not in authors:
                    authors.append(u)
        authors = authors[:opts["max_authors"]]
        site["author_urls"] = len(authors)
        for au in authors:
            if not allowed(au):
                continue
            st, f3, ct3, b3 = get(au, "text/html")
            if st != 200 or "html" not in (ct3 or "").lower():
                continue
            site["author_pages"] += 1
            for im in bf.find_person_images(bf.decode_text(b3, ct3), f3)[:opts["max_images"]]:
                if not allowed(im["url"]):
                    continue
                st, f4, ct4, b4 = get(im["url"], "image/avif;q=0,image/*", max_bytes=4_000_000)
                if st != 200 or not b4:
                    continue
                site["images_tried"] += 1
                sha = bf.sha256(b4)
                rec = det.score(b4, calib)
                rec.update({"host": host, "group": group, "author_page": f3, "img_url": f4,
                            "img_kind": im["kind"], "sha256": sha, "bytes": len(b4)})
                if "geom" in rec:
                    site["single_face"] += 1
                    site["aligned"] += int(rec["aligned"])
                    site["centred_level"] += int(rec["centred_level"])
                    if rec["aligned"] or rec["centred_level"]:
                        rec["saved"] = run.save_flagged(group, sha, b4, ct4 or "")
                run.add_record(rec)
    except Deadline:
        site["status"] = "deadline"
    except Exception as e:
        site["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:100])
    return site


def crawl(run, det, calib, sites, opts, deadline, threads):
    done = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(crawl_site, run, det, calib, h, home, group, qid, opts, deadline)
                for (h, home, group, qid) in sites]
        for fut in as_completed(futs):
            run.add_site(fut.result())
            done += 1
            if done % 100 == 0:
                run.log("sites done", done, "of", len(sites))


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
    return [(h, home, "population", q) for (h, home, q) in hosts]


def summarise(run, calib, negatives):
    groups = {}
    page_count = {}
    for r in run.records:
        if "sha256" in r:
            page_count.setdefault(r["sha256"], set()).add(r.get("author_page"))
    for s in run.sites:
        g = groups.setdefault(s["group"].split(":")[0], {
            "sites": 0, "status": {}, "sites_with_author_page": 0,
            "sites_with_single_face": 0, "sites_flagged_aligned": 0, "sites_flagged_either": 0})
        g["sites"] += 1
        key = s["status"] if not s["status"].startswith("error") else "error"
        g["status"][key] = g["status"].get(key, 0) + 1
        g["sites_with_author_page"] += int(s["author_pages"] > 0)
        g["sites_with_single_face"] += int(s["single_face"] > 0)
        g["sites_flagged_aligned"] += int(s["aligned"] > 0)
        g["sites_flagged_either"] += int(s["aligned"] > 0 or s["centred_level"] > 0)
    for name, g in groups.items():
        recs = [r for r in run.records if r.get("group", "").split(":")[0] == name and "geom" in r]
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
            "g_hist_edges": G_EDGES,
            "g_hist": bf.histogram([r["gan_eye_distance"] for r in u], G_EDGES),
        })
    summary = {"version": bf.__version__, "detector": None, "calibration_n": (calib or {}).get("n", 0),
               "negatives": negatives, "groups": groups,
               "note": "Flags are leads for human review, not findings. flagged/ is not for publication."}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--time-budget-min", type=float, default=160)
    ap.add_argument("--calib-n", type=int, default=60)
    ap.add_argument("--neg-n", type=int, default=1000)
    ap.add_argument("--max-sites", type=int, default=2500)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261117)
    args = ap.parse_args(argv)
    t0 = time.time()
    end = t0 + args.time_budget_min * 60
    run = Run(args.out_dir)
    run.log("byline-faces", bf.__version__, "budget min", args.time_budget_min)
    det = bf.FaceDetector(model_path=os.path.join(args.out_dir, "yunet_2023mar.onnx"))
    run.log("detector", det.name)

    calib = None
    try:
        calib = stage_calibration(run, det, args.calib_n, min(end, time.time() + 8 * 60))
    except Exception as e:
        run.log("calibration failed", type(e).__name__, str(e)[:200])
    negatives = {}
    try:
        negatives = stage_negatives(run, det, calib, args.neg_n, min(end, time.time() + 15 * 60))
    except Exception as e:
        run.log("negatives failed", type(e).__name__, str(e)[:200])

    control_opts = {"max_articles": 6, "max_authors": 40, "max_images": 2}
    pop_opts = {"max_articles": 3, "max_authors": 3, "max_images": 3}
    try:
        crawl(run, det, calib, [(h, home, g, "") for (h, home, g) in CONTROL_SITES], control_opts,
              min(end, time.time() + 15 * 60), threads=1)
    except Exception as e:
        run.log("control failed", type(e).__name__, str(e)[:200])
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
