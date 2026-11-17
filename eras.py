#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""byline-faces survey 2: writer headshots on the same news sites, then and now.

Era A is 2019-2021, before public diffusion models (DALL-E 2: April 2022; Stable Diffusion:
August 2022). Era B is mid-2025 to now. For each outlet, the home page, up to 3 articles and
up to 3 author pages are read from the Wayback Machine near each era's anchor date, and every
candidate headshot is scored by two tests: the StyleGAN eye-placement test (byline_faces.py)
and Community Forensics, a general detector that also covers diffusion models (commfor.py).

Both eras go through the same archive, so compression and format are handled alike. Era A
headshots captured before 2022 cannot be diffusion-made, so their flag rate is a measured
false-positive baseline from the very same outlets. The question is whether headshots that
are new in era B are flagged more often than that baseline (see PREREGISTRATION-2.md).

Also in this run: Community Forensics on known faces (StyleGAN, Stable Diffusion 1.5, real
FFHQ, IMDB-WIKI and LFW photos), the we-news.com control, and the documented fake-persona
cases, now with both tests.

Pure functions (Wayback URL parsing, era windows, the analysis, the HTTP range reader) need
only the standard library; tests: test_eras.py.
"""
import argparse
import json
import math
import os
import random
import re
import tarfile
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import byline_faces as bf
import commfor
import job

__version__ = "0.3.0"

ERAS = {
    "A": {"anchor": "20200701000000", "lo": "20190101000000", "hi": "20211231235959",
          "label": "2019-2021 (before public diffusion models)"},
    "B": {"anchor": "20260801000000", "lo": "20250701000000", "hi": "20261231235959",
          "label": "July 2025 to now"},
}
CF_EDGES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
DFF_BASE = "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/"
DFF_SETS = [("sd15_faces_dff", "text2img.zip"), ("imdbwiki_real_dff", "wiki.zip")]

# Cases documented after survey 1 was written (both sourced; names as published).
EXTRA_DOCUMENTED = [
    {"group": "documented:ventures-media-elle-be-2025",
     "source": "VRT NWS, 26 Jun 2025: Elle Belgium's invented writers; 'Sophie Vermeulen' "
               "(earlier 'Justine') has a photo found in This Person Does Not Exist's "
               "database, 'Marta Peeters' a photo Sightengine rated 99% AI",
     "prefixes": ["elle.be/nl/author/"], "slugs": ["sophie", "marta", "justine"]},
    {"group": "documented:watermark-b2b-2026",
     "source": "Press Gazette, 17 Nov 2026: Watermark Publishing Group's B2B sites, whose "
               "reporters' tiny headshots scan as AI-generated (Made By Humans detector)",
     "prefixes": ["securityguysnews.com/", "thesupplychainer.com/", "warehouseinsider.news/"],
     "slugs": ["paul-epstein", "sarah-oneill", "sarah-o-neill", "paula-vettori",
               "ellie-goldman", "freddie-bolton", "robin-pattinson", "beth-mcmillan",
               "angela-lee", "amy-cohen", "david-donovan", "alex-badmington", "hannah-kohr",
               "evan-porter"]},
]
# The same Watermark sites read live today (named publicly by Press Gazette).
LIVE_DOCUMENTED = [
    ("securityguysnews.com", "https://securityguysnews.com/", "documented-live:watermark-b2b-2026"),
    ("thesupplychainer.com", "https://thesupplychainer.com/", "documented-live:watermark-b2b-2026"),
    ("warehouseinsider.news", "https://warehouseinsider.news/", "documented-live:watermark-b2b-2026"),
]


class Blocked(Exception):
    """The Wayback Machine keeps answering 429: stop asking."""


# ---------------------------------------------------------------------------
# Wayback helpers
# ---------------------------------------------------------------------------

_WB = re.compile(r"^https?://web\.archive\.org/web/(\d{4,14})([a-z]{2}_)?/(.+)$")


def parse_wayback_url(url):
    """(timestamp, flag, original) of a web.archive.org/web/... URL, else None."""
    m = _WB.match(url or "")
    if not m:
        return None
    orig = m.group(3)
    fixed = re.match(r"^(https?):/+(.*)$", orig)
    if fixed:
        orig = fixed.group(1) + "://" + fixed.group(2)
    elif "://" not in orig:
        orig = "http://" + orig
    return m.group(1), m.group(2) or "", orig


def pad_ts(ts):
    return (str(ts or "") + "00000000000000")[:14]


def in_window(ts, era):
    if not ts:
        return False
    e = ERAS[era]
    return e["lo"] <= pad_ts(ts) <= e["hi"]


def image_format(body):
    b = body or b""
    if b[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:4] == b"GIF8":
        return "gif"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    if b[4:12] in (b"ftypavif", b"ftypavis"):
        return "avif"
    return "other"


class WaybackClient:
    """Paced access to archived copies near an era's anchor date (shared by all threads)."""

    def __init__(self, rate=1.5, max_429=5, pause_429=60.0, log=print):
        self.interval = 1.0 / rate if rate > 0 else 0.0
        self.lock = threading.Lock()
        self.next_t = 0.0
        self.pause_until = 0.0
        self.n429 = 0
        self.max_429 = max_429
        self.pause_429 = pause_429
        self.requests = 0
        self.errors = 0
        self.log = log

    def _wait(self):
        with self.lock:
            if self.n429 >= self.max_429:
                raise Blocked()
            now = time.time()
            t = max(now, self.next_t, self.pause_until)
            self.next_t = t + self.interval
            self.requests += 1
        if t > now:
            time.sleep(t - now)

    def get(self, original, era, flag, deadline, max_bytes=3_000_000):
        """(status, capture_ts, original_url, content_type, body) of the capture nearest
        the era's anchor date."""
        if time.time() > deadline:
            raise job.Deadline()
        self._wait()
        url = bf.wayback_url(ERAS[era]["anchor"], original, flag)
        accept = "image/*" if flag == "im_" else "text/html"
        try:
            st, final, ct, body = bf.fetch(url, max_bytes=max_bytes, timeout=45, accept=accept)
        except Exception:
            with self.lock:
                self.errors += 1
            return 0, None, original, "", b""
        if st == 429:
            with self.lock:
                self.n429 += 1
                self.pause_until = time.time() + self.pause_429
                n = self.n429
            self.log("wayback 429 number", n, "; pausing", self.pause_429, "s")
            return 429, None, original, "", b""
        parsed = parse_wayback_url(final)
        if parsed:
            return st, parsed[0], parsed[2], ct, body
        return st, None, original, ct, body


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class Scorer:
    """Eye-placement test plus Community Forensics on single-face images. Keeps the bytes
    of images Community Forensics flags, for private review."""

    def __init__(self, faces, cf, keep=400):
        self.faces, self.cf = faces, cf
        self.name = "%s+%s" % (getattr(faces, "name", "?"), getattr(cf, "name", "none"))
        self.kept = {}
        self.keep = keep
        self.lock = threading.Lock()

    def score(self, data, calib=None):
        rec = self.faces.score(data, calib)
        if "geom" in rec and self.cf is not None:
            try:
                p = self.cf.prob(data)
            except Exception as e:
                rec["cf_error"] = type(e).__name__
                p = None
            rec["cf"] = p
            if p is not None and p >= 0.5:
                with self.lock:
                    if len(self.kept) < self.keep:
                        self.kept[bf.sha256(data)] = data
        return rec


def cf_safe(cf, data):
    try:
        return cf.prob(data)
    except Exception:
        return None


def score_stats(scores, t=0.5):
    vals = sorted(s for s in scores if s is not None)
    n = len(vals)
    out = {"n": n, "failed": len(scores) - n, "flagged_0.5": sum(1 for v in vals if v >= t),
           "hist_edges": CF_EDGES, "hist": bf.histogram(vals, CF_EDGES)}
    out["rate_0.5"] = out["flagged_0.5"] / n if n else None
    out["wilson95"] = wilson(out["flagged_0.5"], n)
    for q in (0.05, 0.5, 0.95):
        out["q%02d" % int(q * 100)] = percentile(vals, q)
    out["mean"] = sum(vals) / n if n else None
    return out


# ---------------------------------------------------------------------------
# Statistics (standard library)
# ---------------------------------------------------------------------------

def wilson(k, n, z=1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def percentile(sorted_vals, q):
    """Nearest-rank percentile of an ascending list."""
    if not sorted_vals:
        return None
    i = max(0, min(len(sorted_vals) - 1, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[i]


def flagged(cf, t, strict):
    return cf > t if strict else cf >= t


def _by_host(items, t, strict):
    g = {}
    for r in items:
        k, n = g.get(r.get("host"), (0, 0))
        g[r.get("host")] = (k + int(flagged(r["cf"], t, strict)), n + 1)
    return list(g.values())


def cluster_boot_diff(a_items, b_items, t, strict, boots=2000, seed=20261117):
    """Percentile bootstrap (resampling outlets) of rate(B) - rate(A)."""
    ga, gb = _by_host(a_items, t, strict), _by_host(b_items, t, strict)
    if not ga or not gb:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(boots):
        ka = na = kb = nb = 0
        for _ in range(len(ga)):
            k, n = ga[rng.randrange(len(ga))]
            ka += k
            na += n
        for _ in range(len(gb)):
            k, n = gb[rng.randrange(len(gb))]
            kb += k
            nb += n
        if na and nb:
            diffs.append(kb / nb - ka / na)
    if not diffs:
        return None
    diffs.sort()
    return [percentile(diffs, 0.025), percentile(diffs, 0.975)]


def analyse_eras(records, boots=2000, seed=20261117):
    """The pre-registered comparison (PREREGISTRATION-2.md).

    Primary images: one face, a Community Forensics score, the page and the image both
    captured inside the era's window; de-duplicated by SHA-256 within each era.
    B_new: primary era-B images whose bytes never appear in era A."""
    prim = {"A": {}, "B": {}}
    seen = {"A": set(), "B": set()}
    for r in records:
        e = r.get("era")
        if e not in prim or not r.get("sha256"):
            continue
        seen[e].add(r["sha256"])
        if r.get("cf") is None or "geom" not in r:
            continue
        if not (r.get("page_in_window") and r.get("img_in_window")):
            continue
        prim[e].setdefault(r["sha256"], r)
    A = list(prim["A"].values())
    B = list(prim["B"].values())
    b_new = [r for r in B if r["sha256"] not in seen["A"]]
    a_sorted = sorted(r["cf"] for r in A)
    t95 = percentile(a_sorted, 0.95) if len(A) >= 40 else None
    out = {"version": __version__, "eras": ERAS,
           "n_primary": {"A": len(A), "B": len(B), "B_new": len(b_new)},
           "hosts_with_primary": {"A": len({r["host"] for r in A}),
                                  "B": len({r["host"] for r in B}),
                                  "B_new": len({r["host"] for r in b_new})},
           "images_seen": {"A": len(seen["A"]), "B": len(seen["B"])},
           "shared_bytes_A_and_B": len(seen["A"] & seen["B"]),
           "t95_A": t95, "thresholds": {}}
    for name, t, strict in (("0.5", 0.5, False), ("t95_A", t95, True)):
        if t is None:
            continue
        res = {"threshold": t, "rule": "cf > t" if strict else "cf >= t"}
        for label, items in (("A", A), ("B", B), ("B_new", b_new)):
            k = sum(1 for r in items if flagged(r["cf"], t, strict))
            n = len(items)
            res[label] = {"k": k, "n": n, "rate": k / n if n else None, "wilson95": wilson(k, n)}
        ra, rb = res["A"]["rate"], res["B_new"]["rate"]
        res["diff_Bnew_minus_A"] = (rb - ra) if ra is not None and rb is not None else None
        res["boot95_outlets"] = cluster_boot_diff(A, b_new, t, strict, boots, seed)
        out["thresholds"][name] = res
    fmt = {}
    for label, items in (("A", A), ("B_new", b_new)):
        for r in items:
            f = fmt.setdefault(r.get("fmt", "other"), {})
            c = f.setdefault(label, {"n": 0, "cf_0.5": 0})
            c["n"] += 1
            c["cf_0.5"] += int(r["cf"] >= 0.5)
    out["by_format"] = fmt
    out["eye_test"] = {label: {"n": len(items),
                               "aligned": sum(1 for r in items if r.get("aligned")),
                               "centred_level": sum(1 for r in items if r.get("centred_level"))}
                       for label, items in (("A", A), ("B", B), ("B_new", b_new))}
    out["cf_hist_edges"] = CF_EDGES
    out["cf_hist"] = {label: bf.histogram([r["cf"] for r in items], CF_EDGES)
                      for label, items in (("A", A), ("B", B), ("B_new", b_new))}
    return out


def cf_group_summary(records):
    """Community Forensics counts per control/documented group (unique images)."""
    out = {}
    for r in records:
        g = bf.group_key(r.get("group", ""))
        head = g.split(":")[0]
        if head not in ("control", "documented", "documented-live") or "geom" not in r:
            continue
        d = out.setdefault(g, {"_seen": set(), "n": 0, "cf_scored": 0, "cf_0.5": 0,
                               "eye_either": 0, "cf_only": 0, "eye_only": 0, "both": 0,
                               "persona_n": 0, "persona_cf_0.5": 0, "persona_cf_only": 0})
        if r.get("sha256") in d["_seen"]:
            continue
        d["_seen"].add(r.get("sha256"))
        eye = bool(r.get("aligned") or r.get("centred_level"))
        cf = r.get("cf")
        cfh = cf is not None and cf >= 0.5
        d["n"] += 1
        d["cf_scored"] += int(cf is not None)
        d["cf_0.5"] += int(cfh)
        d["eye_either"] += int(eye)
        d["cf_only"] += int(cfh and not eye)
        d["eye_only"] += int(eye and not cfh)
        d["both"] += int(eye and cfh)
        if r.get("persona_page"):
            d["persona_n"] += 1
            d["persona_cf_0.5"] += int(cfh)
            d["persona_cf_only"] += int(cfh and not eye)
    for d in out.values():
        d.pop("_seen")
    return out


# ---------------------------------------------------------------------------
# Reading a few members of a large remote zip (HTTP Range requests)
# ---------------------------------------------------------------------------

def http_range_get(url, start, end, timeout=90):
    """Bytes start..end (inclusive) of url, and the total size from Content-Range."""
    req = urllib.request.Request(url, headers={"User-Agent": bf.USER_AGENT,
                                               "Range": "bytes=%d-%d" % (start, end)})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise IOError("server ignored the Range header (status %s)" % r.status)
        cr = r.headers.get("Content-Range", "")
        data = r.read(end - start + 1)
    tail = cr.rsplit("/", 1)[-1] if "/" in cr else ""
    return data, (int(tail) if tail.isdigit() else None)


class RangeFile:
    """A read-only, seekable view of a remote file, enough for zipfile.ZipFile."""

    def __init__(self, url, getter=None, block=1 << 18):
        self.url = url
        self.getter = getter or http_range_get
        self.block = block
        self.pos = 0
        self.cache = {}
        _data, total = self.getter(url, 0, 0)
        if not total:
            raise IOError("no total size in Content-Range")
        self.size = total
        self.requests = 1

    def _block(self, i):
        if i not in self.cache:
            if len(self.cache) >= 32:
                self.cache.clear()
            start = i * self.block
            end = min(self.size, start + self.block) - 1
            self.cache[i] = self.getter(self.url, start, end)[0]
            self.requests += 1
        return self.cache[i]

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n > 4 * self.block:
            data = self.getter(self.url, self.pos, self.pos + n - 1)[0]
            self.requests += 1
            self.pos += len(data)
            return bytes(data)
        out = bytearray()
        while n > 0:
            i, off = divmod(self.pos, self.block)
            chunk = self._block(i)[off:off + n]
            if not chunk:
                break
            out += chunk
            self.pos += len(chunk)
            n -= len(chunk)
        return bytes(out)

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.size + offset
        self.pos = max(0, self.pos)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def close(self):
        self.cache.clear()


def sample_zip_images(zf, n):
    """n image members spread evenly over the sorted names (deterministic)."""
    names = sorted(i.filename for i in zf.infolist()
                   if not i.is_dir() and i.filename.lower().endswith((".jpg", ".jpeg", ".png"))
                   and "__macosx" not in i.filename.lower())
    if not names or n <= 0:
        return names, []
    k = min(n, len(names))
    return names, [names[i * len(names) // k] for i in range(k)]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_cf_calibration(run, cf, data_dir, n, deadline):
    res = {"model": cf.name, "n_per_set": n}
    path = os.path.join(data_dir, "hf140k-test-00000-of-00002.parquet")
    try:
        if job.download(run, job.HF_PARQUET, path, job.HF_SIZE, timeout=300):
            got = {0: [], 1: []}
            for lab, data in job.iter_parquet_faces(path):
                if time.time() > deadline or (len(got[0]) >= n and len(got[1]) >= n):
                    break
                if lab not in got or len(got[lab]) >= n:
                    continue
                got[lab].append(cf_safe(cf, data))
            res["stylegan_140k"] = score_stats(got[1])
            res["ffhq_real_140k"] = score_stats(got[0])
    except Exception as e:
        res["stylegan_140k"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:160])}
    run.log("cf calibration 140k", json.dumps({k: (v.get("rate_0.5") if isinstance(v, dict) else v)
                                               for k, v in res.items()}))
    for key, fname in DFF_SETS:
        if time.time() > deadline:
            res[key] = {"error": "deadline"}
            continue
        try:
            rf = RangeFile(DFF_BASE + fname)
            zf = zipfile.ZipFile(rf)
            names, pick = sample_zip_images(zf, n)
            scores = []
            for name in pick:
                if time.time() > deadline:
                    break
                scores.append(cf_safe(cf, zf.read(name)))
            res[key] = score_stats(scores)
            res[key].update({"source": DFF_BASE + fname, "members": len(names),
                             "examples": pick[:3], "range_requests": rf.requests})
        except Exception as e:
            res[key] = {"error": "%s: %s" % (type(e).__name__, str(e)[:160])}
        run.log("cf calibration", key, json.dumps({k: v for k, v in res[key].items()
                                                   if k not in ("hist", "hist_edges")}))
    scores = []
    for url in job.LFW_URLS:
        if time.time() > deadline or scores:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": bf.USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                tf = tarfile.open(fileobj=r, mode="r|gz")
                idx = 0
                for m in tf:
                    if not m.isfile() or not m.name.lower().endswith(".jpg"):
                        continue
                    idx += 1
                    if idx % 13:
                        continue
                    scores.append(cf_safe(cf, tf.extractfile(m).read()))
                    if len(scores) >= n or time.time() > deadline:
                        break
            res["lfw_real"] = score_stats(scores)
            res["lfw_real"]["source"] = url
        except Exception as e:
            run.log("lfw error", url, type(e).__name__, str(e)[:160])
            if scores:
                res["lfw_real"] = score_stats(scores)
                res["lfw_real"]["source"] = url
    return res


def crawl_site_era(client, scorer, calib, host, home, era, opts, deadline, emit):
    """One outlet in one era, read entirely from the Wayback Machine."""
    site = {"host": host, "era": era, "status": "ok", "home_ts": None, "articles": 0,
            "author_urls": 0, "author_pages": 0, "images": 0, "single_face": 0}
    tried = set()

    def page(url):
        st, ts, orig, ct, body = client.get(url, era, "id_", deadline)
        if st != 200 or not body or not in_window(ts, era):
            return None, None, ts
        if ct and "html" not in ct.lower():
            return None, None, ts
        return orig, bf.decode_text(body, ct), ts

    def images(cands, page_url, page_ts, limit, prefix):
        n = 0
        for im in cands:
            if n >= limit:
                break
            if im["url"] in tried:
                continue
            tried.add(im["url"])
            st, ts, _o, _ct, body = client.get(im["url"], era, "im_", deadline, 4_000_000)
            if st != 200 or not bf.looks_like_image(body):
                continue
            n += 1
            site["images"] += 1
            rec = scorer.score(body, calib)
            rec.update({"era": era, "host": host, "page": page_url, "page_ts": page_ts,
                        "page_in_window": in_window(page_ts, era), "img_url": im["url"],
                        "img_ts": ts, "img_in_window": in_window(ts, era),
                        "img_kind": prefix + im["kind"], "fmt": image_format(body),
                        "sha256": bf.sha256(body), "bytes": len(body)})
            if "geom" in rec:
                site["single_face"] += 1
            emit(rec, body)

    try:
        orig, html, ts = page(home)
        site["home_ts"] = ts
        if html is None:
            site["status"] = "no_capture_in_window"
            return site
        authors = bf.find_author_urls(html, orig)
        arts = bf.find_article_urls(html, orig, opts["max_articles"])
        site["articles"] = len(arts)
        for a in arts:
            o2, h2, ts2 = page(a)
            if h2 is None:
                continue
            if len(authors) < opts["max_authors"]:
                for u in bf.find_author_urls(h2, o2):
                    if u not in authors:
                        authors.append(u)
            images(bf.find_person_images(h2, o2, meta=False), o2, ts2,
                   opts["max_article_images"], "article_")
        authors = authors[:opts["max_authors"]]
        site["author_urls"] = len(authors)
        for au in authors:
            o3, h3, ts3 = page(au)
            if h3 is None:
                continue
            site["author_pages"] += 1
            images(bf.find_person_images(h3, o3), o3, ts3, opts["max_images"], "")
    except job.Deadline:
        site["status"] = "deadline"
    except Blocked:
        site["status"] = "wayback_blocked"
    except Exception as e:
        site["status"] = "error:%s:%s" % (type(e).__name__, str(e)[:100])
    return site


class EraLog:
    def __init__(self, out_dir, save_limit=600):
        self.out = out_dir
        self.lock = threading.Lock()
        self.recf = open(os.path.join(out_dir, "eras_records.jsonl"), "a", encoding="utf-8")
        self.sitef = open(os.path.join(out_dir, "eras_sites.jsonl"), "a", encoding="utf-8")
        self.records, self.sites = [], []
        self.saved = 0
        self.save_limit = save_limit

    def emit(self, rec, body):
        flag = (rec.get("cf") is not None and rec["cf"] >= 0.5) or rec.get("aligned") \
            or rec.get("centred_level")
        if flag and "geom" in rec:
            with self.lock:
                ok = self.saved < self.save_limit
                self.saved += int(ok)
            if ok:
                sub = os.path.join(self.out, "flagged_eras", rec["era"])
                os.makedirs(sub, exist_ok=True)
                path = os.path.join(sub, rec["sha256"][:16] + "." + rec.get("fmt", "img"))
                with open(path, "wb") as fh:
                    fh.write(body)
                rec["saved"] = os.path.relpath(path, self.out)
        with self.lock:
            self.records.append(rec)
            self.recf.write(json.dumps(rec) + "\n")
            self.recf.flush()

    def add_site(self, site):
        with self.lock:
            self.sites.append(site)
            self.sitef.write(json.dumps(site) + "\n")
            self.sitef.flush()

    def close(self):
        self.recf.close()
        self.sitef.close()


def stage_eras(run, elog, client, scorer, calib, sites, opts, deadline, threads):
    done = [0]

    def work(h, home):
        out = []
        for era in ("A", "B"):
            if time.time() > deadline:
                break
            out.append(crawl_site_era(client, scorer, calib, h, home, era, opts, deadline,
                                      elog.emit))
        return out

    with ThreadPoolExecutor(max_workers=threads) as ex:
        futs = [ex.submit(work, h, home) for (h, home, _g, _q, _s) in sites]
        for fut in as_completed(futs):
            for s in fut.result():
                elog.add_site(s)
            done[0] += 1
            if done[0] % 25 == 0:
                run.log("era sites done", done[0], "of", len(sites), "wayback requests",
                        client.requests, "429s", client.n429)


def era_site_summary(sites):
    out = {}
    for s in sites:
        e = out.setdefault(s["era"], {"site_eras": 0, "status": {}, "with_author_page": 0,
                                      "with_single_face": 0})
        e["site_eras"] += 1
        key = s["status"] if not s["status"].startswith("error") else "error"
        e["status"][key] = e["status"].get(key, 0) + 1
        e["with_author_page"] += int(s["author_pages"] > 0)
        e["with_single_face"] += int(s["single_face"] > 0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="byline-faces survey 2 (then vs now)")
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--data-dir", default="/data")
    ap.add_argument("--time-budget-min", type=float, default=220)
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--max-sites", type=int, default=1500)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rate", type=float, default=1.5, help="Wayback requests per second")
    ap.add_argument("--seed", type=int, default=20261117)
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--skip-documented", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    end = t0 + args.time_budget_min * 60
    run = job.Run(args.out_dir)
    run.log("byline-faces eras", __version__, "budget min", args.time_budget_min)
    data_dir = args.data_dir if job.writable_dir(args.data_dir) else "/tmp"
    faces = bf.FaceDetector(model_path=os.path.join(data_dir, "yunet_2023mar.onnx"))
    run.log("face detector", faces.name)

    cf = None
    try:
        wpath = os.path.join(data_dir, "commfor-model-384.safetensors")
        if commfor.ensure_weights(wpath, lambda u, p, s: job.download(run, u, p, s, timeout=600)):
            cf = commfor.CommFor(wpath)
        else:
            run.log("commfor weights download failed")
    except Exception as e:
        run.log("commfor load failed", type(e).__name__, str(e)[:300])
    scorer = Scorer(faces, cf)
    cal = {"model": getattr(cf, "name", None)}
    if cf is not None:
        cal["repro"] = commfor.reproduce(
            cf, lambda u: bf.fetch(u, max_bytes=20_000_000, timeout=60))
        run.log("commfor repro", json.dumps(cal["repro"]))
    run.write_json("calibration_cf.json", cal)

    eye_calib = None
    if not args.skip_calibration:
        try:
            eye_calib = job.stage_calibration(run, faces, data_dir, 300, 300,
                                              min(end, time.time() + 12 * 60))
        except Exception as e:
            run.log("eye calibration failed", type(e).__name__, str(e)[:200])
        if cf is not None:
            try:
                cal.update(stage_cf_calibration(run, cf, data_dir, args.calib_n,
                                                min(end, time.time() + 25 * 60)))
            except Exception as e:
                run.log("cf calibration failed", type(e).__name__, str(e)[:200])
            run.write_json("calibration_cf.json", cal)

    opts_live = {"max_articles": 6, "max_authors": 40, "max_images": 2, "max_seed_images": 60,
                 "max_article_images": 2}
    live = [(h, home, g, "", seeds) for (h, home, g, seeds) in job.CONTROL_SITES]
    live += [(h, home, g, "", ()) for (h, home, g) in LIVE_DOCUMENTED]
    try:
        job.crawl(run, scorer, eye_calib, live, opts_live, min(end, time.time() + 10 * 60),
                  threads=2)
    except Exception as e:
        run.log("live groups failed", type(e).__name__, str(e)[:200])
    if not args.skip_documented:
        try:
            job.DOCUMENTED = list(job.DOCUMENTED) + EXTRA_DOCUMENTED
            job.stage_documented(run, scorer, eye_calib, min(end, time.time() + 30 * 60))
        except Exception as e:
            run.log("documented failed", type(e).__name__, str(e)[:200])

    elog = EraLog(args.out_dir)
    client = WaybackClient(rate=args.rate, log=run.log)
    try:
        sites = job.population_sites(run, args.max_sites, args.seed)
        stage_eras(run, elog, client, scorer, eye_calib, sites,
                   {"max_articles": 3, "max_authors": 3, "max_images": 2,
                    "max_article_images": 2}, end - 6 * 60, args.threads)
    except Exception as e:
        run.log("eras failed", type(e).__name__, str(e)[:200])

    kept_dir = os.path.join(args.out_dir, "flagged_cf")
    os.makedirs(kept_dir, exist_ok=True)
    for sha, data in list(scorer.kept.items()):
        with open(os.path.join(kept_dir, sha[:16] + "." + image_format(data)), "wb") as fh:
            fh.write(data)

    summary = job.summarise(run, eye_calib, {})
    summary.update({"version": __version__, "detector": scorer.name,
                    "cf_repro_ok": (cal.get("repro") or {}).get("ok"),
                    "cf_groups": cf_group_summary(run.records),
                    "era_sites": era_site_summary(elog.sites),
                    "wayback": {"requests": client.requests, "429s": client.n429,
                                "errors": client.errors},
                    "minutes": round((time.time() - t0) / 60, 1)})
    run.write_json("summary.json", summary)
    eras_summary = analyse_eras(elog.records)
    eras_summary["cf_repro_ok"] = summary["cf_repro_ok"]
    run.write_json("eras_summary.json", eras_summary)
    run.log("eras summary", json.dumps(eras_summary)[:4000])
    elog.close()


if __name__ == "__main__":
    main()
