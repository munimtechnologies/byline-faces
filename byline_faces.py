#!/usr/bin/env python3
# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""byline-faces: how often do news sites present GAN-generated faces as their writers?

Faces made by StyleGAN-family generators (thispersondoesnotexist.com, Generated Photos and
the like) are produced on an aligned grid: the two eyes land on almost the same pixels in
every image. Yang, Singh and Menczer (2024, arXiv:2401.02627) used this to estimate how
many Twitter/X profiles use such faces. This module applies the same eye-placement test,
plus a crop-tolerant variant, to the author headshots that news sites publish.

Pure functions (geometry, scoring, HTML parsing) need only the standard library.
FaceDetector needs opencv-python-headless and numpy and is imported lazily.

A flag is a lead, not a verdict: every flagged face must be looked at by a person, and a
real photo that happens to be centred can trip the test. Diffusion-model faces (common
since 2023) do not have the fixed-eye artifact, so this test cannot see them.
"""
import hashlib
import json
import math
import re
import urllib.error
import urllib.request
import urllib.robotparser
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

__version__ = "0.1.0"

USER_AGENT = ("byline-faces/0.1 (research by MV2 of Munim, Inc.; mv2@munimtech.com; "
              "+https://github.com/munimtechnologies/byline-faces)")
ROBOTS_TOKEN = "byline-faces"

# Reference eye positions (image-left eye, image-right eye) in [0, 1] units, from the FFHQ
# alignment recipe: eye distance about a quarter of the width, eyes slightly above centre.
# job.py replaces them with means measured on freshly generated StyleGAN faces.
FALLBACK_REF = ((0.375, 0.475), (0.625, 0.475))
FALLBACK_CALIB = {
    "n": 0,
    "ref": [list(FALLBACK_REF[0]), list(FALLBACK_REF[1])],
    "k": [0.1, 0.03],
    "roll_deg": [0.0, 0.5],
    "centre_offset": [0.0, 0.02],
}
YANG_THRESHOLD = 0.02  # GANEyeDistance cut-off used by Yang, Singh and Menczer (2024)


# ---------------------------------------------------------------------------
# Geometry and scoring
# ---------------------------------------------------------------------------

def eye_geometry(eye_a, eye_b, width, height):
    """Describe where two eye centres (pixel coordinates) sit in a width x height image.

    Returns None if the eyes coincide. Keys:
      left, right      eye centres in [0, 1] units, image-left eye first
      eye_dist_px      distance between the eyes in pixels
      eye_dist_frac    that distance as a fraction of the width
      roll_deg         tilt of the eye line (0 = level)
      centre_offset    horizontal offset of the eye midpoint from the image centre, in eye
                       distances (unchanged by scaling and by centred crops)
      k                height of the eye midpoint above the image centre, in eye distances
                       (unchanged by scaling and by centred crops)
    """
    if width <= 0 or height <= 0:
        raise ValueError("image size must be positive")
    (lx, ly), (rx, ry) = sorted([(float(eye_a[0]), float(eye_a[1])),
                                 (float(eye_b[0]), float(eye_b[1]))])
    dx, dy = rx - lx, ry - ly
    d = math.hypot(dx, dy)
    if d <= 0:
        return None
    mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0
    return {
        "left": (lx / width, ly / height),
        "right": (rx / width, ry / height),
        "eye_dist_px": d,
        "eye_dist_frac": d / width,
        "roll_deg": math.degrees(math.atan2(dy, dx)),
        "centre_offset": (mx - width / 2.0) / d,
        "k": (height / 2.0 - my) / d,
    }


def gan_eye_distance(left, right, ref=FALLBACK_REF):
    """Yang et al.'s GANEyeDistance: mean distance of the two normalised eye centres from
    the reference positions, scaled by 1/sqrt(2) so it lies in [0, 1]."""
    dl = math.hypot(left[0] - ref[0][0], left[1] - ref[0][1])
    dr = math.hypot(right[0] - ref[1][0], right[1] - ref[1][1])
    return (dl + dr) / (2.0 * math.sqrt(2.0))


def _mean_sd(values):
    n = len(values)
    if n == 0:
        return [None, None]
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return [m, sd]


def calibrate(geoms):
    """Reference positions and spreads from eye_geometry() dicts of known GAN faces."""
    geoms = [g for g in geoms if g]
    if not geoms:
        raise ValueError("no geometries to calibrate on")
    lx = _mean_sd([g["left"][0] for g in geoms])
    ly = _mean_sd([g["left"][1] for g in geoms])
    rx = _mean_sd([g["right"][0] for g in geoms])
    ry = _mean_sd([g["right"][1] for g in geoms])
    ref = [[lx[0], ly[0]], [rx[0], ry[0]]]
    return {
        "n": len(geoms),
        "ref": ref,
        "ref_sd": [[lx[1], ly[1]], [rx[1], ry[1]]],
        "k": _mean_sd([g["k"] for g in geoms]),
        "roll_deg": _mean_sd([g["roll_deg"] for g in geoms]),
        "centre_offset": _mean_sd([g["centre_offset"] for g in geoms]),
        "eye_dist_frac": _mean_sd([g["eye_dist_frac"] for g in geoms]),
        "gan_eye_distance": _mean_sd([gan_eye_distance(g["left"], g["right"], ref)
                                      for g in geoms]),
    }


def classify(geom, calib=None, threshold=YANG_THRESHOLD):
    """Score one face. 'aligned' is Yang et al.'s test (uncropped GAN faces);
    'centred_level' is a crop-tolerant test (centred, level, eyes at the GAN height)."""
    calib = calib or FALLBACK_CALIB
    ref = calib["ref"]
    g = gan_eye_distance(geom["left"], geom["right"], ref)
    km, ksd = calib["k"]
    rm, rsd = calib["roll_deg"]
    cm, csd = calib["centre_offset"]
    centred_level = (abs(geom["centre_offset"] - cm) <= max(3 * (csd or 0), 0.03)
                     and abs(geom["roll_deg"] - rm) <= max(3 * (rsd or 0), 1.0)
                     and abs(geom["k"] - km) <= max(3 * (ksd or 0), 0.03))
    return {"gan_eye_distance": g, "aligned": g < threshold, "centred_level": centred_level}


def histogram(values, edges):
    """Counts of values in [edges[i], edges[i+1]); the last bin also takes values >= edges[-1]."""
    counts = [0] * len(edges)
    for v in values:
        if v is None or v < edges[0]:
            continue
        i = len(edges) - 1
        while i > 0 and v < edges[i]:
            i -= 1
        counts[i] += 1
    return counts


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

class _Collector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []   # (href, rel, text)
        self.imgs = []    # attribute dicts; "_in_link" = href of the enclosing <a>
        self.metas = []
        self.ldjson = []
        self._in_ld = False
        self._ld_buf = []
        self._a_stack = []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "a":
            self._a_stack.append([a, []])
        elif tag == "img":
            a["_in_link"] = self._a_stack[-1][0].get("href", "") if self._a_stack else ""
            self.imgs.append(a)
        elif tag == "meta":
            self.metas.append(a)
        elif tag == "link":
            self.links.append((a.get("href", ""), a.get("rel", "").lower(), ""))
        elif tag == "script" and "ld+json" in a.get("type", "").lower():
            self._in_ld = True
            self._ld_buf = []

    def handle_endtag(self, tag):
        if tag == "a" and self._a_stack:
            a, txt = self._a_stack.pop()
            self.links.append((a.get("href", ""), a.get("rel", "").lower(),
                               " ".join("".join(txt).split())))
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.ldjson.append("".join(self._ld_buf))

    def handle_data(self, data):
        if self._in_ld:
            self._ld_buf.append(data)
            return
        for item in self._a_stack:
            item[1].append(data)


def parse_html(text):
    c = _Collector()
    try:
        c.feed(text or "")
        c.close()
    except Exception:  # malformed markup: keep what was collected
        pass
    return c


def ld_objects(raw_blocks):
    """Every dict inside the JSON-LD blocks, nested ones included."""
    out = []
    for raw in raw_blocks:
        try:
            data = json.loads(raw.strip())
        except Exception:
            continue
        stack = [data]
        while stack:
            x = stack.pop()
            if isinstance(x, list):
                stack.extend(reversed(x))
            elif isinstance(x, dict):
                out.append(x)
                for v in x.values():
                    if isinstance(v, (list, dict)):
                        stack.append(v)
    return out


def host_of(url):
    h = (urlparse(url).hostname or "").lower()
    return h[4:] if h.startswith("www.") else h


def same_site(url, base):
    a, b = host_of(url), host_of(base)
    if not a or not b:
        return False
    return a == b or a.endswith("." + b) or b.endswith("." + a)


AUTHOR_PATH = re.compile(
    r"/(author|authors|writer|writers|staff|people|person|profile|profiles|contributor|"
    r"contributors|journalist|journalists|reporter|reporters|columnist|columnists|"
    r"correspondent|correspondents)/[^/?#]+", re.I)


def _clean(url, base):
    full = urljoin(base, (url or "").strip()).split("#")[0]
    return full if full.startswith(("http://", "https://")) else ""


def find_author_urls(html_text, base_url):
    """Same-site author-page URLs named by an article: JSON-LD author url/@id,
    rel=author links, then links whose path looks like an author page."""
    c = parse_html(html_text)
    found = []
    for obj in ld_objects(c.ldjson):
        auth = obj.get("author")
        for a in (auth if isinstance(auth, list) else [auth]):
            if isinstance(a, dict):
                u = a.get("url") or a.get("@id") or ""
                if isinstance(u, list):
                    u = u[0] if u else ""
                if isinstance(u, str) and u:
                    found.append(u)
    for href, rel, _text in c.links:
        if "author" in rel.split():
            found.append(href)
    for href, _rel, _text in c.links:
        if href and AUTHOR_PATH.search(urlparse(urljoin(base_url, href)).path or ""):
            found.append(href)
    out = []
    for u in found:
        full = _clean(u, base_url)
        if not full or not same_site(full, base_url):
            continue
        if not urlparse(full).path.strip("/"):
            continue
        if full not in out:
            out.append(full)
    return out


def find_article_urls(html_text, base_url, limit=5):
    """Same-site links on a home page that look like articles (headline text and a slug,
    a /20xx/ date path or a long numeric id)."""
    c = parse_html(html_text)
    out = []
    for href, _rel, text in c.links:
        full = _clean(href, base_url)
        if not full or not same_site(full, base_url):
            continue
        path = urlparse(full).path
        segs = [s for s in path.split("/") if s]
        if not segs or AUTHOR_PATH.search(path):
            continue
        last = segs[-1]
        slugish = (last.count("-") >= 3 or re.search(r"/20\d\d/", path)
                   or re.search(r"\d{6,}", last))
        if slugish and len(text) >= 20 and full not in out:
            out.append(full)
    return out[:limit]


IMG_HINT = re.compile(r"author|avatar|headshot|byline|profile|bio|portrait|writer|contributor|"
                      r"staff|mugshot|journalist|reporter", re.I)


def find_person_images(html_text, page_url, name_hint=""):
    """Candidate headshot URLs on an author page: JSON-LD Person images, <img> tags with
    author-like class/alt/src, then og:image / twitter:image. Logos and SVGs are skipped."""
    c = parse_html(html_text)
    cands = []
    for obj in ld_objects(c.ldjson):
        t = obj.get("@type")
        types = t if isinstance(t, list) else [t]
        if "Person" in types:
            img = obj.get("image")
            for i in (img if isinstance(img, list) else [img]):
                if isinstance(i, dict):
                    i = i.get("url") or i.get("contentUrl") or ""
                if isinstance(i, str) and i:
                    cands.append(("jsonld", i))
    for a in c.imgs:
        src = ""
        for key in ("src", "data-src", "data-lazy-src", "data-original"):
            v = a.get(key, "")
            if v and not v.startswith("data:"):
                src = v
                break
        if not src and a.get("srcset"):
            first = a["srcset"].split(",")[0].split()
            src = first[0] if first else ""
        if not src or src.startswith("data:"):
            continue
        hay = " ".join([a.get("class", ""), a.get("alt", ""), a.get("id", ""), src,
                        a.get("_in_link", "")])
        if IMG_HINT.search(hay) or (name_hint and name_hint.lower() in a.get("alt", "").lower()):
            cands.append(("img", src))
    for m in c.metas:
        key = (m.get("property") or m.get("name") or "").lower()
        if key in ("og:image", "twitter:image") and m.get("content"):
            cands.append(("og", m["content"]))
    out, seen = [], set()
    for kind, u in cands:
        full = _clean(u, page_url)
        if not full or full in seen:
            continue
        low = full.lower()
        if re.search(r"\.svg(\?|$)", low) or "logo" in low:
            continue
        seen.add(full)
        out.append({"kind": kind, "url": full})
    return out


# ---------------------------------------------------------------------------
# Robots, Wikidata, network
# ---------------------------------------------------------------------------

def robots_allows(robots_text, url, agent=ROBOTS_TOKEN):
    rp = urllib.robotparser.RobotFileParser()
    rp.parse((robots_text or "").splitlines())
    return rp.can_fetch(agent, url)


def wikidata_hosts(sparql_json):
    """[(host, home_url, qid)] from a SPARQL result with ?item and ?site, one per host."""
    out, seen = [], set()
    for b in (sparql_json.get("results", {}) or {}).get("bindings", []):
        site = (b.get("site") or {}).get("value", "")
        item = (b.get("item") or {}).get("value", "")
        if not site.startswith(("http://", "https://")):
            continue
        h = host_of(site)
        if not h or h in seen:
            continue
        seen.add(h)
        out.append((h, "https://" + (urlparse(site).hostname or h) + "/", item.rsplit("/", 1)[-1]))
    return out


def fetch(url, max_bytes=3_000_000, timeout=12, accept="*/*"):
    """GET a public URL. Returns (status, final_url, content_type, body_bytes)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept,
                                               "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(max_bytes + 1)
            if len(body) > max_bytes:
                body = body[:max_bytes]
            return r.status, r.geturl(), r.headers.get("Content-Type", ""), body
    except urllib.error.HTTPError as e:
        return e.code, url, "", b""


def decode_text(body, content_type=""):
    m = re.search(r"charset=([\w-]+)", content_type or "", re.I)
    for enc in ([m.group(1)] if m else []) + ["utf-8", "latin-1"]:
        try:
            return body.decode(enc)
        except (LookupError, UnicodeDecodeError):
            continue
    return body.decode("utf-8", "replace")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Face detection (needs opencv-python-headless + numpy)
# ---------------------------------------------------------------------------

YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx",
]


class FaceDetector:
    """YuNet (OpenCV) face detector with five landmarks; falls back to Haar cascades."""

    def __init__(self, model_path=None, download=True):
        import os
        import threading
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        self.cv2, self.np = cv2, numpy
        self.lock = threading.Lock()
        self.name = None
        model_path = model_path or "yunet_2023mar.onnx"
        if download and not os.path.exists(model_path):
            for u in YUNET_URLS:
                try:
                    st, _f, _ct, body = fetch(u, max_bytes=5_000_000, timeout=60)
                except Exception:
                    continue
                if st == 200 and len(body) > 100_000 and not body.startswith(b"version https://git-lfs"):
                    with open(model_path, "wb") as fh:
                        fh.write(body)
                    break
        if os.path.exists(model_path):
            try:
                self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.8, 0.3, 5000)
                self.name = "yunet-2023mar"
            except Exception:
                self.name = None
        if self.name is None:
            self.face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
            self.name = "haar"

    def detect(self, data):
        """Faces in an encoded image. Returns (info, error); info has w, h (as analysed)
        and faces: [{"eyes": ((x, y), (x, y)), "score": s, "area_frac": a}]."""
        cv2, np = self.cv2, self.np
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return None, "undecodable"
        h0, w0 = img.shape[:2]
        if min(h0, w0) < 48:
            return None, "too_small"
        scale = 1.0
        if max(h0, w0) < 320:
            scale = 320.0 / max(h0, w0)
        elif max(h0, w0) > 1280:
            scale = 1280.0 / max(h0, w0)
        im = img if scale == 1.0 else cv2.resize(
            img, (max(1, round(w0 * scale)), max(1, round(h0 * scale))),
            interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        hh, ww = im.shape[:2]
        faces = []
        with self.lock:
            if self.name.startswith("yunet"):
                self.det.setInputSize((ww, hh))
                _n, res = self.det.detect(im)
                for f in ([] if res is None else res):
                    faces.append({"eyes": ((float(f[4]), float(f[5])), (float(f[6]), float(f[7]))),
                                  "score": float(f[14]),
                                  "area_frac": float(f[2] * f[3]) / float(ww * hh)})
            else:
                grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                for (x, y, w, h) in self.face.detectMultiScale(grey, 1.1, 5, minSize=(40, 40)):
                    roi = grey[y:y + h // 2 + h // 8, x:x + w]
                    eyes = self.eye.detectMultiScale(roi, 1.1, 5)
                    if len(eyes) != 2:
                        continue
                    pts = [(x + ex + ew / 2.0, y + ey + eh / 2.0) for (ex, ey, ew, eh) in eyes]
                    faces.append({"eyes": (pts[0], pts[1]), "score": 1.0,
                                  "area_frac": float(w * h) / float(ww * hh)})
        return {"w": ww, "h": hh, "w0": w0, "h0": h0, "faces": faces}, None

    def score(self, data, calib=None):
        """One record for one image: geometry and flags if exactly one face is found."""
        info, err = self.detect(data)
        if err:
            return {"error": err}
        rec = {"w0": info["w0"], "h0": info["h0"], "n_faces": len(info["faces"])}
        if len(info["faces"]) != 1:
            return rec
        f = info["faces"][0]
        geom = eye_geometry(f["eyes"][0], f["eyes"][1], info["w"], info["h"])
        if not geom:
            return rec
        rec.update({"face_score": f["score"], "area_frac": f["area_frac"],
                    "geom": {k: (list(v) if isinstance(v, tuple) else v) for k, v in geom.items()}})
        rec.update(classify(geom, calib))
        return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Check one headshot for the StyleGAN eye-placement signature.")
    ap.add_argument("image", help="path to an image file")
    ap.add_argument("--calibration", help="calibration.json from job.py (optional)")
    args = ap.parse_args(argv)
    calib = None
    if args.calibration:
        with open(args.calibration, encoding="utf-8") as fh:
            calib = json.load(fh).get("calibration")
    with open(args.image, "rb") as fh:
        data = fh.read()
    print(json.dumps(FaceDetector().score(data, calib), indent=2))


if __name__ == "__main__":
    main()
