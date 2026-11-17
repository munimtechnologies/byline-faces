# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Standard-library tests for eras.py and commfor.py (no torch, no OpenCV, no network).
import io
import os
import tempfile
import time
import unittest
import zipfile
from unittest import mock

import byline_faces as bf
import commfor
import eras

HTML = "text/html; charset=utf-8"
JPEG = b"\xff\xd8\xff\xe0"


class CommForRecipeTests(unittest.TestCase):
    def test_resize_crop_box(self):
        self.assertEqual(commfor.resize_crop_box(250, 250), ((440, 440), (28, 28, 412, 412)))
        self.assertEqual(commfor.resize_crop_box(1024, 768), ((586, 440), (101, 28, 485, 412)))
        self.assertEqual(commfor.resize_crop_box(300, 600), ((440, 880), (28, 248, 412, 632)))
        with self.assertRaises(ValueError):
            commfor.resize_crop_box(0, 10)

    def test_compare_repro(self):
        good = {n: r + 0.005 for n, r in commfor.TEST_IMAGES}
        self.assertTrue(commfor.compare_repro(good)["ok"])
        bad = dict(good)
        bad["00000989.png"] = 0.5
        r = commfor.compare_repro(bad)
        self.assertFalse(r["ok"])
        self.assertAlmostEqual(r["max_abs_diff"], 0.286)
        missing = dict(good)
        missing["00000274.png"] = None
        self.assertFalse(commfor.compare_repro(missing)["ok"])
        self.assertFalse(commfor.compare_repro({})["ok"])

    def test_strip_prefix(self):
        self.assertEqual(commfor.strip_prefix({"vit.head.weight": 1, "other": 2}),
                         {"head.weight": 1, "other": 2})

    def test_reference_scores(self):
        self.assertEqual([r for _n, r in commfor.TEST_IMAGES], [0.9988, 0.9878, 0.9569, 0.9516, 0.7860])


class WaybackParseTests(unittest.TestCase):
    def test_parse_wayback_url(self):
        self.assertEqual(eras.parse_wayback_url(
            "https://web.archive.org/web/20200703120000id_/https://ex.com/a"),
            ("20200703120000", "id_", "https://ex.com/a"))
        self.assertEqual(eras.parse_wayback_url(
            "http://web.archive.org/web/20200703120000im_/http://ex.com:80/i.jpg"),
            ("20200703120000", "im_", "http://ex.com:80/i.jpg"))
        self.assertEqual(eras.parse_wayback_url(
            "https://web.archive.org/web/2020/https:/ex.com/a"), ("2020", "", "https://ex.com/a"))
        self.assertEqual(eras.parse_wayback_url(
            "https://web.archive.org/web/20200101000000/ex.com/a"),
            ("20200101000000", "", "http://ex.com/a"))
        self.assertIsNone(eras.parse_wayback_url("https://ex.com/a"))
        self.assertIsNone(eras.parse_wayback_url(None))

    def test_in_window(self):
        self.assertTrue(eras.in_window("20200703120000", "A"))
        self.assertTrue(eras.in_window("201906", "A"))
        self.assertFalse(eras.in_window("2019", "A"))   # pads to 20190000..., before 1 Jan
        self.assertFalse(eras.in_window("20220101000000", "A"))
        self.assertFalse(eras.in_window("20181231235959", "A"))
        self.assertTrue(eras.in_window("20260801000000", "B"))
        self.assertFalse(eras.in_window("20250630000000", "B"))
        self.assertFalse(eras.in_window(None, "B"))

    def test_eras_do_not_overlap_and_anchor_inside(self):
        a, b = eras.ERAS["A"], eras.ERAS["B"]
        self.assertLess(a["hi"], "20220401000000")  # before DALL-E 2
        self.assertLess(a["hi"], b["lo"])
        for e in (a, b):
            self.assertTrue(e["lo"] <= e["anchor"] <= e["hi"])

    def test_image_format(self):
        self.assertEqual(eras.image_format(JPEG + b"x"), "jpeg")
        self.assertEqual(eras.image_format(b"\x89PNG\r\n\x1a\nx"), "png")
        self.assertEqual(eras.image_format(b"RIFF\x00\x00\x00\x00WEBPVP8 "), "webp")
        self.assertEqual(eras.image_format(b"\x00\x00\x00\x1cftypavif"), "avif")
        self.assertEqual(eras.image_format(b"GIF89a"), "gif")
        self.assertEqual(eras.image_format(None), "other")


class StatsTests(unittest.TestCase):
    def test_wilson(self):
        self.assertIsNone(eras.wilson(0, 0))
        lo, hi = eras.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        lo, hi = eras.wilson(0, 50)
        self.assertAlmostEqual(lo, 0.0)
        self.assertLess(hi, 0.08)

    def test_percentile(self):
        vals = list(range(1, 21))
        self.assertEqual(eras.percentile(vals, 0.95), 19)
        self.assertEqual(eras.percentile(vals, 0.5), 10)
        self.assertEqual(eras.percentile(vals, 0.0), 1)
        self.assertIsNone(eras.percentile([], 0.5))

    def test_score_stats(self):
        s = eras.score_stats([0.1, 0.6, None, 0.9, 0.4])
        self.assertEqual((s["n"], s["failed"], s["flagged_0.5"]), (4, 1, 2))
        self.assertAlmostEqual(s["rate_0.5"], 0.5)
        self.assertEqual(sum(s["hist"]), 4)
        self.assertEqual(eras.score_stats([])["rate_0.5"], None)


def rec(era, host, sha, cf, page_in=True, img_in=True, fmt="jpeg", aligned=False, face=True):
    r = {"era": era, "host": host, "sha256": sha, "cf": cf, "page_in_window": page_in,
         "img_in_window": img_in, "fmt": fmt, "aligned": aligned, "centred_level": False}
    if face:
        r["geom"] = {}
    return r


class AnalysisTests(unittest.TestCase):
    def test_analyse_eras_small(self):
        records = [
            rec("A", "a1.com", "s1", 0.1, aligned=True), rec("A", "a1.com", "s2", 0.2),
            rec("A", "a1.com", "s3", 0.9), rec("A", "a2.com", "s4", 0.3),
            rec("A", "a2.com", "s5", 0.05), rec("A", "a1.com", "s1", 0.1),   # duplicate bytes
            rec("A", "a2.com", "sout", 0.99, img_in=False),                     # outside window
            rec("A", "a2.com", "noface", 0.99, face=False),                     # no single face
            rec("B", "b1.com", "s1", 0.1),                                      # same bytes as A
            rec("B", "b1.com", "b2", 0.8, fmt="webp"), rec("B", "b2.com", "b3", 0.6),
            rec("B", "b2.com", "b4", 0.2), rec("B", "b2.com", "sout", 0.95),
            rec("B", "b2.com", "b5", None),                                     # not scored
        ]
        out = eras.analyse_eras(records, boots=200)
        self.assertEqual(out["n_primary"], {"A": 5, "B": 5, "B_new": 3})
        self.assertEqual(out["hosts_with_primary"], {"A": 2, "B": 2, "B_new": 2})
        self.assertEqual(out["shared_bytes_A_and_B"], 2)
        self.assertIsNone(out["t95_A"])
        self.assertEqual(list(out["thresholds"]), ["0.5"])
        t = out["thresholds"]["0.5"]
        self.assertEqual((t["A"]["k"], t["A"]["n"]), (1, 5))
        self.assertEqual((t["B_new"]["k"], t["B_new"]["n"]), (2, 3))
        self.assertEqual((t["B"]["k"], t["B"]["n"]), (3, 5))
        self.assertAlmostEqual(t["diff_Bnew_minus_A"], 2 / 3 - 0.2)
        lo, hi = t["boot95_outlets"]
        self.assertTrue(-1 <= lo <= hi <= 1)
        self.assertEqual(out["by_format"]["webp"]["B_new"], {"n": 1, "cf_0.5": 1})
        self.assertEqual(out["by_format"]["jpeg"]["A"], {"n": 5, "cf_0.5": 1})
        self.assertEqual(out["eye_test"]["A"]["aligned"], 1)
        self.assertEqual(sum(out["cf_hist"]["B_new"]), 3)

    def test_analyse_eras_t95(self):
        records = [rec("A", "a%d.com" % (i % 7), "a%d" % i, i / 50) for i in range(50)]
        records += [rec("B", "b%d.com" % (i % 4), "b%d" % i, 0.95 if i < 3 else 0.1)
                    for i in range(10)]
        out = eras.analyse_eras(records, boots=200)
        self.assertAlmostEqual(out["t95_A"], 0.94)
        t = out["thresholds"]["t95_A"]
        self.assertEqual(t["rule"], "cf > t")
        self.assertEqual((t["A"]["k"], t["A"]["n"]), (2, 50))
        self.assertEqual((t["B_new"]["k"], t["B_new"]["n"]), (3, 10))
        self.assertAlmostEqual(t["diff_Bnew_minus_A"], 0.3 - 0.04)

    def test_bootstrap_is_deterministic(self):
        a = [rec("A", "a%d" % (i % 5), "x%d" % i, 0.9 if i % 9 == 0 else 0.1) for i in range(40)]
        b = [rec("B", "b%d" % (i % 5), "y%d" % i, 0.9 if i % 3 == 0 else 0.1) for i in range(40)]
        r1 = eras.cluster_boot_diff(a, b, 0.5, False, boots=300, seed=7)
        r2 = eras.cluster_boot_diff(a, b, 0.5, False, boots=300, seed=7)
        self.assertEqual(r1, r2)
        self.assertIsNone(eras.cluster_boot_diff([], b, 0.5, False))

    def test_cf_group_summary(self):
        records = [
            {"group": "documented:x", "sha256": "1", "geom": {}, "cf": 0.9, "aligned": False,
             "centred_level": False, "persona_page": True},
            {"group": "documented:x", "sha256": "1", "geom": {}, "cf": 0.9},     # duplicate
            {"group": "documented:x", "sha256": "2", "geom": {}, "cf": 0.2, "aligned": True,
             "centred_level": False, "persona_page": True},
            {"group": "documented:x", "sha256": "3", "geom": {}, "cf": 0.7, "aligned": True},
            {"group": "control:we (y)", "sha256": "4", "geom": {}, "cf": None},
            {"group": "documented-live:w", "sha256": "5", "geom": {}, "cf": 0.6},
            {"group": "population", "sha256": "6", "geom": {}, "cf": 0.99},
            {"group": "documented:x", "sha256": "7", "cf": 0.99},               # no face
        ]
        out = eras.cf_group_summary(records)
        self.assertEqual(sorted(out), ["control:we (y)", "documented-live", "documented:x"])
        d = out["documented:x"]
        self.assertEqual((d["n"], d["cf_0.5"], d["eye_either"], d["cf_only"], d["eye_only"],
                          d["both"]), (3, 2, 2, 1, 1, 1))
        self.assertEqual((d["persona_n"], d["persona_cf_0.5"], d["persona_cf_only"]), (2, 1, 1))
        self.assertEqual((out["control:we (y)"]["n"], out["control:we (y)"]["cf_scored"]), (1, 0))


class RangeFileTests(unittest.TestCase):
    def test_zip_over_ranges(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            for i in range(5):
                z.writestr("faces/%03d.png" % i, b"IMG%d" % i * 50)
            z.writestr("faces/readme.txt", b"x")
            z.writestr("__MACOSX/faces/._000.png", b"junk")
        data = buf.getvalue()
        calls = []

        def getter(url, start, end):
            calls.append((start, end))
            return data[start:end + 1], len(data)

        rf = eras.RangeFile("https://x/y.zip", getter, block=64)
        self.assertEqual(rf.size, len(data))
        zf = zipfile.ZipFile(rf)
        names, pick = eras.sample_zip_images(zf, 3)
        self.assertEqual(names, ["faces/%03d.png" % i for i in range(5)])
        self.assertEqual(pick, ["faces/000.png", "faces/001.png", "faces/003.png"])
        self.assertEqual(zf.read("faces/003.png"), b"IMG3" * 50)
        self.assertGreater(len(calls), 2)
        self.assertEqual(eras.sample_zip_images(zf, 0)[1], [])

    def test_seek_and_read(self):
        data = bytes(range(256)) * 4

        def getter(url, start, end):
            return data[start:end + 1], len(data)

        rf = eras.RangeFile("u", getter, block=16)
        rf.seek(-10, 2)
        self.assertEqual(rf.read(), data[-10:])
        rf.seek(5)
        rf.seek(3, 1)
        self.assertEqual(rf.tell(), 8)
        self.assertEqual(rf.read(40), data[8:48])
        rf.seek(0)
        self.assertEqual(rf.read(200), data[:200])   # large read in one request
        self.assertEqual(rf.read(0), b"")


class FakeScorer:
    name = "fake"

    def score(self, data, calib=None):
        cf = 0.9 if b"AI" in data else 0.1
        return {"n_faces": 1, "geom": {}, "aligned": False, "centred_level": False, "cf": cf}


def wb_fetch(routes, log):
    """routes: request URL -> (status, final URL, content type, body)."""
    def fetch(url, max_bytes=3_000_000, timeout=12, accept="*/*"):
        log.append(url)
        if url in routes:
            return routes[url]
        return 404, url, "", b""
    return fetch


class CrawlEraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.elog = eras.EraLog(self.tmp.name)

    def tearDown(self):
        self.elog.close()
        self.tmp.cleanup()

    def test_crawl_site_era(self):
        A = eras.ERAS["A"]["anchor"]
        art = "https://ex.com/2020/07/01/council-votes-to-close-the-library"
        home_html = ('<a href="/2020/07/01/council-votes-to-close-the-library">'
                     'Council votes to close the main library early</a>')
        art_html = ('<script type="application/ld+json">{"@type":"NewsArticle","author":'
                    '{"@type":"Person","url":"https://ex.com/author/ann"}}</script>'
                    '<img class="byline-avatar" src="/img/ann-small.jpg">')

        def wb(ts, orig, flag):
            return "https://web.archive.org/web/%s%s/%s" % (ts, flag, orig)

        routes = {
            bf.wayback_url(A, "https://ex.com/", "id_"):
                (200, wb("20200703120000", "https://ex.com/", "id_"), HTML, home_html.encode()),
            bf.wayback_url(A, art, "id_"):
                (200, wb("20200702000000", art, "id_"), HTML, art_html.encode()),
            bf.wayback_url(A, "https://ex.com/author/ann", "id_"):   # nearest copy is too late
                (200, wb("20230101000000", "https://ex.com/author/ann", "id_"), HTML,
                 b'<img class="author-photo" src="/img/ann.jpg">'),
            bf.wayback_url(A, "https://ex.com/img/ann-small.jpg", "im_"):
                (200, wb("20200702000000", "https://ex.com/img/ann-small.jpg", "im_"),
                 "image/jpeg", JPEG + b"AI"),
        }
        calls = []
        client = eras.WaybackClient(rate=0)
        with mock.patch.object(bf, "fetch", wb_fetch(routes, calls)):
            site = eras.crawl_site_era(client, FakeScorer(), None, "ex.com", "https://ex.com/",
                                       "A", {"max_articles": 3, "max_authors": 3, "max_images": 2,
                                             "max_article_images": 2}, time.time() + 60,
                                       self.elog.emit)
            site_b = eras.crawl_site_era(client, FakeScorer(), None, "ex.com", "https://ex.com/",
                                         "B", {"max_articles": 3, "max_authors": 3,
                                               "max_images": 2, "max_article_images": 2},
                                         time.time() + 60, self.elog.emit)
        self.assertEqual(site["status"], "ok")
        self.assertEqual(site["home_ts"], "20200703120000")
        self.assertEqual((site["articles"], site["author_urls"], site["author_pages"],
                          site["images"], site["single_face"]), (1, 1, 0, 1, 1))
        self.assertEqual(len(self.elog.records), 1)
        r = self.elog.records[0]
        self.assertEqual((r["era"], r["img_kind"], r["fmt"], r["page_in_window"],
                          r["img_in_window"], r["cf"]), ("A", "article_img", "jpeg", True, True, 0.9))
        self.assertTrue(r["saved"].startswith("flagged_eras"))
        self.assertTrue(os.path.exists(os.path.join(self.tmp.name, r["saved"])))
        self.assertEqual(site_b["status"], "no_capture_in_window")
        self.assertEqual(client.requests, 5)   # A: home, article, image, author; B: home
        summ = eras.era_site_summary([site, site_b])
        self.assertEqual(summ["A"]["with_single_face"], 1)
        self.assertEqual(summ["B"]["status"], {"no_capture_in_window": 1})

    def test_wayback_429_then_blocked(self):
        client = eras.WaybackClient(rate=0, max_429=1, pause_429=0, log=lambda *a: None)
        with mock.patch.object(bf, "fetch", lambda url, **kw: (429, url, "", b"")):
            st = client.get("https://ex.com/", "A", "id_", time.time() + 60)[0]
            self.assertEqual(st, 429)
            with self.assertRaises(eras.Blocked):
                client.get("https://ex.com/", "A", "id_", time.time() + 60)
            site = eras.crawl_site_era(client, FakeScorer(), None, "ex.com", "https://ex.com/",
                                       "B", {"max_articles": 3, "max_authors": 3, "max_images": 2,
                                             "max_article_images": 2}, time.time() + 60,
                                       self.elog.emit)
        self.assertEqual(site["status"], "wayback_blocked")

    def test_deadline(self):
        client = eras.WaybackClient(rate=0)
        site = eras.crawl_site_era(client, FakeScorer(), None, "ex.com", "https://ex.com/", "A",
                                   {"max_articles": 3, "max_authors": 3, "max_images": 2,
                                    "max_article_images": 2}, time.time() - 1, self.elog.emit)
        self.assertEqual(site["status"], "deadline")


class MetaTests(unittest.TestCase):
    def test_version_and_licence(self):
        self.assertEqual(eras.__version__, "0.3.0")
        for mod in (eras, commfor):
            with open(mod.__file__, encoding="utf-8") as fh:
                self.assertIn("Apache-2.0", fh.read(400))

    def test_extra_documented_are_sourced(self):
        for spec in eras.EXTRA_DOCUMENTED:
            self.assertTrue(spec["group"].startswith("documented:"))
            self.assertTrue(spec["source"] and spec["prefixes"] and spec["slugs"])
        for host, home, group in eras.LIVE_DOCUMENTED:
            self.assertTrue(home.startswith("https://") and host in home)
            self.assertTrue(group.startswith("documented-live:"))


if __name__ == "__main__":
    unittest.main()
