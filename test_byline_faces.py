# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Standard-library tests for byline_faces and job.py (no OpenCV, no network: fetch is faked).
import json
import math
import tempfile
import time
import unittest
from unittest import mock

import byline_faces as bf
import job


class GeometryTests(unittest.TestCase):
    def test_eye_geometry_orders_and_measures(self):
        g = bf.eye_geometry((640, 486), (384, 486), 1024, 1024)  # passed right eye first
        self.assertEqual(g["left"], (0.375, 486 / 1024))
        self.assertEqual(g["right"], (0.625, 486 / 1024))
        self.assertAlmostEqual(g["eye_dist_px"], 256.0)
        self.assertAlmostEqual(g["eye_dist_frac"], 0.25)
        self.assertAlmostEqual(g["roll_deg"], 0.0)
        self.assertAlmostEqual(g["centre_offset"], 0.0)
        self.assertAlmostEqual(g["k"], 26 / 256)

    def test_coincident_eyes(self):
        self.assertIsNone(bf.eye_geometry((5, 5), (5, 5), 10, 10))
        with self.assertRaises(ValueError):
            bf.eye_geometry((1, 1), (2, 2), 0, 10)

    def test_k_and_centre_survive_centred_crop_and_scale(self):
        full = bf.eye_geometry((384, 486), (640, 486), 1024, 1024)
        crop = bf.eye_geometry((384 - 112, 486 - 112), (640 - 112, 486 - 112), 800, 800)
        small = bf.eye_geometry((384 / 8, 486 / 8), (640 / 8, 486 / 8), 128, 128)
        for g in (crop, small):
            self.assertAlmostEqual(g["k"], full["k"])
            self.assertAlmostEqual(g["centre_offset"], full["centre_offset"])
            self.assertAlmostEqual(g["roll_deg"], full["roll_deg"])
        # a centred crop moves the normalised eyes, so Yang's test no longer matches
        self.assertGreater(bf.gan_eye_distance(crop["left"], crop["right"],
                                               (full["left"], full["right"])), 0.02)
        self.assertAlmostEqual(bf.gan_eye_distance(small["left"], small["right"],
                                                   (full["left"], full["right"])), 0.0)

    def test_roll(self):
        g = bf.eye_geometry((0, 0), (10, 10), 100, 100)
        self.assertAlmostEqual(g["roll_deg"], 45.0)

    def test_gan_eye_distance(self):
        self.assertEqual(bf.gan_eye_distance(*bf.FALLBACK_REF), 0.0)
        left = (bf.FALLBACK_REF[0][0] + 0.03, bf.FALLBACK_REF[0][1] + 0.04)
        g = bf.gan_eye_distance(left, bf.FALLBACK_REF[1])
        self.assertAlmostEqual(g, 0.05 / (2 * math.sqrt(2)))
        self.assertAlmostEqual(bf.gan_eye_distance((0, 0), (0, 0), ((1, 1), (1, 1))), 1.0)

    def test_calibrate(self):
        a = bf.eye_geometry((370, 470), (620, 470), 1000, 1000)
        b = bf.eye_geometry((380, 480), (630, 480), 1000, 1000)
        c = bf.calibrate([a, None, b])
        self.assertEqual(c["n"], 2)
        self.assertAlmostEqual(c["ref"][0][0], 0.375)
        self.assertAlmostEqual(c["ref"][0][1], 0.475)
        self.assertAlmostEqual(c["ref"][1][0], 0.625)
        self.assertAlmostEqual(c["ref_sd"][0][0], math.sqrt(2 * 0.005 ** 2))
        self.assertAlmostEqual(c["k"][0], 0.1)  # (0.12 + 0.08) / 2
        with self.assertRaises(ValueError):
            bf.calibrate([])

    def test_classify(self):
        exact = bf.eye_geometry((375, 475), (625, 475), 1000, 1000)
        r = bf.classify(exact)
        self.assertTrue(r["aligned"])
        self.assertTrue(r["centred_level"])
        self.assertAlmostEqual(r["gan_eye_distance"], 0.0)
        # a typical real portrait: eyes high and off-centre
        real = bf.eye_geometry((300, 330), (460, 350), 1000, 1000)
        r = bf.classify(real)
        self.assertFalse(r["aligned"])
        self.assertFalse(r["centred_level"])
        # centred crop of a GAN face: Yang's test misses it, the crop-tolerant test keeps it
        crop = bf.eye_geometry((375 - 100, 475 - 100), (625 - 100, 475 - 100), 800, 800)
        r = bf.classify(crop)
        self.assertFalse(r["aligned"])
        self.assertTrue(r["centred_level"])

    def test_histogram(self):
        edges = [0.0, 0.02, 0.05]
        self.assertEqual(bf.histogram([0.0, 0.01, 0.02, 0.049, 0.05, 0.9, -1, None], edges), [2, 2, 2])

    def test_rate_summary(self):
        s = bf.rate_summary([{"aligned": True, "centred_level": False},
                             {"aligned": False, "centred_level": True},
                             {"aligned": False, "centred_level": False},
                             {"aligned": True, "centred_level": True}])
        self.assertEqual((s["n"], s["aligned"], s["centred_level"], s["either"]), (4, 2, 2, 3))
        self.assertAlmostEqual(s["rate_either"], 0.75)
        self.assertIsNone(bf.rate_summary([])["rate_aligned"])

    def test_group_key(self):
        self.assertEqual(bf.group_key("population"), "population")
        self.assertEqual(bf.group_key("control:we-news (x)"), "control:we-news (x)")
        self.assertEqual(bf.group_key("documented:si"), "documented:si")
        self.assertEqual(bf.group_key("other:thing"), "other")


ARTICLE = """<html><head>
<link rel="author" href="/people/jane-roe">
<script type="application/ld+json">
{"@context":"https://schema.org","@type":"NewsArticle","headline":"x",
 "author":[{"@type":"Person","name":"Ann Writer","url":"https://www.example.com/author/ann-writer"},
           {"@type":"Person","name":"Bob","@id":"https://example.com/#/schema/person/1"}]}
</script></head><body>
<a href="https://other.org/author/zed">Zed</a>
<a href="/author/ann-writer">Ann Writer</a>
<a href="https://news.example.com/staff/carl">Carl</a>
<a href="/about">About</a>
</body></html>"""

HOME = """<html><body>
<a href="/2026/11/16/council-votes-to-close-library">Council votes to close the main library early</a>
<a href="/news/council-votes-to-close-library-branch">Short</a>
<a href="/local/story-123456789">Storm knocks out power to thousands overnight</a>
<a href="/author/ann-writer">Ann Writer, a reporter with a long name</a>
<a href="https://other.org/2026/11/16/a-b-c-d">Off-site headline that is long enough</a>
<a href="/about-us-and-our-team">About us and our team and more</a>
</body></html>"""

AUTHOR_PAGE = """<html><head>
<meta property="og:image" content="https://cdn.example.com/site-logo.png">
<meta name="twitter:image" content="/img/ann-large.jpg">
<script type="application/ld+json">
{"@type":"ProfilePage","mainEntity":{"@type":"Person","name":"Ann Writer",
 "image":{"@type":"ImageObject","url":"https://cdn.example.com/ann.jpg"}}}
</script></head><body>
<img class="author-avatar lazy" src="data:image/gif;base64,R0lGOD" data-src="/img/ann-120.jpg" alt="Ann Writer">
<img src="/img/icon.svg" class="author-icon">
<img src="/img/ad-banner.jpg" alt="Advert">
<img srcset="/img/ann-2x.jpg 2x, /img/ann-1x.jpg 1x" alt="Ann Writer">
<a href="/author/ann-writer"><img src="/img/thumb.jpg"></a>
</body></html>"""


class HtmlTests(unittest.TestCase):
    def test_same_site(self):
        self.assertTrue(bf.same_site("https://www.example.com/a", "https://example.com/"))
        self.assertTrue(bf.same_site("https://news.example.com/a", "https://www.example.com/"))
        self.assertFalse(bf.same_site("https://other.org/a", "https://example.com/"))
        self.assertFalse(bf.same_site("mailto:x@example.com", "https://example.com/"))

    def test_find_author_urls(self):
        got = bf.find_author_urls(ARTICLE, "https://www.example.com/2026/11/16/story")
        self.assertEqual(got, [
            "https://www.example.com/author/ann-writer",
            "https://www.example.com/people/jane-roe",
            "https://news.example.com/staff/carl",
        ])

    def test_find_article_urls(self):
        got = bf.find_article_urls(HOME, "https://www.example.com/")
        self.assertEqual(got, [
            "https://www.example.com/2026/11/16/council-votes-to-close-library",
            "https://www.example.com/local/story-123456789",
            "https://www.example.com/about-us-and-our-team",
        ])
        self.assertEqual(len(bf.find_article_urls(HOME, "https://www.example.com/", limit=1)), 1)

    def test_find_person_images(self):
        got = bf.find_person_images(AUTHOR_PAGE, "https://www.example.com/author/ann-writer",
                                    name_hint="Ann Writer")
        self.assertEqual([g["url"] for g in got], [
            "https://cdn.example.com/ann.jpg",
            "https://www.example.com/img/ann-120.jpg",
            "https://www.example.com/img/ann-2x.jpg",
            "https://www.example.com/img/thumb.jpg",
            "https://www.example.com/img/ann-large.jpg",
        ])
        self.assertEqual(got[0]["kind"], "jsonld")
        self.assertEqual(got[-1]["kind"], "og")
        no_meta = bf.find_person_images(AUTHOR_PAGE, "https://www.example.com/author/ann-writer",
                                        name_hint="Ann Writer", meta=False)
        self.assertEqual([g["url"] for g in no_meta], [g["url"] for g in got][:4])

    def test_we_news_style_avatars(self):
        page = ('<a href="https://we-news.com/chiara-lombardi">'
                '<img src="https://we-news.com/uploads/_r/160/ia-avatar-chiara-lombardi-602f73.webp" '
                'alt="Chiara Lombardi"></a>')
        got = bf.find_person_images(page, "https://we-news.com/la-redazione", meta=False)
        self.assertEqual([g["url"] for g in got],
                         ["https://we-news.com/uploads/_r/160/ia-avatar-chiara-lombardi-602f73.webp"])

    def test_bad_json_ld_is_ignored(self):
        page = '<script type="application/ld+json">{not json</script><a href="/author/x">x</a>'
        self.assertEqual(bf.find_author_urls(page, "https://example.com/a-b"),
                         ["https://example.com/author/x"])

    def test_ld_objects_nested(self):
        objs = bf.ld_objects([json.dumps({"@graph": [{"@type": "Person", "name": "A"},
                                                    {"@type": "WebPage", "author": {"name": "B"}}]})])
        names = sorted(o.get("name") for o in objs if "name" in o)
        self.assertEqual(names, ["A", "B"])


class WaybackTests(unittest.TestCase):
    def test_cdx_query_url(self):
        u = bf.cdx_query_url("si.com/review/author/", limit=50)
        self.assertTrue(u.startswith("https://web.archive.org/cdx/search/cdx?"))
        for part in ("url=si.com%2Freview%2Fauthor%2F", "matchType=prefix", "output=json",
                     "filter=statuscode%3A200", "filter=mimetype%3Atext%2Fhtml",
                     "collapse=urlkey", "limit=50"):
            self.assertIn(part, u)

    def test_cdx_rows(self):
        data = [["timestamp", "original", "statuscode", "mimetype"],
                ["20231101000000", "https://www.si.com/review/author/drew-ortiz/", "200", "text/html"],
                ["short"]]
        self.assertEqual(bf.cdx_rows(data),
                         [("20231101000000", "https://www.si.com/review/author/drew-ortiz/")])
        self.assertEqual(bf.cdx_rows([]), [])
        self.assertEqual(bf.cdx_rows({"x": 1}), [])
        self.assertEqual(bf.cdx_rows([["a", "b"], ["1", "2"]]), [])

    def test_wayback_url(self):
        self.assertEqual(bf.wayback_url("20250101000000", "https://x.com/a.jpg", "im_"),
                         "https://web.archive.org/web/20250101000000im_/https://x.com/a.jpg")

    def test_profile_candidates(self):
        rows = [("1", "http://x.com/author/a"),
                ("2", "https://www.x.com/author/a/"),          # same page, later capture
                ("3", "https://x.com/author/a/feed/"),          # feed: skipped
                ("4", "x.com/masthead"),                        # masthead: last tier
                ("5", "https://x.com/2025/01/some-story-here"),  # article: skipped
                ("6", "https://x.com/jane-fake/"),              # documented persona slug: first
                ("7", "https://x.com/author/b/page/2")]         # pagination: skipped
        got = bf.profile_candidates(rows, ["Jane-Fake"])
        self.assertEqual(got, [("6", "https://x.com/jane-fake/", True),
                               ("1", "http://x.com/author/a", False),
                               ("4", "x.com/masthead", False)])
        self.assertEqual(len(bf.profile_candidates(rows, ["jane-fake"], limit=1)), 1)

    def test_looks_like_image(self):
        self.assertTrue(bf.looks_like_image(b"\xff\xd8\xff\xe0rest"))
        self.assertTrue(bf.looks_like_image(b"\x89PNG\r\n\x1a\nrest"))
        self.assertTrue(bf.looks_like_image(b"RIFF\x00\x00\x00\x00WEBPVP8 "))
        self.assertFalse(bf.looks_like_image(b"<!DOCTYPE html>"))
        self.assertFalse(bf.looks_like_image(None))


class MiscTests(unittest.TestCase):
    def test_robots(self):
        robots = "User-agent: *\nDisallow: /private/\n\nUser-agent: byline-faces\nDisallow: /authors/\n"
        self.assertTrue(bf.robots_allows(robots, "https://example.com/news/a"))
        self.assertFalse(bf.robots_allows(robots, "https://example.com/authors/ann"))
        self.assertTrue(bf.robots_allows("", "https://example.com/anything"))

    def test_wikidata_hosts(self):
        data = {"results": {"bindings": [
            {"item": {"value": "http://www.wikidata.org/entity/Q1"}, "site": {"value": "https://www.paper.com/"}},
            {"item": {"value": "http://www.wikidata.org/entity/Q2"}, "site": {"value": "http://paper.com/news"}},
            {"item": {"value": "http://www.wikidata.org/entity/Q3"}, "site": {"value": "ftp://x.org"}},
            {"item": {"value": "http://www.wikidata.org/entity/Q4"}, "site": {"value": "https://Other.co.uk"}},
        ]}}
        self.assertEqual(bf.wikidata_hosts(data), [
            ("paper.com", "https://www.paper.com/", "Q1"),
            ("other.co.uk", "https://other.co.uk/", "Q4"),
        ])

    def test_decode_text(self):
        self.assertEqual(bf.decode_text("café".encode("latin-1"), "text/html; charset=ISO-8859-1"), "café")
        self.assertEqual(bf.decode_text("café".encode("utf-8")), "café")

    def test_version_and_licence(self):
        self.assertEqual(bf.__version__, "0.2.0")
        for mod in (bf, job):
            with open(mod.__file__, encoding="utf-8") as fh:
                self.assertIn("Apache-2.0", fh.read(400))


# ---------------------------------------------------------------------------
# job.py end to end, with a faked network and a faked face detector
# ---------------------------------------------------------------------------

JPEG = b"\xff\xd8\xff\xe0"
HTML = "text/html; charset=utf-8"


class FakeDetector:
    """'Detects' one face: GAN-aligned eyes if the bytes contain GANFACE, else a real pose."""
    name = "fake"

    def score(self, data, calib=None):
        if b"GANFACE" in data:
            geom = bf.eye_geometry((375, 475), (625, 475), 1000, 1000)
        else:
            geom = bf.eye_geometry((300, 330), (460, 350), 1000, 1000)
        rec = {"w0": 1000, "h0": 1000, "n_faces": 1, "geom": geom}
        rec.update(bf.classify(geom, calib))
        return rec


def fake_fetch(routes, log):
    def fetch(url, max_bytes=3_000_000, timeout=12, accept="*/*"):
        log.append(url)
        if url in routes:
            st, ct, body = routes[url]
            return st, url, ct, body
        return 404, url, "", b""
    return fetch


SITE_HOME = """<html><body>
<a href="/2026/11/16/council-votes-to-close-the-library">Council votes to close the main library early</a>
</body></html>"""
SITE_ARTICLE = """<html><head><script type="application/ld+json">
{"@type":"NewsArticle","author":{"@type":"Person","name":"Ann","url":"https://ex.com/author/ann"}}
</script></head><body><img class="byline-avatar" src="/img/ann-small.jpg"></body></html>"""
SITE_AUTHOR = """<html><head><meta property="og:image" content="/img/og.jpg"></head>
<body><img class="author-photo" src="/img/ann.jpg"></body></html>"""


class JobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.run = job.Run(self.tmp.name)

    def tearDown(self):
        for fh in (self.run.logf, self.run.recf, self.run.sitef):
            fh.close()
        self.tmp.cleanup()

    def test_crawl_site_population(self):
        routes = {
            "https://ex.com/": (200, HTML, SITE_HOME.encode()),
            "https://ex.com/2026/11/16/council-votes-to-close-the-library": (200, HTML, SITE_ARTICLE.encode()),
            "https://ex.com/author/ann": (200, HTML, SITE_AUTHOR.encode()),
            "https://ex.com/img/ann-small.jpg": (200, "image/jpeg", JPEG + b"REAL"),
            "https://ex.com/img/ann.jpg": (200, "image/jpeg", JPEG + b"GANFACE"),
            "https://ex.com/img/og.jpg": (200, "image/jpeg", JPEG + b"REAL2"),
        }
        calls = []
        with mock.patch.object(bf, "fetch", fake_fetch(routes, calls)):
            site = job.crawl_site(self.run, FakeDetector(), None, "ex.com", "https://ex.com/",
                                  "population", "Q1",
                                  {"max_articles": 3, "max_authors": 3, "max_images": 3,
                                   "max_article_images": 2}, time.time() + 60)
        self.assertEqual(site["status"], "ok")
        self.assertEqual((site["articles"], site["author_urls"], site["author_pages"]), (1, 1, 1))
        self.assertEqual((site["images_tried"], site["single_face"], site["aligned"],
                          site["centred_level"]), (3, 3, 1, 1))
        self.assertEqual(calls[0], "https://ex.com/robots.txt")
        kinds = sorted(r["img_kind"] for r in self.run.records)
        self.assertEqual(kinds, ["article_img", "img", "og"])
        flagged = [r for r in self.run.records if r.get("saved")]
        self.assertEqual(len(flagged), 1)
        self.assertTrue(flagged[0]["saved"].startswith("flagged"))
        self.run.add_site(site)
        s = job.summarise(self.run, None, {})
        g = s["groups"]["population"]
        self.assertEqual((g["sites"], g["unique_single_face_images"], g["unique_aligned"],
                          g["sites_flagged_either"]), (1, 3, 1, 1))
        self.assertEqual(g["by_kind"]["img"], {"n": 1, "either": 1})

    def test_robots_disallow(self):
        routes = {"https://ex.com/robots.txt": (200, "text/plain",
                                                b"User-agent: *\nDisallow: /\n")}
        with mock.patch.object(bf, "fetch", fake_fetch(routes, [])):
            site = job.crawl_site(self.run, FakeDetector(), None, "ex.com", "https://ex.com/",
                                  "population", "Q1", {"max_articles": 3, "max_authors": 3,
                                                       "max_images": 3}, time.time() + 60)
        self.assertEqual(site["status"], "robots_disallow")
        self.assertEqual(self.run.records, [])

    def test_control_seed_page(self):
        seed = ('<a href="/chiara-lombardi"><img src="/uploads/_r/160/ia-avatar-chiara-lombardi-1.webp"'
                ' alt="Chiara Lombardi"></a><a href="/luca-moretti"><img '
                'src="/uploads/_r/160/ia-avatar-luca-moretti-2.webp" alt="Luca Moretti"></a>')
        routes = {
            "https://we.test/la-redazione": (200, HTML, seed.encode()),
            "https://we.test/": (200, HTML, b"<html></html>"),
            "https://we.test/uploads/_r/160/ia-avatar-chiara-lombardi-1.webp": (200, "image/webp", JPEG + b"A"),
            "https://we.test/uploads/_r/160/ia-avatar-luca-moretti-2.webp": (200, "image/webp", JPEG + b"GANFACE"),
        }
        with mock.patch.object(bf, "fetch", fake_fetch(routes, [])):
            site = job.crawl_site(self.run, FakeDetector(), None, "we.test", "https://we.test/",
                                  "control:we (labelled)", "",
                                  {"max_articles": 6, "max_authors": 40, "max_images": 2,
                                   "max_seed_images": 60}, time.time() + 60,
                                  ["https://we.test/la-redazione"])
        self.assertEqual((site["images_tried"], site["single_face"], site["aligned"]), (2, 2, 1))
        self.assertTrue(all(r["img_kind"] == "seed_img" for r in self.run.records))
        saved = [r["saved"] for r in self.run.records if r.get("saved")]
        self.assertEqual(len(saved), 1)
        self.assertNotIn("(", saved[0])
        self.run.add_site(site)
        self.assertIn("control:we (labelled)", job.summarise(self.run, None, {})["groups"])

    def test_documented_stage(self):
        spec = {"group": "documented:test", "source": "a public report", "prefixes": ["fake.com/"],
                "slugs": ["jane-fake"]}
        cdx = [["timestamp", "original", "statuscode", "mimetype"],
               ["20250101000000", "https://fake.com/jane-fake/", "200", "text/html"],
               ["20250101000000", "https://fake.com/author/bob", "200", "text/html"],
               ["20250101000000", "https://fake.com/2025/01/story-a-b-c", "200", "text/html"]]
        ts = "20250101000000"
        routes = {
            bf.cdx_query_url("fake.com/"): (200, "application/json", json.dumps(cdx).encode()),
            bf.wayback_url(ts, "https://fake.com/jane-fake/"): (
                200, HTML, b'<img class="author-avatar" src="/up/jane.jpg">'),
            bf.wayback_url(ts, "https://fake.com/author/bob"): (
                200, HTML, b'<img class="author-avatar" src="/up/bob.jpg">'),
            bf.wayback_url(ts, "https://fake.com/up/jane.jpg", "im_"): (200, "image/jpeg", JPEG + b"GANFACE"),
            bf.wayback_url(ts, "https://fake.com/up/bob.jpg", "im_"): (200, "image/jpeg", JPEG + b"REAL"),
        }
        with mock.patch.object(bf, "fetch", fake_fetch(routes, [])), \
                mock.patch.object(job, "DOCUMENTED", [spec]):
            job.stage_documented(self.run, FakeDetector(), None, time.time() + 60, pause=0)
        site = self.run.sites[0]
        self.assertEqual(site["status"], "ok")
        self.assertEqual((site["author_urls"], site["author_pages"], site["persona_pages"]), (2, 2, 1))
        self.assertEqual((site["images_tried"], site["single_face"], site["aligned"],
                          site["persona_single_face"]), (2, 2, 1, 1))
        g = job.summarise(self.run, None, {})["groups"]["documented:test"]
        self.assertEqual((g["persona_unique_single_face"], g["persona_either"]), (1, 1))
        self.assertEqual(g["source"], "a public report")

    def test_documented_specs_are_sourced(self):
        for spec in job.DOCUMENTED:
            self.assertTrue(spec["group"].startswith("documented:"))
            self.assertTrue(spec["source"] and spec["prefixes"] and spec["slugs"])


if __name__ == "__main__":
    unittest.main()
