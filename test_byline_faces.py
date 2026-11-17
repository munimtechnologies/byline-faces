# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Standard-library tests for the pure functions of byline_faces (no OpenCV, no network).
import json
import math
import unittest

import byline_faces as bf


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

    def test_bad_json_ld_is_ignored(self):
        page = '<script type="application/ld+json">{not json</script><a href="/author/x">x</a>'
        self.assertEqual(bf.find_author_urls(page, "https://example.com/a-b"),
                         ["https://example.com/author/x"])

    def test_ld_objects_nested(self):
        objs = bf.ld_objects([json.dumps({"@graph": [{"@type": "Person", "name": "A"},
                                                    {"@type": "WebPage", "author": {"name": "B"}}]})])
        names = sorted(o.get("name") for o in objs if "name" in o)
        self.assertEqual(names, ["A", "B"])


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
        self.assertEqual(bf.__version__, "0.1.0")
        with open(bf.__file__, encoding="utf-8") as fh:
            self.assertIn("Apache-2.0", fh.read(400))


if __name__ == "__main__":
    unittest.main()
