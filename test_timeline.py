# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Tests for timeline.py (survey 4). Standard library only, no network."""
import unittest

import timeline as tl


def ob(persona, capture, sha, dhash, cf, aligned=False, img_capture=None, page=None):
    return {"persona": persona, "case": tl.PERSONA_BY_ID[persona]["case"],
            "page": page or tl.PERSONA_BY_ID[persona]["pages"][0], "capture": capture,
            "img_capture": img_capture, "sha256": sha, "dhash": dhash, "cf": cf,
            "aligned": aligned, "centred_level": False, "wayback_img": ""}


ZERO = "0" * 16
ONES = "f" * 16


def synthetic():
    return [
        # Kuznetsov: an old real-looking picture (2016-2017), then an AI-looking one (2024)
        ob("nikolai-kuznetsov", "20160501000000", "k1", ZERO, 0.1),
        ob("nikolai-kuznetsov", "20170301000000", "k1", ZERO, 0.1),
        ob("nikolai-kuznetsov", "20240101000000", "k2", ONES, 0.9),
        # Jackson: 2019 -> 2024, up
        ob("reuben-jackson", "20190601000000", "j1", ZERO, 0.2),
        ob("reuben-jackson", "20240601000000", "j2", ONES, 0.8),
        # Aureliano: one old picture, not flagged
        ob("luis-aureliano", "20170101000000", "a1", ZERO, 0.05),
        # Liebkind: one picture in three re-encodings (same dhash), all before Dec 2018
        ob("joe-liebkind", "20160101000000", "l1", ZERO, 0.1),
        ob("joe-liebkind", "20170101000000", "l2", "0000000000000003", 0.2),
        ob("joe-liebkind", "20171101000000", "l3", ZERO, 0.15),
        # Robson: one AI-looking picture, 2025
        ob("daniel-robson", "20250901000000", "r1", ONES, 0.95),
        # alice-johnson: 2024 -> 2025, up
        ob("alice-johnson", "20240301000000", "x1", ZERO, 0.3),
        ob("alice-johnson", "20250301000000", "x2", ONES, 0.7),
    ]


class TestTables(unittest.TestCase):
    def test_personas(self):
        ids = [p["id"] for p in tl.PERSONAS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(1 for p in tl.PERSONAS if p["case"] == tl.PG_CASE), 4)
        for p in tl.PERSONAS:
            self.assertIn(p["case"], tl.CASE_SOURCES)
            self.assertTrue(p["pages"] and p["tokens"] and p["hint"])
            for page in p["pages"]:
                self.assertFalse(page.startswith("http"), page)
                self.assertFalse(page.startswith("www."), page)
                self.assertNotIn(" ", page)

    def test_cdx_and_wayback(self):
        u = tl.cdx_exact_url("forbes.com/sites/x")
        self.assertTrue(u.startswith("https://web.archive.org/cdx/search/cdx?"))
        self.assertIn("matchType=exact", u)
        self.assertIn("output=json", u)
        self.assertEqual(tl.wayback("2016", "http://a.com/x", "id_"),
                         "https://web.archive.org/web/2016id_/http://a.com/x")
        self.assertEqual(tl.variants("a.com/x/"), ["a.com/x", "a.com/x/"])
        self.assertEqual(tl.variants("a.com/x"), ["a.com/x", "a.com/x/"])
        self.assertEqual(tl.variants(""), [])

    def test_parse_cdx(self):
        rows = tl.parse_cdx([["timestamp", "original"], ["20160101000000", "http://a.com/x"],
                             ["bad"]])
        self.assertEqual(rows, [("20160101000000", "http://a.com/x")])
        self.assertEqual(tl.parse_cdx([]), [])
        self.assertEqual(tl.parse_cdx({"x": 1}), [])


class TestPick(unittest.TestCase):
    def test_month_dedupe_and_spread(self):
        rows = [("20160105000000", "a"), ("20160102000000", "b"), ("20160301000000", "c"),
                ("20170101000000", "d"), ("20180101000000", "e"), ("20190101000000", "f"),
                ("junk", "g")]
        self.assertEqual(tl.pick_captures(rows, 10),
                         [("20160102000000", "b"), ("20160301000000", "c"),
                          ("20170101000000", "d"), ("20180101000000", "e"),
                          ("20190101000000", "f")])
        self.assertEqual([r[1] for r in tl.pick_captures(rows, 3)], ["b", "d", "f"])
        self.assertEqual([r[1] for r in tl.pick_captures(rows, 1)], ["b"])
        self.assertEqual(tl.pick_captures(rows, 0), [])
        ten = [("2016%02d01000000" % m, str(m)) for m in range(1, 11)]
        self.assertEqual([r[1] for r in tl.pick_captures(ten, 4)], ["1", "4", "7", "10"])


class TestHash(unittest.TestCase):
    def test_dhash(self):
        dec = [list(range(9, 0, -1)) for _ in range(8)]
        self.assertEqual(tl.dhash_bits(dec), 2 ** 64 - 1)
        flat = [[5] * 9 for _ in range(8)]
        self.assertEqual(tl.dhash_bits(flat), 0)
        with self.assertRaises(ValueError):
            tl.dhash_bits([[1] * 9])
        self.assertEqual(tl.hamming(0, 2 ** 64 - 1), 64)
        self.assertEqual(tl.parse_hash("ff"), 255)
        self.assertIsNone(tl.parse_hash("zz"))
        self.assertIsNone(tl.parse_hash(None))

    def test_host_and_order(self):
        self.assertEqual(tl.host_of("https://www.Forbes.com/sites/x"), "forbes.com")
        self.assertEqual(tl.host_of("tech.eu/author/x"), "tech.eu")
        cands = [{"kind": "og", "url": "https://a.com/og.jpg"},
                 {"kind": "img", "url": "https://a.com/avatar.jpg"},
                 {"kind": "jsonld", "url": "https://a.com/p.jpg"},
                 {"kind": "img", "url": "https://a.com/Kuznetsov-head.jpg"}]
        got = [c["url"] for c in tl.order_candidates(cands, ["kuznetsov"])]
        self.assertEqual(got, ["https://a.com/Kuznetsov-head.jpg", "https://a.com/p.jpg",
                               "https://a.com/avatar.jpg", "https://a.com/og.jpg"])


class TestCluster(unittest.TestCase):
    def test_clusters(self):
        obs = [o for o in synthetic() if o["persona"] == "joe-liebkind"]
        cl = tl.cluster_observations(obs)
        self.assertEqual(len(cl), 1)
        s = tl.cluster_summary(cl[0])
        self.assertEqual(s["images"], 3)
        self.assertEqual(s["observations"], 3)
        self.assertEqual(s["cf_median"], 0.15)
        self.assertFalse(s["cf_flag"])
        self.assertEqual(s["first_seen"], "20160101000000")
        self.assertEqual(s["last_seen"], "20171101000000")
        # same SHA without a hash still joins; a far hash starts a new cluster
        more = obs + [ob("joe-liebkind", "20180101000000", "l1", None, 0.1),
                      ob("joe-liebkind", "20200101000000", "l9", ONES, 0.6)]
        cl2 = tl.cluster_observations(more)
        self.assertEqual(len(cl2), 2)
        self.assertEqual(tl.median([3, 1, 2, 4]), 2.5)
        self.assertIsNone(tl.median([]))


class TestAnalyseGrade(unittest.TestCase):
    def test_synthetic(self):
        s = tl.analyse(synthetic(), cf_repro_ok=True)
        P = s["personas"]
        self.assertEqual(P["nikolai-kuznetsov"]["distinct_headshots"], 2)
        self.assertEqual(P["nikolai-kuznetsov"]["direction"], "up")
        self.assertEqual(P["joe-liebkind"]["distinct_headshots"], 1)
        self.assertIsNone(P["joe-liebkind"]["direction"])
        self.assertEqual(P["michael-harrison"]["distinct_headshots"], 0)
        c = s["control_pre2019"]
        self.assertEqual(c["n"], 5)          # k1, a1, l1, l2, l3
        self.assertEqual(c["cf_flagged"], 0)
        d = s["diffusion_era_flagged"]
        self.assertEqual(d["n"], 4)          # k2, j2, r1, x2
        self.assertEqual(d["eye_flagged"], 0)
        g = tl.grade(s)["grades"]
        self.assertEqual(g["T1"], {"verdict": "true", "value": 4})
        self.assertEqual(g["T2"], {"verdict": "true", "value": 2})
        self.assertEqual(g["T3"]["verdict"], "true")
        self.assertEqual((g["T3"]["up"], g["T3"]["n"]), (3, 3))
        self.assertAlmostEqual(g["T3"]["p_one_sided"], 0.125)
        self.assertEqual(g["T4"]["verdict"], "true")
        self.assertEqual(g["T5"]["verdict"], "true")
        self.assertEqual(g["T6"]["verdict"], "true")
        self.assertEqual(g["T7"]["verdict"], "true")
        self.assertEqual(g["T8"]["verdict"], "true")
        self.assertEqual(len(tl.public_lines(synthetic())), 12)

    def test_control_needs_both_captures_early(self):
        obs = [ob("joe-liebkind", "20170101000000", "l1", ZERO, 0.9,
                  img_capture="20200101000000")]
        self.assertEqual(tl.analyse(obs, cf_repro_ok=True)["control_pre2019"]["n"], 0)

    def test_failures_and_invalid(self):
        obs = synthetic() + [ob("joe-liebkind", "20200101000000", "l9", ONES, 0.6,
                                aligned=True)]
        g = tl.grade(tl.analyse(obs, cf_repro_ok=True))["grades"]
        self.assertEqual(g["T7"]["verdict"], "false")
        bad = tl.grade(tl.analyse(synthetic(), cf_repro_ok=False))["grades"]
        self.assertEqual(bad["T1"]["verdict"], "true")
        for name in ("T3", "T4", "T5", "T6", "T7", "T8"):
            self.assertEqual(bad[name]["verdict"], "invalid")

    def test_empty(self):
        g = tl.grade(tl.analyse([], cf_repro_ok=True))["grades"]
        self.assertEqual(g["T1"]["verdict"], "false")
        self.assertEqual(g["T2"]["verdict"], "no_data")
        for name in ("T3", "T4", "T5", "T6", "T7", "T8"):
            self.assertEqual(g[name]["verdict"], "no_data", name)

    def test_binom(self):
        self.assertAlmostEqual(tl.binom_tail(3, 3), 0.125)
        self.assertAlmostEqual(tl.binom_tail(0, 4), 1.0)
        self.assertIsNone(tl.binom_tail(0, 0))


if __name__ == "__main__":
    unittest.main()
