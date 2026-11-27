# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Standard-library tests for survey5.py (no network, no torch, no opencv)."""
import json
import unittest

import survey5 as s5

CHATGPT = ("https://www.videogamer.com/wp-content/uploads/2025/10/"
           "ChatGPT-Image-Oct-20-2025-11_57_34-AM-300×300.png")


def rec(period, sha, cf, aligned=False, role="main", img_url="", nbytes=1000, slug="writer",
        geom=True):
    r = {"survey5_role": role, "period": period, "sha256": sha, "cf": cf, "aligned": aligned,
         "centred_level": False, "img_url": img_url, "bytes": nbytes, "slug": slug,
         "capture": "20260301000000", "author_page": "https://www.videogamer.com/author/%s/" % slug}
    if geom:
        r["geom"] = {}
    return r


def before(cfs):
    return [rec("before", "b%d" % i, v) for i, v in enumerate(cfs, 1)]


def after(cfs, aligned_first=True):
    return [rec("after", "a%d" % i, v, aligned=(aligned_first and i == 1))
            for i, v in enumerate(cfs, 1)]


def merrygold(cf, nbytes=1000, sha="m1"):
    return rec(None, sha, cf, role="secondary", img_url=CHATGPT, nbytes=nbytes)


class TestPeriods(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(s5.period_of_ts("20250831235959"), "before")
        self.assertEqual(s5.period_of_ts("20250901000000"), "between")
        self.assertEqual(s5.period_of_ts("20260218235959"), "between")
        self.assertEqual(s5.period_of_ts("20260219000000"), "after")
        self.assertEqual(s5.period_of_ts("2026"), "between")
        self.assertIsNone(s5.period_of_ts(""))
        self.assertIsNone(s5.period_of_ts(None))

    def test_author_slug(self):
        self.assertEqual(s5.author_slug("https://www.videogamer.com/author/brian-merrygold/"),
                         "brian-merrygold")
        self.assertEqual(s5.author_slug("http://videogamer.com/author/Jane-Doe"), "jane-doe")
        self.assertEqual(s5.author_slug("http://videogamer.com/author/jane/?amp=1"), "jane")
        self.assertEqual(s5.author_slug("videogamer.com/author/jane/"), "jane")
        self.assertIsNone(s5.author_slug("https://videogamer.com/author/x/page/2/"))
        self.assertIsNone(s5.author_slug("http://videogamer.com/author/"))
        self.assertIsNone(s5.author_slug("http://videogamer.com/author/jane/feed/"))
        self.assertIsNone(s5.author_slug(""))

    def test_cdx_url(self):
        u = s5.cdx_url(s5.PREFIX, "20230101", "20250831")
        self.assertTrue(u.startswith("https://web.archive.org/cdx/search/cdx?"))
        self.assertIn("url=videogamer.com%2Fauthor%2F", u)
        self.assertIn("matchType=prefix", u)
        self.assertIn("from=20230101", u)
        self.assertIn("to=20250831", u)
        self.assertNotIn("collapse", u)
        self.assertNotIn("to=", s5.cdx_url(s5.PREFIX, "20260219", None))

    def test_cdx_rows(self):
        data = [["timestamp", "original", "statuscode", "mimetype"],
                ["20240101000000", "https://videogamer.com/author/a/", "200", "text/html"],
                "junk"]
        self.assertEqual(s5.cdx_rows(data), [("20240101000000", "https://videogamer.com/author/a/")])
        self.assertEqual(s5.cdx_rows([]), [])


class TestSelect(unittest.TestCase):
    ROWS = [
        ("20240105000000", "http://www.videogamer.com/author/alpha/"),
        ("20250815000000", "https://videogamer.com/author/alpha/"),
        ("20251001000000", "https://videogamer.com/author/alpha/"),
        ("20260301000000", "https://videogamer.com/author/alpha/"),
        ("20260220000000", "https://videogamer.com/author/alpha/"),
        ("20260222000000", "https://videogamer.com/author/beta/"),
        ("20240101000000", "https://videogamer.com/author/beta/page/2/"),
        ("20250101000000", "https://videogamer.com/author/"),
    ]

    def test_latest_before_earliest_after_interleaved(self):
        got = [(p["period"], p["slug"], p["ts"]) for p in s5.select_pages(self.ROWS)]
        self.assertEqual(got, [("before", "alpha", "20250815000000"),
                               ("after", "alpha", "20260220000000"),
                               ("after", "beta", "20260222000000")])

    def test_cap(self):
        got = [(p["period"], p["slug"]) for p in s5.select_pages(self.ROWS, cap=1)]
        self.assertEqual(got, [("before", "alpha"), ("after", "alpha")])


class TestHelpers(unittest.TestCase):
    def test_auc(self):
        self.assertAlmostEqual(s5.auc([0.9, 0.8], [0.1, 0.8]), 0.875)
        self.assertIsNone(s5.auc([], [1.0]))

    def test_full_size_variant(self):
        self.assertEqual(
            s5.full_size_variant("https://icobench.com/wp-content/uploads/2025/05/"
                                 "avatar_user_175_1747725199-50x50.png"),
            "https://icobench.com/wp-content/uploads/2025/05/avatar_user_175_1747725199.png")
        self.assertEqual(s5.full_size_variant("https://a.com/x-150x150.jpg?ver=2"),
                         "https://a.com/x.jpg?ver=2")
        self.assertEqual(s5.full_size_variant(CHATGPT), CHATGPT)

    def test_units_dedupe_and_filters(self):
        recs = before([0.1, 0.2]) + [rec("before", "b1", 0.1), rec("between", "z", 0.9),
                                     rec("after", "nf", 0.9, geom=False), merrygold(0.7)]
        u = s5.units(recs)
        self.assertEqual(len(u["before"]), 2)
        self.assertEqual(u["after"], [])


class TestGrade(unittest.TestCase):
    B = [0.1, 0.2, 0.05, 0.3, 0.4]

    def test_all_pass(self):
        recs = (before(self.B) + after([0.9, 0.8, 0.7, 0.3, 0.95])
                + [rec("after", "a1", 0.9), merrygold(0.7), merrygold(0.2, nbytes=500, sha="m2")])
        g = s5.grade(recs, True)
        self.assertEqual(g["periods"]["after"]["n"], 5)
        self.assertEqual(g["periods"]["before"]["cf_flagged"], 0)
        self.assertEqual(g["grades"]["V1"]["verdict"], "pass")
        self.assertAlmostEqual(g["grades"]["V1"]["after_share"], 0.8)
        self.assertEqual(g["grades"]["V2"]["verdict"], "pass")
        self.assertAlmostEqual(g["grades"]["V2"]["auc"], 0.94)
        self.assertEqual(g["grades"]["V3"]["verdict"], "pass")
        self.assertAlmostEqual(g["grades"]["V3"]["chosen_cf"], 0.7)
        self.assertEqual(g["grades"]["V4"]["verdict"], "pass")
        self.assertEqual(g["grades"]["V4"]["after_eye_flagged"], 1)
        self.assertEqual(g["missed_after_faces"], [])
        self.assertEqual(g["same_picture_in_both_periods"], 0)

    def test_failed_repro_invalidates_cf_only(self):
        recs = before(self.B) + after([0.9, 0.8, 0.7, 0.3, 0.95]) + [merrygold(0.7)]
        g = s5.grade(recs, False)
        for k in ("V1", "V2", "V3"):
            self.assertEqual(g["grades"][k]["verdict"], "invalid")
        self.assertEqual(g["grades"]["V4"]["verdict"], "pass")
        self.assertEqual(s5.grade(recs, None)["grades"]["V1"]["verdict"], "invalid")

    def test_too_few(self):
        recs = before(self.B[:4]) + after([0.9, 0.8, 0.7, 0.3, 0.95]) + [merrygold(0.7)]
        g = s5.grade(recs, True)
        for k in ("V1", "V2", "V4"):
            self.assertEqual(g["grades"][k]["verdict"], "unresolved")
            self.assertIn("before 4, after 5", g["grades"][k]["note"])
        self.assertEqual(g["grades"]["V3"]["verdict"], "pass")

    def test_fail_lists_missed(self):
        recs = before(self.B) + after([0.9, 0.3, 0.2, 0.1, 0.4], aligned_first=False) + [merrygold(0.7)]
        g = s5.grade(recs, True)
        self.assertEqual(g["grades"]["V1"]["verdict"], "fail")
        self.assertAlmostEqual(g["grades"]["V1"]["after_share"], 0.2)
        self.assertEqual(g["grades"]["V2"]["verdict"], "fail")
        self.assertAlmostEqual(g["grades"]["V2"]["auc"], 0.68)
        self.assertEqual([m["sha256"] for m in g["missed_after_faces"]], ["a2", "a3", "a4", "a5"])
        self.assertEqual(g["grades"]["V4"]["after_eye_flagged"], 0)

    def test_v3_unresolved_and_fail(self):
        recs = before(self.B) + after([0.9, 0.8, 0.7, 0.3, 0.95])
        g = s5.grade(recs, True)
        self.assertEqual(g["grades"]["V3"]["verdict"], "unresolved")
        self.assertIn("ChatGPT", g["grades"]["V3"]["note"])
        g2 = s5.grade(recs + [merrygold(0.3)], True)
        self.assertEqual(g2["grades"]["V3"]["verdict"], "fail")

    def test_v4_fail(self):
        recs = before(self.B) + [rec("after", "a%d" % i, 0.9, aligned=(i <= 2)) for i in range(1, 6)]
        self.assertEqual(s5.grade(recs, True)["grades"]["V4"]["verdict"], "fail")


class TestPublic(unittest.TestCase):
    def test_no_before_faces_or_real_names(self):
        recs = (before([0.1]) + [rec("after", "a1", 0.9, slug="brian-merrygold"),
                                 rec("after", "a2", 0.2, slug="realperson"),
                                 merrygold(0.7), merrygold(0.7)])
        recs[-1]["persona"] = recs[-2]["persona"] = "Brian Merrygold"
        out = s5.public_faces(recs)
        text = json.dumps(out)
        self.assertNotIn("realperson", text)
        self.assertNotIn("b1", [o.get("sha256") for o in out])
        roles = [o["role"] for o in out]
        self.assertEqual(roles, ["after", "after", "secondary"])
        self.assertEqual(out[0]["persona_slug"], "brian-merrygold")
        self.assertTrue(out[0]["archived_page"].startswith("https://web.archive.org/web/20260301000000/"))
        self.assertNotIn("archived_page", out[1])


if __name__ == "__main__":
    unittest.main()
