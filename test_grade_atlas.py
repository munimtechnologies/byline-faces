# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Tests for grade_atlas.py on a small hand-checked fixture (standard library only)."""
import json
import os
import tempfile
import unittest

import grade_atlas as ga


def meta(period, origin):
    return {"period": period, "origin": origin}


CASES = {
    "oliver-taylor-2020": meta("2019-2021", "stylegan"),
    "uae-network-labba-2020": meta("2019-2021", "stylegan"),
    "uae-network-badani-2020": meta("2019-2021", "stolen"),
    "si-advon-2023": meta("2022-2024", "stylegan"),
    "irish-times-2023": meta("2022-2024", "diffusion"),
    "dc-weekly-2023": meta("2022-2024", "stolen"),
    "watermark-b2b-2026": meta("2025-2026", "ai-unspecified"),
    "paradise-media-pa-2026": meta("2025-2026", "ai-unspecified"),
    "we-news-2026": meta("2025-2026", "ai-unspecified"),
}


def face(case, sha, eye=False, cf=None, page=""):
    return {"case": case, "sha256": sha, "eye_aligned": eye, "eye_centred_level": False,
            "cf": cf, "archived_page": page}


def fixture_faces():
    return [
        face("oliver-taylor-2020", "a1", eye=True, cf=0.6),
        face("oliver-taylor-2020", "a2", eye=True),
        face("uae-network-labba-2020", "b1", eye=True),
        face("uae-network-labba-2020", "b2", cf=0.2),
        face("uae-network-badani-2020", "c1", cf=0.1),
        face("uae-network-badani-2020", "c2", cf=0.3),
        face("si-advon-2023", "d1", eye=True,
             page="https://web.archive.org/web/2023/https://www.si.com/review/author/drew-ortiz/"),
        face("si-advon-2023", "d2",
             page="https://web.archive.org/web/2023/https://www.si.com/review/author/sora-tanaka/"),
        face("irish-times-2023", "e1", cf=0.9),
        face("dc-weekly-2023", "f1", cf=0.1),
        face("watermark-b2b-2026", "g1", cf=0.8),
        face("watermark-b2b-2026", "g2", cf=0.7),
        face("watermark-b2b-2026", "g3", eye=True, cf=0.9),
        face("paradise-media-pa-2026", "h1", cf=0.2),
        face("we-news-2026", "i1", cf=0.95),
        # the same image as a1, reused by a later case: pooled once, in 2019-2021
        face("we-news-2026", "a1", eye=True, cf=0.6),
        # a face of a case that is not in the summary is ignored
        face("not-a-case", "z1", eye=True),
    ]


def summary(cf_ok=True, primary_diff=-0.3):
    return {"version": "0.4.0", "cf_repro_ok": cf_ok, "cases": dict(CASES),
            "primary": {"diff_late_minus_early": primary_diff}}


class GradeTests(unittest.TestCase):
    def setUp(self):
        self.r = ga.grade(summary(), fixture_faces())
        self.p = self.r["predictions"]

    def test_all_predictions_present(self):
        self.assertEqual(list(self.p), ["A%d" % i for i in range(1, 12)])
        self.assertTrue(self.r["cf_valid"])
        self.assertEqual(self.r["faces_graded"], 17)

    def test_a1_a2(self):
        self.assertEqual(self.p["A1"]["verdict"], "true")
        self.assertEqual(self.p["A2"]["cases_with_persona_faces"], 9)
        self.assertEqual(self.p["A2"]["verdict"], "false")  # 9 < 10

    def test_a3_gan_era(self):
        # a1, a2, b1 of a1, a2, b1, b2 carry the eye signature
        self.assertEqual((self.p["A3"]["k"], self.p["A3"]["n"]), (3, 4))
        self.assertEqual(self.p["A3"]["verdict"], "true")

    def test_a4_shift(self):
        a4 = self.p["A4"]
        # early pool a1 a2 b1 b2 c1 c2: 3 of 6; late pool g1 g2 g3 h1 i1 (a1 counted early): 1 of 5
        self.assertEqual((a4["eye_2019_2021"]["k"], a4["eye_2019_2021"]["n"]), (3, 6))
        self.assertEqual((a4["eye_2025_2026"]["k"], a4["eye_2025_2026"]["n"]), (1, 5))
        self.assertAlmostEqual(a4["diff_pooled"], -0.3)
        # case means: early (1 + 0.5 + 0) / 3; late (1/3 + 0 + 1/2) / 3
        self.assertAlmostEqual(a4["case_mean_2019_2021"], 0.5)
        self.assertAlmostEqual(a4["case_mean_2025_2026"], (1 / 3 + 0.5) / 3)
        self.assertAlmostEqual(a4["diff_case_mean"], (1 / 3 + 0.5) / 3 - 0.5)
        self.assertEqual(a4["verdict"], "true")
        self.assertTrue(a4["matches_job_summary"])

    def test_a4_mismatch_with_job_summary_is_reported(self):
        r = ga.grade(summary(primary_diff=-0.1), fixture_faces())
        self.assertFalse(r["predictions"]["A4"]["matches_job_summary"])

    def test_a5_new_faces(self):
        # g1, g2, i1 flagged by CF only, of 5 late faces
        self.assertEqual((self.p["A5"]["k"], self.p["A5"]["n"]), (3, 5))
        self.assertEqual(self.p["A5"]["verdict"], "true")

    def test_a6_a7_a8_single_personas(self):
        self.assertEqual(self.p["A6"]["verdict"], "true")
        self.assertEqual(self.p["A6"]["flagged"], 0)
        self.assertEqual(self.p["A7"]["verdict"], "true")
        self.assertEqual((self.p["A8"]["k"], self.p["A8"]["n"]), (1, 1))
        self.assertEqual(self.p["A8"]["verdict"], "true")

    def test_a9_a10(self):
        # AI-labelled pool: a1 a2 b1 b2 d1 d2 e1 g1 g2 g3 h1 i1; flagged: all but b2 d2 h1
        self.assertEqual((self.p["A9"]["k"], self.p["A9"]["n"]), (9, 12))
        self.assertEqual(self.p["A9"]["verdict"], "true")
        self.assertEqual((self.p["A10"]["k"], self.p["A10"]["n"]), (0, 3))
        self.assertEqual(self.p["A10"]["verdict"], "true")

    def test_a11(self):
        cases = self.p["A11"]["cases"]
        self.assertEqual(cases["watermark-b2b-2026"]["verdict"], "true")
        self.assertEqual(cases["paradise-media-pa-2026"]["verdict"], "false")
        self.assertEqual(self.p["A11"]["verdict"], "false")

    def test_cf_invalid(self):
        p = ga.grade(summary(cf_ok=False), fixture_faces())["predictions"]
        self.assertEqual(p["A1"]["verdict"], "false")
        for key in ("A5", "A7", "A11"):
            self.assertEqual(p[key]["verdict"], "invalid", key)
        self.assertEqual(p["A6"]["verdict"], "true")
        self.assertIn("eye test only", p["A6"]["basis"])
        # eye only: a1 a2 b1 d1 g3 of 12
        self.assertEqual((p["A9"]["k"], p["A9"]["n"]), (5, 12))
        self.assertEqual(p["A9"]["verdict"], "false")
        self.assertEqual(p["A10"]["verdict"], "true")

    def test_missing_repro_counts_as_invalid(self):
        s = summary()
        del s["cf_repro_ok"]
        self.assertFalse(ga.grade(s, fixture_faces())["cf_valid"])

    def test_tie_is_undetermined(self):
        faces = fixture_faces() + [face("irish-times-2023", "e2", cf=0.1)]
        self.assertEqual(ga.grade(summary(), faces)["predictions"]["A7"]["verdict"],
                         "undetermined")

    def test_no_faces(self):
        p = ga.grade(summary(), [])["predictions"]
        self.assertEqual(p["A2"]["verdict"], "false")
        for key in ("A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11"):
            self.assertEqual(p[key]["verdict"], "no_data", key)
        self.assertNotIn("matches_job_summary", p["A4"])

    def test_no_cases(self):
        self.assertEqual(ga.grade({}, [])["predictions"]["A2"]["verdict"], "no_data")

    def test_a11_partial(self):
        faces = [f for f in fixture_faces() if f["case"] != "paradise-media-pa-2026"]
        self.assertEqual(ga.grade(summary(), faces)["predictions"]["A11"]["verdict"], "partial")

    def test_review_keeps_only_marked_headshots(self):
        review = {f["sha256"]: True for f in fixture_faces()}
        review["d1"] = False
        r = ga.grade(summary(), fixture_faces(), review)
        self.assertTrue(r["review_applied"])
        self.assertEqual(r["predictions"]["A8"]["verdict"], "no_data")
        self.assertNotIn("matches_job_summary", r["predictions"]["A4"])

    def test_wilson(self):
        self.assertIsNone(ga.wilson(0, 0))
        lo, hi = ga.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertTrue(0.25 < hi < 0.35)

    def test_markdown(self):
        md = ga.to_markdown(self.r)
        self.assertIn("| A4 | true |", md)
        self.assertIn("watermark-b2b-2026: true (3/3)", md)


class FileTests(unittest.TestCase):
    def test_main_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            sp, fp = os.path.join(d, "s.json"), os.path.join(d, "f.jsonl")
            rp, op = os.path.join(d, "r.jsonl"), os.path.join(d, "g.json")
            with open(sp, "w", encoding="utf-8") as fh:
                json.dump(summary(), fh)
            with open(fp, "w", encoding="utf-8") as fh:
                for f in fixture_faces():
                    fh.write(json.dumps(f) + "\n")
                fh.write("\n")
            with open(rp, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"sha256": "e1", "is_persona_headshot": True}) + "\n")
                fh.write(json.dumps({"sha256": "d1", "is_persona_headshot": False}) + "\n")
            result = ga.main([sp, fp, "--review", rp, "--out", op])
            with open(op, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["mechanical"]["predictions"]["A4"]["verdict"], "true")
            self.assertEqual(result["reviewed"]["faces_graded"], 1)
            self.assertEqual(saved["reviewed"]["predictions"]["A7"]["verdict"], "true")


if __name__ == "__main__":
    unittest.main()
