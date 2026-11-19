# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Tests for grade_surveys12.py on made-up inputs (standard library only, no network)."""
import json
import os
import tempfile
import unittest

import grade_surveys12 as gs

CTRL = "control:we-news (site labels its writers AI agents)"


def s1_inputs():
    calibration = {"holdout": {"n": 150, "rate_aligned": 0.92},
                   "real_ffhq": {"n": 290, "rate_aligned": 0.85}}
    negatives = {"n": 900, "rate_aligned": 0.05, "rate_centred_level": 0.20}
    summary = {"version": "0.2.0", "groups": {
        CTRL: {"by_kind": {"seed_img": {"n": 40, "either": 4},
                           "article_img": {"n": 10, "either": 9}}},
        "population": {"sites": 10, "status": {"ok": 4, "home_404": 6},
                       "sites_with_single_face": 2, "unique_single_face_images": 200,
                       "unique_aligned": 3},
        "documented:si-advon-2023": {"persona_unique_single_face": 2, "persona_either": 0},
        "documented:south-florida-standard-2026": {"persona_unique_single_face": 2,
                                                   "persona_either": 2},
    }}
    return calibration, negatives, summary


def s2_inputs():
    cal = {"repro": {"ok": True, "max_abs_diff": 0.004},
           "sd15_faces_dff": {"n": 200, "rate_0.5": 0.9},
           "stylegan_140k": {"n": 200, "rate_0.5": 0.79},
           "lfw_real": {"n": 200, "rate_0.5": 0.02},
           "ffhq_real_140k": {"n": 200, "rate_0.5": 0.10},
           "imdbwiki_real_dff": {"error": "HTTPError: 404"}}
    es = {"version": "0.3.0", "t95_A": 0.41,
          "thresholds": {
              "0.5": {"A": {"k": 4, "n": 50}, "diff_Bnew_minus_A": 0.01,
                      "boot95_outlets": [-0.02, 0.04]},
              "t95_A": {"A": {"k": 2, "n": 50}, "diff_Bnew_minus_A": 0.015,
                        "boot95_outlets": [-0.01, 0.05]}},
          "by_format": {"jpeg": {"A": {"n": 48, "cf_0.5": 3}, "B_new": {"n": 30, "cf_0.5": 2}},
                        "webp": {"A": {"n": 2, "cf_0.5": 0}, "B_new": {"n": 20, "cf_0.5": 5}}}}
    sm = {"cf_groups": {
        CTRL: {"cf_scored": 10, "cf_0.5": 2},
        "documented:si-advon-2023": {"persona_n": 3, "persona_cf_0.5": 2, "persona_cf_only": 1},
        "documented:ventures-media-elle-be-2025": {"persona_n": 1, "persona_cf_0.5": 0,
                                                   "persona_cf_only": 0},
        "documented-live": {"cf_scored": 6, "cf_0.5": 3}}}
    return cal, es, sm


class Survey1(unittest.TestCase):
    def test_mechanical(self):
        r = gs.grade_survey1(*s1_inputs())["predictions"]
        self.assertEqual(r["P1"]["verdict"], "true")           # 0.92 >= 0.90
        self.assertEqual(r["P1b"]["verdict"], "true")          # 0.85 >= 0.80
        self.assertEqual(r["P2"]["verdict"], "false")          # Test B 0.20 > 0.15
        self.assertEqual(r["P2"]["test_a"]["verdict"], "true")
        self.assertEqual(r["P3"]["n"], 40)                     # staff-page avatars only
        self.assertEqual(r["P3"]["k"], 4)
        self.assertEqual(r["P3"]["verdict"], "true")           # 0.10 < 0.50
        self.assertEqual(r["P4"]["n"], 4)                      # summary fallback: status ok
        self.assertEqual(r["P4"]["verdict"], "true")           # 2/4 >= 0.25
        self.assertEqual(r["P5"]["test_a_unique"]["verdict"], "true")   # 3/200 < 0.02
        self.assertEqual(r["P5"]["verdict"], "pending_review")
        self.assertEqual(r["P6"]["verdict"], "true")
        self.assertEqual(r["P6"]["persona_single_face"], 4)
        self.assertEqual(r["P7"]["verdict"], "false")          # 2/4 = 0.50, not < 0.50
        self.assertEqual(r["P8"]["verdict"], "false")          # SI: 2 found, 0 flagged

    def test_p4_from_sites_and_reviewed_p5(self):
        cal, neg, sm = s1_inputs()
        sites = [{"group": "population", "status": "ok", "single_face": 1},
                 {"group": "population", "status": "deadline", "articles": 2, "single_face": 0},
                 {"group": "population", "status": "deadline"},          # never reached
                 {"group": "population", "status": "home_403"},
                 {"group": CTRL, "status": "ok", "single_face": 5}]
        records = [{"sha256": "a" * 64, "group": "population", "host": "x.example"},
                   {"sha256": "b" * 64, "group": "population", "host": "y.example"}]
        review = {"a" * 64: True, "b" * 64: False}
        r = gs.grade_survey1(cal, neg, sm, sites, records, review)["predictions"]
        self.assertEqual(r["P4"]["n"], 2)
        self.assertEqual(r["P4"]["k"], 1)
        self.assertEqual(r["P4"]["basis"], "sites.jsonl")
        b = r["P5"]["reviewed_sites"]
        self.assertEqual((b["k"], b["n"]), (1, 2))             # 1 GAN host of 2 with faces
        self.assertEqual(b["verdict"], "false")                # 0.5 >= 0.01
        self.assertEqual(r["P5"]["verdict"], "false")

    def test_empty(self):
        r = gs.grade_survey1({}, {}, {})["predictions"]
        for key in ("P1", "P1b", "P2", "P3", "P4", "P6", "P7", "P8"):
            self.assertEqual(r[key]["verdict"], "no_data", key)
        self.assertEqual(r["P5"]["verdict"], "pending_review")


class Survey2(unittest.TestCase):
    def test_mechanical(self):
        res = gs.grade_survey2(*s2_inputs())
        r = res["predictions"]
        self.assertTrue(res["cf_valid"])
        self.assertEqual(r["E1"]["verdict"], "true")
        self.assertEqual(r["E2"]["verdict"], "false")          # StyleGAN 0.79 < 0.80
        self.assertEqual(r["E3"]["verdict"], "partial")        # 2 true, IMDB-WIKI missing
        self.assertEqual(r["E3"]["imdbwiki"]["error"], "HTTPError: 404")
        self.assertEqual(r["E4"]["verdict"], "true")           # 4/50 = 0.08 <= 0.10
        self.assertEqual(r["E5"]["verdict"], "true")           # 0.015 < 0.02, CI has 0
        self.assertEqual(r["E5"]["outcome"], "null")
        self.assertEqual(r["E5"]["rare_formats_in_A"], ["webp"])
        self.assertEqual(r["E6"]["basis"], "summary.json, all control images")
        self.assertEqual(r["E6"]["verdict"], "false")          # 2/10 < 0.70
        self.assertEqual(r["E7"]["persona"]["n"], 4)
        self.assertEqual(r["E7"]["persona"]["k"], 2)
        self.assertEqual(r["E7"]["verdict"], "true")           # 2/4 >= 0.50 and 1 CF-only
        self.assertEqual(r["E8"]["verdict"], "no_data")        # no records
        self.assertEqual(r["E9"]["verdict"], "true")           # 3/6 >= 0.50

    def test_rise_explained_by_rare_format_is_unresolved(self):
        cal, es, sm = s2_inputs()
        es["thresholds"]["t95_A"].update({"diff_Bnew_minus_A": 0.06,
                                          "boot95_outlets": [0.01, 0.11]})
        r = gs.grade_survey2(cal, es, sm)["predictions"]["E5"]
        self.assertEqual(r["verdict"], "false")
        # without webp: B_new 2/30 - A 3/48 = 0.0667 - 0.0625 > 0, so the rise stands
        self.assertEqual(r["outcome"], "rise")
        es["by_format"]["jpeg"]["B_new"]["cf_0.5"] = 1         # 1/30 - 3/48 < 0
        r = gs.grade_survey2(cal, es, sm)["predictions"]["E5"]
        self.assertEqual(r["outcome"], "unresolved")

    def test_invalid_when_repro_fails(self):
        cal, es, sm = s2_inputs()
        cal["repro"] = {"ok": False, "max_abs_diff": 0.3}
        r = gs.grade_survey2(cal, es, sm)["predictions"]
        self.assertEqual(r["E1"]["verdict"], "false")
        for key in ("E2", "E3", "E4", "E5", "E6", "E7", "E9"):
            self.assertEqual(r[key]["verdict"], "invalid", key)
        self.assertEqual(r["E5"]["outcome"], "invalid")

    def test_records_for_e6_and_e8(self):
        cal, es, sm = s2_inputs()
        g = {"x": 1}
        records = [
            {"group": CTRL, "img_kind": "seed_img", "sha256": "1", "geom": g, "cf": 0.9},
            {"group": CTRL, "img_kind": "seed_img", "sha256": "1", "geom": g, "cf": 0.9},
            {"group": CTRL, "img_kind": "seed_img", "sha256": "2", "geom": g, "cf": 0.8},
            {"group": CTRL, "img_kind": "seed_img", "sha256": "3", "geom": g, "cf": 0.1},
            {"group": CTRL, "img_kind": "article_img", "sha256": "4", "geom": g, "cf": 0.1},
            {"group": CTRL, "img_kind": "seed_img", "sha256": "5", "cf": None},
            {"group": gs.ELLE, "persona_page": True, "sha256": "6", "geom": g, "cf": 0.7,
             "aligned": True, "author_page": "elle.be/nl/author/sophie-vermeulen/"},
            {"group": gs.ELLE, "persona_page": True, "sha256": "7", "geom": g, "cf": 0.7,
             "aligned": False, "centred_level": False,
             "author_page": "elle.be/nl/author/marta-peeters/"},
        ]
        r = gs.grade_survey2(cal, es, sm, records)["predictions"]
        self.assertEqual((r["E6"]["k"], r["E6"]["n"]), (2, 3))  # unique staff avatars
        self.assertEqual(r["E6"]["verdict"], "false")           # 0.667 < 0.70
        self.assertEqual((r["E8"]["k"], r["E8"]["n"]), (1, 1))  # Marta is not E8
        self.assertEqual(r["E8"]["verdict"], "true")

    def test_no_t95(self):
        cal, es, sm = s2_inputs()
        del es["thresholds"]["t95_A"]
        r = gs.grade_survey2(cal, es, sm)["predictions"]
        self.assertEqual(r["E5"]["verdict"], "no_data")
        self.assertIsNone(r["E5"]["outcome"])


class Cli(unittest.TestCase):
    def test_main_survey2(self):
        cal, es, sm = s2_inputs()
        with tempfile.TemporaryDirectory() as d:
            for name, obj in (("calibration_cf.json", cal), ("eras_summary.json", es),
                              ("summary.json", sm)):
                with open(os.path.join(d, name), "w", encoding="utf-8") as fh:
                    json.dump(obj, fh)
            out = os.path.join(d, "g.json")
            res = gs.main(["survey2", d, "--out", out])
            self.assertEqual(res["predictions"]["E1"]["verdict"], "true")
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["predictions"]["E9"]["verdict"], "true")
            self.assertIn("| E5 | true |", gs.to_markdown(res))


if __name__ == "__main__":
    unittest.main()
