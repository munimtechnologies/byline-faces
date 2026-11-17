# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
"""Tests for atlas.py (survey 3). Standard library only."""
import re
import unittest

import atlas


def rec(case_id, sha, page, cf=None, aligned=False, centred=False, persona_page=False,
        face=True):
    r = {"group": "documented:" + case_id, "sha256": sha, "author_page": page,
         "capture": "20200601000000", "persona_page": persona_page,
         "aligned": aligned, "centred_level": centred, "cf": cf}
    if face:
        r["geom"] = {}
    return r


class CaseTable(unittest.TestCase):
    def test_every_case_is_sourced_and_valid(self):
        ids = set()
        for c in atlas.CASES:
            self.assertNotIn(c["id"], ids)
            ids.add(c["id"])
            self.assertRegex(c["id"], r"^[a-z0-9-]+-20\d\d$")
            self.assertTrue(c["id"].endswith(str(c["year"])))
            self.assertIn(c["origin"], atlas.ORIGIN_CLASSES)
            self.assertIsNotNone(atlas.period_of(c["year"]))
            self.assertGreater(len(c["source"]), 60)
            self.assertRegex(c["source"], r"(19|20)\d\d")
            self.assertTrue(c["prefixes"])
            for p in c["prefixes"]:
                self.assertNotIn("://", p)
                self.assertFalse(p.startswith("www."), p)
            if c.get("persona_path_re"):
                re.compile(c["persona_path_re"])
            # Every case must be able to mark persona pages.
            self.assertTrue(c.get("slugs") or c.get("whole_site_fake")
                            or c.get("persona_path_re"))

    def test_every_period_has_cases(self):
        periods = {atlas.period_of(c["year"]) for c in atlas.CASES}
        self.assertEqual(periods, {"2019-2021", "2022-2024", "2025-2026"})

    def test_case_count_and_stolen_controls(self):
        # Pre-registration 3 with amendment 1: 17 cases, two labelled "stolen".
        self.assertEqual(len(atlas.CASES), 17)
        stolen = sorted(c["id"] for c in atlas.CASES if c["origin"] == "stolen")
        self.assertEqual(stolen, ["dc-weekly-2023", "uae-network-badani-2020"])

    def test_group_round_trip(self):
        for c in atlas.CASES:
            self.assertIs(atlas.case_of_group(atlas.group_of(c)), c)
        self.assertIsNone(atlas.case_of_group("population"))
        self.assertIsNone(atlas.case_of_group("documented:nope-2020"))
        self.assertIsNone(atlas.case_of_group(None))

    def test_documented_specs(self):
        specs = atlas.documented_specs()
        self.assertEqual(len(specs), len(atlas.CASES))
        for s in specs:
            self.assertEqual(set(s), {"group", "source", "prefixes", "slugs"})
            self.assertTrue(s["group"].startswith("documented:"))

    def test_periods(self):
        self.assertEqual(atlas.period_of(2019), "2019-2021")
        self.assertEqual(atlas.period_of(2021), "2019-2021")
        self.assertEqual(atlas.period_of(2022), "2022-2024")
        self.assertEqual(atlas.period_of(2026), "2025-2026")
        self.assertIsNone(atlas.period_of(2018))


class PersonaRules(unittest.TestCase):
    def test_slug_in_path(self):
        c = atlas.CASE_BY_ID["si-advon-2023"]
        self.assertTrue(atlas.is_persona(
            {"author_page": "https://www.si.com/review/author/drew-ortiz/"}, c))
        self.assertTrue(atlas.is_persona(
            {"author_page": "https://www.thestreet.com/author/denise-mcnamara"}, c))
        self.assertFalse(atlas.is_persona(
            {"author_page": "https://www.si.com/review/author/real-person/"}, c))
        self.assertTrue(atlas.is_persona({"author_page": "x", "persona_page": True}, c))

    def test_path_rule_and_whole_site(self):
        pd = atlas.CASE_BY_ID["peace-data-2020"]
        self.assertTrue(atlas.is_persona({"author_page": "peacedata.net/about-us/"}, pd))
        self.assertFalse(atlas.is_persona({"author_page": "peacedata.net/author/jane/"}, pd))
        sfs = atlas.CASE_BY_ID["south-florida-standard-2026"]
        self.assertTrue(atlas.is_persona({"author_page": "southfloridastandard.com/x"}, sfs))

    def test_amendment_cases(self):
        dc = atlas.CASE_BY_ID["dc-weekly-2023"]
        self.assertTrue(atlas.is_persona({"author_page": "dcweekly.org/author/someone/"}, dc))
        self.assertTrue(atlas.is_persona(
            {"author_page": "https://dcweekly.org/jessica-devlin/"}, dc))
        self.assertFalse(atlas.is_persona(
            {"author_page": "dcweekly.org/2023/11/03/some-story/"}, dc))
        pm = atlas.CASE_BY_ID["paradise-media-pa-2026"]
        self.assertTrue(atlas.is_persona(
            {"author_page": "https://northeasttimes.com/authors/maya-brooks/"}, pm))
        self.assertFalse(atlas.is_persona(
            {"author_page": "https://northeasttimes.com/authors/a-real-reporter/"}, pm))
        dr = atlas.CASE_BY_ID["daniel-robson-2026"]
        self.assertTrue(atlas.is_persona(
            {"author_page": "policyoptions.irpp.org/authors/daniel-robson/"}, dr))

    def test_eye_flag(self):
        self.assertTrue(atlas.eye_flag({"aligned": True}))
        self.assertTrue(atlas.eye_flag({"centred_level": True}))
        self.assertFalse(atlas.eye_flag({}))


class Analysis(unittest.TestCase):
    def records(self):
        return [
            # 2020 GAN case: two persona faces, one shared by two outlets.
            rec("oliver-taylor-2020", "a", "blogs.timesofisrael.com/author/oliver-taylor/",
                cf=0.9, aligned=True, persona_page=True),
            rec("oliver-taylor-2020", "a", "algemeiner.com/author/oliver-taylor/",
                cf=0.9, aligned=True, persona_page=True),
            rec("peace-data-2020", "b", "peacedata.net/about/", cf=0.2, centred=True),
            rec("peace-data-2020", "c", "peacedata.net/author/real-writer/", cf=0.1),
            # 2026: diffusion-like face (CF only) and a stolen-looking one (neither).
            rec("south-florida-standard-2026", "d", "southfloridastandard.com/author/x/",
                cf=0.8),
            rec("bogus-bylines-2026", "e", "fairobserver.com/author/alice-johnson/",
                cf=0.1, persona_page=True),
            rec("bogus-bylines-2026", "f", "fairobserver.com/author/alice-johnson/",
                cf=None, persona_page=True),
            # Ignored: no face, unknown group.
            rec("bogus-bylines-2026", "g", "fairobserver.com/author/alice-johnson/",
                cf=0.99, persona_page=True, face=False),
            {"group": "population", "sha256": "h", "geom": {}, "cf": 0.9},
        ]

    def test_faces_by_case(self):
        fc = atlas.faces_by_case(self.records())
        self.assertEqual(len(fc["oliver-taylor-2020"]["persona"]), 1)   # deduplicated
        self.assertEqual([r["sha256"] for r in fc["peace-data-2020"]["persona"]], ["b"])
        self.assertEqual([r["sha256"] for r in fc["peace-data-2020"]["other"]], ["c"])
        self.assertEqual(len(fc["bogus-bylines-2026"]["persona"]), 2)
        self.assertNotIn("population", fc)

    def test_dedup_upgrades_to_persona(self):
        rs = [rec("peace-data-2020", "z", "peacedata.net/author/someone/", cf=0.3),
              rec("peace-data-2020", "z", "peacedata.net/team/", cf=0.3)]
        fc = atlas.faces_by_case(rs)
        self.assertEqual(len(fc["peace-data-2020"]["persona"]), 1)
        self.assertEqual(len(fc["peace-data-2020"]["other"]), 0)

    def test_analyse(self):
        out = atlas.analyse_atlas(self.records())
        early = out["periods"]["2019-2021"]
        self.assertEqual(early["n"], 2)
        self.assertEqual(early["eye"], 2)
        self.assertEqual(early["eye_share"], 1.0)
        self.assertEqual(early["cf_flagged"], 1)
        self.assertEqual(early["cases_with_persona_faces"], 2)
        self.assertEqual(early["case_mean_eye_share"], 1.0)
        late = out["periods"]["2025-2026"]
        self.assertEqual(late["n"], 3)
        self.assertEqual(late["eye"], 0)
        self.assertEqual(late["cf_scored"], 2)
        self.assertEqual(late["cf_flagged"], 1)
        self.assertEqual(late["cf_only"], 1)
        self.assertEqual(late["neither"], 2)
        self.assertAlmostEqual(late["cf_only_share"], 1 / 3)
        self.assertEqual(late["cases_with_persona_faces"], 2)
        self.assertAlmostEqual(late["case_mean_cf_share"], 0.5)   # (1/1 + 0/1) / 2
        mid = out["periods"]["2022-2024"]
        self.assertEqual(mid["n"], 0)
        self.assertIsNone(mid["eye_share"])
        self.assertIsNone(mid["case_mean_eye_share"])
        p = out["primary"]
        self.assertEqual(p["diff_late_minus_early"], -1.0)
        self.assertEqual(p["case_mean_diff"], -1.0)
        self.assertEqual(out["cases"]["oliver-taylor-2020"]["persona_cf_scores"], [0.9])
        self.assertEqual(out["cases"]["peace-data-2020"]["other_writers_same_pages"]["n"], 1)
        self.assertEqual(out["by_origin"]["stylegan"]["n"], 2)
        self.assertEqual(out["by_origin"]["mixed"]["n"], 2)
        self.assertEqual(len(out["cases"]), len(atlas.CASES))

    def test_counts_both_and_only(self):
        items = [{"aligned": True, "cf": 0.7}, {"aligned": True, "cf": 0.2},
                 {"cf": 0.6}, {"cf": None}]
        c = atlas._counts(items, 0.5)
        self.assertEqual((c["both"], c["eye_only"], c["cf_only"], c["neither"]), (1, 1, 1, 1))
        self.assertEqual(c["cf_scored"], 3)

    def test_wilson(self):
        self.assertIsNone(atlas.wilson(0, 0))
        lo, hi = atlas.wilson(5, 10)
        self.assertTrue(0.2 < lo < 0.5 < hi < 0.8)

    def test_public_faces(self):
        lines = atlas.public_faces(self.records())
        self.assertEqual(len(lines), 5)
        first = [x for x in lines if x["case"] == "oliver-taylor-2020"][0]
        self.assertTrue(first["archived_page"].startswith(
            "https://web.archive.org/web/20200601000000/"))
        self.assertTrue(first["eye_aligned"])
        self.assertEqual(first["cf"], 0.9)
        self.assertTrue(all(set(x) == {"case", "archived_page", "wayback_image", "sha256",
                                       "eye_aligned", "eye_centred_level", "cf"}
                            for x in lines))


if __name__ == "__main__":
    unittest.main()
