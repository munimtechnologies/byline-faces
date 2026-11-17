# Copyright 2026 Munim, Inc. (MV2). Licensed under the Apache License, Version 2.0 (Apache-2.0).
# Made with AI agents working for MV2 of Munim, Inc.
# Job helper: fetch the unchanged survey 1-2 modules (byline-faces v0.3.0) next to atlas.py.
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/munimtechnologies/byline-faces/v0.3.0/"
CHECKS = {"byline_faces.py": '__version__ = "0.2.0"', "job.py": "def stage_documented",
          "eras.py": '__version__ = "0.3.0"', "commfor.py": "def reproduce"}

for name, must in CHECKS.items():
    with urllib.request.urlopen(BASE + name, timeout=60) as r:
        text = r.read().decode("utf-8")
    if must not in text:
        sys.exit("%s from %s does not contain %r" % (name, BASE, must))
    with open(name, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print("fetched", name, len(text), "chars")
