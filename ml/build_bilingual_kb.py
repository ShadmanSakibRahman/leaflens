"""Pre-bake the full bilingual advice for every class, so the static (in-browser)
app needs no server or API key at runtime. Reuses the existing advisor (grounded
KB facts + Groq Bangla translation) once per class at build time.

Output: frontend/public/model/advice.json  (keyed by class label, plus __uncertain__)
"""

import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.join(HERE, "..", "backend")
sys.path.insert(0, BACKEND)

from app import advisor, model as vision  # noqa: E402

OUT = os.path.join(HERE, "..", "frontend", "public", "model", "advice.json")
labels = json.load(open(os.path.join(BACKEND, "model", "labels.json"), encoding="utf-8"))

out = {}
for label in labels:
    crop, disease, is_healthy = vision._prettify(label)
    print(f"advising {label} ...")
    out[label] = advisor.advise(label, crop, disease, is_healthy=is_healthy, uncertain=False)

# uncertain template (low-confidence path)
out["__uncertain__"] = advisor.advise("", "Unknown", "Unknown", uncertain=True)

json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"\nWrote {len(out)} advice entries to {OUT}")
# quick sanity: how many got Groq Bangla vs fallback
groq = sum(1 for v in out.values() if v.get("source") == "groq")
print(f"groq-translated: {groq}, other: {len(out)-groq}")
