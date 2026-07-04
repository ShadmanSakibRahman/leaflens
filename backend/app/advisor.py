"""Advisory layer: retrieve grounded treatment facts and present them bilingually.

Design choice that matters: the *facts* (symptoms, organic/chemical treatment,
prevention) come verbatim from a curated knowledge base — the LLM never invents
them. Groq is used only to (a) translate those facts into Bangla and (b) write a
short friendly summary. If Groq is unavailable, we return the English KB facts as-is
so the feature degrades gracefully instead of failing. This is the anti-hallucination
guarantee.
"""

import os
import json

KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

_kb = None


def _load_kb():
    global _kb
    if _kb is None:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            _kb = json.load(f)
    return _kb


def retrieve(raw_label):
    """Return the curated KB entry for a class label, or None if absent."""
    return _load_kb().get(raw_label)


def _healthy_advice(crop):
    return {
        "grounded": True,
        "source": "kb",
        "disease": "Healthy",
        "crop": crop,
        "summary_en": f"Good news — the {crop.lower()} leaf looks healthy. No disease detected.",
        "summary_bn": f"সুখবর — {crop} পাতাটি সুস্থ মনে হচ্ছে। কোনো রোগ ধরা পড়েনি।",
        "symptoms_en": "No disease symptoms detected on the leaf.",
        "organic_en": ["Keep monitoring the field regularly."],
        "chemical_en": [],
        "prevention_en": [
            "Maintain balanced watering and spacing.",
            "Remove weeds and crop debris that harbour pests.",
            "Scout the field weekly for early symptoms.",
        ],
        "expert_note_en": "No action needed now. Re-check if you notice spots or wilting.",
        "organic_bn": ["নিয়মিত ক্ষেত পর্যবেক্ষণ করুন।"],
        "chemical_bn": [],
        "prevention_bn": [
            "সুষম সেচ ও গাছের মধ্যে দূরত্ব বজায় রাখুন।",
            "আগাছা ও ফসলের অবশিষ্ট অংশ পরিষ্কার রাখুন।",
            "প্রতি সপ্তাহে ক্ষেত পরীক্ষা করুন।",
        ],
        "expert_note_bn": "এখন কিছু করার দরকার নেই। দাগ বা নেতিয়ে পড়া দেখলে আবার পরীক্ষা করুন।",
    }


def _uncertain_advice():
    return {
        "grounded": True,
        "source": "kb",
        "disease": "Uncertain",
        "crop": "Unknown",
        "summary_en": "I'm not confident about this one. Please retake a clear, close photo of a single affected leaf in good light, or consult a local agriculture officer.",
        "summary_bn": "এই ছবিটি নিয়ে আমি নিশ্চিত নই। ভালো আলোতে একটি আক্রান্ত পাতার স্পষ্ট, কাছের ছবি আবার তুলুন, অথবা স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।",
        "symptoms_en": "",
        "organic_en": [],
        "chemical_en": [],
        "prevention_en": [],
        "expert_note_en": "Low model confidence — no diagnosis is given to avoid a wrong recommendation.",
        "organic_bn": [],
        "chemical_bn": [],
        "prevention_bn": [],
        "expert_note_bn": "মডেলের আত্মবিশ্বাস কম — ভুল পরামর্শ এড়াতে কোনো রোগ নির্ণয় দেওয়া হয়নি।",
    }


def _fallback_from_kb(entry, crop, disease):
    """Structure a response purely from KB English facts (no LLM)."""
    return {
        "grounded": True,
        "source": "kb-fallback",
        "disease": disease,
        "crop": crop,
        "summary_en": entry.get("summary_en", f"{disease} detected on {crop.lower()}."),
        "summary_bn": "",
        "symptoms_en": entry.get("symptoms", ""),
        "organic_en": entry.get("organic", []),
        "chemical_en": entry.get("chemical", []),
        "prevention_en": entry.get("prevention", []),
        "expert_note_en": entry.get("expert_note", ""),
        "organic_bn": [],
        "chemical_bn": [],
        "prevention_bn": [],
        "expert_note_bn": "",
    }


def _groq_translate(entry, crop, disease):
    """Ask Groq to translate the grounded facts to Bangla + write summaries.

    Returns a dict of Bangla fields + summaries, or raises on any failure so the
    caller can fall back to KB-only English.
    """
    from groq import Groq  # imported lazily so the app boots without the SDK/key

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")

    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    client = Groq(api_key=api_key)

    facts = {
        "crop": crop,
        "disease": disease,
        "symptoms": entry.get("symptoms", ""),
        "organic": entry.get("organic", []),
        "chemical": entry.get("chemical", []),
        "prevention": entry.get("prevention", []),
        "expert_note": entry.get("expert_note", ""),
    }

    system = (
        "You are an agriculture extension assistant for Bangladeshi farmers. "
        "You will be given verified facts about a crop disease as JSON. "
        "Use ONLY these facts. Do NOT add treatments, chemicals, dosages, or claims "
        "that are not present in the facts. Your job is to (1) translate the given "
        "lists and notes into simple, respectful Bangla a rural farmer can follow, and "
        "(2) write a two-sentence plain-language summary in both English and Bangla. "
        "Return ONLY a JSON object with these exact keys: summary_en, summary_bn, "
        "organic_bn (array), chemical_bn (array), prevention_bn (array), expert_note_bn. "
        "Each Bangla array item must correspond to the same-index English item you were given."
    )
    user = json.dumps(facts, ensure_ascii=False)

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
        timeout=25,
    )
    out = json.loads(resp.choices[0].message.content)
    return out


def advise(raw_label, crop, disease, is_healthy=False, uncertain=False):
    """Top-level advice builder used by the API."""
    if uncertain:
        return _uncertain_advice()
    if is_healthy:
        return _healthy_advice(crop)

    entry = retrieve(raw_label)
    if not entry:
        # Diagnosed a class we have no KB entry for — stay grounded, don't invent.
        return {
            "grounded": True,
            "source": "kb",
            "disease": disease,
            "crop": crop,
            "summary_en": f"{disease} detected on {crop.lower()}. I don't have a verified treatment sheet for this one yet — please consult a local agriculture officer.",
            "summary_bn": f"{crop} গাছে {disease} শনাক্ত হয়েছে। এই রোগের যাচাইকৃত চিকিৎসা তথ্য এখনো নেই — স্থানীয় কৃষি কর্মকর্তার পরামর্শ নিন।",
            "symptoms_en": "",
            "organic_en": [], "chemical_en": [], "prevention_en": [],
            "expert_note_en": "No verified treatment sheet available for this class.",
            "organic_bn": [], "chemical_bn": [], "prevention_bn": [], "expert_note_bn": "",
        }

    # English facts are always grounded from KB.
    base = _fallback_from_kb(entry, crop, disease)
    try:
        bn = _groq_translate(entry, crop, disease)
        base.update(
            {
                "source": "groq",
                "summary_en": bn.get("summary_en") or base["summary_en"],
                "summary_bn": bn.get("summary_bn", ""),
                "organic_bn": bn.get("organic_bn", []),
                "chemical_bn": bn.get("chemical_bn", []),
                "prevention_bn": bn.get("prevention_bn", []),
                "expert_note_bn": bn.get("expert_note_bn", ""),
            }
        )
    except Exception:  # noqa: BLE001 - any Groq/parse failure -> KB English fallback
        pass
    return base
