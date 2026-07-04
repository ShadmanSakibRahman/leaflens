"use client";

import { useEffect, useRef, useState } from "react";

// Empty string => same-origin (single-service deploy). Local dev sets this to
// http://localhost:8000 via .env.local.
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "";

// All UI strings in both languages so the toggle flips everything.
const T = {
  en: {
    tagline: "AI crop doctor for Bangladeshi farmers",
    uploadTitle: "Diagnose a crop leaf",
    uploadSub: "Take or upload a clear photo of a single affected leaf.",
    dropHint: "Tap to take a photo or choose from gallery",
    dropTypes: "JPG or PNG, up to 8 MB",
    analyze: "Diagnose",
    analyzing: "Diagnosing…",
    another: "Diagnose another leaf",
    waking: "Waking up the server… the first request can take up to a minute on free hosting.",
    healthy: "Healthy",
    disease: "Disease detected",
    uncertain: "Not sure",
    confidence: "Confidence",
    symptoms: "Symptoms",
    organic: "Organic / cultural control",
    chemical: "Chemical control",
    prevention: "Prevention",
    expert: "Note",
    history: "Recent diagnoses",
    noHistory: "No diagnoses yet.",
    errGeneric: "Something went wrong. Please try again.",
    errImage: "Please choose an image file first.",
    poweredKb: "Advice grounded in a curated treatment knowledge base.",
    poweredGroq: "Bangla translation by Groq LLM, grounded in the knowledge base.",
    poweredFallback: "Showing verified English advice (Bangla service unavailable).",
  },
  bn: {
    tagline: "বাংলাদেশের কৃষকদের জন্য এআই ফসল ডাক্তার",
    uploadTitle: "ফসলের পাতা পরীক্ষা করুন",
    uploadSub: "একটি আক্রান্ত পাতার স্পষ্ট ছবি তুলুন বা আপলোড করুন।",
    dropHint: "ছবি তুলতে বা গ্যালারি থেকে বেছে নিতে চাপ দিন",
    dropTypes: "JPG বা PNG, সর্বোচ্চ ৮ MB",
    analyze: "রোগ নির্ণয় করুন",
    analyzing: "পরীক্ষা করা হচ্ছে…",
    another: "আরেকটি পাতা পরীক্ষা করুন",
    waking: "সার্ভার চালু হচ্ছে… ফ্রি হোস্টিংয়ে প্রথম অনুরোধে এক মিনিট পর্যন্ত লাগতে পারে।",
    healthy: "সুস্থ",
    disease: "রোগ শনাক্ত হয়েছে",
    uncertain: "নিশ্চিত নয়",
    confidence: "আত্মবিশ্বাস",
    symptoms: "লক্ষণ",
    organic: "জৈব / পরিচর্যা ব্যবস্থা",
    chemical: "রাসায়নিক ব্যবস্থা",
    prevention: "প্রতিরোধ",
    expert: "দ্রষ্টব্য",
    history: "সাম্প্রতিক রোগ নির্ণয়",
    noHistory: "এখনো কোনো রোগ নির্ণয় হয়নি।",
    errGeneric: "কিছু একটা সমস্যা হয়েছে। আবার চেষ্টা করুন।",
    errImage: "প্রথমে একটি ছবি বেছে নিন।",
    poweredKb: "যাচাইকৃত চিকিৎসা তথ্যভান্ডারের উপর ভিত্তি করে পরামর্শ।",
    poweredGroq: "গ্রাউন্ডেড তথ্যের উপর ভিত্তি করে Groq LLM দিয়ে বাংলা অনুবাদ।",
    poweredFallback: "যাচাইকৃত ইংরেজি পরামর্শ দেখানো হচ্ছে (বাংলা সেবা এখন বন্ধ)।",
  },
};

export default function Home() {
  const [lang, setLang] = useState("bn");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [waking, setWaking] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const inputRef = useRef(null);
  const t = T[lang];

  // Ping backend on load: wakes a sleeping free-tier server + loads history.
  useEffect(() => {
    let cancelled = false;
    const ping = async () => {
      setWaking(true);
      try {
        await fetch(`${API_BASE}/health`, { cache: "no-store" });
        if (!cancelled) loadHistory();
      } catch (_) {
        /* backend not up yet; the diagnose call will retry */
      } finally {
        if (!cancelled) setWaking(false);
      }
    };
    ping();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadHistory = async () => {
    try {
      const r = await fetch(`${API_BASE}/history?limit=8`, { cache: "no-store" });
      if (r.ok) {
        const d = await r.json();
        setHistory(d.scans || []);
      }
    } catch (_) {}
  };

  const onPick = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  };

  const diagnose = async () => {
    if (!file) {
      setError(t.errImage);
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch(`${API_BASE}/diagnose`, { method: "POST", body: fd });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || t.errGeneric);
      }
      const data = await r.json();
      setResult(data);
      loadHistory();
    } catch (err) {
      setError(err.message || t.errGeneric);
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
    if (inputRef.current) inputRef.current.value = "";
  };

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <span className="leaf">🌿</span>
            <div>
              <h1>LeafLens</h1>
              <p>{t.tagline}</p>
            </div>
          </div>
          <div className="lang-toggle">
            <button
              className={lang === "bn" ? "active" : ""}
              onClick={() => setLang("bn")}
            >
              বাংলা
            </button>
            <button
              className={lang === "en" ? "active" : ""}
              onClick={() => setLang("en")}
            >
              EN
            </button>
          </div>
        </div>
      </header>

      <main className="wrap">
        {waking && <div className="alert info">{t.waking}</div>}

        <section className="card">
          <h2>{t.uploadTitle}</h2>
          <p className="subtle">{t.uploadSub}</p>

          <div className="dropzone" onClick={() => inputRef.current?.click()}>
            <div className="icon">📷</div>
            <p>
              {t.dropHint}
              <span>{t.dropTypes}</span>
            </p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            capture="environment"
            onChange={onPick}
            style={{ display: "none" }}
          />

          {preview && (
            <div className="preview">
              <img src={preview} alt="selected leaf" />
            </div>
          )}

          {!result && (
            <button className="btn" onClick={diagnose} disabled={loading || !file}>
              {loading ? <span className="spinner" /> : null}
              {loading ? t.analyzing : t.analyze}
            </button>
          )}
          {result && (
            <button className="btn secondary" onClick={reset}>
              {t.another}
            </button>
          )}

          {error && <div className="alert err">{error}</div>}
        </section>

        {result && <ResultCard data={result} t={t} lang={lang} />}

        <section className="card">
          <h2>{t.history}</h2>
          {history.length === 0 ? (
            <p className="subtle" style={{ marginTop: 8 }}>
              {t.noHistory}
            </p>
          ) : (
            <div>
              {history.map((h) => (
                <div className="history-item" key={h.id}>
                  <div>
                    <div className="h-disease">
                      {h.crop} — {h.disease}
                    </div>
                    <div className="h-meta">
                      {new Date(h.created_at).toLocaleString()}
                    </div>
                  </div>
                  <div className="h-meta">{Math.round(h.confidence * 100)}%</div>
                </div>
              ))}
            </div>
          )}
        </section>

        <footer>LeafLens · Md. Shadman Sakib Rahman · 2026</footer>
      </main>
    </>
  );
}

function ResultCard({ data, t, lang }) {
  const p = data.prediction;
  const a = data.advice;
  const isBn = lang === "bn";

  let pillClass = "disease";
  let pillText = t.disease;
  if (p.uncertain) {
    pillClass = "uncertain";
    pillText = t.uncertain;
  } else if (p.is_healthy) {
    pillClass = "healthy";
    pillText = t.healthy;
  }

  const summary = isBn ? a.summary_bn || a.summary_en : a.summary_en;
  const symptoms = isBn ? a.symptoms_bn || a.symptoms_en : a.symptoms_en;
  const organic = isBn && a.organic_bn?.length ? a.organic_bn : a.organic_en;
  const chemical = isBn && a.chemical_bn?.length ? a.chemical_bn : a.chemical_en;
  const prevention =
    isBn && a.prevention_bn?.length ? a.prevention_bn : a.prevention_en;
  const expert = isBn ? a.expert_note_bn || a.expert_note_en : a.expert_note_en;

  let sourceLabel = t.poweredKb;
  if (a.source === "groq") sourceLabel = t.poweredGroq;
  else if (a.source === "kb-fallback") sourceLabel = t.poweredFallback;

  const conf = Math.round(p.confidence * 100);

  return (
    <section className="card">
      <div className="diagnosis">
        <span className={`pill ${pillClass}`}>{pillText}</span>
        <div>
          <div className="crop-name">{p.crop}</div>
          <div className="disease-name">
            {p.uncertain ? "—" : isBn ? a.summary_bn ? p.disease : p.disease : p.disease}
          </div>
        </div>
      </div>

      {!p.uncertain && (
        <>
          <div className="confbar">
            <span style={{ width: `${conf}%` }} />
          </div>
          <div className="conflabel">
            {t.confidence}: {conf}%
          </div>
        </>
      )}

      <div className="advice">
        {summary && <p className="summary" style={{ marginTop: 14 }}>{summary}</p>}

        {symptoms && !p.uncertain && (
          <>
            <h3>🔍 {t.symptoms}</h3>
            <p style={{ fontSize: 14, margin: 0 }}>{symptoms}</p>
          </>
        )}

        {organic?.length > 0 && (
          <>
            <h3>🌱 {t.organic}</h3>
            <ul>
              {organic.map((x, i) => (
                <li key={i}>{x}</li>
              ))}
            </ul>
          </>
        )}

        {chemical?.length > 0 && (
          <>
            <h3>🧪 {t.chemical}</h3>
            <ul>
              {chemical.map((x, i) => (
                <li key={i}>{x}</li>
              ))}
            </ul>
          </>
        )}

        {prevention?.length > 0 && (
          <>
            <h3>🛡️ {t.prevention}</h3>
            <ul>
              {prevention.map((x, i) => (
                <li key={i}>{x}</li>
              ))}
            </ul>
          </>
        )}

        {expert && (
          <p className="note">
            <strong>{t.expert}:</strong> {expert}
          </p>
        )}

        <p className="source-tag">{sourceLabel}</p>
      </div>
    </section>
  );
}
