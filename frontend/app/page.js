"use client";

import { useEffect, useRef, useState } from "react";

// Base path for assets (set to "/leaflens" for GitHub Pages, "" for local).
const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
const CONF_THRESHOLD = 0.55;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const HISTORY_KEY = "leaflens_history";

const T = {
  en: {
    tagline: "AI crop doctor for Bangladeshi farmers",
    uploadTitle: "Diagnose a crop leaf",
    uploadSub: "Take or upload a clear photo of a single affected leaf.",
    dropHint: "Tap to take a photo or choose from gallery",
    dropTypes: "JPG or PNG",
    analyze: "Diagnose",
    analyzing: "Diagnosing…",
    another: "Diagnose another leaf",
    loadingModel: "Loading the AI model (first time only, a few seconds)…",
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
    errImage: "Please choose an image file first.",
    runsInBrowser: "Runs entirely in your browser — the AI model and advice are on-device.",
  },
  bn: {
    tagline: "বাংলাদেশের কৃষকদের জন্য এআই ফসল ডাক্তার",
    uploadTitle: "ফসলের পাতা পরীক্ষা করুন",
    uploadSub: "একটি আক্রান্ত পাতার স্পষ্ট ছবি তুলুন বা আপলোড করুন।",
    dropHint: "ছবি তুলতে বা গ্যালারি থেকে বেছে নিতে চাপ দিন",
    dropTypes: "JPG বা PNG",
    analyze: "রোগ নির্ণয় করুন",
    analyzing: "পরীক্ষা করা হচ্ছে…",
    another: "আরেকটি পাতা পরীক্ষা করুন",
    loadingModel: "এআই মডেল লোড হচ্ছে (শুধু প্রথমবার, কয়েক সেকেন্ড)…",
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
    errImage: "প্রথমে একটি ছবি বেছে নিন।",
    runsInBrowser: "সম্পূর্ণ আপনার ব্রাউজারেই চলে — এআই মডেল ও পরামর্শ ডিভাইসেই।",
  },
};

function prettify(label) {
  let crop = label, condition = label;
  if (label.includes("___")) {
    [crop, condition] = label.split("___");
  }
  crop = crop.replace(/_/g, " ").trim();
  crop = crop.charAt(0).toUpperCase() + crop.slice(1);
  const isHealthy = condition.trim().toLowerCase() === "healthy";
  let disease = condition.replace(/_/g, " ").trim();
  disease = isHealthy ? "Healthy" : disease.charAt(0).toUpperCase() + disease.slice(1);
  return { crop, disease, isHealthy };
}

function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

// torchvision Resize(256) + CenterCrop(224) + normalize, done on a canvas.
function preprocess(img) {
  const size = 224;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  const w = img.naturalWidth, h = img.naturalHeight;
  const shorter = Math.min(w, h);
  const cropOrig = shorter * (224 / 256);
  const sx = (w - cropOrig) / 2, sy = (h - cropOrig) / 2;
  ctx.drawImage(img, sx, sy, cropOrig, cropOrig, 0, 0, size, size);
  const { data } = ctx.getImageData(0, 0, size, size);
  const out = new Float32Array(3 * size * size);
  const plane = size * size;
  for (let i = 0; i < plane; i++) {
    for (let c = 0; c < 3; c++) {
      const v = data[i * 4 + c] / 255;
      out[c * plane + i] = (v - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
    }
  }
  return out;
}

export default function Home() {
  const [lang, setLang] = useState("bn");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [history, setHistory] = useState([]);
  const inputRef = useRef(null);
  const sessionRef = useRef(null);
  const labelsRef = useRef(null);
  const adviceRef = useRef(null);
  const ortRef = useRef(null);
  const t = T[lang];

  useEffect(() => {
    setHistory(loadHistory());
    let cancelled = false;
    (async () => {
      try {
        const ort = await import("onnxruntime-web");
        ort.env.wasm.numThreads = 1; // GitHub Pages isn't cross-origin isolated
        ort.env.wasm.wasmPaths = `${BASE}/ort/`;
        const [session, labels, advice] = await Promise.all([
          ort.InferenceSession.create(`${BASE}/model/leaflens.onnx`, {
            executionProviders: ["wasm"],
          }),
          fetch(`${BASE}/model/labels.json`).then((r) => r.json()),
          fetch(`${BASE}/model/advice.json`).then((r) => r.json()),
        ]);
        if (cancelled) return;
        ortRef.current = ort;
        sessionRef.current = session;
        labelsRef.current = labels;
        adviceRef.current = advice;
        setModelReady(true);
      } catch (e) {
        if (!cancelled) setError("Could not load the AI model. Please refresh.");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

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
    if (!modelReady) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const ort = ortRef.current;
      const img = await loadImage(preview);
      const input = preprocess(img);
      const tensor = new ort.Tensor("float32", input, [1, 3, 224, 224]);
      const output = await sessionRef.current.run({ input: tensor });
      const logits = Array.from(output.logits.data);
      const probs = softmax(logits);
      let best = 0;
      for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
      const label = labelsRef.current[best];
      const conf = probs[best];
      const { crop, disease, isHealthy } = prettify(label);
      const uncertain = conf < CONF_THRESHOLD;
      const advice = uncertain
        ? adviceRef.current["__uncertain__"]
        : adviceRef.current[label];
      const prediction = {
        crop,
        disease,
        is_healthy: isHealthy,
        confidence: Number(conf.toFixed(4)),
        uncertain,
        raw_label: label,
      };
      const res = { prediction, advice };
      setResult(res);
      const entry = {
        id: Date.now(),
        crop,
        disease: uncertain ? "Uncertain" : disease,
        confidence: prediction.confidence,
        created_at: new Date().toISOString(),
      };
      const next = [entry, ...history].slice(0, 12);
      setHistory(next);
      saveHistory(next);
    } catch (e) {
      setError("Diagnosis failed. Please try another photo.");
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
            <button className={lang === "bn" ? "active" : ""} onClick={() => setLang("bn")}>
              বাংলা
            </button>
            <button className={lang === "en" ? "active" : ""} onClick={() => setLang("en")}>
              EN
            </button>
          </div>
        </div>
      </header>

      <main className="wrap">
        {!modelReady && !error && <div className="alert info">{t.loadingModel}</div>}

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
            <button className="btn" onClick={diagnose} disabled={loading || !file || !modelReady}>
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
                    <div className="h-meta">{new Date(h.created_at).toLocaleString()}</div>
                  </div>
                  <div className="h-meta">{Math.round(h.confidence * 100)}%</div>
                </div>
              ))}
            </div>
          )}
        </section>

        <footer>
          {t.runsInBrowser}
          <br />
          LeafLens · Md. Shadman Sakib Rahman · 2026
        </footer>
      </main>
    </>
  );
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function loadHistory() {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveHistory(h) {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
  } catch {}
}

function ResultCard({ data, t, lang }) {
  const p = data.prediction;
  const a = data.advice || {};
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
  const prevention = isBn && a.prevention_bn?.length ? a.prevention_bn : a.prevention_en;
  const expert = isBn ? a.expert_note_bn || a.expert_note_en : a.expert_note_en;
  const conf = Math.round(p.confidence * 100);

  return (
    <section className="card">
      <div className="diagnosis">
        <span className={`pill ${pillClass}`}>{pillText}</span>
        <div>
          <div className="crop-name">{p.crop}</div>
          <div className="disease-name">{p.uncertain ? "—" : p.disease}</div>
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
        {summary && (
          <p className="summary" style={{ marginTop: 14 }}>
            {summary}
          </p>
        )}

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
      </div>
    </section>
  );
}
