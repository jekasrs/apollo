/* global fetch, document, FileReader, Blob, URL, SVGNS */

const SPEAKER_COLORS = [
  "#3b82f6",
  "#f97316",
  "#22c55e",
  "#ef4444",
  "#a855f7",
  "#ec4899",
  "#14b8a6",
  "#eab308",
];

const EMOTIONS = {
  neutral: { fill: "#64748b", emoji: "😐", short: "нейтральная, спокойная речь без яркого тона" },
  surprise: { fill: "#f59e0b", emoji: "😮", short: "удивление, внезапная реакция, изумление, неожиданность" },
  fear: { fill: "#7c3aed", emoji: "😨", short: "страх, тревога, беспокойство, ожидание чего-то плохого" },
  sadness: { fill: "#2563eb", emoji: "😢", short: "печаль, грусть, сниженный тон, сожаление" },
  joy: { fill: "#22c55e", emoji: "😊", short: "радость, лёгкость, теплота, позитив в голосе" },
  disgust: { fill: "#a16207", emoji: "🤢", short: "отвращение, неприязнь, осуждение, отталкивание" },
  anger: { fill: "#dc2626", emoji: "😠", short: "злость, раздражение, упрёк, враждебный настрой" },
};

const EMOJI_FONT =
  'ui-sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif';

const RU = {
  neutral: "Нейтральная",
  surprise: "Удивление",
  fear: "Страх",
  sadness: "Печаль",
  joy: "Радость",
  disgust: "Отвращение",
  anger: "Злость",
};

const FORMAT = "eleos-dialogue-v1";
const LEGACY_FORMAT = "apollo-dialogue-v1";

/** Demo: neutral start → conflict → happy ending (3 speakers, 17 lines). */
const DEFAULT_UTTERANCES = [
  { speaker: 0, text: "Morning — I'm grabbing coffee. What does everyone want?" },
  { speaker: 1, text: "Black americano for me, no sugar. Thanks." },
  { speaker: 2, text: "Cappuccino if that's okay." },
  { speaker: 0, text: "Sounds good. Meet at the office at ten?" },
  { speaker: 1, text: "Yep. I already dropped the docs in the shared channel." },
  { speaker: 0, text: "Wait — the spreadsheet has last quarter's numbers again!" },
  { speaker: 1, text: "Wasn't me. I uploaded the latest version yesterday." },
  { speaker: 2, text: "I didn't revert anything — I only opened it read-only." },
  { speaker: 0, text: "Then who did?! I'm tired of cleaning up other people's messes!" },
  { speaker: 1, text: "Don't raise your voice at me — I'm not the one who broke this." },
  { speaker: 2, text: "Hey, let's calm down — we'll open version history and sort it in a minute." },
  { speaker: 0, text: "Sorry… deadline's on fire and I'm on edge. I didn't mean to snap." },
  { speaker: 1, text: "It's fine, I get it. I'll re-upload the file and post in the thread so it's visible." },
  { speaker: 2, text: "I'm here — we can double-check together in five, okay?" },
  { speaker: 0, text: "Thank you both — you really saved me. I'm breathing easier already." },
  { speaker: 1, text: "Happens to everyone. When we're done, pizza's on me — deal?" },
  { speaker: 2, text: "Love it! Let's crush this and then actually celebrate." },
];

const numSp = document.getElementById("numSp");
const selSpeaker = document.getElementById("selSpeaker");
const inputText = document.getElementById("inputText");
const messages = document.getElementById("messages");
const btnAdd = document.getElementById("btnAdd");
const btnNew = document.getElementById("btnNew");
const btnAnalyze = document.getElementById("btnAnalyze");
const btnExport = document.getElementById("btnExport");
const fileIn = document.getElementById("fileIn");
const jsonPreview = document.getElementById("jsonPreview");
const err = document.getElementById("err");
const resultBlock = document.getElementById("resultBlock");
const resHint = document.getElementById("resHint");
const diagSvg = document.getElementById("diag-svg");
const graphWrap = document.getElementById("graphWrap");
const legendTable = document.getElementById("legendTable");
const statusBar = document.getElementById("statusBar");
const speakerStats = document.getElementById("speakerStats");

/** @type {{ text: string, speaker: number }[]} */
let utterances = DEFAULT_UTTERANCES.map((u) => ({ text: u.text, speaker: u.speaker }));

function updateSpeakerOptions() {
  const n = Math.max(1, parseInt(numSp.value, 10) || 1);
  const cur = Math.min(parseInt(selSpeaker.value, 10) || 0, n - 1);
  selSpeaker.innerHTML = "";
  for (let i = 0; i < n; i++) {
    const o = document.createElement("option");
    o.value = String(i);
    o.textContent = `Участник ${i}`;
    if (i === cur) o.selected = true;
    selSpeaker.appendChild(o);
  }
}

function renderMessages() {
  messages.innerHTML = "";
  if (utterances.length === 0) {
    messages.textContent = "Пока пусто — добавьте реплики снизу.";
    return;
  }
  utterances.forEach((u, i) => {
    const d = document.createElement("div");
    d.className = "msg";
    const spCol = SPEAKER_COLORS[u.speaker % SPEAKER_COLORS.length];
    d.style.borderLeftColor = spCol;
    d.innerHTML = `<div class="msg-head" style="color:${spCol}">#${i} · говорит ${u.speaker}</div><div>${escapeHtml(
      u.text
    )}</div>`;
    messages.appendChild(d);
  });
  messages.scrollTop = messages.scrollHeight;
}

function escapeHtml(s) {
  return s
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function toDocument() {
  return {
    format: FORMAT,
    num_speakers: Math.max(1, parseInt(numSp.value, 10) || 1),
    utterances: utterances.map((u) => ({ text: u.text, speaker: u.speaker })),
  };
}

function fromDocument(doc) {
  const fmt = doc.format;
  const fmtOk = !fmt || fmt === FORMAT || fmt === LEGACY_FORMAT;
  if (!fmtOk) {
    throw new Error("Неверный format (ожидается eleos-dialogue-v1 или apollo-dialogue-v1)");
  }
  if (!Array.isArray(doc.utterances)) {
    throw new Error("Нет поля utterances");
  }
  const ns = Math.max(1, parseInt(doc.num_speakers, 10) || 1);
  numSp.value = String(ns);
  utterances = doc.utterances.map((t) => ({
    text: String(t.text || "").trim(),
    speaker: Math.max(0, Math.min(ns - 1, parseInt(t.speaker, 10) || 0)),
  }));
  for (const u of utterances) {
    if (!u.text) throw new Error("Пустой текст");
  }
  updateSpeakerOptions();
  renderMessages();
  syncJsonPreview();
}

function syncJsonPreview() {
  jsonPreview.textContent = JSON.stringify(toDocument(), null, 2);
}

function renderLegend() {
  const keys = Object.keys(EMOTIONS);
  let h = '<div class="legend-grid">';
  for (const k of keys) {
    const e = EMOTIONS[k];
    h += `<div class="legend-item" title="${escapeHtml(e.short)}">`;
    h += `<span class="legend-swatch swatch-emoji" style="background:${e.fill}">${e.emoji}</span>`;
    h += `<span class="legend-label"><span class="legend-name">${RU[k] || k}</span><code class="legend-code">${k}</code></span>`;
    h += "</div>";
  }
  h += "</div>";
  legendTable.innerHTML = h;
}

function renderSpeakerStats(stats) {
  if (!speakerStats) return;
  if (!stats || stats.length === 0) {
    speakerStats.innerHTML = "";
    return;
  }
  let h =
    '<h3 class="stats-heading">Итог по участникам</h3><p class="sub stats-note">Доли считаются по <strong>числу реплик</strong> каждого говорящего. «Хорошо» — позитив (в основном радость); «плохо» — негатив (злость, отвращение, страх, печаль); «спокойно» — нейтральная эмоция и удивление.</p>';
  for (const s of stats) {
    const spCol = SPEAKER_COLORS[s.speaker % SPEAKER_COLORS.length];
    const tp = s.tone_percent || { positive: 0, negative: 0, neutral: 0 };
    const ep = s.emotion_percent || {};
    h += `<div class="speaker-stat-card">`;
    h += `<div class="speaker-stat-head" style="color:${spCol}">Участник ${s.speaker}</div>`;
    h += `<p class="sub" style="margin:0 0 0.5rem">Реплик в диалоге: <strong>${s.utterances}</strong></p>`;
    h += `<div class="tone-bar" title="Позитив / Негатив / Спокойный тон" role="img">`;
    h += `<span class="tone-seg tone-pos" style="width:${tp.positive}%"></span>`;
    h += `<span class="tone-seg tone-neg" style="width:${tp.negative}%"></span>`;
    h += `<span class="tone-seg tone-neu" style="width:${tp.neutral}%"></span>`;
    h += `</div>`;
    h += `<ul class="tone-legend-row">`;
    h += `<li><span class="tone-dot tone-pos"></span> Хорошо / позитив <strong>${tp.positive}%</strong></li>`;
    h += `<li><span class="tone-dot tone-neg"></span> Плохо / негатив <strong>${tp.negative}%</strong></li>`;
    h += `<li><span class="tone-dot tone-neu"></span> Спокойно <strong>${tp.neutral}%</strong></li>`;
    h += `</ul>`;
    h += `<table class="stats-emotions"><thead><tr><th>Эмоция</th><th>% реплик</th></tr></thead><tbody>`;
    for (const k of Object.keys(EMOTIONS)) {
      const pct = ep[k] != null ? ep[k] : 0;
      h += `<tr><td>${EMOTIONS[k].emoji} ${RU[k] || k}</td><td>${pct}%</td></tr>`;
    }
    h += `</tbody></table></div>`;
  }
  speakerStats.innerHTML = h;
}

/** Вертикальное колёсико над графом → прокрутка graph-wrap по горизонтали (страница не едет). */
function ensureGraphWrapWheelScroll() {
  if (!graphWrap || graphWrap.dataset.eleosWheelBound === "1") return;
  graphWrap.dataset.eleosWheelBound = "1";
  graphWrap.addEventListener(
    "wheel",
    (e) => {
      const el = graphWrap;
      if (!el || el.scrollWidth <= el.clientWidth + 1) return;
      if (e.shiftKey) return;
      if (Math.abs(e.deltaX) >= Math.abs(e.deltaY)) return;
      e.preventDefault();
      el.scrollLeft += e.deltaY;
    },
    { passive: false }
  );
}

function drawGraph(apiUtterances) {
  const n = apiUtterances.length;
  const pad = 48;
  const nodeGap = 100;
  const w = n <= 1 ? 400 : pad * 2 + (n - 1) * nodeGap;
  const h = 180;
  const cy = 90;
  const r = 18;
  diagSvg.setAttribute("viewBox", `0 0 ${w} ${h}`);
  diagSvg.setAttribute("width", String(w));
  diagSvg.setAttribute("height", String(h));
  diagSvg.setAttribute("preserveAspectRatio", "xMinYMid meet");
  diagSvg.style.minWidth = `${w}px`;

  while (diagSvg.firstChild) diagSvg.removeChild(diagSvg.firstChild);

  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "arrowend");
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "8");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "4");
  marker.setAttribute("markerHeight", "4");
  marker.setAttribute("orient", "auto");
  const p = document.createElementNS("http://www.w3.org/2000/svg", "path");
  p.setAttribute("d", "M0,0 L10,5 L0,10z");
  p.setAttribute("fill", "#64748b");
  marker.appendChild(p);
  defs.appendChild(marker);
  diagSvg.appendChild(defs);

  const xAt = (i) => {
    if (n <= 1) return w / 2;
    return pad + i * nodeGap;
  };

  for (let i = 0; i < n - 1; i++) {
    const x0 = xAt(i) + r * 0.4;
    const x1 = xAt(i + 1) - r * 0.4;
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", String(x0));
    line.setAttribute("y1", String(cy));
    line.setAttribute("x2", String(x1));
    line.setAttribute("y2", String(cy));
    line.setAttribute("stroke", "#475569");
    line.setAttribute("stroke-width", "1.2");
    line.setAttribute("marker-end", "url(#arrowend)");
    diagSvg.appendChild(line);
  }

  const svgNS = "http://www.w3.org/2000/svg";

  for (let i = 0; i < n; i++) {
    const u = apiUtterances[i];
    const em = u.emotion || "neutral";
    const emo = EMOTIONS[em] || EMOTIONS.neutral;
    const fill = emo.fill;
    const face = emo.emoji || "😐";
    const sp = u.speaker;
    const spC = SPEAKER_COLORS[sp % SPEAKER_COLORS.length];
    const cx = xAt(i);
    const fullLine = ((u.text || "").trim() || "—").replace(/\s+/g, " ");
    const clipW = n <= 1 ? Math.min(280, w - 80) : Math.max(36, nodeGap - 14);
    const clipId = `utt-clip-${i}`;
    const clip = document.createElementNS(svgNS, "clipPath");
    clip.setAttribute("id", clipId);
    const crect = document.createElementNS(svgNS, "rect");
    crect.setAttribute("x", String(cx - clipW / 2));
    crect.setAttribute("y", String(cy + 36));
    crect.setAttribute("width", String(clipW));
    crect.setAttribute("height", "16");
    clip.appendChild(crect);
    defs.appendChild(clip);

    const g = document.createElementNS(svgNS, "g");
    g.setAttribute("class", "graph-utt-group");
    const tip = document.createElementNS(svgNS, "title");
    tip.textContent = fullLine;
    g.appendChild(tip);

    const circ = document.createElementNS(svgNS, "circle");
    circ.setAttribute("cx", String(cx));
    circ.setAttribute("cy", String(cy - 8));
    circ.setAttribute("r", String(r));
    circ.setAttribute("fill", fill);
    circ.setAttribute("fill-opacity", "0.85");
    circ.setAttribute("stroke", "#0f172a");
    circ.setAttribute("stroke-width", "1.5");
    g.appendChild(circ);
    const faceEl = document.createElementNS(svgNS, "text");
    faceEl.setAttribute("x", String(cx));
    faceEl.setAttribute("y", String(cy - 8));
    faceEl.setAttribute("text-anchor", "middle");
    faceEl.setAttribute("dominant-baseline", "central");
    faceEl.setAttribute("font-size", "16");
    faceEl.setAttribute("font-family", EMOJI_FONT);
    faceEl.setAttribute("aria-hidden", "true");
    faceEl.textContent = face;
    g.appendChild(faceEl);
    const t0 = document.createElementNS(svgNS, "text");
    t0.setAttribute("x", String(cx));
    t0.setAttribute("y", String(cy + 28));
    t0.setAttribute("text-anchor", "middle");
    t0.setAttribute("font-size", "14");
    t0.setAttribute("font-weight", "700");
    t0.setAttribute("fill", spC);
    t0.textContent = String(sp);
    g.appendChild(t0);
    const t1 = document.createElementNS(svgNS, "text");
    t1.setAttribute("x", String(cx - clipW / 2));
    t1.setAttribute("y", String(cy + 48));
    t1.setAttribute("text-anchor", "start");
    t1.setAttribute("font-size", "10");
    t1.setAttribute("fill", "#94a3b8");
    t1.setAttribute("clip-path", `url(#${clipId})`);
    t1.textContent = fullLine;
    g.appendChild(t1);
    diagSvg.appendChild(g);
  }

  ensureGraphWrapWheelScroll();
}

numSp.addEventListener("change", () => {
  updateSpeakerOptions();
  syncJsonPreview();
});
btnNew.addEventListener("click", () => {
  utterances = [];
  renderMessages();
  err.textContent = "";
  resultBlock.classList.add("hidden");
  if (speakerStats) speakerStats.innerHTML = "";
  syncJsonPreview();
});
btnAdd.addEventListener("click", () => {
  err.textContent = "";
  const t = (inputText.value || "").trim();
  if (!t) {
    err.textContent = "Введите текст.";
    return;
  }
  const sp = parseInt(selSpeaker.value, 10);
  utterances.push({ text: t, speaker: sp });
  inputText.value = "";
  renderMessages();
  syncJsonPreview();
});
btnExport.addEventListener("click", () => {
  const blob = new Blob([JSON.stringify(toDocument(), null, 2)], {
    type: "application/json",
  });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "dialogue.json";
  a.click();
  URL.revokeObjectURL(a.href);
});
fileIn.addEventListener("change", () => {
  const f = fileIn.files && fileIn.files[0];
  if (!f) return;
  const r = new FileReader();
  r.onload = () => {
    try {
      fromDocument(JSON.parse(String(r.result)));
      err.textContent = "";
    } catch (e) {
      err.textContent = (e && e.message) || String(e);
    }
  };
  r.readAsText(f);
  fileIn.value = "";
});
btnAnalyze.addEventListener("click", async () => {
  err.textContent = "";
  if (utterances.length === 0) {
    err.textContent = "Сначала добавьте реплики.";
    return;
  }
  btnAnalyze.disabled = true;
  const body = toDocument();
  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        num_speakers: body.num_speakers,
        utterances: body.utterances,
      }),
    });
    if (!res.ok) {
      const t = await res.text();
      let msg = t;
      try {
        const j = JSON.parse(t);
        if (j.detail) msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
      } catch (_) {}
      err.textContent = msg;
      return;
    }
    const j = await res.json();
    const uu = j.utterances;
    resHint.textContent = `Реплик: ${uu.length} · смотрите цепочку стрелок (порядок обмена).`;
    drawGraph(uu);
    renderSpeakerStats(j.speaker_stats);
    resultBlock.classList.remove("hidden");
  } catch (e) {
    err.textContent = (e && e.message) || String(e);
  } finally {
    btnAnalyze.disabled = false;
  }
});

if (numSp) numSp.value = "3";
updateSpeakerOptions();
renderMessages();
syncJsonPreview();
renderLegend();

fetch("/api/status")
  .then((r) => r.json())
  .then((s) => {
    if (s.ready) {
      statusBar.textContent = "Модель загружена · " + (s.checkpoint || "");
      statusBar.className = "status-pill status-ok";
    } else {
      statusBar.textContent =
        "Нет готовой модели: " + (s.load_error || "см. APOLLO_CHECKPOINT");
      statusBar.className = "status-pill status-bad";
    }
  })
  .catch(() => {
    statusBar.textContent = "Сервер недоступен (запустите uvicorn).";
    statusBar.className = "status-pill status-bad";
  });
