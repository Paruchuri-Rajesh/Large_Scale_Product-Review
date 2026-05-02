const $ = (id) => document.getElementById(id);

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

async function postJSON(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${await r.text()}`);
  return r.json();
}

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (x * 100).toFixed(1) + "%";
}
function fmtNum(x, d = 2) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(d);
}

function parseNum(v) {
  if (v === null || v === undefined) return NaN;
  if (typeof v === "string") {
    const t = v.trim();
    if (t === "" || /^nan$/i.test(t)) return NaN;
    const n = Number(t);
    return Number.isFinite(n) ? n : NaN;
  }
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}

function normalizeTask(t) {
  return String(t ?? "")
    .trim()
    .toLowerCase();
}

function modelName(row) {
  return String(row.model_name ?? row.model ?? "")
    .trim();
}

function metricClose(a, b) {
  if (Number.isNaN(a) || Number.isNaN(b)) return false;
  return Math.abs(a - b) < 1e-9;
}

function chartSeriesFromPreview(rows) {
  if (!rows || !rows.length) return null;
  const sentiment = [];
  const fraud = [];
  for (const r of rows) {
    const task = normalizeTask(r.task);
    if (task === "sentiment") {
      sentiment.push({
        model_name: modelName(r),
        f1_macro: parseNum(r.f1_macro),
      });
    }
    if (task === "fraud") {
      fraud.push({
        model_name: modelName(r),
        f1: parseNum(r.f1),
        roc_auc: parseNum(r.roc_auc),
      });
    }
  }
  return { sentiment, fraud };
}

function resolveBaselineHighlight(meta, preview) {
  const fb = pickBaselineWinners(preview);
  let bestSent = fb.bestSent;
  let bestFraud = fb.bestFraud;
  let bestSentScore = fb.bestSentScore;
  let bestFraudScore = fb.bestFraudScore;
  const bs = meta.best_sentiment_baseline;
  const bf = meta.best_fraud_baseline;
  if (bs && bs.model_name) {
    bestSent = {
      model: String(bs.model_name).trim(),
      f1Macro: parseNum(bs.f1_macro),
    };
    bestSentScore = parseNum(bs.f1_macro);
  }
  if (bf && bf.model_name) {
    bestFraud = {
      model: String(bf.model_name).trim(),
      f1: parseNum(bf.f1),
    };
    bestFraudScore = parseNum(bf.f1);
  }
  return { bestSent, bestFraud, bestSentScore, bestFraudScore };
}

function renderMlOverview(meta) {
  const el = $("ml-overview-kpis");
  if (!el) return;
  const nTrain = meta.n_train;
  const nTest = meta.n_test;
  const bs = meta.best_sentiment_baseline;
  const bf = meta.best_fraud_baseline;
  const sel = meta.selected_thresholds;
  const roc =
    bf && bf.roc_auc != null && !Number.isNaN(parseNum(bf.roc_auc))
      ? parseNum(bf.roc_auc)
      : null;
  const thr =
    sel && sel.best_f1_threshold != null
      ? parseNum(sel.best_f1_threshold)
      : NaN;

  const cells = [
    {
      k: "Train rows",
      v: nTrain != null ? fmtNum(nTrain, 0) : null,
      sub: null,
    },
    {
      k: "Test rows",
      v: nTest != null ? fmtNum(nTest, 0) : null,
      sub: null,
    },
    {
      k: "Best sentiment baseline",
      v: bs && bs.model_name ? esc(String(bs.model_name)) : null,
      sub:
        bs && !Number.isNaN(parseNum(bs.f1_macro))
          ? `f1_macro ${fmtNum(bs.f1_macro, 4)}`
          : null,
    },
    {
      k: "Best fraud baseline",
      v: bf && bf.model_name ? esc(String(bf.model_name)) : null,
      sub:
        bf && !Number.isNaN(parseNum(bf.f1))
          ? `f1 ${fmtNum(bf.f1, 4)}`
          : null,
    },
    {
      k: "Selected fraud threshold",
      v: !Number.isNaN(thr) ? fmtNum(thr, 4) : null,
      sub: null,
    },
    {
      k: "Fraud ROC-AUC (best baseline)",
      v: roc != null ? fmtNum(roc, 4) : null,
      sub: null,
    },
  ];

  el.innerHTML = cells
    .map((c) => {
      const main =
        c.v != null && c.v !== ""
          ? `<div class="kpi-value">${c.v}</div>`
          : `<div class="kpi-value na">Not available</div>`;
      const sub = c.sub
        ? `<div class="kpi-sub">${c.sub}</div>`
        : "";
      return `<div class="kpi-card"><div class="kpi-label">${esc(c.k)}</div>${main}${sub}</div>`;
    })
    .join("");
}

function buildHBarChartSvg(title, items, valueKey, accent) {
  const rows = (items || [])
    .map((it) => ({
      name: String(it.model_name ?? "").trim(),
      v: parseNum(it[valueKey]),
    }))
    .filter((it) => it.name && !Number.isNaN(it.v));
  if (!rows.length) {
    return `<div class="bar-chart-panel"><p class="mini-chart-title">${esc(title)}</p>${na()}</div>`;
  }
  rows.sort((a, b) => b.v - a.v);
  const maxV = Math.max(...rows.map((r) => r.v), 1e-9);
  const labelW = 168;
  const barMax = 440;
  const rowH = 24;
  const gap = 6;
  const padT = 28;
  const H = padT + rows.length * (rowH + gap) + 8;
  const W = labelW + barMax + 56;
  let y = padT;
  const rects = rows
    .map((r, i) => {
      const w = (r.v / maxV) * barMax;
      const fill = i === 0 ? accent : "#30363d";
      const op = i === 0 ? 1 : 0.82;
      const rx = `<rect x="${labelW + 8}" y="${y}" width="${w.toFixed(1)}" height="${rowH - 2}" rx="3" fill="${fill}" opacity="${op}"/>`;
      const tx = `<text x="${labelW + 16 + w}" y="${y + 15}" fill="#8b949e" font-size="11">${esc(fmtNum(r.v, 4))}</text>`;
      const lbl = `<text x="${labelW - 6}" y="${y + 15}" fill="#e6edf3" font-size="11" text-anchor="end">${esc(r.name.length > 26 ? r.name.slice(0, 24) + "…" : r.name)}</text>`;
      y += rowH + gap;
      return lbl + rx + tx;
    })
    .join("");
  return `<div class="bar-chart-panel"><p class="mini-chart-title">${esc(title)}</p><svg class="bar-chart-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">${rects}</svg></div>`;
}

function buildDriftDeltaBars(rows) {
  if (!rows || !rows.length) return "";
  const absD = (r) => {
    const d = parseNum(r.mean_delta);
    return Number.isNaN(d) ? 0 : Math.abs(d);
  };
  const vals = rows.map(absD);
  const maxV = Math.max(...vals, 1e-12);
  const labelW = 210;
  const barMax = 420;
  const rowH = 26;
  const gap = 8;
  const padT = 26;
  const H = padT + rows.length * (rowH + gap) + 10;
  const W = labelW + barMax + 72;
  let y = padT;
  const parts = rows
    .map((r, i) => {
      const av = absD(r);
      const w = (av / maxV) * barMax;
      const fill = i === 0 ? "#ff9900" : "#3d444d";
      const stroke = i === 0 ? "#ffa726" : "#30363d";
      const hi =
        i === 0
          ? `<rect x="2" y="${y - 2}" width="${W - 4}" height="${rowH + 4}" rx="4" fill="none" stroke="rgba(255,153,0,0.35)" stroke-width="1"/>`
          : "";
      const rx = `${hi}<rect x="${labelW + 8}" y="${y}" width="${w.toFixed(1)}" height="${rowH - 2}" rx="3" fill="${fill}" stroke="${stroke}" stroke-width="${i === 0 ? 1.5 : 0}"/>`;
      const feat = String(r.feature ?? "")
        .trim()
        .slice(0, 32);
      const lbl = `<text x="${labelW - 6}" y="${y + 16}" fill="#e6edf3" font-size="11" text-anchor="end">${esc(feat)}</text>`;
      const val = `<text x="${labelW + 16 + w}" y="${y + 16}" fill="#8b949e" font-size="11">${esc(fmtNum(r.mean_delta, 4))}</text>`;
      y += rowH + gap;
      return lbl + rx + val;
    })
    .join("");
  return `<div class="drift-bar-wrap"><p class="mini-chart-title">Largest mean shifts (|Δ|)</p><svg class="bar-chart-svg drift-bar-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" aria-label="Drift bar chart">${parts}</svg></div>`;
}

function chosenThresholdHero(sel) {
  if (!sel || sel.best_f1_threshold == null) {
    return `<div class="chosen-threshold-hero"><span class="chosen-label">Chosen threshold (best F1)</span><span class="chosen-na">Not available</span></div>`;
  }
  const t = parseNum(sel.best_f1_threshold);
  if (Number.isNaN(t)) {
    return `<div class="chosen-threshold-hero"><span class="chosen-label">Chosen threshold (best F1)</span><span class="chosen-na">Not available</span></div>`;
  }
  return `<div class="chosen-threshold-hero"><span class="chosen-label">Chosen threshold (best F1)</span><span class="chosen-val">${esc(fmtNum(t, 4))}</span></div>`;
}

function renderFraudExplanation(fe) {
  const wrap = $("score-explanation");
  if (!wrap) return;
  if (fe === null || fe === undefined) {
    wrap.innerHTML = `<div class="fraud-explain-panel fraud-explain-na">Explanation not available</div>`;
    return;
  }
  if (typeof fe !== "object") {
    wrap.innerHTML = `<div class="fraud-explain-panel fraud-explain-na">Explanation not available</div>`;
    return;
  }
  const risk = String(fe.risk_level || "").toLowerCase();
  const badgeClass =
    risk === "high"
      ? "risk-badge risk-high"
      : risk === "medium"
        ? "risk-badge risk-medium"
        : "risk-badge risk-low";
  const reasons = Array.isArray(fe.reasons) ? fe.reasons : [];
  const bullets = reasons.map((t) => `<li>${esc(String(t))}</li>`).join("");
  const summary = fe.summary ? `<p class="fraud-explain-summary">${esc(String(fe.summary))}</p>` : "";
  wrap.innerHTML = `<div class="fraud-explain-panel">
    <div class="fraud-explain-head"><span class="${badgeClass}">${esc(risk || "unknown")} risk</span></div>
    ${summary}
    <ul class="fraud-reasons">${bullets}</ul>
  </div>`;
}

$("score-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const body = $("body").value.trim();
  if (!body) return;
  const star = $("star").value;
  const payload = {
    review_body: body,
    star_rating: star ? Number(star) : null,
    verified_purchase: $("verified").value === "true",
  };
  $("score-out").textContent = "scoring…";
  renderFraudExplanation(undefined);
  try {
    const out = await postJSON("/predict", payload);
    renderFraudExplanation(out.fraud_explanation);
    $("score-out").textContent = JSON.stringify(out, null, 2);
  } catch (err) {
    renderFraudExplanation(null);
    $("score-out").textContent = String(err);
  }
});

async function loadProducts() {
  const tbody = document.querySelector("#products-table tbody");
  try {
    const rows = await getJSON("/aggregates/products?limit=15");
    tbody.innerHTML = rows
      .map(
        (r) => `<tr>
        <td><code>${r.product_id ?? ""}</code></td>
        <td>${r.product_category ?? ""}</td>
        <td>${r.review_count ?? 0}</td>
        <td>${fmtNum(r.avg_rating, 2)}</td>
        <td>${fmtPct(r.pct_5star)}</td>
        <td>${fmtPct(r.fraud_rate)}</td>
      </tr>`
      )
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6">${err}</td></tr>`;
  }
}

async function loadReviewers() {
  const tbody = document.querySelector("#reviewers-table tbody");
  try {
    const rows = await getJSON("/aggregates/fraud-reviewers?limit=15");
    tbody.innerHTML = rows
      .map(
        (r) => `<tr class="${r.fraud_rate > 0.5 ? "flag-fraud" : ""}">
        <td><code>${r.reviewer_id ?? ""}</code></td>
        <td>${r.review_count ?? 0}</td>
        <td>${fmtNum(r.avg_rating, 2)}</td>
        <td>${fmtPct(r.fraud_rate)}</td>
        <td>${fmtPct(r.verified_share)}</td>
      </tr>`
      )
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="5">${err}</td></tr>`;
  }
}

function sentimentBadge(s) {
  return `<span class="badge ${s}">${s}</span>`;
}

async function loadStream() {
  const tbody = document.querySelector("#stream-table tbody");
  try {
    const rows = await getJSON("/stream/recent?limit=20");
    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="6">no streamed records yet — start the streaming scorer + drip producer.</td></tr>`;
      return;
    }
    tbody.innerHTML = rows
      .map(
        (r) => `<tr class="${r.fraud_flag ? "flag-fraud" : ""}">
        <td><code>${(r.review_id ?? "").slice(0, 8)}</code></td>
        <td><code>${r.product_id ?? ""}</code></td>
        <td>${r.star_rating ?? ""}</td>
        <td>${sentimentBadge(r.sentiment ?? "neutral")}</td>
        <td>${fmtNum(r.fraud_proba, 3)}</td>
        <td>${r.fraud_flag ? "🚩" : ""}</td>
      </tr>`
      )
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6">${err}</td></tr>`;
  }
}

function esc(s) {
  if (s === null || s === undefined) return "";
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function na(msg) {
  return `<p class="na">${esc(msg || "Not available")}</p>`;
}

function pickBaselineWinners(rows) {
  let bestSent = null;
  let bestSentScore = NaN;
  let bestFraud = null;
  let bestFraudScore = NaN;

  for (const row of rows) {
    const task = normalizeTask(row.task);
    if (task === "sentiment") {
      const score = parseNum(row.f1_macro);
      if (Number.isNaN(score)) continue;
      if (Number.isNaN(bestSentScore) || score > bestSentScore) {
        bestSentScore = score;
        bestSent = { model: modelName(row), f1Macro: score };
      }
    } else if (task === "fraud") {
      const score = parseNum(row.f1);
      if (Number.isNaN(score)) continue;
      if (Number.isNaN(bestFraudScore) || score > bestFraudScore) {
        bestFraudScore = score;
        bestFraud = { model: modelName(row), f1: score };
      }
    }
  }

  return { bestSent, bestFraud, bestSentScore, bestFraudScore };
}

function baselineSummaryBlock(bestSent, bestFraud) {
  let html = '<div class="baseline-summary">';
  if (bestSent) {
    html += `<div class="baseline-task-card is-winner">
      <p class="task-label">Sentiment · best by f1_macro</p>
      <p class="task-model"><code>${esc(bestSent.model)}</code></p>
      <p class="task-metric">f1_macro <strong>${esc(fmtNum(bestSent.f1Macro, 4))}</strong></p>
    </div>`;
  } else {
    html += `<div class="baseline-task-card"><p class="task-label">Sentiment</p><p class="na">Not available</p></div>`;
  }
  if (bestFraud) {
    html += `<div class="baseline-task-card is-winner">
      <p class="task-label">Fraud · best by f1</p>
      <p class="task-model"><code>${esc(bestFraud.model)}</code></p>
      <p class="task-metric">f1 <strong>${esc(fmtNum(bestFraud.f1, 4))}</strong></p>
    </div>`;
  } else {
    html += `<div class="baseline-task-card"><p class="task-label">Fraud</p><p class="na">Not available</p></div>`;
  }
  html += "</div>";
  return html;
}

function renderBaseline(meta, el) {
  const preview = meta.baseline_report_preview;
  if (!preview || !preview.length) {
    el.innerHTML = na();
    return;
  }
  const series = meta.baseline_chart_series;
  const fromPrev =
    chartSeriesFromPreview(preview) || { sentiment: [], fraud: [] };
  const chartSent =
    series && Array.isArray(series.sentiment) && series.sentiment.length
      ? series.sentiment
      : fromPrev.sentiment;
  const chartFraud =
    series && Array.isArray(series.fraud) && series.fraud.length
      ? series.fraud
      : fromPrev.fraud;

  const { bestSent, bestFraud, bestSentScore, bestFraudScore } =
    resolveBaselineHighlight(meta, preview);

  function isBestRow(row) {
    const task = normalizeTask(row.task);
    if (task === "sentiment") {
      if (meta.best_sentiment_baseline && meta.best_sentiment_baseline.model_name) {
        return (
          modelName(row) ===
          String(meta.best_sentiment_baseline.model_name).trim()
        );
      }
      const s = parseNum(row.f1_macro);
      return (
        bestSent != null &&
        !Number.isNaN(bestSentScore) &&
        !Number.isNaN(s) &&
        metricClose(s, bestSentScore)
      );
    }
    if (task === "fraud") {
      if (meta.best_fraud_baseline && meta.best_fraud_baseline.model_name) {
        return (
          modelName(row) === String(meta.best_fraud_baseline.model_name).trim()
        );
      }
      const f = parseNum(row.f1);
      return (
        bestFraud != null &&
        !Number.isNaN(bestFraudScore) &&
        !Number.isNaN(f) &&
        metricClose(f, bestFraudScore)
      );
    }
    return false;
  }

  const keys = Object.keys(preview[0]);
  const thead = `<thead><tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead>`;
  const tbody = preview
    .map((row) => {
      const cls = isBestRow(row) ? ' class="ml-row-best"' : "";
      return `<tr${cls}>${keys.map((k) => `<td>${esc(row[k])}</td>`).join("")}</tr>`;
    })
    .join("");

  const sentBars = buildHBarChartSvg(
    "Sentiment baselines · F1 macro",
    chartSent,
    "f1_macro",
    "#58a6ff"
  );
  const fraudBars = buildHBarChartSvg(
    "Fraud baselines · F1",
    chartFraud,
    "f1",
    "#ff9900"
  );

  el.innerHTML =
    baselineSummaryBlock(bestSent, bestFraud) +
    `<div class="chart-grid duo-chart">${sentBars}${fraudBars}</div>` +
    `<p class="ml-subhead table-detail-label">Raw CSV preview (first rows)</p>` +
    `<details class="table-details"><summary>Preview table</summary>` +
    `<div class="table-wrap table-wrap-muted"><table class="ml-table ml-table-compact">${thead}<tbody>${tbody}</tbody></table></div>` +
    `</details>`;
}

const THR_COL_P = "#58a6ff";
const THR_COL_R = "#56d364";
const THR_COL_F = "#ff9900";

function computeThresholdYRange(pts) {
  const vals = [];
  for (const p of pts) {
    [p.precision, p.recall, p.f1].forEach((v) => {
      if (v === null || v === undefined || Number.isNaN(parseNum(v))) return;
      vals.push(Number(v));
    });
  }
  if (!vals.length) {
    return { yMin: 0, yMax: 1, zoomed: false, flat: false };
  }
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const spread = hi - lo;
  if (spread < 1e-9) {
    const pad = 0.045;
    return {
      yMin: Math.max(0, lo - pad),
      yMax: Math.min(1, hi + pad),
      zoomed: true,
      flat: true,
    };
  }
  if (spread < 0.14) {
    const pad = Math.max(0.01, spread * 0.18);
    return {
      yMin: Math.max(0, lo - pad),
      yMax: Math.min(1, hi + pad),
      zoomed: true,
      flat: false,
    };
  }
  return { yMin: 0, yMax: 1, zoomed: false, flat: false };
}

function thresholdHintText(yr) {
  const base =
    "Harder synthetic evaluation is tuned to show a modest precision/recall tradeoff on holdout; curves are not assumed flat. ";
  if (yr.flat && yr.zoomed) {
    return (
      base +
      "Here all preview points sit in a very narrow band—the Y axis is zoomed so small differences stay visible."
    );
  }
  if (yr.zoomed) {
    return (
      base +
      "The Y axis is zoomed to the observed metric range so overlapping lines remain readable."
    );
  }
  return base + "Full 0–1 scale when metrics span a wider band.";
}

function buildThresholdChartSvg(pts) {
  const W = 680;
  const H = 260;
  const pad = { l: 44, r: 16, t: 16, b: 34 };
  const plotW = W - pad.l - pad.r;
  const plotH = H - pad.t - pad.b;
  let tMin = pts[0].t;
  let tMax = pts[pts.length - 1].t;
  if (tMin === tMax) {
    tMin -= 0.05;
    tMax += 0.05;
  }
  const xScale = (t) => pad.l + ((t - tMin) / (tMax - tMin)) * plotW;

  const yr = computeThresholdYRange(pts);
  const yMin = yr.yMin;
  const yMax = yr.yMax;
  const ySpan = Math.max(yMax - yMin, 1e-12);
  const yScale = (v) => {
    const nv = Number(v);
    if (Number.isNaN(nv)) return pad.t + plotH;
    const t = (nv - yMin) / ySpan;
    const c = Math.max(0, Math.min(1, t));
    return pad.t + plotH - c * plotH;
  };

  function linePath(getter) {
    const parts = [];
    for (let i = 0; i < pts.length; i++) {
      const v = getter(pts[i]);
      if (Number.isNaN(v)) continue;
      const x = xScale(pts[i].t);
      const y = yScale(v);
      parts.push(`${parts.length ? " L " : "M "}${x.toFixed(1)} ${y.toFixed(1)}`);
    }
    return parts.join(" ");
  }

  const gridLevels = [0, 0.25, 0.5, 0.75, 1].map((u) => yMin + u * (yMax - yMin));
  const grids = gridLevels
    .map(
      (g) =>
        `<line x1="${pad.l}" y1="${yScale(g).toFixed(1)}" x2="${W - pad.r}" y2="${yScale(g).toFixed(1)}" stroke="#30363d" stroke-width="1" stroke-dasharray="3 4" opacity="0.45"/>`
    )
    .join("");

  const pPath = linePath((p) => p.precision);
  const rPath = linePath((p) => p.recall);
  const fPath = linePath((p) => p.f1);
  const lblLo = `<text x="${pad.l}" y="${H - 8}" fill="#8b949e" font-size="10">${esc(fmtNum(tMin, 2))}</text>`;
  const lblHi = `<text x="${W - pad.r}" y="${H - 8}" fill="#8b949e" font-size="10" text-anchor="end">${esc(fmtNum(tMax, 2))}</text>`;
  const yTop = `<text x="8" y="${pad.t + 8}" fill="#8b949e" font-size="10">${esc(fmtNum(yMax, 3))}</text>`;
  const yMid = `<text x="8" y="${(pad.t + plotH / 2 + 4).toFixed(0)}" fill="#8b949e" font-size="10">${esc(fmtNum((yMin + yMax) / 2, 3))}</text>`;
  const yBot = `<text x="8" y="${H - pad.b}" fill="#8b949e" font-size="10">${esc(fmtNum(yMin, 3))}</text>`;

  const band =
    yr.zoomed || yr.flat
      ? `<p class="chart-y-band">Y-axis: ${esc(fmtNum(yMin, 4))}–${esc(fmtNum(yMax, 4))}${yr.zoomed ? " (zoomed)" : ""}</p>`
      : `<p class="chart-y-band">Y-axis: 0–1</p>`;

  const svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" aria-label="Threshold metrics chart">
    ${grids}
    <line x1="${pad.l}" y1="${H - pad.b}" x2="${W - pad.r}" y2="${H - pad.b}" stroke="#30363d" stroke-width="1"/>
    ${yTop}${yMid}${yBot}
    ${lblLo}${lblHi}
    ${pPath ? `<path d="${pPath}" fill="none" stroke="${THR_COL_P}" stroke-width="2.25"/>` : ""}
    ${rPath ? `<path d="${rPath}" fill="none" stroke="${THR_COL_R}" stroke-width="2.25"/>` : ""}
    ${fPath ? `<path d="${fPath}" fill="none" stroke="${THR_COL_F}" stroke-width="2.25"/>` : ""}
  </svg>`;

  return { bandHtml: band, svgHtml: svg, yRange: yr };
}

function renderThresholdStudy(rows, el, meta) {
  const sel = meta && meta.selected_thresholds;
  if (!rows || !rows.length) {
    el.innerHTML =
      chosenThresholdHero(sel) +
      `<p class="ml-subhead">Metrics vs threshold</p>` +
      `<div class="chart-wrap chart-wrap-lg"><p class="na">Not available</p></div>`;
    return;
  }
  const pts = rows
    .map((r) => ({
      t: parseNum(r.threshold),
      precision: parseNum(r.precision),
      recall: parseNum(r.recall),
      f1: parseNum(r.f1),
    }))
    .filter((r) => !Number.isNaN(r.t))
    .sort((a, b) => a.t - b.t);
  let chartBlock = `<p class="na">Not available</p>`;
  if (
    pts.length &&
    pts.some(
      (p) =>
        !Number.isNaN(p.precision) ||
        !Number.isNaN(p.recall) ||
        !Number.isNaN(p.f1)
    )
  ) {
    const built = buildThresholdChartSvg(pts);
    chartBlock = `${built.bandHtml}${built.svgHtml}<div class="chart-legend" role="list">
      <span role="listitem"><i style="background:${THR_COL_P}"></i> precision</span>
      <span role="listitem"><i style="background:${THR_COL_R}"></i> recall</span>
      <span role="listitem"><i style="background:${THR_COL_F}"></i> f1</span>
    </div>
    <p class="hint">${esc(thresholdHintText(built.yRange))}</p>`;
  }

  const keys = Object.keys(rows[0]);
  const thead = `<thead><tr>${keys.map((k) => `<th>${esc(k)}</th>`).join("")}</tr></thead>`;
  const tbody = rows
    .map(
      (row) =>
        `<tr>${keys.map((k) => `<td>${esc(row[k])}</td>`).join("")}</tr>`
    )
    .join("");
  el.innerHTML =
    `${chosenThresholdHero(sel)}
    <p class="ml-subhead">Metrics vs threshold</p>
    <div class="chart-wrap chart-wrap-lg">${chartBlock}</div>
    <p class="ml-subhead table-detail-label">Threshold sweep (detail)</p>
    <details class="table-details"><summary>Preview rows</summary>
    <div class="table-wrap table-wrap-muted"><table class="ml-table ml-table-compact">${thead}<tbody>${tbody}</tbody></table></div>
    </details>`;
}

function renderDrift(summary, el) {
  if (!summary || typeof summary !== "object") {
    el.innerHTML = na();
    return;
  }
  const nums = summary.numeric_mean_comparison;
  const txt = summary.text_length_summary;
  let cards = `
    <div class="drift-cards">
      <div class="drift-card"><p class="label">Reference rows</p><p class="val">${esc(summary.row_count_reference ?? "—")}</p></div>
      <div class="drift-card"><p class="label">Current rows</p><p class="val">${esc(summary.row_count_current ?? "—")}</p></div>
    </div>`;
  if (txt && typeof txt === "object" && Object.keys(txt).length) {
    cards += `
      <div class="drift-cards">
        <div class="drift-card"><p class="label">Ref avg text len</p><p class="val">${esc(fmtNum(txt.reference_avg_length, 1))}</p></div>
        <div class="drift-card"><p class="label">Cur avg text len</p><p class="val">${esc(fmtNum(txt.current_avg_length, 1))}</p></div>
        <div class="drift-card"><p class="label">Δ avg length</p><p class="val">${esc(fmtNum(txt.avg_length_delta, 2))}</p></div>
      </div>`;
  }
  let tableHtml = "";
  if (nums && nums.length) {
    const absD = (r) => {
      const d = parseNum(r.mean_delta);
      return Number.isNaN(d) ? 0 : Math.abs(d);
    };
    const top5 = [...nums].sort((a, b) => absD(b) - absD(a)).slice(0, 5);
    const driftBars = buildDriftDeltaBars(top5);
    tableHtml = `<p class="drift-table-note">Top 5 numeric features by |Δ mean|</p>
    ${driftBars || `<p class="na">Not available</p>`}
    <p class="ml-subhead table-detail-label">Breakdown</p>
    <details class="table-details"><summary>Compact table</summary>
    <div class="table-wrap table-wrap-muted"><table class="ml-table ml-table-compact"><thead><tr>
      <th>feature</th><th>ref mean</th><th>cur mean</th><th>Δ</th>
    </tr></thead><tbody>${top5
      .map((r, idx) => {
        const trc =
          idx === 0 ? ' class="ml-row-best"' : "";
        return `<tr${trc}><td><code>${esc(r.feature)}</code></td><td>${esc(fmtNum(r.reference_mean, 4))}</td><td>${esc(fmtNum(r.current_mean, 4))}</td><td>${esc(fmtNum(r.mean_delta, 4))}</td></tr>`;
      })
      .join("")}</tbody></table></div></details>`;
  } else {
    tableHtml = `<p class="na">Not available</p>`;
  }
  el.innerHTML = cards + tableHtml;
}

function metaRunIdHtml(id) {
  if (!id) return `<span class="na">Not available</span>`;
  const s = String(id);
  return `<span class="meta-run-id" title="${esc(s)}">${esc(s.slice(0, 10))}…</span>`;
}

function renderMetaSummary(meta, el) {
  if (!el) return;
  const sent = meta.sentiment || {};
  const fraud = meta.fraud || {};
  const sel = meta.selection || {};
  const nFeat = Array.isArray(meta.numeric_fraud_features)
    ? meta.numeric_fraud_features.length
    : null;

  const mk = (label, inner) =>
    `<div class="kpi-card"><div class="kpi-label">${esc(label)}</div><div class="kpi-value">${inner}</div></div>`;

  el.innerHTML = `
    <div class="meta-summary-heading">Registered models</div>
    <div class="kpi-grid meta-kpi-compact">
      ${mk(
        "Sentiment model",
        sel.sentiment_model_name
          ? esc(sel.sentiment_model_name)
          : `<span class="na">Not available</span>`
      )}
      ${mk(
        "Fraud model",
        sel.fraud_model_name
          ? esc(sel.fraud_model_name)
          : `<span class="na">Not available</span>`
      )}
    </div>
    <div class="meta-summary-heading">Sentiment (holdout)</div>
    <div class="kpi-grid meta-kpi-compact">
      ${mk("MLflow run", metaRunIdHtml(sent.run_id))}
      ${mk("f1_macro", fmtNum(sent.f1_macro, 4))}
      ${mk("f1_weighted", fmtNum(sent.f1_weighted, 4))}
    </div>
    <div class="meta-summary-heading">Fraud (holdout)</div>
    <div class="kpi-grid meta-kpi-compact">
      ${mk("MLflow run", metaRunIdHtml(fraud.run_id))}
      ${mk("Precision", fmtNum(fraud.precision, 4))}
      ${mk("Recall", fmtNum(fraud.recall, 4))}
      ${mk("F1", fmtNum(fraud.f1, 4))}
      ${mk("ROC-AUC", fmtNum(fraud.roc_auc, 4))}
    </div>
    <div class="meta-summary-heading">Features</div>
    <div class="kpi-grid meta-kpi-compact">
      ${mk(
        "Fraud numeric inputs",
        nFeat != null ? esc(String(nFeat)) : `<span class="na">Not available</span>`
      )}
    </div>
  `;
}

function renderSelectedThresholds(obj, el) {
  if (!obj || typeof obj !== "object") {
    el.innerHTML = na();
    return;
  }
  const bf = obj.best_f1_metrics || {};
  const pt = obj.precision_target_metrics;
  const ptTh = obj.precision_target_threshold;
  let html = `<div class="threshold-cards"><div class="threshold-box"><h3>Best F1 · threshold ${esc(fmtNum(obj.best_f1_threshold, 4))}</h3>
    <div class="threshold-grid">
      <div>Precision<strong>${esc(fmtNum(bf.precision, 4))}</strong></div>
      <div>Recall<strong>${esc(fmtNum(bf.recall, 4))}</strong></div>
      <div>F1<strong>${esc(fmtNum(bf.f1, 4))}</strong></div>
    </div></div>`;
  if (ptTh != null && pt && typeof pt === "object") {
    html += `<div class="threshold-box"><h3>Precision target (≥0.95) · ${esc(fmtNum(ptTh, 4))}</h3>
      <div class="threshold-grid">
        <div>Precision<strong>${esc(fmtNum(pt.precision, 4))}</strong></div>
        <div>Recall<strong>${esc(fmtNum(pt.recall, 4))}</strong></div>
        <div>F1<strong>${esc(fmtNum(pt.f1, 4))}</strong></div>
      </div></div>`;
  } else {
    html += `<div class="threshold-box threshold-box-muted"><h3>Precision target (≥0.95)</h3><p class="na" style="margin:0">Not available</p></div>`;
  }
  html += "</div>";
  el.innerHTML = html;
}

async function loadMeta() {
  try {
    const meta = await getJSON("/metadata");
    renderMetaSummary(meta, $("meta-summary"));

    renderMlOverview(meta);
    renderBaseline(meta, $("baseline-preview"));
    renderThresholdStudy(
      meta.threshold_report_preview,
      $("threshold-preview"),
      meta
    );
    renderDrift(meta.drift_summary, $("drift-summary"));
    renderSelectedThresholds(meta.selected_thresholds, $("selected-thresholds"));
  } catch (err) {
    const ms = $("meta-summary");
    if (ms) ms.innerHTML = `<p class="na">${esc(String(err))}</p>`;
    const ko = $("ml-overview-kpis");
    if (ko) ko.innerHTML = na();
    $("baseline-preview").innerHTML = na();
    $("threshold-preview").innerHTML = na();
    $("drift-summary").innerHTML = na();
    $("selected-thresholds").innerHTML = na();
  }
}

loadProducts();
loadReviewers();
loadStream();
loadMeta();
setInterval(loadStream, 5000);
