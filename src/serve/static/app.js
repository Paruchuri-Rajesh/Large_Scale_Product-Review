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
  try {
    const out = await postJSON("/predict", payload);
    $("score-out").textContent = JSON.stringify(out, null, 2);
  } catch (err) {
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

async function loadMeta() {
  try {
    const meta = await getJSON("/metadata");
    $("meta-out").textContent = JSON.stringify(meta, null, 2);
  } catch (err) {
    $("meta-out").textContent = String(err);
  }
}

loadProducts();
loadReviewers();
loadStream();
loadMeta();
setInterval(loadStream, 5000);
