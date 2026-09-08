async function renderFullAnalysis(paperId) {
  const modal = document.querySelector("#view-modal");
  if (!modal) return;
  const isAnalysisRoute = location.pathname.startsWith("/papers/") && location.pathname.endsWith("/analysis");
  modal.innerHTML = `<div class="modal__content card--modal analysis-modal"><div class="analysis-skeleton">${skeletonMarkup(12)}</div></div>`;
  modal.setAttribute("open", "");
  document.body.style.overflow = "hidden";

  let analysis;
  try {
    analysis = await getAnalysis(paperId);
  } catch (error) {
    modal.innerHTML = errorModalMarkup(error, isAnalysisRoute);
    return;
  }

  if (!analysis || !analysis.paper) {
    modal.innerHTML = emptyModalMarkup(isAnalysisRoute);
    return;
  }

  modal.innerHTML = buildAnalysisPage(analysis, isAnalysisRoute);

  if (isAnalysisRoute) {
    document.title = `${analysis.paper.title || "Paper Analysis"} \u2014 Analysis`;
  }
  wireAnalysisActions(modal, analysis);
}

function errorModalMarkup(error, isAnalysisRoute) {
  const back = isAnalysisRoute
    ? `<button class="btn btn--ghost" data-nav="dashboard">${icon("arrow-right", 16)} Back to Dashboard</button>`
    : `<button class="btn btn--ghost" data-close>${icon("arrow-right", 16)}</button>`;
  return `<div class="modal__content card--modal analysis-modal">
    <div class="analysis-page">
      <div class="analysis-topbar">${back}</div>
      ${alertMarkup(error, { title: "Could not load full analysis" })}
    </div></div>`;
}

function emptyModalMarkup(isAnalysisRoute) {
  const back = isAnalysisRoute
    ? `<button class="btn btn--ghost" data-nav="dashboard">${icon("arrow-right", 16)} Back to Dashboard</button>`
    : `<button class="btn btn--ghost" data-close>${icon("arrow-right", 16)}</button>`;
  return `<div class="modal__content card--modal analysis-modal">
    <div class="analysis-page">
      <div class="analysis-topbar">${back}</div>
      ${emptyMarkup("No analysis data was returned for this paper.")}
    </div></div>`;
}

function buildAnalysisPage(analysis, isAnalysisRoute) {
  const { paper, prediction, review, attention, similar, model, groundTruth } = analysis;
  const backBtn = isAnalysisRoute
    ? `<button class="btn btn--ghost" data-nav="dashboard">${icon("arrow-right", 16)} Back to Dashboard</button>`
    : `<button class="btn btn--ghost" data-close>${icon("arrow-right", 16)}</button>`;
  return `<div class="modal__content card--modal analysis-modal">
    <div class="analysis-page">
      <div class="analysis-topbar">
        ${backBtn}
        <div class="analysis-topbar__actions">
          <button class="btn btn--ghost" data-action="download-report">${icon("extract", 16)} Download Report</button>
          <button class="btn btn--ghost" data-action="export-json">${icon("database", 16)} Export JSON</button>
        </div>
      </div>
      <div class="analysis-content">
        ${analysisHeader(paper, prediction, review)}
        ${paperInfoCard(paper, model)}
        ${classificationResultSection(prediction)}
        ${reviewAssessmentSection(prediction, review)}
        ${classificationEvidenceSection(attention)}
        ${hierarchicalAnalysisSection(attention)}
        ${similarPapersSection(similar)}
        ${modelPerformanceSection(model)}
        ${groundTruthSection(prediction, groundTruth)}
        ${askSection(paper)}
      </div>
    </div>
  </div>`;
}

function analysisHeader(paper, prediction, review) {
  const meta = [paper.authors_short, paper.year, paper.venue].filter(Boolean).join(" \u00b7 ");
  const words = paper.word_count ? `${paper.word_count.toLocaleString()} words` : "";
  const reviewBadge = review.needs_review
    ? `<span class="tag tag--warn">${icon("warn", 11)} Flagged for Review</span>`
    : `<span class="tag tag--good">${icon("trend", 11)} Confident</span>`;
  return `<section class="analysis-header">
    <div class="analysis-header__main">
      <h1 class="analysis-header__title">${escapeHtml(paper.title || "Untitled Paper")}</h1>
      <p class="analysis-header__subtitle">Detailed model prediction, attention analysis, and classification evidence</p>
      <div class="analysis-header__meta">${meta || ""} ${words ? `\u00b7 ${words}` : ""}</div>
      ${reviewBadge}
    </div>
    <div class="analysis-header__thumb" aria-hidden="true">
      <svg width="100%" height="100%" viewBox="0 0 96 124">
        <rect width="96" height="124" fill="var(--bg-inset)"/>
        ${Array.from({length:15},(_,i)=>{const y=14+i*7;const w=i===0?56:i%4===3?44:72;return `<rect x="${i===0?20:12}" y="${y}" width="${w}" height="2.2" rx="1.1" fill="var(--border-strong)"/>`;}).join("")}
      </svg>
    </div>
  </section>`;
}

function paperInfoCard(paper, model) {
  const fields = [["Paper ID", paper.paper_id]];
  if (paper.n_authors !== null && paper.n_authors !== undefined) fields.push(["Authors", String(paper.n_authors) + (paper.authors_short ? ` (${paper.authors_short})` : "")]);
  if (paper.n_references !== null && paper.n_references !== undefined) fields.push(["References", String(paper.n_references)]);
  if (paper.split) fields.push(["Split", paper.split]);
  const left = fields.map(([k,v]) => `<tr><th>${escapeHtml(k)}</th><td>${escapeHtml(v)}</td></tr>`).join("");
  const right = `
    <tr><th>Inference Status</th><td><span class="tag tag--good">Completed</span></td></tr>
    <tr><th>Model</th><td>${escapeHtml(model.model_display_name || "Unknown")}</td></tr>
    <tr><th>Confidence Type</th><td>${escapeHtml(model.confidence_kind || "unavailable")}</td></tr>
    <tr><th>Similarity Method</th><td>${escapeHtml(model.similarity_method || "unavailable")}</td></tr>`;
  return `<section class="card analysis-paper-info">
    <h3 class="section-title">Paper Information</h3>
    <div class="analysis-paper-info__grid">
      <table class="analysis-table">${left}</table>
      <table class="analysis-table">${right}</table>
    </div>
  </section>`;
}

function classificationResultSection(prediction) {
  const probs = Array.isArray(prediction.probabilities) ? prediction.probabilities : [];
  if (!probs.length) {
    return `<section class="card"><h3 class="section-title">Classification Result</h3>${emptyMarkup("No prediction is available for this paper.")}</section>`;
  }
  const top = probs[0];
  const kind = prediction.confidence_kind || "probability";
  const headValue = kind === "probability" ? pct(top.score) : `${top.score.toFixed(2)} margin`;
  const rows = probs.map((s) => barRow(s.label, kind === "probability" ? pct(s.score) : s.score.toFixed(2), s.score, hueFor(s.label), { stacked: true })).join("");
  return `<section class="card analysis-classification">
    <h3 class="section-title">Classification Result</h3>
    <div class="analysis-classification__main">
      <div class="confidence-ring" data-fraction="${top.score}">
        <svg viewBox="0 0 36 36" width="120" height="120">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--track)" stroke-width="2.5"/>
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--accent)" stroke-width="2.5" stroke-dasharray="${(top.score * 100).toFixed(1)} 100" stroke-linecap="round" transform="rotate(-90 18 18)"/>
        </svg>
        <div class="confidence-ring__value">${headValue}</div>
        <div class="confidence-ring__label">${escapeHtml(top.label)}</div>
      </div>
      <div class="analysis-classification__bars">
        <h4>Top Predicted Domains</h4>
        <div class="bars">${rows}</div>
      </div>
    </div>
  </section>`;
}

function reviewAssessmentSection(prediction, review) {
  if (!review || !review.top_label) {
    return `<section class="card"><h3 class="section-title">Review Assessment</h3>${emptyMarkup("Review assessment is not available for this paper.")}</section>`;
  }
  const gap = review.confidence_gap != null ? review.confidence_gap : (review.top_score != null && review.second_score != null ? review.top_score - review.second_score : null);
  const gapPct = gap != null ? pct(gap) : "\u2014";
  const band = review.review_band || (review.needs_review ? "review_recommended" : "high");
  const bandLabel = band === "high" ? "High Confidence" : band === "review_recommended" ? "Moderate Confidence" : band === "review_required" ? "Low Confidence" : band;
  const bandClass = band === "high" ? "tag--good" : band === "review_recommended" ? "tag--warn" : "tag--critical";
  let reason = review.review_reason;
  if (!reason) {
    if (review.needs_review && gap != null && gap < 0.2) reason = "The top prediction has moderate confidence and is relatively close to the second-highest prediction.";
    else if (review.needs_review) reason = "The top prediction falls below the configured confidence threshold for automatic acceptance.";
    else reason = "The top prediction meets the configured confidence threshold.";
  }
  return `<section class="card analysis-review">
    <h3 class="section-title">Review Assessment</h3>
    <div class="analysis-review__status">
      <span class="tag ${bandClass}">Review Required: ${review.needs_review ? "YES" : "NO"}</span>
      <span class="tag">${escapeHtml(bandLabel)}</span>
    </div>
    <p class="analysis-review__reason">${escapeHtml(reason)}</p>
    <div class="analysis-review__metrics">
      <div class="metric"><span class="metric__label">Top Prediction</span><span class="metric__value">${pct(review.top_score)}</span></div>
      <div class="metric"><span class="metric__label">Second Prediction</span><span class="metric__value">${review.second_score != null ? pct(review.second_score) : "\u2014"}</span></div>
      <div class="metric"><span class="metric__label">Confidence Gap</span><span class="metric__value">${gapPct}</span></div>
    </div>
  </section>`;
}
