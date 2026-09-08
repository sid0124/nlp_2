/* ==========================================================================
   Full Paper Analysis Page Module
   ==========================================================================
   Implements the comprehensive, academic research-grade Full Paper Analysis
   page for:
   "Academic Research Paper Classification using Hierarchical Attention
   Networks and Transformer Embeddings"

   Demonstrates the complete pipeline:
   Paper -> Text Extraction -> Transformer Embeddings -> Hierarchical
   Attention Network -> Section/Sentence/Word Attention -> Classification
   -> Prediction + Confidence -> Interpretability / Evidence
   ========================================================================== */

import { ask, getAnalysis } from "./api.js";
import { domainColor, isRegisteredDomain, readToken } from "./domains.js";
import { icon } from "./icons.js";

const $ = (sel, parent = document) => parent.querySelector(sel);
const $$ = (sel, parent = document) => Array.from(parent.querySelectorAll(sel));

function escapeHtml(value) {
  return String(value ?? "").replace(
    /[&<>"']/g,
    (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c],
  );
}

const pct = (value) =>
  typeof value === "number" && !Number.isNaN(value) ? `${(value * 100).toFixed(1)}%` : "—";

const signed = (value, digits = 3) => {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${value < 0 ? "−" : "+"}${Math.abs(value).toFixed(digits)}`;
};

/** Clamp a probability-like number into [0, 1] before it becomes a percentage. */
const clamp01 = (value) => (typeof value === "number" && !Number.isNaN(value) ? Math.max(0, Math.min(1, value)) : 0);

const hueFor = (label) =>
  isRegisteredDomain(label) ? domainColor(label) : readToken("--series-other", "#64748b");

/** Current active analysis state for interactive components (sections, tabs, ask) */
let activeAnalysis = null;
let activeSelectedSection = null;
let activeSelectedSentence = null;
let embeddingViewMode = "domain"; // "domain" | "similarity"
let onNavigateBackCallback = null;
let onSelectPaperCallback = null;

export function initAnalysisPage({ onNavigateBack, onSelectPaper }) {
  onNavigateBackCallback = onNavigateBack;
  onSelectPaperCallback = onSelectPaper;
}

/**
 * Render the full analysis page into the mount container.
 *
 * @param {HTMLElement} mountContainer
 * @param {string} paperId
 */
export async function renderAnalysisPage(mountContainer, paperId) {
  if (!mountContainer) return;

  // Show rich skeleton state while loading
  mountContainer.innerHTML = renderSkeletonAnalysis(paperId);

  let analysisData = null;
  try {
    analysisData = await getAnalysis(paperId);
  } catch (error) {
    mountContainer.innerHTML = `
      <div class="analysis-page" style="padding: var(--sp-4) 0;">
        <div class="analysis-topbar">
          <button class="btn btn--ghost" id="analysis-err-back">${icon("arrow_right", 16)} Back to Dashboard</button>
        </div>
        <div style="margin-top: var(--sp-4);">
          <div class="card" style="padding: var(--sp-5); border-color: var(--status-critical, #ef4444);">
            <h3 style="color: var(--status-critical, #ef4444); margin-bottom: var(--sp-2);">Could not load full analysis</h3>
            <p style="color: var(--text-secondary); margin-bottom: var(--sp-4);">${escapeHtml(error.message || "The analysis payload could not be fetched from the API.")}</p>
            <button class="btn btn--ghost" id="analysis-err-retry">${icon("refresh", 16)} Retry Loading</button>
          </div>
        </div>
      </div>`;

    $("#analysis-err-back", mountContainer)?.addEventListener("click", () => {
      if (onNavigateBackCallback) onNavigateBackCallback();
    });
    $("#analysis-err-retry", mountContainer)?.addEventListener("click", () => {
      renderAnalysisPage(mountContainer, paperId);
    });
    return;
  }

  if (!analysisData || !analysisData.paper) {
    mountContainer.innerHTML = `
      <div class="analysis-page" style="padding: var(--sp-4) 0;">
        <div class="analysis-topbar">
          <button class="btn btn--ghost" id="analysis-empty-back">${icon("arrow_right", 16)} Back to Dashboard</button>
        </div>
        <div class="card" style="padding: var(--sp-5); margin-top: var(--sp-4); text-align: center;">
          <p style="color: var(--text-muted);">No analysis data found for paper ID: <code>${escapeHtml(paperId)}</code>.</p>
        </div>
      </div>`;
    $("#analysis-empty-back", mountContainer)?.addEventListener("click", () => {
      if (onNavigateBackCallback) onNavigateBackCallback();
    });
    return;
  }

  activeAnalysis = analysisData;
  const sections = analysisData.attention?.section_attention?.sections || [];
  activeSelectedSection = sections.length ? sections[0].name || sections[0].canonical_name : null;

  const sentences = analysisData.attention?.sentences || [];
  activeSelectedSentence = sentences.length ? sentences[0] : null;

  mountContainer.innerHTML = buildFullAnalysisMarkup(analysisData);
  wireAnalysisInteractions(mountContainer, analysisData);
}

/* ==========================================================================
   Markup Generation
   ========================================================================== */

function buildFullAnalysisMarkup(data) {
  const { paper, prediction, review, attention, model, similar_papers, ground_truth, embedding, insights } = data;

  return `
    <div class="analysis" style="display: flex; flex-direction: column; gap: var(--sp-5); max-width: 1200px; margin: 0 auto; width: 100%; padding-bottom: var(--sp-6);">
      <!-- 1. TOP HEADER & PAPER METADATA -->
      ${renderTopHeader(paper, prediction, review)}

      <!-- 2. PAPER INFORMATION CARD -->
      ${renderPaperInfoCard(paper, model)}

      <!-- 3. CLASSIFICATION RESULT & 4. REVIEW ASSESSMENT (GRID) -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: var(--sp-4);">
        ${renderClassificationResult(prediction)}
        ${renderReviewAssessment(prediction, review)}
      </div>

      <!-- 10. HAN MODEL PIPELINE VISUALIZATION -->
      ${renderHANPipelineVisualization(model)}

      <!-- 5. CLASSIFICATION EVIDENCE (TABS) -->
      ${renderClassificationEvidence(attention, prediction, embedding)}

      <!-- 6, 7, 8. HIERARCHICAL ATTENTION ANALYSIS (SECTIONS -> SENTENCES -> TOKENS) -->
      ${renderHierarchicalAttentionSection(attention, paper)}

      <!-- 9. TRANSFORMER EMBEDDING ANALYSIS -->
      ${renderTransformerEmbeddingSection(embedding, paper)}

      <!-- 11. SIMILAR PAPERS & 13. PREDICTION VERIFICATION (GRID) -->
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: var(--sp-4);">
        ${renderSimilarPapersSection(similar_papers)}
        <div style="display: flex; flex-direction: column; gap: var(--sp-4);">
          ${renderGroundTruthVerification(ground_truth, prediction)}
          ${renderPaperInsightsSection(insights, prediction)}
        </div>
      </div>

      <!-- 12. MODEL PERFORMANCE (DATASET-LEVEL) -->
      ${renderModelPerformanceSection(model)}

      <!-- 15. ASK THIS PAPER -->
      ${renderAskPaperSection(paper)}
    </div>
  `;
}

/* --------------------------------------------------------------------------
   1. Top Header
   -------------------------------------------------------------------------- */
function renderTopHeader(paper, prediction, review) {
  const title = paper.title || "Untitled paper";
  const year = paper.year != null ? String(paper.year) : "—";
  const venue = paper.venue || "—";
  const wordCount = paper.word_count ? paper.word_count.toLocaleString() : "—";
  const needsReview = review?.needs_review ?? prediction?.needs_review ?? false;
  const reviewBand = review?.review_band || prediction?.review_band || (needsReview ? "review_recommended" : "high");

  const statusBadge = needsReview
    ? `<span class="tag tag--warn" style="border-color: rgba(245, 158, 11, 0.4); color: #f59e0b; background: rgba(245, 158, 11, 0.08); font-size: var(--fs-xs); font-weight: 600; padding: 3px 9px;">⚠ Flagged for Review (${escapeHtml(reviewBand.replace('_', ' '))})</span>`
    : `<span class="tag tag--good" style="border-color: rgba(34, 197, 94, 0.4); color: #22c55e; background: rgba(34, 197, 94, 0.08); font-size: var(--fs-xs); font-weight: 600; padding: 3px 9px;">✓ Confident Prediction</span>`;

  return `
    <header class="analysis__header" style="border-bottom: 1px solid var(--border); padding-bottom: var(--sp-4);">
      <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: var(--sp-3); flex-wrap: wrap;">
        <div style="display: flex; flex-direction: column; gap: var(--sp-1);">
          <button class="btn btn--ghost" id="analysis-back-btn" style="align-self: flex-start; margin-bottom: var(--sp-2); font-size: var(--fs-xs); padding: 4px 10px;">
            ← Back to Dashboard
          </button>
          <div style="display: flex; align-items: center; gap: var(--sp-2); flex-wrap: wrap;">
            <h1 style="font-size: var(--fs-2xl); font-weight: 700; color: var(--text-primary); margin: 0; line-height: 1.2;">
              Paper Analysis
            </h1>
            ${statusBadge}
          </div>
          <p style="font-size: var(--fs-sm); color: var(--text-secondary); margin: 0;">
            Detailed model prediction, attention analysis, and classification evidence
          </p>
        </div>

        <div style="display: flex; gap: var(--sp-2); align-items: center; flex-wrap: wrap; margin-top: var(--sp-2);">
          <button class="btn btn--ghost" id="analysis-download-btn" title="Generate and print full technical report">
            ${icon("extract", 16)} Download Report
          </button>
          <button class="btn btn--ghost" id="analysis-export-json-btn" title="Export complete structured analysis object as JSON">
            ${icon("database", 16)} Export JSON
          </button>
        </div>
      </div>

      <!-- Paper Title and Metadata Strip -->
      <div style="margin-top: var(--sp-3); padding: var(--sp-3) var(--sp-4); background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--r-md);">
        <div style="font-size: var(--fs-xs); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px;">
          Active Document
        </div>
        <div style="font-size: var(--fs-md); font-weight: 600; color: var(--text-primary); margin-bottom: 4px;">
          ${escapeHtml(title)}
        </div>
        <div style="font-size: var(--fs-xs); color: var(--text-secondary); display: flex; gap: var(--sp-2); flex-wrap: wrap; align-items: center;">
          <span>${escapeHtml(String(year))}</span>
          <span style="color: var(--text-muted)">·</span>
          <span>${escapeHtml(venue)}</span>
          <span style="color: var(--text-muted)">·</span>
          <span>${wordCount === "—" ? "—" : `${wordCount} words`}</span>
          <span style="color: var(--text-muted)">·</span>
          <span style="font-family: monospace; color: var(--text-muted);">${escapeHtml(paper.paper_id)}</span>
          ${paper.split ? `<span class="tag" style="margin-left: var(--sp-1);">${escapeHtml(paper.split)}</span>` : ""}
        </div>
      </div>
    </header>
  `;
}

/* --------------------------------------------------------------------------
   2. Paper Information Card
   -------------------------------------------------------------------------- */
function renderPaperInfoCard(paper, model) {
  const thumbSvg = `
    <div class="doc-thumb" style="width: 76px; height: 104px; background: var(--bg-inset); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 8px; display: flex; flex-direction: column; gap: 4px; flex-shrink: 0;" aria-hidden="true">
      <div style="width: 40px; height: 4px; background: var(--accent); border-radius: 2px; margin-bottom: 4px;"></div>
      <div style="width: 100%; height: 2px; background: var(--border-strong); border-radius: 1px;"></div>
      <div style="width: 88%; height: 2px; background: var(--border-strong); border-radius: 1px;"></div>
      <div style="width: 95%; height: 2px; background: var(--border-strong); border-radius: 1px;"></div>
      <div style="width: 70%; height: 2px; background: var(--border-strong); border-radius: 1px; margin-bottom: 6px;"></div>
      <div style="width: 100%; height: 2px; background: var(--border); border-radius: 1px;"></div>
      <div style="width: 92%; height: 2px; background: var(--border); border-radius: 1px;"></div>
      <div style="width: 80%; height: 2px; background: var(--border); border-radius: 1px;"></div>
    </div>
  `;

  const authors = paper.authors_short || (paper.n_authors ? `${paper.n_authors} authors` : "Not specified");
  const year = paper.year != null ? String(paper.year) : "—";
  const venue = paper.venue || "—";
  const wordCount = paper.word_count ? paper.word_count.toLocaleString() : "—";
  const numSections = paper.n_sections != null ? String(paper.n_sections) : "—";
  const numReferences = paper.n_references != null ? String(paper.n_references) : "—";
  const modelName = model?.model_display_name || model?.model_name || "Not available";
  const now = new Date().toISOString().replace("T", " ").slice(0, 19) + " UTC";

  return `
    <section class="card" aria-labelledby="paper-info-title">
      <div class="card__head">
        <h2 class="card__title" id="paper-info-title">${icon("papers", 18)} Paper Information</h2>
      </div>
      <div class="card__body" style="display: grid; grid-template-columns: auto 1fr auto; gap: var(--sp-4); align-items: center; flex-wrap: wrap;">
        ${thumbSvg}

        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: var(--sp-3); flex: 1;">
          <div class="analysis-fact">
            <span class="analysis-fact__label">Authors</span>
            <span class="analysis-fact__value" title="${escapeHtml(authors)}">${escapeHtml(authors)}</span>
          </div>
          <div class="analysis-fact">
            <span class="analysis-fact__label">Publication Year</span>
            <span class="analysis-fact__value">${escapeHtml(String(year))}</span>
          </div>
          <div class="analysis-fact">
            <span class="analysis-fact__label">Venue / Journal</span>
            <span class="analysis-fact__value" title="${escapeHtml(venue)}">${escapeHtml(venue)}</span>
          </div>
          <div class="analysis-fact">
            <span class="analysis-fact__label">Word Count</span>
            <span class="analysis-fact__value">${wordCount === "—" ? "—" : `${wordCount} words`}</span>
          </div>
          <div class="analysis-fact">
            <span class="analysis-fact__label">Document Sections</span>
            <span class="analysis-fact__value">${numSections}</span>
          </div>
          <div class="analysis-fact">
            <span class="analysis-fact__label">References</span>
            <span class="analysis-fact__value">${numReferences === "—" ? "—" : `${numReferences} citations`}</span>
          </div>
        </div>

        <div style="border-left: 1px solid var(--border); padding-left: var(--sp-4); display: flex; flex-direction: column; gap: var(--sp-2); min-width: 170px;">
          <div>
            <span class="analysis-fact__label">Inference Status</span>
            <span style="display: inline-flex; align-items: center; gap: 4px; color: #22c55e; font-size: var(--fs-xs); font-weight: 600; margin-top: 2px;">
              <span style="width: 6px; height: 6px; border-radius: 50%; background: #22c55e;"></span> COMPLETED
            </span>
          </div>
          <div>
            <span class="analysis-fact__label">Active Model</span>
            <span style="font-size: var(--fs-xs); font-weight: 500; color: var(--text-primary); display: block; margin-top: 2px;">${escapeHtml(modelName)}</span>
          </div>
          <div>
            <span class="analysis-fact__label">Analysis Timestamp</span>
            <span style="font-size: var(--fs-xs); color: var(--text-muted); font-family: monospace; display: block; margin-top: 2px;">${now}</span>
          </div>
        </div>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   3. Classification Result Card
   -------------------------------------------------------------------------- */
function renderClassificationResult(prediction) {
  const label = prediction.predicted_label || "Unclassified";
  const confidence = typeof prediction.confidence === "number" ? prediction.confidence : 0;
  const kind = prediction.confidence_kind || "probability";
  const confPercent = kind === "probability" ? pct(confidence) : `${confidence.toFixed(2)} margin`;
  const domainColorCode = hueFor(label);

  const probabilities = Array.isArray(prediction.probabilities) ? prediction.probabilities : [];
  const probRows = probabilities.map((p) => {
    const scoreVal = typeof p.score === "number" ? p.score : 0;
    const scorePct = kind === "probability" ? pct(scoreVal) : scoreVal.toFixed(2);
    const fillPct = Math.min(100, Math.max(0, scoreVal * 100));
    const isTop = p.label === label;
    const color = isTop ? domainColorCode : "var(--border-strong)";

    return `
      <div class="bar-row" style="margin-bottom: var(--sp-2);">
        <div class="bar-row__label" style="font-size: var(--fs-sm); color: ${isTop ? "var(--text-primary)" : "var(--text-secondary)"}; font-weight: ${isTop ? "600" : "400"};" title="${escapeHtml(p.label)}">
          ${escapeHtml(p.label)}
        </div>
        <div class="meter" style="height: 7px; background: var(--track); border-radius: var(--r-pill); overflow: hidden;">
          <div class="meter__fill" style="width: ${fillPct.toFixed(1)}%; height: 100%; background: ${color}; border-radius: var(--r-pill); transition: width 0.6s ease;"></div>
        </div>
        <div class="bar-row__value" style="font-size: var(--fs-xs); font-weight: ${isTop ? "700" : "400"}; color: ${isTop ? "var(--accent-ink)" : "var(--text-muted)"};">
          ${scorePct}
        </div>
      </div>
    `;
  }).join("");

  // Circular progress ring SVG
  const strokeDash = (clamp01(confidence) * 100).toFixed(1);

  return `
    <section class="card" aria-labelledby="classification-title" style="display: flex; flex-direction: column;">
      <div class="card__head">
        <h2 class="card__title" id="classification-title">${icon("flag", 18)} Classification Result</h2>
      </div>
      <div class="card__body" style="display: flex; flex-direction: column; gap: var(--sp-4); flex: 1;">
        <!-- Primary Prediction Display -->
        <div style="display: flex; align-items: center; gap: var(--sp-4); padding: var(--sp-3); background: var(--bg-inset); border-radius: var(--r-md); border: 1px solid var(--border);">
          <!-- Circular Progress Ring -->
          <div class="confidence-ring" style="width: 96px; height: 96px; position: relative; display: grid; place-items: center; flex-shrink: 0;">
            <svg viewBox="0 0 36 36" width="96" height="96" style="transform: rotate(-90deg);">
              <circle cx="18" cy="18" r="15.9" fill="none" stroke="var(--track)" stroke-width="2.8"/>
              <circle cx="18" cy="18" r="15.9" fill="none" stroke="${domainColorCode}" stroke-width="2.8"
                stroke-dasharray="${strokeDash} 100" stroke-linecap="round"/>
            </svg>
            <div style="position: absolute; text-align: center;">
              <div style="font-size: var(--fs-md); font-weight: 800; color: var(--text-primary); line-height: 1;">${confPercent}</div>
              <div style="font-size: 9px; color: var(--text-muted); text-transform: uppercase; margin-top: 2px;">${kind === "probability" ? "Conf." : "Margin"}</div>
            </div>
          </div>

          <!-- Label & Subtext -->
          <div style="flex: 1; min-width: 0;">
            <div style="font-size: var(--fs-xs); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 2px;">
              Primary Predicted Domain
            </div>
            <div style="font-size: var(--fs-lg); font-weight: 700; color: var(--text-primary); line-height: 1.25; word-break: break-word;">
              ${escapeHtml(label)}
            </div>
            <div style="font-size: var(--fs-xs); color: var(--text-secondary); margin-top: 4px;">
              Predicted via Hierarchical Attention Network over long document representations.
            </div>
          </div>
        </div>

        <!-- Dynamic Top Predicted Domains -->
        <div>
          <div style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); margin-bottom: var(--sp-2);">
            Top Predicted Domains (${probabilities.length} Classes Evaluated)
          </div>
          <div class="prob-bars">
            ${probRows || '<div style="font-size: var(--fs-xs); color: var(--text-muted);">No distribution scores available.</div>'}
          </div>
        </div>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   4. Confidence & Review Assessment Card
   -------------------------------------------------------------------------- */
function renderReviewAssessment(prediction, review) {
  const needsReview = review?.needs_review ?? prediction?.needs_review ?? false;
  const topScore = review?.top_score ?? prediction?.confidence ?? 0;
  const secondScore = review?.second_score ?? null;
  const gap = review?.confidence_gap != null
    ? review.confidence_gap
    : (secondScore != null && typeof topScore === "number" ? Math.max(0, topScore - secondScore) : null);

  const gapDisplay = gap != null ? pct(gap) : "—";
  const band = review?.review_band || prediction?.review_band || (needsReview ? "review_recommended" : "high");

  let gaugeLevel = "High Confidence";
  let gaugeColor = "#22c55e";

  if (band === "review_required" || (typeof topScore === "number" && topScore < 0.5)) {
    gaugeLevel = "Low Confidence";
    gaugeColor = "var(--status-critical, #ef4444)";
  } else if (needsReview || band === "review_recommended" || (gap != null && gap < 0.2)) {
    gaugeLevel = "Moderate Confidence";
    gaugeColor = "#f59e0b";
  }

  let reason = review?.review_reason;
  if (!reason) {
    if (needsReview && gap != null && gap < 0.2) {
      reason = "The top prediction has moderate confidence and is relatively close to the second-highest prediction.";
    } else if (needsReview) {
      reason = "The top prediction falls below the automated threshold and human review is recommended.";
    } else {
      reason = "The top prediction meets model certainty criteria with clear separation from secondary categories.";
    }
  }

  return `
    <section class="card" aria-labelledby="review-assessment-title" style="display: flex; flex-direction: column;">
      <div class="card__head">
        <h2 class="card__title" id="review-assessment-title">${icon("shield", 18)} Review Assessment</h2>
      </div>
      <div class="card__body" style="display: flex; flex-direction: column; gap: var(--sp-4); flex: 1;">
        <!-- Status Box -->
        <div style="display: flex; align-items: center; justify-content: space-between; gap: var(--sp-3); padding: var(--sp-3); background: var(--bg-inset); border-radius: var(--r-md); border: 1px solid var(--border);">
          <div>
            <span style="font-size: var(--fs-xs); color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em;">Review Required</span>
            <div style="font-size: var(--fs-lg); font-weight: 700; color: ${needsReview ? '#f59e0b' : '#22c55e'};">
              ${needsReview ? "YES — Human Review Recommended" : "NO — Automated Confidence"}
            </div>
          </div>
          <span class="tag" style="border-color: ${gaugeColor}; color: ${gaugeColor}; font-weight: 600;">
            ${gaugeLevel}
          </span>
        </div>

        <!-- Reason Text -->
        <div>
          <span style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; color: var(--text-muted); letter-spacing: 0.04em; display: block; margin-bottom: 4px;">
            Assessment Rationale
          </span>
          <p style="font-size: var(--fs-sm); color: var(--text-secondary); line-height: 1.5; margin: 0; background: var(--bg-surface); padding: var(--sp-2) var(--sp-3); border-radius: var(--r-sm); border-left: 3px solid ${gaugeColor};">
            “${escapeHtml(reason)}”
          </p>
        </div>

        <!-- Dynamic Gap Metrics -->
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: var(--sp-2); text-align: center;">
          <div style="background: var(--bg-inset); padding: var(--sp-2); border-radius: var(--r-sm); border: 1px solid var(--border);">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Top Prediction</div>
            <div style="font-size: var(--fs-md); font-weight: 700; color: var(--text-primary); margin-top: 2px;">${pct(topScore)}</div>
          </div>
          <div style="background: var(--bg-inset); padding: var(--sp-2); border-radius: var(--r-sm); border: 1px solid var(--border);">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Second Pred.</div>
            <div style="font-size: var(--fs-md); font-weight: 700; color: var(--text-secondary); margin-top: 2px;">${secondScore != null ? pct(secondScore) : "—"}</div>
          </div>
          <div style="background: var(--bg-inset); padding: var(--sp-2); border-radius: var(--r-sm); border: 1px solid var(--border);">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Confidence Gap</div>
            <div style="font-size: var(--fs-md); font-weight: 700; color: var(--accent-ink); margin-top: 2px;">${gapDisplay}</div>
          </div>
        </div>

        <!-- Scientific Caveat -->
        <div style="font-size: 11px; color: var(--text-muted); line-height: 1.4; border-top: 1px solid var(--border); padding-top: var(--sp-2);">
          <em>Scientific Note:</em> Model confidence represents prediction certainty, not guaranteed empirical correctness. Low confidence gap indicates borderline domain alignment.
        </div>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   10. HAN Model Pipeline Visualization
   -------------------------------------------------------------------------- */
function renderHANPipelineVisualization(model) {
  const isHan = Boolean(model?.is_han);
  const displayName = model?.model_display_name || model?.model_name || "Current model";

  // The diagram must reflect the run that actually produced this prediction:
  // a HAN run shows the hierarchical encoder pipeline, any other run shows the
  // fitted pipeline it really used. Inventing the other family's stages would
  // present one architecture as another (see spec's "adapt the diagram" note).
  const modelSteps = isHan
    ? [
        { label: "Paper", sub: "Document Input" },
        { label: "Text Extraction", sub: "Sections + Sentences" },
        { label: "Word Encoder", sub: "SciBERT Embeddings" },
        { label: "Sentence Encoder", sub: "Bi-GRU Context" },
        { label: "Section Encoder", sub: "Bi-GRU Context" },
        { label: "Hierarchical Attention", sub: "Two-Level Attention", active: true },
        { label: "Document Vector", sub: "Attention-Weighted" },
        { label: "Classifier", sub: "Softmax Head" },
        { label: "Predicted Domain", sub: "Target Class Output", active: true },
      ]
    : [
        { label: "Paper", sub: "Document Input" },
        { label: "Text Extraction", sub: "Sections + Body" },
        { label: "TF-IDF Vectorizer", sub: "Vocabulary Weights" },
        { label: "Term Weights", sub: "Class Contributions", active: true },
        { label: "Linear Classifier", sub: "Decision Function" },
        { label: "Probabilities", sub: "Calibrated Scores" },
        { label: "Predicted Domain", sub: "Target Class Output", active: true },
      ];

  const stepsHtml = modelSteps.map((s, idx) => `
    <div style="display: flex; align-items: center; gap: var(--sp-1); flex-shrink: 0;">
      <div style="background: ${s.active ? 'var(--accent-wash, rgba(99, 102, 241, 0.15))' : 'var(--bg-inset)'}; border: 1px solid ${s.active ? 'var(--accent)' : 'var(--border)'}; border-radius: var(--r-md); padding: 8px 12px; text-align: center; min-width: 95px;">
        <div style="width: 16px; height: 16px; border-radius: 50%; background: ${s.active ? 'var(--accent)' : 'var(--border-strong)'}; color: #fff; font-size: 9px; font-weight: 700; display: inline-flex; align-items: center; justify-content: center; margin-bottom: 2px;">${idx + 1}</div>
        <div style="font-size: var(--fs-xs); font-weight: 600; color: ${s.active ? 'var(--accent-ink)' : 'var(--text-primary)'}; white-space: nowrap;">${escapeHtml(s.label)}</div>
        <div style="font-size: 10px; color: var(--text-muted); white-space: nowrap;">${escapeHtml(s.sub)}</div>
      </div>
      ${idx < modelSteps.length - 1 ? `<div style="color: var(--text-muted); font-size: 14px; padding: 0 2px;">→</div>` : ''}
    </div>
  `).join("");

  return `
    <section class="card" aria-labelledby="pipeline-title">
      <div class="card__head">
        <h2 class="card__title" id="pipeline-title">${icon("trend", 18)} ${isHan ? "Hierarchical Attention Network (HAN) Pipeline" : "Document Classification Pipeline (Active Run)"}</h2>
        <span class="tag" style="font-size: var(--fs-xs);">${escapeHtml(displayName)}</span>
      </div>
      <div class="card__body">
        <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin-top: 0; margin-bottom: var(--sp-3);">
          ${isHan
            ? "The hierarchical pipeline encodes words, sentences, and sections bottom-up, then weights them with learned attention to form a document vector that the classifier consumes."
            : "This run's fitted pipeline: text is vectorised with TF-IDF, per-term weights drive the linear decision, and calibrated probabilities produce the prediction."}
        </p>
        <div style="display: flex; align-items: center; gap: 4px; overflow-x: auto; padding: var(--sp-2) 0; scrollbar-width: thin;">
          ${stepsHtml}
        </div>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   5. Classification Evidence (Tabs)
   -------------------------------------------------------------------------- */
function renderClassificationEvidence(attention, prediction, embedding) {
  const terms = Array.isArray(attention?.terms) ? attention.terms : [];
  const evidenceLabel = attention?.evidence_label || "Model evidence";
  const method = attention?.method || "model_evidence";
  const isAttention = method === "han_hierarchical_attention";
  const topPos = terms.filter((t) => (t.contribution ?? t.weight ?? 0) >= 0).slice(0, 10);
  const topNeg = terms.filter((t) => (t.contribution ?? t.weight ?? 0) < 0).slice(0, 8);

  const termRows = topPos.map((t) => {
    const w = typeof t.weight === "number" ? t.weight : 0.5;
    const contrib = typeof t.contribution === "number" ? t.contribution : w;
    return `
      <div style="display: flex; align-items: center; justify-content: space-between; gap: var(--sp-2); padding: 4px 8px; background: var(--bg-surface); border-radius: var(--r-sm); border: 1px solid var(--border);">
        <span style="font-size: var(--fs-sm); font-family: monospace; color: var(--text-primary);">${escapeHtml(t.term)}</span>
        <div style="display: flex; align-items: center; gap: var(--sp-2);">
          <div style="width: 80px; height: 5px; background: var(--track); border-radius: var(--r-pill); overflow: hidden;">
            <div style="width: ${(w * 100).toFixed(0)}%; height: 100%; background: #22c55e; border-radius: var(--r-pill);"></div>
          </div>
          <span style="font-size: var(--fs-xs); color: #22c55e; font-weight: 600; min-width: 44px; text-align: right;">${signed(contrib, 3)}</span>
        </div>
      </div>
    `;
  }).join("");

  const negTermRows = topNeg.map((t) => {
    const w = typeof t.weight === "number" ? t.weight : 0.5;
    const contrib = typeof t.contribution === "number" ? t.contribution : -w;
    return `
      <div style="display: flex; align-items: center; justify-content: space-between; gap: var(--sp-2); padding: 4px 8px; background: var(--bg-surface); border-radius: var(--r-sm); border: 1px solid var(--border);">
        <span style="font-size: var(--fs-sm); font-family: monospace; color: var(--text-secondary);">${escapeHtml(t.term)}</span>
        <div style="display: flex; align-items: center; gap: var(--sp-2);">
          <div style="width: 80px; height: 5px; background: var(--track); border-radius: var(--r-pill); overflow: hidden;">
            <div style="width: ${(w * 100).toFixed(0)}%; height: 100%; background: var(--text-muted); border-radius: var(--r-pill);"></div>
          </div>
          <span style="font-size: var(--fs-xs); color: var(--text-muted); font-weight: 600; min-width: 44px; text-align: right;">${signed(contrib, 3)}</span>
        </div>
      </div>
    `;
  }).join("");

  return `
    <section class="card" aria-labelledby="evidence-title">
      <div class="card__head">
        <h2 class="card__title" id="evidence-title">${icon("extract", 18)} Classification Evidence</h2>
        <span class="info-tooltip" title="Attention scores indicate where the model placed greater focus. They should not be interpreted as definitive causal explanations.">
          ${icon("info", 14)} Causal Caveat
        </span>
      </div>
      <div class="card__body">
        <div class="evidence-tabs" role="tablist">
          <button class="evidence-tab" role="tab" id="tab-attention" aria-selected="true" data-tab="panel-attention">${isAttention ? "Attention" : "Feature"} Evidence</button>
          <button class="evidence-tab" role="tab" id="tab-keyword" aria-selected="false" data-tab="panel-keyword">Keyword Evidence</button>
          <button class="evidence-tab" role="tab" id="tab-embedding" aria-selected="false" data-tab="panel-embedding">Embedding Evidence</button>
        </div>

        <!-- Panel 1: Attention / Feature Evidence -->
        <div class="evidence-panel" id="panel-attention" aria-hidden="false">
          <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin-top: 0; margin-bottom: var(--sp-3);">
            ${isAttention ? "Attention" : "Feature"} evidence for the predicted domain <strong>${escapeHtml(prediction.predicted_label || "target")}</strong>.
            Evidence type: <strong>${escapeHtml(evidenceLabel)}</strong>. These weights indicate where the model placed its focus — they are not a causal explanation of the paper.
          </p>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: var(--sp-3);">
            <div>
              <div style="font-size: var(--fs-xs); font-weight: 600; color: #22c55e; margin-bottom: var(--sp-1);">Supporting Evidence (Positive Contributions)</div>
              <div style="display: flex; flex-direction: column; gap: 4px;">
                ${termRows || '<div style="font-size: var(--fs-xs); color: var(--text-muted);">No supporting terms found.</div>'}
              </div>
            </div>
            <div>
              <div style="font-size: var(--fs-xs); font-weight: 600; color: var(--text-muted); margin-bottom: var(--sp-1);">Counter-Evidence (Diverging Terms)</div>
              <div style="display: flex; flex-direction: column; gap: 4px;">
                ${negTermRows || '<div style="font-size: var(--fs-xs); color: var(--text-muted);">No counter-terms found.</div>'}
              </div>
            </div>
          </div>
        </div>

        <!-- Panel 2: Keyword Evidence -->
        <div class="evidence-panel" id="panel-keyword" aria-hidden="true" style="display: none;">
          <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin-top: 0; margin-bottom: var(--sp-3);">
            Ranked TF-IDF feature vocabulary aligned with the paper's canonical section taxonomy.
          </p>
          <div style="display: flex; flex-wrap: wrap; gap: 6px;">
            ${terms.slice(0, 20).map((t) => `<span class="tag" style="font-family: monospace; font-size: var(--fs-xs);">${escapeHtml(t.term)} <small style="color: var(--text-muted);">(${signed(t.contribution ?? t.weight ?? 0, 2)})</small></span>`).join("")}
          </div>
        </div>

        <!-- Panel 3: Embedding Evidence -->
        <div class="evidence-panel" id="panel-embedding" aria-hidden="true" style="display: none;">
          ${embedding && Array.isArray(embedding.points)
            ? `
            <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin-top: 0; margin-bottom: var(--sp-3);">
              Document vectors embedded with <strong>${escapeHtml(embedding.model_name || "the active run's encoder")}</strong> and projected to 2D via ${escapeHtml(embedding.method || "dimensionality reduction")}. Nearby points share representation proximity, which is not the same as methodological equivalence.
            </p>
            <div style="padding: var(--sp-3); background: var(--bg-inset); border-radius: var(--r-sm); font-size: var(--fs-xs); color: var(--text-secondary);">
              ${escapeHtml(embedding.representation || "Document-level representation.")} The embedding section above visualises these points in the semantic space.
            </div>`
            : `
            <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin-top: 0; margin-bottom: var(--sp-3);">
              Embedding evidence is not available for this run.
            </p>
            <div style="padding: var(--sp-3); background: var(--bg-inset); border-radius: var(--r-sm); font-size: var(--fs-xs); color: var(--text-secondary);">
              No document-vector projection data was returned by the backend, so no embedding evidence can be shown here.
            </div>`}
        </div>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   6, 7, 8. Hierarchical Attention (Sections -> Sentences -> Tokens)
   -------------------------------------------------------------------------- */
function renderHierarchicalAttentionSection(attention, paper) {
  const sections = Array.isArray(attention?.section_attention?.sections) ? attention.section_attention.sections : [];
  const sentences = Array.isArray(attention?.sentences) ? attention.sentences : [];

  const maxSecW = Math.max(...sections.map((s) => s.weight || 0.001), 0.001);

  const sectionsListHtml = sections.map((s) => {
    const secName = s.name || s.canonical_name || "Section";
    const secW = typeof s.weight === "number" ? s.weight : 0.1;
    const normW = Math.min(100, Math.max(0, (secW / maxSecW) * 100));
    const isSelected = activeSelectedSection === secName;

    return `
      <div class="section-item" data-section-name="${escapeHtml(secName)}" style="border: 1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}; border-radius: var(--r-sm); margin-bottom: 6px; cursor: pointer; transition: border-color 0.15s;">
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background: ${isSelected ? 'var(--accent-wash, rgba(99,102,241,0.1))' : 'var(--bg-surface)'};">
          <div style="font-size: var(--fs-sm); font-weight: ${isSelected ? '600' : '500'}; color: ${isSelected ? 'var(--accent-ink)' : 'var(--text-primary)'};">
            ${escapeHtml(secName.toUpperCase())}
          </div>
          <div style="display: flex; align-items: center; gap: var(--sp-2);">
            <div style="width: 70px; height: 5px; background: var(--track); border-radius: var(--r-pill); overflow: hidden;">
              <div style="width: ${normW.toFixed(0)}%; height: 100%; background: var(--accent); border-radius: var(--r-pill);"></div>
            </div>
            <span style="font-size: var(--fs-xs); font-family: monospace; color: var(--text-secondary); min-width: 36px; text-align: right;">${secW.toFixed(3)}</span>
          </div>
        </div>
      </div>
    `;
  }).join("");

  // Filter sentences for the selected section
  const currentSecSentences = sentences.filter((s) => {
    if (!activeSelectedSection) return true;
    const matchName = (s.section_name || s.canonical_name || "").toLowerCase();
    const selName = activeSelectedSection.toLowerCase();
    return matchName === selName || matchName.includes(selName) || selName.includes(matchName);
  });

  const displaySentences = currentSecSentences.length ? currentSecSentences.slice(0, 6) : sentences.slice(0, 6);

  const sentencesHtml = displaySentences.map((s, idx) => {
    const sentW = typeof s.weight === "number" ? s.weight : 0.1;
    const isSelected = activeSelectedSentence === s || (activeSelectedSentence && activeSelectedSentence.text === s.text);

    return `
      <div class="sentence-item" data-sentence-index="${idx}" style="padding: 10px; background: ${isSelected ? 'var(--accent-wash, rgba(99,102,241,0.1))' : 'var(--bg-surface)'}; border: 1px solid ${isSelected ? 'var(--accent)' : 'var(--border)'}; border-left: 4px solid var(--accent); border-radius: var(--r-sm); margin-bottom: 8px; cursor: pointer; transition: background 0.15s;">
        <div style="display: flex; justify-content: space-between; align-items: center; font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px;">
          <span>Sentence ${idx + 1} (${escapeHtml(s.canonical_name || s.section_name || "Section")})</span>
          <span class="tag" style="font-size: 10px; padding: 1px 6px;">Attention: ${sentW.toFixed(3)}</span>
        </div>
        <div style="font-size: var(--fs-sm); color: var(--text-primary); line-height: 1.5;">
          “${escapeHtml(s.text)}”
        </div>
      </div>
    `;
  }).join("");

  // Token level highlighting on active sentence
  const activeSentenceText = activeSelectedSentence?.text || (sentences.length ? sentences[0].text : (paper.text ? paper.text.slice(0, 180) : "Hierarchical Attention Networks classify research papers using section and sentence representations."));
  const highlightedTokensHtml = renderTokenAttentionHighlighting(activeSentenceText, attention?.terms || []);

  return `
    <section class="card" aria-labelledby="han-section-title">
      <div class="card__head">
        <h2 class="card__title" id="han-section-title">${icon("cube", 18)} Hierarchical Attention Analysis</h2>
      </div>
      <div class="card__body">
        <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin-top: 0; margin-bottom: var(--sp-4);">
          Visual representation of hierarchical document attention. Click any <strong>Section</strong> to inspect its sentence-level attention, and click a <strong>Sentence</strong> to view token-level attention highlighting.
        </p>

        <!-- Hierarchy Flow Breadcrumb -->
        <div style="display: flex; align-items: center; gap: 6px; overflow-x: auto; padding: var(--sp-2) 0; margin-bottom: var(--sp-4); font-size: var(--fs-xs); color: var(--text-muted); font-weight: 500;">
          <span style="color: var(--text-primary);">DOCUMENT</span> <span>→</span>
          <span style="color: var(--accent-ink); font-weight: 600;">SECTIONS</span> <span>→</span>
          <span style="color: var(--accent-ink); font-weight: 600;">SENTENCES</span> <span>→</span>
          <span style="color: var(--accent-ink); font-weight: 600;">WORDS</span> <span>→</span>
          <span>ATTENTION WEIGHTS</span> <span>→</span>
          <span>DOCUMENT VECTOR</span> <span>→</span>
          <span style="color: #22c55e; font-weight: 600;">CLASSIFIER</span>
        </div>

        <div style="display: grid; grid-template-columns: 280px 1fr; gap: var(--sp-4); align-items: start;">
          <!-- Left: Section-Level Attention -->
          <div>
            <div style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: var(--sp-2);">
              Section-Level Attention
            </div>
            <div class="section-list">
              ${sectionsListHtml || '<div style="font-size: var(--fs-xs); color: var(--text-muted);">No section attention data.</div>'}
            </div>
          </div>

          <!-- Right: Sentence-Level Attention & Token Attention -->
          <div style="display: flex; flex-direction: column; gap: var(--sp-4);">
            <!-- Sentence-Level Attention -->
            <div>
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--sp-2);">
                <span style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; color: var(--text-muted);">
                  Sentence-Level Attention (${escapeHtml(activeSelectedSection || "All Sections")})
                </span>
                <span style="font-size: 11px; color: var(--text-muted);">Click sentence to highlight tokens</span>
              </div>
              <div class="sentence-list" id="analysis-sentence-mount">
                ${sentencesHtml || '<div style="font-size: var(--fs-xs); color: var(--text-muted);">No extracted sentences for this section.</div>'}
              </div>
            </div>

            <!-- Token-Level Attention -->
            <div style="padding: var(--sp-3); background: var(--bg-inset); border-radius: var(--r-md); border: 1px solid var(--border);">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: var(--sp-2);">
                <span style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; color: var(--text-muted);">
                  Token-Level Attention Highlighting
                </span>
                <span class="info-tooltip" title="Token highlighting represents relative model attention within the selected text.">
                  ${icon("info", 13)} Relative Focus
                </span>
              </div>
              <div class="token-display" id="analysis-token-display" style="font-size: var(--fs-sm); line-height: 1.8; color: var(--text-primary); padding: var(--sp-3); background: var(--bg-surface); border-radius: var(--r-sm); border: 1px solid var(--border);">
                ${highlightedTokensHtml}
              </div>
              <div class="token-legend" style="display: flex; gap: var(--sp-3); margin-top: var(--sp-2); font-size: 11px; color: var(--text-muted); align-items: center;">
                <div class="token-legend__item" style="display: flex; align-items: center; gap: 4px;">
                  <span style="width: 12px; height: 12px; border-radius: 2px; background: rgba(99, 102, 241, 0.15);"></span> Low Focus
                </div>
                <div class="token-legend__item" style="display: flex; align-items: center; gap: 4px;">
                  <span style="width: 12px; height: 12px; border-radius: 2px; background: rgba(99, 102, 241, 0.35);"></span> Moderate Focus
                </div>
                <div class="token-legend__item" style="display: flex; align-items: center; gap: 4px;">
                  <span style="width: 12px; height: 12px; border-radius: 2px; background: #6366f1; color: #fff;"></span> High Focus
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  `;
}

function renderTokenAttentionHighlighting(sentenceText, terms) {
  if (!sentenceText) return "No sentence selected.";
  const termMap = new Map();
  for (const t of terms) {
    if (t.term) {
      termMap.set(t.term.toLowerCase(), Math.abs(t.contribution ?? t.weight ?? 0.5));
    }
  }

  const words = sentenceText.split(/(\s+|[.,;:?!()]+)/);
  return words.map((w) => {
    if (!w.trim() || /^[.,;:?!()]+$/.test(w)) return escapeHtml(w);
    const clean = w.toLowerCase().replace(/[^a-z0-9_-]/g, "");
    const score = termMap.get(clean) || 0;

    if (score >= 0.7) {
      return `<span class="token token--high" style="background: #6366f1; color: #ffffff; padding: 2px 4px; border-radius: 3px; font-weight: 600;" title="High Attention: ${score.toFixed(3)}">${escapeHtml(w)}</span>`;
    }
    if (score >= 0.35) {
      return `<span class="token token--medium" style="background: rgba(99, 102, 241, 0.35); color: var(--text-primary); padding: 2px 4px; border-radius: 3px; font-weight: 500;" title="Medium Attention: ${score.toFixed(3)}">${escapeHtml(w)}</span>`;
    }
    if (score > 0.1) {
      return `<span class="token token--low" style="background: rgba(99, 102, 241, 0.15); padding: 2px 4px; border-radius: 3px;" title="Low Attention">${escapeHtml(w)}</span>`;
    }
    return escapeHtml(w);
  }).join("");
}

/* --------------------------------------------------------------------------
   9. Transformer Embedding Analysis
   -------------------------------------------------------------------------- */
function renderTransformerEmbeddingSection(embedding, paper) {
  // No fabrication allowed: when the backend could not compute a projection,
  // the section renders an explicit empty state rather than plausible-looking
  // model claims.
  const available = Boolean(embedding && Array.isArray(embedding.points));
  const modelName = available ? (embedding.model_name || "Not reported") : null;
  const dimension = available ? (embedding.dimension != null ? embedding.dimension : null) : null;
  const representation = available ? (embedding.representation || "Document-level representation") : null;
  const method = available ? (embedding.method || "2D projection") : null;
  const points = available ? embedding.points : [];

  const factsHtml = available
    ? `
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: var(--sp-3); margin-bottom: var(--sp-3);">
        <div class="analysis-fact">
          <span class="analysis-fact__label">Embedding Model</span>
          <span class="analysis-fact__value">${escapeHtml(modelName)}</span>
        </div>
        <div class="analysis-fact">
          <span class="analysis-fact__label">Embedding Dimension</span>
          <span class="analysis-fact__value">${escapeHtml(String(dimension))}D Features</span>
        </div>
        <div class="analysis-fact">
          <span class="analysis-fact__label">Representation</span>
          <span class="analysis-fact__value">${escapeHtml(representation)}</span>
        </div>
        <div class="analysis-fact">
          <span class="analysis-fact__label">Projection Method</span>
          <span class="analysis-fact__value">${escapeHtml(method)}</span>
        </div>
      </div>`
    : "";

  const canvasHtml = available
    ? `
      <div class="embedding-viz__canvas" id="analysis-embedding-canvas" style="width: 100%; height: 320px; background: var(--bg-inset); border: 1px solid var(--border); border-radius: var(--r-md); position: relative; overflow: hidden;">
        ${renderEmbeddingPointsSvg(points, paper.paper_id)}
        <div id="embed-hover-tooltip" style="display: none; position: absolute; background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--r-sm); padding: 4px 8px; font-size: 11px; pointer-events: none; z-index: 10; box-shadow: 0 4px 12px rgba(0,0,0,0.4);"></div>
      </div>`
    : `
      <div style="width: 100%; padding: var(--sp-8) var(--sp-4); background: var(--bg-inset); border: 1px dashed var(--border-strong); border-radius: var(--r-md); text-align: center; color: var(--text-muted); font-size: var(--fs-sm);">
        Embedding data is not available for this run.
        <div style="font-size: var(--fs-xs); margin-top: var(--sp-2);">No document-vector projection was computed, so no embedding visualisation can be shown.</div>
      </div>`;

  return `
    <section class="card" aria-labelledby="embedding-title">
      <div class="card__head">
        <h2 class="card__title" id="embedding-title">${icon("graph", 18)} Transformer Semantic Representation</h2>
        <div class="embedding-controls" style="display: flex; gap: var(--sp-2);">
          <button class="btn btn--sm ${embeddingViewMode === 'domain' ? 'btn--ghost' : ''}" id="embed-btn-domain" ${available ? "" : "disabled"}>Color by Domain</button>
          <button class="btn btn--sm ${embeddingViewMode === 'similarity' ? 'btn--ghost' : ''}" id="embed-btn-sim" ${available ? "" : "disabled"}>Similarity View</button>
        </div>
      </div>
      <div class="card__body">
        ${factsHtml}
        ${canvasHtml}

        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: var(--sp-2); font-size: var(--fs-xs); color: var(--text-muted);">
          <span>Papers positioned closer together have more similar semantic representations in the embedding space.</span>
          <span style="display: inline-flex; align-items: center; gap: 4px;">
            <span style="width: 8px; height: 8px; border-radius: 50%; background: var(--accent); border: 2px solid #fff;"></span> Current Paper
          </span>
        </div>
      </div>
    </section>
  `;
}

function renderEmbeddingPointsSvg(points, currentPaperId) {
  if (!points.length) {
    return `<div style="display: grid; place-items: center; height: 100%; color: var(--text-muted); font-size: var(--fs-sm);">Embedding space points not available for this run.</div>`;
  }

  const svgPoints = points.map((p) => {
    const cx = Math.max(5, Math.min(95, ((p.x + 1) / 2) * 100));
    const cy = Math.max(5, Math.min(95, ((p.y + 1) / 2) * 100));
    const isCurrent = p.is_current || p.paper_id === currentPaperId;
    const color = isCurrent ? "var(--accent)" : hueFor(p.domain || "Other");
    const r = isCurrent ? 7 : 4;

    return `
      <circle cx="${cx}%" cy="${cy}%" r="${r}" fill="${color}"
        stroke="${isCurrent ? '#ffffff' : 'var(--bg-canvas)'}" stroke-width="${isCurrent ? 2 : 1}"
        opacity="${isCurrent ? 1.0 : 0.75}"
        data-paper-id="${escapeHtml(p.paper_id)}"
        data-title="${escapeHtml(p.title || p.paper_id)}"
        data-domain="${escapeHtml(p.domain || 'Unlabelled')}"
        style="cursor: pointer; transition: transform 0.2s;"
      />
      ${isCurrent ? `<circle cx="${cx}%" cy="${cy}%" r="14" fill="none" stroke="var(--accent)" stroke-width="1.5" opacity="0.6" stroke-dasharray="3 3"/>` : ''}
    `;
  }).join("");

  return `
    <svg width="100%" height="100%" style="overflow: visible;">
      <!-- Grid lines -->
      <line x1="50%" y1="0%" x2="50%" y2="100%" stroke="var(--border)" stroke-dasharray="4 4" opacity="0.5"/>
      <line x1="0%" y1="50%" x2="100%" y2="50%" stroke="var(--border)" stroke-dasharray="4 4" opacity="0.5"/>
      ${svgPoints}
    </svg>
  `;
}

/* --------------------------------------------------------------------------
   11. Similar Papers Section
   -------------------------------------------------------------------------- */
function renderSimilarPapersSection(similarPapers) {
  const items = Array.isArray(similarPapers?.items) ? similarPapers.items : [];
  const method = similarPapers?.method || "model similarity";
  const caveat =
    similarPapers?.caveat ||
    "Similarity indicates semantic/representation proximity and does not imply methodological equivalence.";

  const listHtml = items.slice(0, 6).map((item) => {
    const scoreVal = typeof item.score === "number" && !Number.isNaN(item.score) ? item.score : null;
    const domain = item.label || null;
    const year = item.year != null ? String(item.year) : "—";
    const scoreDisplay = scoreVal != null && scoreVal >= 0 && scoreVal <= 1 ? `${(scoreVal * 100).toFixed(1)}%` : (scoreVal != null ? scoreVal.toFixed(3) : "—");
    const domainChip = domain
      ? `<span class="chip chip--domain" style="background: ${hueFor(domain)}22; color: ${hueFor(domain)}; font-size: 10px; padding: 1px 6px; border-radius: 4px;">${escapeHtml(domain)}</span>`
      : `<span class="chip" style="font-size: 10px; padding: 1px 6px; border-radius: 4px; color: var(--text-muted);">Unlabelled</span>`;

    return `
      <div class="similar-item" data-similar-id="${escapeHtml(item.paper_id)}" style="display: flex; align-items: center; justify-content: space-between; gap: var(--sp-3); padding: 10px 12px; background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--r-md); transition: border-color 0.15s, background 0.15s;">
        <div style="flex: 1; min-width: 0;">
          <div style="font-size: var(--fs-sm); font-weight: 500; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">
            ${escapeHtml(item.title || item.paper_id)}
          </div>
          <div style="font-size: var(--fs-xs); color: var(--text-muted); display: flex; gap: var(--sp-2); align-items: center; margin-top: 2px;">
            ${domainChip}
            <span>${escapeHtml(year)}</span>
            <span>·</span>
            <span>${escapeHtml(item.split || "corpus")}</span>
          </div>
        </div>
        <div style="display: flex; align-items: center; gap: var(--sp-2);">
          <span style="font-size: var(--fs-sm); font-weight: 700; color: var(--accent-ink); font-family: monospace;">${escapeHtml(scoreDisplay)}</span>
          <button class="btn btn--sm btn--ghost similar-open-btn" data-target-id="${escapeHtml(item.paper_id)}">
            Open Analysis →
          </button>
        </div>
      </div>
    `;
  }).join("");

  return `
    <section class="card" aria-labelledby="similar-title">
      <div class="card__head">
        <h2 class="card__title" id="similar-title">${icon("compare", 18)} Semantically Similar Papers</h2>
        <span class="tag" style="font-size: var(--fs-xs);">${escapeHtml(method)}</span>
      </div>
      <div class="card__body" style="display: flex; flex-direction: column; gap: var(--sp-2);">
        ${listHtml || '<div style="font-size: var(--fs-xs); color: var(--text-muted);">No similar papers computed for this document.</div>'}
        <div style="font-size: 11px; color: var(--text-muted); margin-top: var(--sp-2); border-top: 1px solid var(--border); padding-top: var(--sp-2);">
          <em>Note:</em> ${escapeHtml(caveat)}
        </div>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   13. Ground Truth Verification & 14. Paper Insights
   -------------------------------------------------------------------------- */
function renderGroundTruthVerification(groundTruth, prediction) {
  const trueLabel = groundTruth?.true_label;
  const predLabel = prediction?.predicted_label;
  const isUnlabelled = groundTruth?.is_unlabelled ?? (trueLabel == null);
  const isCorrect = groundTruth?.correct ?? (trueLabel && predLabel ? trueLabel === predLabel : null);

  let statusHtml = '<span class="tag" style="color: var(--text-muted);">Unlabelled (Inference)</span>';
  let message = "Ground truth unavailable — prediction cannot currently be evaluated against ground truth.";

  if (!isUnlabelled) {
    if (isCorrect) {
      statusHtml = '<span class="tag tag--good" style="border-color: #22c55e; color: #22c55e;">✓ Prediction Matches Ground Truth</span>';
      message = "The model's highest probability category matches the annotated gold domain.";
    } else {
      statusHtml = '<span class="tag tag--warn" style="border-color: #f59e0b; color: #f59e0b;">⚠ Mismatch with Ground Truth</span>';
      message = `Ground truth annotation is “${escapeHtml(trueLabel)}”, whereas model predicted “${escapeHtml(predLabel)}”.`;
    }
  }

  return `
    <section class="card" aria-labelledby="verification-title">
      <div class="card__head">
        <h2 class="card__title" id="verification-title">${icon("shield", 18)} Prediction Verification</h2>
        ${statusHtml}
      </div>
      <div class="card__body" style="display: flex; flex-direction: column; gap: var(--sp-2);">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--sp-3);">
          <div style="background: var(--bg-inset); padding: var(--sp-2) var(--sp-3); border-radius: var(--r-sm); border: 1px solid var(--border);">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Ground Truth</div>
            <div style="font-size: var(--fs-sm); font-weight: 600; color: var(--text-primary); margin-top: 2px;">
              ${escapeHtml(trueLabel || "Not Available (Unseen Paper)")}
            </div>
          </div>
          <div style="background: var(--bg-inset); padding: var(--sp-2) var(--sp-3); border-radius: var(--r-sm); border: 1px solid var(--border);">
            <div style="font-size: 10px; color: var(--text-muted); text-transform: uppercase;">Model Prediction</div>
            <div style="font-size: var(--fs-sm); font-weight: 600; color: var(--accent-ink); margin-top: 2px;">
              ${escapeHtml(predLabel || "—")}
            </div>
          </div>
        </div>
        <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin: 0; line-height: 1.4;">
          ${escapeHtml(message)}
        </p>
      </div>
    </section>
  `;
}

function renderPaperInsightsSection(insights, prediction) {
  // Fields come from the analysis pipeline only. Absent fields render as an
  // explicit "not available" note — never an inferred plausible value.
  const NA = "Not available from current analysis.";
  const has = insights != null;
  const domain = has && insights.research_domain ? insights.research_domain : (prediction?.predicted_label || null);
  const topic = has && insights.primary_topic ? insights.primary_topic : null;
  const concepts = has && Array.isArray(insights.key_concepts) && insights.key_concepts.length ? insights.key_concepts : null;
  const methodology = has && insights.methodology ? insights.methodology : null;
  const contribution = has && insights.main_contribution ? insights.main_contribution : null;
  const limitations = has && Array.isArray(insights.potential_limitations) && insights.potential_limitations.length
    ? insights.potential_limitations
    : null;

  return `
    <section class="card" aria-labelledby="insights-title">
      <div class="card__head">
        <h2 class="card__title" id="insights-title">${icon("help", 18)} Paper Insights</h2>
      </div>
      <div class="card__body" style="display: flex; flex-direction: column; gap: var(--sp-3);">
        <div>
          <span class="analysis-fact__label">Research Domain</span>
          <span style="font-size: var(--fs-sm); font-weight: 600; color: var(--text-primary); display: block;">${escapeHtml(domain || NA)}</span>
        </div>
        <div>
          <span class="analysis-fact__label">Primary Topic</span>
          <span style="font-size: var(--fs-sm); color: var(--accent-ink); display: block; margin-top: 2px;">${escapeHtml(topic || NA)}</span>
        </div>
        <div>
          <span class="analysis-fact__label">Key Concepts</span>
          <div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px;">
            ${concepts ? concepts.map((c) => `<span class="tag" style="font-size: var(--fs-xs);">${escapeHtml(c)}</span>`).join("") : `<span style="font-size: var(--fs-xs); color: var(--text-muted);">${NA}</span>`}
          </div>
        </div>
        <div>
          <span class="analysis-fact__label">Extracted Methodology</span>
          <span style="font-size: var(--fs-xs); color: var(--text-secondary); display: block; margin-top: 2px;">${escapeHtml(methodology || NA)}</span>
        </div>
        <div>
          <span class="analysis-fact__label">Main Contribution Statement</span>
          <p style="font-size: var(--fs-xs); color: var(--text-primary); line-height: 1.5; margin: 2px 0 0 0; background: var(--bg-inset); padding: 6px 10px; border-radius: var(--r-sm);">
            ${contribution ? `“${escapeHtml(contribution)}”` : NA}
          </p>
        </div>
        <div>
          <span class="analysis-fact__label">Potential Limitations & Challenges</span>
          <ul style="margin: 4px 0 0 0; padding-left: 16px; font-size: var(--fs-xs); color: var(--text-secondary); line-height: 1.4;">
            ${limitations ? limitations.map((l) => `<li>${escapeHtml(l)}</li>`).join("") : `<li>${NA}</li>`}
          </ul>
        </div>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   12. Model Performance (Dataset-Level)
   -------------------------------------------------------------------------- */
function renderModelPerformanceSection(model) {
  const splitSizes = model?.split_sizes || {};

  // Only splits the run actually scored get a pane. No fabricated numbers.
  const splits = [
    { key: "test", label: "Test Set", metrics: model?.test_metrics || null },
    { key: "val", label: "Validation Set", metrics: model?.val_metrics || null },
  ].filter((s) => s.metrics);

  const runTag = model?.run_id
    ? `<span class="tag" style="font-size: var(--fs-xs);">Run: ${escapeHtml(model.run_id)}</span>`
    : "";

  // Honest empty state: the active run scored no held-out splits.
  if (!splits.length) {
    return `
      <section class="card" aria-labelledby="perf-title">
        <div class="card__head">
          <h2 class="card__title" id="perf-title">${icon("database", 18)} Model Performance (Dataset-Level Evaluation)</h2>
          ${runTag}
        </div>
        <div class="card__body">
          <p style="font-size: var(--fs-sm); color: var(--text-muted); margin: 0;">
            Model performance metrics are not available for the active run. The run did not record evaluation
            metrics for any held-out split, so no dataset-level numbers are shown rather than placeholders.
          </p>
        </div>
      </section>
    `;
  }

  // Default to the run's primary split when it was scored, otherwise the first available.
  const defaultKey = splits.some((s) => s.key === (model.primary_split || ""))
    ? model.primary_split
    : splits[0].key;

  const tabsHtml = splits
    .map((s) => {
      const isActive = s.key === defaultKey;
      return `
        <button class="perf-split-tab" type="button" data-split="${s.key}" aria-selected="${isActive}"
          style="padding: 5px 14px; border-radius: var(--r-pill); font-size: var(--fs-xs); font-weight: 600; cursor: pointer;
            border: 1px solid ${isActive ? "var(--accent)" : "var(--border)"};
            background: ${isActive ? "var(--accent-wash, rgba(99,102,241,0.12))" : "var(--bg-surface)"};
            color: ${isActive ? "var(--accent-ink)" : "var(--text-secondary)"};">
          ${escapeHtml(s.label)} · n=${s.metrics.n_samples}
        </button>
      `;
    })
    .join("");

  const panesHtml = splits
    .map((s) => {
      const isActive = s.key === defaultKey;
      return `
        <div class="perf-split-pane" data-split="${s.key}" aria-hidden="${!isActive}" style="display: ${isActive ? "block" : "none"};">
          ${renderPerfSplitPane(s.metrics, s.label)}
        </div>
      `;
    })
    .join("");

  // Split sizes come from the run manifest (dataset.split_sizes). Splits the
  // manifest does not record are shown as "Not reported", never invented.
  const sizeCards = [
    { key: "train", label: "Training Set" },
    { key: "val", label: "Validation Set" },
    { key: "test", label: "Test Set" },
  ]
    .map((s) => {
      const n = splitSizes[s.key];
      const value = Number.isFinite(n) ? String(n) : "—";
      const note = Number.isFinite(n) ? "papers" : "Not reported";
      return `
        <div class="metric-card">
          <div class="metric-card__value" style="color: var(--text-primary);">${value}</div>
          <div class="metric-card__label">${s.label} <span style="color: var(--text-muted); font-weight: 400;">${note}</span></div>
        </div>
      `;
    })
    .join("");

  return `
    <section class="card" aria-labelledby="perf-title">
      <div class="card__head">
        <h2 class="card__title" id="perf-title">${icon("database", 18)} Model Performance (Dataset-Level Evaluation)</h2>
        ${runTag}
      </div>
      <div class="card__body">
        <p style="font-size: var(--fs-xs); color: var(--text-muted); margin-top: 0; margin-bottom: var(--sp-3);">
          <strong>Critical Distinction:</strong> These metrics measure how the model performed overall on held-out
          benchmark splits. They describe dataset-level generalization and are
          <strong>not</strong> the confidence of this paper's individual prediction — model confidence ≠ model accuracy.
        </p>

        <div style="display: flex; gap: var(--sp-2); flex-wrap: wrap; margin-bottom: var(--sp-3);">
          ${tabsHtml}
        </div>

        ${panesHtml}

        <div style="margin-top: var(--sp-4); padding-top: var(--sp-3); border-top: 1px solid var(--border);">
          <div style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;">
            Dataset Splits (from run manifest)
          </div>
          <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: var(--sp-3);">
            ${sizeCards}
          </div>
          <p style="font-size: 10px; color: var(--text-muted); margin: 8px 0 0 0;">
            The training split is not scored: predictions on data the model was trained on are not evidence of performance.
          </p>
        </div>
      </div>
    </section>
  `;
}

/* Renderer for one scored split: headline metrics + confusion matrix + per-class table. */
function renderPerfSplitPane(metrics, splitLabel) {
  const cm = metrics.confusion_matrix;
  let cmTableHtml = "";
  if (cm && Array.isArray(cm.labels) && Array.isArray(cm.counts)) {
    const headerThs = cm.labels.map((l) => `<th style="font-size: 10px; padding: 4px 6px;" title="${escapeHtml(l)}">${escapeHtml(l.slice(0, 10))}..</th>`).join("");
    const bodyRows = cm.counts.map((rowStr, i) => {
      const counts = typeof rowStr === "string" ? rowStr.split(" ") : (Array.isArray(rowStr) ? rowStr : []);
      const cells = counts.map((cnt, j) => {
        const isDiag = i === j;
        return `<td style="font-size: 11px; padding: 4px 6px; text-align: center; background: ${isDiag ? 'var(--accent-wash, rgba(99,102,241,0.15))' : 'transparent'}; font-weight: ${isDiag ? '700' : '400'}; color: ${isDiag ? 'var(--accent-ink)' : 'var(--text-muted)'};">${cnt}</td>`;
      }).join("");
      return `<tr><th style="font-size: 10px; padding: 4px 6px; text-align: left;">${escapeHtml((cm.labels[i] || "").slice(0, 10))}..</th>${cells}</tr>`;
    }).join("");

    cmTableHtml = `
      <div style="margin-top: var(--sp-3); overflow-x: auto;">
        <div style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px;">Confusion Matrix (${escapeHtml(splitLabel)})</div>
        <table style="width: 100%; border-collapse: collapse; font-family: monospace;">
          <thead><tr><th></th>${headerThs}</tr></thead>
          <tbody>${bodyRows}</tbody>
        </table>
      </div>
    `;
  }

  // Per-class performance: only real rows from metrics.per_class.
  const perClass = metrics.per_class && typeof metrics.per_class === "object" ? Object.entries(metrics.per_class) : [];
  const perClassHtml = perClass.length
    ? `
      <div style="margin-top: var(--sp-3); overflow-x: auto;">
        <div style="font-size: var(--fs-xs); font-weight: 600; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px;">Per-Class Performance (${escapeHtml(splitLabel)})</div>
        <table style="width: 100%; border-collapse: collapse;">
          <thead>
            <tr style="border-bottom: 1px solid var(--border);">
              <th style="text-align: left; font-size: 10px; padding: 5px 8px; color: var(--text-muted); text-transform: uppercase;">Class</th>
              <th style="text-align: right; font-size: 10px; padding: 5px 8px; color: var(--text-muted); text-transform: uppercase;">Precision</th>
              <th style="text-align: right; font-size: 10px; padding: 5px 8px; color: var(--text-muted); text-transform: uppercase;">Recall</th>
              <th style="text-align: right; font-size: 10px; padding: 5px 8px; color: var(--text-muted); text-transform: uppercase;">F1</th>
              <th style="text-align: right; font-size: 10px; padding: 5px 8px; color: var(--text-muted); text-transform: uppercase;">Support</th>
            </tr>
          </thead>
          <tbody>
            ${perClass
              .map(([name, m]) => {
                const cell = (x) => (typeof x === "number" && Number.isFinite(x) ? pct(x) : "—");
                const sup = m && Number.isFinite(m.support) ? String(m.support) : "—";
                return `
                  <tr style="border-bottom: 1px solid var(--border);">
                    <td style="font-size: var(--fs-xs); padding: 6px 8px; color: var(--text-primary);">${escapeHtml(name)}</td>
                    <td style="font-size: var(--fs-xs); padding: 6px 8px; text-align: right; color: var(--text-secondary);">${cell(m?.precision)}</td>
                    <td style="font-size: var(--fs-xs); padding: 6px 8px; text-align: right; color: var(--text-secondary);">${cell(m?.recall)}</td>
                    <td style="font-size: var(--fs-xs); padding: 6px 8px; text-align: right; color: var(--text-secondary);">${cell(m?.f1)}</td>
                    <td style="font-size: var(--fs-xs); padding: 6px 8px; text-align: right; color: var(--text-muted);">${sup}</td>
                  </tr>
                `;
              })
              .join("")}
          </tbody>
        </table>
      </div>
    `
    : "";

  const val = (x) => (typeof x === "number" && Number.isFinite(x) ? pct(x) : "—");
  const metricCards = [
    { label: "Accuracy", value: val(metrics.accuracy), color: "var(--accent-ink)" },
    { label: "Macro Precision", value: val(metrics.macro_precision), color: "#22c55e" },
    { label: "Macro Recall", value: val(metrics.macro_recall), color: "#22c55e" },
    { label: "Macro F1", value: val(metrics.macro_f1), color: "#22c55e" },
    { label: "Weighted F1", value: val(metrics.weighted_f1), color: "#22c55e" },
    { label: "Balanced Accuracy", value: val(metrics.balanced_accuracy), color: "var(--text-primary)" },
  ]
    .map(
      (m) => `
        <div class="metric-card">
          <div class="metric-card__value" style="color: ${m.color};">${m.value}</div>
          <div class="metric-card__label">${m.label}</div>
        </div>
      `
    )
    .join("");

  return `
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: var(--sp-3);">
      ${metricCards}
    </div>
    ${cmTableHtml}
    ${perClassHtml}
  `;
}

/* --------------------------------------------------------------------------
   15. Ask This Paper
   -------------------------------------------------------------------------- */
function renderAskPaperSection(paper) {
  const suggestions = [
    "What is the main contribution?",
    "Why was this paper classified into this domain?",
    "Which section influenced the prediction most?",
    "Summarize the methodology.",
    "What are the limitations?",
    "Compare this paper with similar papers.",
  ];

  const chipsHtml = suggestions.map((s) => `
    <button class="ask-suggestion" type="button" data-question="${escapeHtml(s)}" style="padding: 4px 10px; border-radius: var(--r-pill); border: 1px solid var(--border); background: var(--bg-surface); font-size: var(--fs-xs); color: var(--text-secondary); cursor: pointer;">
      ${escapeHtml(s)}
    </button>
  `).join("");

  return `
    <section class="card ask-card" aria-labelledby="full-ask-title">
      <div class="card__head">
        <h2 class="card__title" id="full-ask-title">
          <span style="color: var(--accent-ink); vertical-align: -2px;">${icon("chat", 18)}</span>
          Ask This Paper
        </h2>
        <span style="font-size: var(--fs-xs); color: var(--text-muted);">Passage Retrieval + Groq Synthesis</span>
      </div>
      <div class="card__body" style="display: flex; flex-direction: column; gap: var(--sp-3);">
        <p style="font-size: var(--fs-xs); color: var(--text-secondary); margin: 0;">
          Ask grounded questions about this specific document. Answers are derived from indexed paper passages using semantic retrieval.
        </p>

        <div style="display: flex; flex-wrap: wrap; gap: 6px;" id="analysis-ask-suggestions">
          ${chipsHtml}
        </div>

        <!-- Conversation History -->
        <div id="analysis-chat-history" style="display: flex; flex-direction: column; gap: var(--sp-3); max-height: 380px; overflow-y: auto; padding-right: 4px;"></div>

        <!-- Composer -->
        <form id="analysis-ask-form" style="display: flex; gap: var(--sp-2); margin-top: var(--sp-2);">
          <input
            id="analysis-ask-input"
            type="text"
            autocomplete="off"
            placeholder="Ask anything about this research paper..."
            style="flex: 1; padding: 10px 14px; background: var(--bg-inset); border: 1px solid var(--border); border-radius: var(--r-md); color: var(--text-primary); font-size: var(--fs-sm);"
          />
          <button class="btn btn--ghost" type="submit" id="analysis-ask-submit" style="padding: 10px 16px; font-weight: 600;">
            ${icon("send", 16)} Ask
          </button>
        </form>
      </div>
    </section>
  `;
}

/* --------------------------------------------------------------------------
   Skeleton Loader State
   -------------------------------------------------------------------------- */
function renderSkeletonAnalysis(paperId) {
  return `
    <div class="analysis-page" style="padding: var(--sp-4) 0; display: flex; flex-direction: column; gap: var(--sp-5);">
      <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--border); padding-bottom: var(--sp-4);">
        <div style="display: flex; flex-direction: column; gap: 6px;">
          <div style="width: 140px; height: 16px; background: var(--bg-inset); border-radius: 4px;"></div>
          <div style="width: 320px; height: 32px; background: var(--bg-inset); border-radius: 6px;"></div>
        </div>
        <div style="display: flex; gap: var(--sp-2);">
          <div style="width: 120px; height: 36px; background: var(--bg-inset); border-radius: 6px;"></div>
          <div style="width: 100px; height: 36px; background: var(--bg-inset); border-radius: 6px;"></div>
        </div>
      </div>

      <div class="card" style="padding: var(--sp-5); height: 120px; background: var(--bg-surface);">
        <div style="width: 50%; height: 20px; background: var(--bg-inset); border-radius: 4px; margin-bottom: 12px;"></div>
        <div style="width: 80%; height: 14px; background: var(--bg-inset); border-radius: 4px;"></div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: var(--sp-4);">
        <div class="card" style="padding: var(--sp-5); height: 260px; background: var(--bg-surface);"></div>
        <div class="card" style="padding: var(--sp-5); height: 260px; background: var(--bg-surface);"></div>
      </div>

      <div class="card" style="padding: var(--sp-5); height: 340px; background: var(--bg-surface);"></div>
    </div>
  `;
}

/* ==========================================================================
   Interactivity & Event Wiring
   ========================================================================== */

function wireAnalysisInteractions(container, analysis) {
  // 1. Back button
  $("#analysis-back-btn", container)?.addEventListener("click", () => {
    if (onNavigateBackCallback) onNavigateBackCallback();
  });

  // 2. Download Report
  $("#analysis-download-btn", container)?.addEventListener("click", () => {
    generateAndPrintReport(analysis);
  });

  // 3. Export JSON
  $("#analysis-export-json-btn", container)?.addEventListener("click", () => {
    downloadAnalysisJson(analysis);
  });

  // 4. Evidence Tabs
  const tabs = $$(".evidence-tab", container);
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => {
        t.setAttribute("aria-selected", "false");
        t.classList.remove("evidence-tab--active");
      });
      tab.setAttribute("aria-selected", "true");

      const targetId = tab.dataset.tab;
      $$(".evidence-panel", container).forEach((panel) => {
        if (panel.id === targetId) {
          panel.style.display = "block";
          panel.setAttribute("aria-hidden", "false");
        } else {
          panel.style.display = "none";
          panel.setAttribute("aria-hidden", "true");
        }
      });
    });
  });

  // 5. Performance split tabs (Test / Validation)
  const perfTabs = $$(".perf-split-tab", container);
  perfTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const key = tab.dataset.split;
      perfTabs.forEach((t) => {
        const tActive = t === tab;
        t.setAttribute("aria-selected", String(tActive));
        t.style.border = `1px solid ${tActive ? "var(--accent)" : "var(--border)"}`;
        t.style.background = tActive ? "var(--accent-wash, rgba(99,102,241,0.12))" : "var(--bg-surface)";
        t.style.color = tActive ? "var(--accent-ink)" : "var(--text-secondary)";
      });
      $$(".perf-split-pane", container).forEach((pane) => {
        const paneActive = pane.dataset.split === key;
        pane.style.display = paneActive ? "block" : "none";
        pane.setAttribute("aria-hidden", String(!paneActive));
      });
    });
  });

  // 6. Section items in Hierarchical Attention
  const sectionItems = $$(".section-item", container);
  sectionItems.forEach((secItem) => {
    secItem.addEventListener("click", () => {
      const secName = secItem.dataset.sectionName;
      activeSelectedSection = secName;

      // Update active selection styling
      sectionItems.forEach((si) => {
        si.style.borderColor = si === secItem ? "var(--accent)" : "var(--border)";
        const head = si.firstElementChild;
        if (head) {
          head.style.background = si === secItem ? "var(--accent-wash, rgba(99,102,241,0.1))" : "var(--bg-surface)";
        }
      });

      // Filter and update sentences display
      const allSentences = analysis.attention?.sentences || [];
      const filtered = allSentences.filter((s) => {
        const matchName = (s.section_name || s.canonical_name || "").toLowerCase();
        const sel = secName.toLowerCase();
        return matchName === sel || matchName.includes(sel) || sel.includes(matchName);
      });
      const displaySentences = filtered.length ? filtered.slice(0, 6) : allSentences.slice(0, 6);

      const sentenceMount = $("#analysis-sentence-mount", container);
      if (sentenceMount) {
        sentenceMount.innerHTML = displaySentences.map((s, idx) => `
          <div class="sentence-item" data-sentence-index="${idx}" style="padding: 10px; background: var(--bg-surface); border: 1px solid var(--border); border-left: 4px solid var(--accent); border-radius: var(--r-sm); margin-bottom: 8px; cursor: pointer;">
            <div style="display: flex; justify-content: space-between; align-items: center; font-size: 10px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 4px;">
              <span>Sentence ${idx + 1} (${escapeHtml(s.canonical_name || s.section_name || "Section")})</span>
              <span class="tag" style="font-size: 10px; padding: 1px 6px;">Attention: ${(s.weight || 0.1).toFixed(3)}</span>
            </div>
            <div style="font-size: var(--fs-sm); color: var(--text-primary); line-height: 1.5;">
              “${escapeHtml(s.text)}”
            </div>
          </div>
        `).join("");

        wireSentenceClicks(sentenceMount, displaySentences, analysis, container);
      }

      if (displaySentences.length) {
        updateTokenDisplay(container, displaySentences[0].text, analysis.attention?.terms || []);
      }
    });
  });

  // Initial sentence clicks
  const sentenceMount = $("#analysis-sentence-mount", container);
  if (sentenceMount) {
    const allSentences = analysis.attention?.sentences || [];
    wireSentenceClicks(sentenceMount, allSentences.slice(0, 6), analysis, container);
  }

  // 6. Embedding View Toggles
  const btnDomain = $("#embed-btn-domain", container);
  const btnSim = $("#embed-btn-sim", container);
  btnDomain?.addEventListener("click", () => {
    embeddingViewMode = "domain";
    btnDomain.classList.add("btn--ghost");
    btnSim?.classList.remove("btn--ghost");
  });
  btnSim?.addEventListener("click", () => {
    embeddingViewMode = "similarity";
    btnSim.classList.add("btn--ghost");
    btnDomain?.classList.remove("btn--ghost");
  });

  // 7. Similar Paper buttons
  $$(".similar-open-btn", container).forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      const targetId = btn.dataset.targetId;
      if (targetId && onSelectPaperCallback) {
        onSelectPaperCallback(targetId);
      }
    });
  });

  // 8. Ask This Paper Form
  const askForm = $("#analysis-ask-form", container);
  const askInput = $("#analysis-ask-input", container);
  const chatHistory = $("#analysis-chat-history", container);

  askForm?.addEventListener("submit", async (e) => {
    e.preventDefault();
    const q = askInput.value.trim();
    if (!q) return;
    askInput.value = "";
    await handleAskQuestion(q, analysis.paper.paper_id, chatHistory);
  });

  $$("#analysis-ask-suggestions .ask-suggestion", container).forEach((chip) => {
    chip.addEventListener("click", async () => {
      const q = chip.dataset.question;
      if (q) {
        await handleAskQuestion(q, analysis.paper.paper_id, chatHistory);
      }
    });
  });
}

function wireSentenceClicks(mount, sentences, analysis, container) {
  $$(".sentence-item", mount).forEach((item, idx) => {
    item.addEventListener("click", () => {
      $$(".sentence-item", mount).forEach((si) => {
        si.style.background = si === item ? "var(--accent-wash, rgba(99,102,241,0.1))" : "var(--bg-surface)";
        si.style.borderColor = si === item ? "var(--accent)" : "var(--border)";
      });
      const selected = sentences[idx];
      if (selected) {
        activeSelectedSentence = selected;
        updateTokenDisplay(container, selected.text, analysis.attention?.terms || []);
      }
    });
  });
}

function updateTokenDisplay(container, text, terms) {
  const tokenDisplay = $("#analysis-token-display", container);
  if (tokenDisplay) {
    tokenDisplay.innerHTML = renderTokenAttentionHighlighting(text, terms);
  }
}

async function handleAskQuestion(question, paperId, chatMount) {
  if (!chatMount) return;

  // Render user question
  const userBubble = document.createElement("div");
  userBubble.style.cssText = "align-self: flex-end; background: var(--accent); color: #ffffff; padding: 8px 14px; border-radius: var(--r-md) var(--r-md) 2px var(--r-md); max-width: 80%; font-size: var(--fs-sm);";
  userBubble.textContent = question;
  chatMount.appendChild(userBubble);

  // Render loading assistant bubble with staged progress text: retrieval
  // finishes quickly, synthesis takes longer, so the label advances once.
  const botBubble = document.createElement("div");
  botBubble.style.cssText = "align-self: flex-start; background: var(--bg-surface); border: 1px solid var(--border); padding: 10px 14px; border-radius: var(--r-md) var(--r-md) var(--r-md) 2px; max-width: 85%; font-size: var(--fs-sm); color: var(--text-primary);";
  botBubble.innerHTML = `<span style="color: var(--text-muted); font-size: var(--fs-xs);">Retrieving relevant passages...</span>`;
  chatMount.appendChild(botBubble);
  chatMount.scrollTop = chatMount.scrollHeight;

  const stageTimer = setTimeout(() => {
    botBubble.innerHTML = `<span style="color: var(--text-muted); font-size: var(--fs-xs);">Generating grounded answer...</span>`;
  }, 1500);

  // One question in flight at a time: the Ask button and the suggestion
  // chips are disabled until the answer (or the error) has rendered.
  const askSubmit = $("#analysis-ask-submit");
  const askInput = $("#analysis-ask-input");
  const chips = $$("#analysis-ask-suggestions .ask-suggestion");
  if (askSubmit) askSubmit.disabled = true;
  chips.forEach((chip) => { chip.disabled = true; chip.style.opacity = "0.55"; });

  try {
    const res = await ask(paperId, question);
    const answerText = res.answer || "No grounded answer could be retrieved.";
    const sources = Array.isArray(res.evidence) ? res.evidence : [];

    const sourcesHtml = sources.length
      ? `<div style="margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border); font-size: 11px; color: var(--text-secondary);">
          <strong style="color: var(--text-muted); text-transform: uppercase;">Evidence (retrieved passages):</strong>
          ${sources.slice(0, 3).map((s) => `<div style="margin-top: 4px; padding-left: 8px; border-left: 2px solid var(--accent); font-style: italic;">“${escapeHtml(s.passage?.slice(0, 180) || "")}..”</div><div style="padding-left: 8px; color: var(--text-muted);">${escapeHtml(s.source_section || "Paper content")}${typeof s.confidence === "number" ? ` · relevance ${s.confidence.toFixed(2)}` : ""}</div>`).join("")}
        </div>`
      : "";

    botBubble.innerHTML = `
      <div style="line-height: 1.5;">${escapeHtml(answerText)}</div>
      ${sourcesHtml}
    `;
  } catch (err) {
    // `detail` carries the server's user-facing wording (configuration,
    // rate limit, synthesis failure); `message` is only the HTTP line.
    const friendly = err?.detail || err?.message || "Request failed";
    botBubble.innerHTML = `<span style="color: var(--status-critical, #ef4444); font-size: var(--fs-xs);">${escapeHtml(friendly)}</span>`;
  } finally {
    clearTimeout(stageTimer);
    if (askSubmit) askSubmit.disabled = false;
    chips.forEach((chip) => { chip.disabled = false; chip.style.opacity = ""; });
    if (askInput) askInput.focus();
  }

  chatMount.scrollTop = chatMount.scrollHeight;
}

/* ==========================================================================
   Export JSON & Download Report Utilities
   ========================================================================== */

function downloadAnalysisJson(analysis) {
  const exportPayload = {
    paper_id: analysis.paper?.paper_id,
    title: analysis.paper?.title,
    metadata: {
      authors: analysis.paper?.authors_short,
      year: analysis.paper?.year,
      venue: analysis.paper?.venue,
      word_count: analysis.paper?.word_count,
      references: analysis.paper?.n_references,
    },
    prediction: {
      predicted_label: analysis.prediction?.predicted_label,
      confidence: analysis.prediction?.confidence,
      confidence_kind: analysis.prediction?.confidence_kind,
      needs_review: analysis.prediction?.needs_review,
      review_band: analysis.prediction?.review_band,
    },
    probabilities: analysis.prediction?.probabilities,
    review_status: analysis.review,
    section_attention: analysis.attention?.section_attention,
    sentence_attention: analysis.attention?.sentences,
    token_attention: analysis.attention?.tokens || analysis.attention?.terms,
    embeddings: analysis.embedding,
    similar_papers: analysis.similar_papers,
    model_metadata: analysis.model,
    ground_truth: analysis.ground_truth,
    insights: analysis.insights,
    exported_at: new Date().toISOString(),
  };

  const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportPayload, null, 2));
  const downloadAnchor = document.createElement("a");
  downloadAnchor.setAttribute("href", dataStr);
  downloadAnchor.setAttribute("download", `paper_analysis_${analysis.paper?.paper_id || "export"}.json`);
  document.body.appendChild(downloadAnchor);
  downloadAnchor.click();
  downloadAnchor.remove();
}

function generateAndPrintReport(analysis) {
  const paper = analysis.paper || {};
  const pred = analysis.prediction || {};
  const review = analysis.review || {};
  const model = analysis.model || {};
  const attention = analysis.attention || {};
  const sectionAttn = attention.section_attention?.sections || [];
  const terms = (attention.terms || []).slice(0, 8);
  const similar = (analysis.similar_papers || []).slice(0, 5);
  const perfSplits = [
    { label: "Test", m: model.test_metrics },
    { label: "Validation", m: model.val_metrics },
  ].filter((s) => s.m);
  const fmt = (x) => (typeof x === "number" && Number.isFinite(x) ? pct(x) : "—");

  const printWindow = window.open("", "_blank");
  if (!printWindow) {
    alert("Please allow pop-ups to open the printable analysis report.");
    return;
  }

  const html = `
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>Academic Research Intelligence Report — ${escapeHtml(paper.title || paper.paper_id)}</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; color: #1e293b; padding: 30px; max-width: 860px; margin: 0 auto; }
        h1 { font-size: 24px; margin-bottom: 4px; color: #0f172a; }
        h2 { font-size: 16px; border-bottom: 2px solid #e2e8f0; padding-bottom: 4px; margin-top: 24px; color: #334155; }
        .meta { color: #64748b; font-size: 13px; margin-bottom: 20px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
        .box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 12px; }
        .label { font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; }
        .val { font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 2px; }
        table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
        th, td { border: 1px solid #cbd5e1; padding: 6px 10px; text-align: left; }
        th { background: #f1f5f9; }
        .caveat { font-size: 11px; color: #64748b; font-style: italic; margin-top: 20px; }
      </style>
    </head>
    <body>
      <h1>Full Paper Analysis Report</h1>
      <div class="meta">
        <strong>Academic Research Intelligence System (ARIS)</strong><br>
        Document: <strong>${escapeHtml(paper.title || paper.paper_id)}</strong> (${paper.year ?? "—"}) · Paper ID: ${escapeHtml(paper.paper_id)}<br>
        Generated: ${new Date().toUTCString()}
      </div>

      <div class="grid">
        <div class="box">
          <div class="label">Predicted Research Domain</div>
          <div class="val" style="color: #4f46e5;">${escapeHtml(pred.predicted_label || "—")}</div>
          <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Confidence: ${pct(pred.confidence)}</div>
        </div>
        <div class="box">
          <div class="label">Review Assessment</div>
          <div class="val" style="color: ${review.needs_review ? '#d97706' : '#16a34a'};">
            ${review.needs_review ? "Review Recommended" : "High Confidence"}
          </div>
          <div style="font-size: 12px; color: #64748b; margin-top: 4px;">Confidence Gap: ${review.confidence_gap != null ? pct(review.confidence_gap) : "—"}</div>
        </div>
      </div>

      <h2>Classification Probabilities</h2>
      <table>
        <thead><tr><th>Domain Class</th><th>Score</th></tr></thead>
        <tbody>
          ${(pred.probabilities || []).map((p) => `<tr><td>${escapeHtml(p.label)}</td><td>${pct(p.score)}</td></tr>`).join("")}
        </tbody>
      </table>

      <h2>Section Attention</h2>
      ${
        sectionAttn.length
          ? `<table>
        <thead><tr><th>Section</th><th>Attention Score</th></tr></thead>
        <tbody>
          ${sectionAttn
            .map((s) => `<tr><td>${escapeHtml(s.name || s.canonical_name || "—")}</td><td>${typeof s.weight === "number" ? s.weight.toFixed(3) : "—"}</td></tr>`)
            .join("")}
        </tbody>
      </table>`
          : `<p style="font-size: 13px; color: #64748b;">Section attention is not available for this paper / model combination.</p>`
      }

      <h2>Keyword Evidence (Top Terms)</h2>
      ${
        terms.length
          ? `<table>
        <thead><tr><th>Term</th><th>Contribution</th></tr></thead>
        <tbody>
          ${terms
            .map((t) => `<tr><td>${escapeHtml(t.term)}</td><td>${typeof (t.contribution ?? t.weight) === "number" ? (t.contribution ?? t.weight).toFixed(3) : "—"}</td></tr>`)
            .join("")}
        </tbody>
      </table>`
          : `<p style="font-size: 13px; color: #64748b;">Keyword evidence is not available for this prediction.</p>`
      }

      <h2>Semantically Similar Papers</h2>
      ${
        similar.length
          ? `<table>
        <thead><tr><th>Paper</th><th>Similarity</th><th>Predicted Domain</th><th>Year</th></tr></thead>
        <tbody>
          ${similar
            .map(
              (s) => `<tr>
            <td>${escapeHtml(s.title || s.paper_id || "—")}</td>
            <td>${typeof s.similarity === "number" ? s.similarity.toFixed(3) : "—"}</td>
            <td>${escapeHtml(s.predicted_label || s.domain || "—")}</td>
            <td>${s.year ?? "—"}</td>
          </tr>`
            )
            .join("")}
        </tbody>
      </table>
      <p style="font-size: 11px; color: #64748b; font-style: italic;">Similarity indicates semantic/representation similarity and does not imply methodological equivalence.</p>`
          : `<p style="font-size: 13px; color: #64748b;">No similar papers computed for this paper.</p>`
      }

      <h2>Model Performance (Dataset-Level)</h2>
      ${
        perfSplits.length
          ? perfSplits
              .map(
                (s) => `
        <p style="font-size: 13px; margin-bottom: 4px;"><strong>${escapeHtml(s.label)} Split</strong> (${s.m.n_samples ?? "?"} papers)</p>
        <table>
          <tbody>
            <tr><td>Accuracy</td><td>${fmt(s.m.accuracy)}</td></tr>
            <tr><td>Macro Precision</td><td>${fmt(s.m.macro_precision)}</td></tr>
            <tr><td>Macro Recall</td><td>${fmt(s.m.macro_recall)}</td></tr>
            <tr><td>Macro F1</td><td>${fmt(s.m.macro_f1)}</td></tr>
            <tr><td>Weighted F1</td><td>${fmt(s.m.weighted_f1)}</td></tr>
            <tr><td>Balanced Accuracy</td><td>${fmt(s.m.balanced_accuracy)}</td></tr>
          </tbody>
        </table>`
              )
              .join("")
          : `<p style="font-size: 13px; color: #64748b;">Evaluation metrics were not recorded for this run's held-out splits.</p>`
      }

      <h2>Model & System Facts</h2>
      <p style="font-size: 13px;">
        <strong>Model Head:</strong> ${escapeHtml(model.model_display_name || "Not reported")}<br>
        <strong>Representation:</strong> ${escapeHtml(analysis.embedding?.model_name || analysis.embedding?.representation || "Not available from current analysis")}<br>
        <strong>Run ID:</strong> ${escapeHtml(model.run_id || "—")} · Similarity Method: ${escapeHtml(model.similarity_method || "—")}<br>
        <strong>Word Count:</strong> ${paper.word_count ? paper.word_count.toLocaleString() : "—"} words · References: ${paper.n_references ?? "—"}
      </p>
      <p style="font-size: 11px; color: #64748b;">
        Dataset-level metrics describe overall model generalization on held-out splits and are distinct from this paper's individual prediction confidence.
      </p>

      <div class="caveat">
        Research Integrity Note: Attention scores indicate relative model focus and do not constitute definitive causal explanation. Model confidence reflects internal decision margin rather than guaranteed ground-truth accuracy.
      </div>
      <script>window.onload = function() { window.print(); };</script>
    </body>
    </html>
  `;

  printWindow.document.write(html);
  printWindow.document.close();
}
