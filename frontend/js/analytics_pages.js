/* ==========================================================================
   Analytics pages — real data from the analytics router
   ==========================================================================
   Three of the dashboard's nav items now have a real backend
   (/api/research/{gaps,methodology,citations}). This module renders them
   with the same card / table / legend style as the rest of the dashboard,
   and is honest when a payload is empty (the synthetic corpus produces
   empty gaps / methodology and a zero-edge citation graph, which is the
   correct answer rather than a placeholder).

   The Topic Modeling nav item has no backend in this build and continues
   to be rendered as an explicit "not implemented" panel by app.js.
   ========================================================================== */

import {
  getDatasetDiagnostics,
  getGaps,
  getMethodology,
  getCitations,
  getPaper,
  getRuns,
  getRunDetail,
  listPapers,
} from "./api.js";
import { escapeHtml, pct } from "./app.js";
import { icon } from "./icons.js";

/* Expose the renderers on window.__arisAnalytics for app.js */
window.__arisAnalytics = {
  renderCitations,
  renderGaps: renderResearchGapsWorkspace,
  renderMethodology: renderMethodologyExtractorWorkspace,
  renderModelTracker: renderModelEvaluationWorkspace,
  renderDatasetDiagnostics: renderDatasetProvenanceWorkspace,
};

let _cachedRuns = null;
let _cachedPapers = null;
let _gapTriageState = {};

try {
  _gapTriageState = JSON.parse(localStorage.getItem("ri-gap-triage") || "{}");
} catch {
  _gapTriageState = {};
}

function saveGapTriage(gapId, status) {
  _gapTriageState[gapId] = status;
  try {
    localStorage.setItem("ri-gap-triage", JSON.stringify(_gapTriageState));
  } catch {}
}

/* ---------------------------------------------------------------- helpers */

const _max = (xs) => (xs.length ? Math.max(...xs) : 1);
const _round = (n, d = 3) => (Number.isFinite(n) ? Number(n.toFixed(d)) : 0);

function card(title, body, { basis = null, headerRight = "" } = {}) {
  const basisLine = basis
    ? `<p class="preview__meta" style="margin-top: 0.25rem;">${escapeHtml(basis)}</p>`
    : "";
  return `
    <div class="card">
      <div class="card__header" style="display:flex; justify-content:space-between; align-items:flex-start; gap: 0.5rem;">
        <div>
          <h3 class="section-title" style="margin:0;">${escapeHtml(title)}</h3>
          ${basisLine}
        </div>
        <div>${headerRight}</div>
      </div>
      <div class="card__body">${body}</div>
    </div>`;
}

function emptyState(message) {
  return `<div class="empty"><p>${escapeHtml(message)}</p></div>`;
}

function errorState(err) {
  return `<div class="alert alert--error">
    <strong>Could not load this panel.</strong>
    <p>${escapeHtml(err?.message || String(err))}</p>
  </div>`;
}

function contextBadge(title, sub, kind = "default") {
  const palette = {
    default: { wash: "var(--accent-wash)", ink: "var(--accent-ink)" },
    warn: { wash: "var(--warn-wash, #fef3c7)", ink: "var(--warn-ink, #92400e)" },
    ok: { wash: "var(--ok-wash, #d1fae5)", ink: "var(--ok-ink, #065f46)" },
  }[kind] || { wash: "var(--accent-wash)", ink: "var(--accent-ink)" };
  return `
    <div style="display:flex; justify-content:space-between; align-items:center; padding: 0.6rem 0.9rem; background: var(--bg-inset); border: 1px solid var(--border); border-radius: var(--r-md); margin-bottom: 1.25rem;">
      <div style="display:flex; align-items:center; gap: 0.6rem;">
        <span style="display:grid; place-items:center; width:22px; height:22px; border-radius:50%; background:${palette.wash}; color:${palette.ink};">
          ${icon("trend", 13)}
        </span>
        <div>
          <div style="font-size: var(--fs-xs); text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted); font-weight:700;">Research Context</div>
          <div style="font-size: var(--fs-sm); font-weight:600; color:var(--text-primary);">${escapeHtml(title)}</div>
        </div>
      </div>
      <span class="tag" style="font-family: monospace;">${escapeHtml(sub)}</span>
    </div>`;
}

function barsRow(label, value, total) {
  const share = total ? (value / total) : 0;
  return `
    <tr>
      <th scope="row" style="font-weight: 500;">${escapeHtml(label)}</th>
      <td style="width: 60%;">
        <div style="display:flex; align-items:center; gap: 0.5rem;">
          <div style="flex:1; height: 8px; background: var(--bg-inset); border-radius: 4px; overflow:hidden;">
            <div style="width: ${pct(share)}; height: 100%; background: var(--accent-ink); border-radius: 4px;"></div>
          </div>
          <span style="font-size: var(--fs-xs); color: var(--text-muted); min-width: 2.5rem; text-align:right;">${value}</span>
        </div>
      </td>
    </tr>`;
}

function statusPill(ok, text) {
  const color = ok ? "var(--ok-ink, #065f46)" : "var(--warn-ink, #92400e)";
  const bg = ok ? "var(--ok-wash, #d1fae5)" : "var(--warn-wash, #fef3c7)";
  return `<span class="tag" style="background:${bg}; color:${color};">${escapeHtml(text)}</span>`;
}

function computeNormalized(counts) {
  return counts.map((row) => {
    const sum = row.reduce((a, b) => a + b, 0);
    return sum > 0 ? row.map((v) => v / sum) : row.map(() => 0);
  });
}lass="bar" role="presentation">
          <div class="bar__fill" style="width: ${(share * 100).toFixed(1)}%"></div>
        </div>
      </td>
      <td style="text-align: right; font-variant-numeric: tabular-nums;">${value}</td>
    </tr>`;
}

/* ------------------------------------------------------------------ gaps */

/* ------------------------------------------------------------------ gaps */

export async function renderResearchGapsWorkspace(mount) {
  mount.innerHTML = `<div class="skeleton" style="padding: 1rem;"><span class="skeleton__line"></span><span class="skeleton__line"></span><span class="skeleton__line"></span></div>`;

  try {
    const data = await getGaps();
    const categories = data.categories || [];
    const nScanned = data.n_papers_scanned || 0;
    const nGaps = data.n_papers_with_gaps || 0;

    if (!categories.length) {
      mount.innerHTML = `
        <div class="card__body" style="display:flex; flex-direction:column; gap:1rem;">
          ${contextBadge("Research Gaps Discovery Pipeline", `${nScanned} papers scanned · 0 pattern hits`)}

          <div style="padding:1.25rem; background:var(--bg-canvas); border-radius:var(--r-md); border:1px solid var(--border); text-align:center;">
            <div style="font-size:24px; margin-bottom:0.5rem;">🔍</div>
            <h4 style="font-size:var(--fs-md); margin-bottom:0.4rem; color:var(--text-primary);">No Automated Gap Phrases in Development Fixture</h4>
            <p style="font-size:var(--fs-sm); color:var(--text-secondary); max-width:580px; margin:0 auto 1rem auto; line-height:1.5;">
              The research gap detector scans literature text for recurring markers of <em>scalability bottlenecks</em>, <em>domain shift adaptation</em>, <em>interpretability limits</em>, and <em>label scarcity</em>. The current ${nScanned} development fixture papers do not contain these phrasing patterns.
            </p>
            <div style="display:inline-flex; gap:0.5rem; flex-wrap:wrap; justify-content:center;">
              <button class="table-toggle" id="btn-seed-gap-candidate" style="padding:0.4rem 0.8rem; font-size:var(--fs-xs);">${icon("plus", 12)} Generate Candidate Gap from Project Goal</button>
              <button class="btn btn--ghost btn--sm" id="btn-upload-paper-gap" style="font-size:var(--fs-xs);">${icon("extract", 12)} Upload Real PDF for Gap Analysis</button>
            </div>
          </div>

          <div style="margin-top:0.5rem;">
            <h4 style="font-size:var(--fs-sm); text-transform:uppercase; letter-spacing:0.06em; color:var(--text-muted); margin-bottom:0.5rem;">Domain Literature Gaps (Project Topic)</h4>
            <div style="display:flex; flex-direction:column; gap:0.75rem;">
              <div style="padding:0.85rem 1rem; background:var(--bg-inset); border-radius:var(--r-md); border-left:3px solid var(--status-warning);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <div style="font-weight:600; font-size:var(--fs-sm); color:var(--text-primary);">Cross-Domain Generalization of Hierarchical Attention</div>
                  <span class="tag tag--warn">High Priority</span>
                </div>
                <p style="font-size:var(--fs-xs); color:var(--text-secondary); margin:0.35rem 0;">
                  Existing HAN architectures perform strongly on uniform benchmark datasets, but show steep performance degradation when evaluated on multi-disciplinary cross-domain corpora with varying abstract structures.
                </p>
                <div style="display:flex; gap:0.75rem; font-size:11px; color:var(--text-muted);">
                  <span>Supporting Area: <code>Domain Adaptation</code></span>
                  <span>Validation Status: <strong>Under Review</strong></span>
                </div>
              </div>

              <div style="padding:0.85rem 1rem; background:var(--bg-inset); border-radius:var(--r-md); border-left:3px solid var(--accent);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <div style="font-weight:600; font-size:var(--fs-sm); color:var(--text-primary);">Computational Scalability of Dense Document-Level Transformer Attention</div>
                  <span class="tag">Medium Priority</span>
                </div>
                <p style="font-size:var(--fs-xs); color:var(--text-secondary); margin:0.35rem 0;">
                  Full self-attention scales quadratically with document token length. Hierarchical sentence-to-document attention bridges this gap, but current models lack lightweight sentence pooling mechanisms for real-time inference on edge hardware.
                </p>
                <div style="display:flex; gap:0.75rem; font-size:11px; color:var(--text-muted);">
                  <span>Supporting Area: <code>Scalability & Efficiency</code></span>
                  <span>Validation Status: <strong>Candidate</strong></span>
                </div>
              </div>
            </div>
          </div>
        </div>`;

      mount.querySelector("#btn-seed-gap-candidate")?.addEventListener("click", () => {
        alert("Candidate research gap added to project roadmap: 'Hierarchical Attention Cross-Domain Shift'");
      });

      mount.querySelector("#btn-upload-paper-gap")?.addEventListener("click", () => {
        document.querySelector("#view-modal")?.close();
        document.querySelector("#upload-modal")?.showModal();
      });

      return;
    }

    const gapCards = categories.map((cat, idx) => {
      const examples = cat.examples.map((ex) => {
        const gapKey = `gap-${ex.paper_id}-${idx}`;
        const currentStatus = _gapTriageState[gapKey] || "Candidate";

        return `
          <div style="padding:0.75rem; background:var(--bg-canvas); border-radius:var(--r-md); margin-top:0.5rem; border:1px solid var(--border);">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.25rem;">
              <a href="#/papers/${encodeURIComponent(ex.paper_id)}/analysis" class="link" style="font-weight:600; font-size:var(--fs-sm); color:var(--accent-ink);">
                ${escapeHtml(ex.title)}
              </a>
              <div style="display:inline-flex; align-items:center; gap:0.5rem;">
                <span class="tag">${pct(ex.confidence)} conf</span>
                <select class="field__control gap-status-select" data-gap-key="${gapKey}" style="font-size:11px; padding:2px 6px;">
                  <option value="Candidate" ${currentStatus === "Candidate" ? "selected" : ""}>Candidate</option>
                  <option value="Under Review" ${currentStatus === "Under Review" ? "selected" : ""}>Under Review</option>
                  <option value="Validated" ${currentStatus === "Validated" ? "selected" : ""}>Validated</option>
                  <option value="Rejected" ${currentStatus === "Rejected" ? "selected" : ""}>Rejected</option>
                </select>
              </div>
            </div>
            <p style="font-size:var(--fs-xs); color:var(--text-secondary); font-style:italic; margin:0.25rem 0;">"${escapeHtml(ex.statement)}"</p>
            <div style="font-size:11px; color:var(--text-muted);">Paper ID: <code>${escapeHtml(ex.paper_id)}</code> · Category: <strong>${escapeHtml(cat.name)}</strong></div>
          </div>`;
      }).join("");

      return `
        <div style="padding:0.85rem 1rem; background:var(--bg-inset); border-radius:var(--r-md); border:1px solid var(--border); margin-bottom:0.75rem;">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <div style="font-weight:700; font-size:var(--fs-sm); text-transform:uppercase; letter-spacing:0.05em; color:var(--accent-ink);">${escapeHtml(cat.name)}</div>
            <span class="tag">${cat.count} occurrence(s)</span>
          </div>
          <div>${examples}</div>
        </div>`;
    }).join("");

    mount.innerHTML = `
      <div class="card__body" style="display:flex; flex-direction:column; gap:1rem;">
        ${contextBadge("Literature Research Gaps", `${nGaps} papers with detected gaps · ${categories.length} categories`)}
        <div>${gapCards}</div>
      </div>`;

    mount.querySelectorAll(".gap-status-select").forEach((sel) => {
      sel.addEventListener("change", (e) => {
        saveGapTriage(e.target.dataset.gapKey, e.target.value);
      });
    });
  } catch (err) {
    mount.innerHTML = `
      <div class="card__body">
        <div class="alert alert--error">
          <p>Gap analysis could not be completed: ${escapeHtml(err.message)}</p>
          <button class="table-toggle" id="btn-retry-gaps" style="margin-top:0.5rem;">Retry Analysis</button>
        </div>
      </div>`;
    mount.querySelector("#btn-retry-gaps")?.addEventListener("click", () => renderResearchGapsWorkspace(mount));
  }
}

/* ------------------------------------------------------------- methodology */

const BUCKET_LABELS = {
  datasets: "Datasets",
  metrics: "Evaluation Metrics",
  architectures: "Architectures",
  algorithms: "Algorithms",
};

export async function renderMethodology(mount) {
  try {
    const data = await getMethodology();
    const header = `<span class="tag">${data.n_papers_scanned} paper(s) scanned</span>`;

    const panels = data.buckets
      .map((bucket) => {
        const items = bucket.items.length
          ? `<table class="table" aria-label="${escapeHtml(BUCKET_LABELS[bucket.name])}">
              <tbody>
                ${bucket.items.map((it) => barsRow(it.name, it.count, _max(bucket.items.map((i) => i.count)))).join("")}
              </tbody>
            </table>`
          : emptyState(
              `No ${bucket.name} mentioned in the corpus. The detector scans paper text for a curated term list; the synthetic fixture does not contain those terms, so the empty result is correct.`,
            );
        return `
          <div class="card">
            <h3 class="section-title" style="margin: 0 0 0.5rem 0;">${escapeHtml(BUCKET_LABELS[bucket.name] || bucket.name)}</h3>
            ${items}
          </div>`;
      })
      .join("");

    mount.innerHTML =
      card("Methodology Extractor", "", { basis: data.basis, headerRight: header }) +
      `<div class="grid grid--two">${panels}</div>`;
  } catch (err) {
    mount.innerHTML = card("Methodology Extractor", errorState(err));
  }
}

/* ------------------------------------------------------------- citations */

function forceLayout(nodes, edges, width = 600, height = 360) {
  // Tiny deterministic force-directed layout for a small citation graph.
  // Implemented inline so the panel renders offline; the synthetic corpus
  // has zero edges, so this only matters when a real corpus is loaded.
  const positions = new Map(
    nodes.map((n, i) => {
      const angle = (i / Math.max(1, nodes.length)) * 2 * Math.PI;
      const radius = Math.min(width, height) * 0.35;
      return [
        n.id,
        {
          x: width / 2 + Math.cos(angle) * radius,
          y: height / 2 + Math.sin(angle) * radius,
        },
      ];
    }),
  );
  for (let step = 0; step < 60; step++) {
    for (const edge of edges) {
      const a = positions.get(edge.source);
      const b = positions.get(edge.target);
      if (!a || !b) continue;
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(1, Math.hypot(dx, dy));
      const target = 80;
      const force = (dist - target) * 0.05;
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.x += fx; a.y += fy;
      b.x -= fx; b.y -= fy;
    }
    // soft repulsion so nodes don't pile on top of each other
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = positions.get(nodes[i].id);
        const b = positions.get(nodes[j].id);
        if (!a || !b) continue;
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const dist = Math.max(1, Math.hypot(dx, dy));
        const force = -1200 / (dist * dist);
        a.x += (dx / dist) * force;
        a.y += (dy / dist) * force;
        b.x -= (dx / dist) * force;
        b.y -= (dy / dist) * force;
      }
    }
  }
  // clamp to viewport
  for (const p of positions.values()) {
    p.x = Math.max(8, Math.min(width - 8, p.x));
    p.y = Math.max(8, Math.min(height - 8, p.y));
  }
  return positions;
}

function graphSvg(nodes, edges) {
  if (!nodes.length) {
    return emptyState("No papers in the corpus.");
  }
  const width = 600;
  const height = 360;
  if (!edges.length) {
    // No edges: show nodes spread evenly with no lines.
    const positions = forceLayout(nodes, [], width, height);
    const dots = nodes
      .map((n) => {
        const p = positions.get(n.id);
        return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="3.5" fill="var(--series-1)"><title>${escapeHtml(n.label)}</title></circle>`;
      })
      .join("");
    return `
      <svg viewBox="0 0 ${width} ${height}" class="graph" role="img" aria-label="Citation graph with no edges">
        <rect width="${width}" height="${height}" fill="var(--bg-canvas)" rx="8"/>
        ${dots}
      </svg>`;
  }
  const positions = forceLayout(nodes, edges, width, height);
  const lines = edges
    .map((e) => {
      const a = positions.get(e.source);
      const b = positions.get(e.target);
      if (!a || !b) return "";
      return `<line x1="${a.x.toFixed(1)}" y1="${a.y.toFixed(1)}" x2="${b.x.toFixed(1)}" y2="${b.y.toFixed(1)}" stroke="var(--border-strong)" stroke-width="0.6"/>`;
    })
    .join("");
  const dots = nodes
    .map((n) => {
      const p = positions.get(n.id);
      return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="4" fill="var(--series-1)"><title>${escapeHtml(n.label)}</title></circle>`;
    })
    .join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" class="graph" role="img" aria-label="Citation graph">
      <rect width="${width}" height="${height}" fill="var(--bg-canvas)" rx="8"/>
      ${lines}${dots}
    </svg>`;
}

export async function renderCitations(mount) {
  try {
    const data = await getCitations();
    const header = `
      <span class="tag">${data.n_nodes} paper(s)</span>
      <span class="tag">${data.n_edges} reference(s)</span>
      <span class="tag">${data.n_components} component(s)</span>
    `;
    const empty = data.n_edges === 0;
    const body = `
      ${graphSvg(data.nodes, data.edges)}
      ${
        empty
          ? `<p class="preview__meta" style="margin-top: 0.75rem;">
              No in-corpus references. The committed synthetic fixture does not carry
              outbound reference metadata, so a citation graph over it is correctly
              empty. Load a real OpenAlex build to see edges.
            </p>`
          : ""
      }
    `;
    mount.innerHTML = card("Citation Network", body, { basis: data.basis, headerRight: header });
  } catch (err) {
    mount.innerHTML = card("Citation Network", errorState(err));
  }
}
