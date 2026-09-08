/* ==========================================================================
   Research Workspace Sidebar
   ==========================================================================
   A contextual research command-center that answers "What should I research
   next?" instead of merely "Where are the pages?".

   Rules:
   1. Every count comes from a live API call. No hardcoded numbers.
   2. Unavailable features show "--" or omit the badge -- never fake data.
   3. All existing openNavView(key) routes are preserved unchanged.
   4. Collapse state is persisted in localStorage.
   ========================================================================== */

import { getGaps, listPapers } from "./api.js";
import { icon } from "./icons.js";

/* --------------------------------------------------------------------------
   Nav structure
   -------------------------------------------------------------------------- */

const WORKSPACE_GROUPS = [
  {
    group: "Research",
    collapseKey: "research",
    items: [
      { id: "dashboard",  label: "Overview",       icon: "home",     built: true },
      { id: "papers",     label: "Papers",          icon: "papers",   built: true, capability: "corpus",        badge: "totalPapers" },
      { id: "search",     label: "Discover",        icon: "search",   built: true, capability: "corpus" },
      { id: "review",     label: "Reading Queue",   icon: "flag",     built: true, capability: "corpus",        badge: "reviewPapers", badgeTone: "warn" },
      { id: "trends",     label: "Research Map",    icon: "trend",    built: true, capability: "trends" },
    ],
  },
  {
    group: "Analysis",
    collapseKey: "analysis",
    items: [
      { id: "compare",   label: "Compare Papers",   icon: "compare", built: true, capability: "comparison" },
      { id: "trends",    label: "Research Trends",  icon: "bars",    built: true, capability: "trends" },
      { id: "topics",    label: "Topics",           icon: "pie",     built: true },
      { id: "citations", label: "Citation Network", icon: "graph",   built: true },
      { id: "gaps",      label: "Research Gaps",    icon: "gap",     built: true, capability: "research_gaps", badge: "gapsCount", badgeTone: "critical" },
    ],
  },
  {
    group: "Extraction",
    collapseKey: "extraction",
    items: [
      { id: "methodology", label: "Methodology", icon: "extract",  built: true },
      { id: "datasets",    label: "Datasets",    icon: "database", built: true },
      { id: "models",      label: "Models",      icon: "cube",     built: true },
      { id: "evidence",    label: "Evidence",    icon: "info",     built: true },
    ],
  },
  {
    group: "Project",
    collapseKey: "project",
    items: [
      { id: "roadmap",    label: "Research Roadmap", icon: "trend",  built: true },
      { id: "objectives", label: "Objectives",       icon: "flag",   built: true },
    ],
  },
];

/* --------------------------------------------------------------------------
   Internal state
   -------------------------------------------------------------------------- */

let _activeNavId = "dashboard";
const _badges = { totalPapers: null, reviewPapers: null, gapsCount: null };
let _meta = null;
let _capabilities = new Map();

const LS_GROUPS   = "ri-sidebar-groups";
const LS_SIDEBAR  = "ri-sidebar-minimized";

/* --------------------------------------------------------------------------
   LocalStorage helpers
   -------------------------------------------------------------------------- */

function getCollapsedGroups() {
  try { return JSON.parse(localStorage.getItem(LS_GROUPS) || "[]"); } catch { return []; }
}

function setCollapsedGroup(key, collapsed) {
  const groups = getCollapsedGroups();
  if (collapsed && !groups.includes(key)) groups.push(key);
  if (!collapsed) { const i = groups.indexOf(key); if (i !== -1) groups.splice(i, 1); }
  localStorage.setItem(LS_GROUPS, JSON.stringify(groups));
}

function isSidebarMinimized() {
  return localStorage.getItem(LS_SIDEBAR) === "true";
}

function setSidebarMinimized(val) {
  localStorage.setItem(LS_SIDEBAR, String(val));
}

/* --------------------------------------------------------------------------
   Utility
   -------------------------------------------------------------------------- */

function esc(str) {
  return String(str ?? "").replace(/[&<>"']/g,
    c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function badgePill(count, tone) {
  if (count === null || count === undefined || count <= 0) return "";
  const cls = tone === "critical" ? "nav-badge nav-badge--critical"
            : tone === "warn"     ? "nav-badge nav-badge--warn"
            :                       "nav-badge";
  return `<span class="${cls}">${count > 999 ? "999+" : count}</span>`;
}

/* --------------------------------------------------------------------------
   Workspace header
   -------------------------------------------------------------------------- */

function renderWorkspaceHeader(meta) {
  const run = meta?.run;
  const runLabel = run ? run.model_display_name : "No run loaded";
  const isReady  = run?.model_ready;
  const dotCls   = !run ? "ws-dot ws-dot--critical"
                 : isReady ? "ws-dot ws-dot--good"
                 : "ws-dot ws-dot--warn";
  const statusText = !run ? "No run" : isReady ? "Active" : "Degraded";

  return `
<div class="ws-header" id="ws-header">
  <div class="ws-header__top">
    <div class="ws-brand">
      <div class="ws-brand__mark" aria-hidden="true">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor"
             stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M12 3l9 5.2v7.6L12 21l-9-5.2V8.2z"/>
          <path d="M12 12l9-5.2M12 12v9M12 12L3 6.8"/>
        </svg>
      </div>
      <span class="ws-brand__name">Research Intelligence</span>
    </div>
    <button class="sidebar-toggle-btn" id="sidebar-minimize-btn" type="button"
            aria-label="Collapse sidebar" title="Collapse sidebar">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor"
           stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M15 18l-6-6 6-6"/>
      </svg>
    </button>
  </div>

  <button class="ws-project-btn" id="ws-project-btn"
          aria-label="Switch research workspace" aria-expanded="false">
    <div class="ws-project-btn__inner">
      <span class="ws-project-name">Academic Paper Classification</span>
      <span class="ws-project-sub">HAN + Transformer Embeddings</span>
    </div>
    <svg class="ws-project-btn__chevron" width="12" height="12" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
         stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
  </button>

  <div class="ws-run-row" title="${esc(run?.run_id ?? '')}">
    <span class="${dotCls}" aria-hidden="true"></span>
    <span class="ws-run-label">${esc(runLabel)}</span>
    <span class="ws-run-status">${esc(statusText)}</span>
  </div>
</div>`;
}

/* --------------------------------------------------------------------------
   Project progress
   -------------------------------------------------------------------------- */

function computeObjectives(meta, badges) {
  const run = meta?.run;
  const total  = badges.totalPapers  ?? 0;
  const review = badges.reviewPapers ?? null;
  const gaps   = badges.gapsCount    ?? 0;
  const hasMetrics = run && (run.metrics?.test || run.metrics?.val);

  return [
    { label: "Papers collected",       done: !!(run && total > 0) },
    { label: "Papers classified",      done: !!(run?.model_ready) },
    { label: "Baseline evaluated",     done: !!hasMetrics },
    { label: "Literature reviewed",    done: review !== null && review === 0 && total > 0 },
    { label: "Research gaps found",    done: gaps > 0 },
    { label: "Methodology extracted",  done: false },
  ];
}

function renderProgress(meta, badges) {
  const objs  = computeObjectives(meta, badges);
  const done  = objs.filter(o => o.done).length;
  const total = objs.length;
  const pct   = total ? Math.round((done / total) * 100) : 0;

  const items = objs.map(o => `
    <div class="prog-obj${o.done ? " prog-obj--done" : ""}">
      <span class="prog-obj__icon" aria-hidden="true">${o.done
        ? '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"><path d="M20 6L9 17l-5-5"/></svg>'
        : '<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="8"/></svg>'
      }</span>
      <span>${esc(o.label)}</span>
    </div>`).join("");

  return `
<div class="ws-progress" id="ws-progress">
  <button class="ws-progress__toggle" id="ws-progress-toggle"
          type="button" aria-expanded="false" aria-controls="ws-progress-detail">
    <span class="ws-progress__title">Project Progress</span>
    <span class="ws-progress__frac">${done}/${total}</span>
    <svg class="ws-progress__caret" width="11" height="11" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <path d="M6 9l6 6 6-6"/>
    </svg>
  </button>
  <div class="prog-bar" role="progressbar" aria-valuenow="${pct}"
       aria-valuemin="0" aria-valuemax="100" aria-label="Research ${pct}% complete">
    <div class="prog-bar__fill" style="width:${pct}%"></div>
  </div>
  <div class="ws-progress__detail" id="ws-progress-detail" hidden>
    ${items}
  </div>
</div>`;
}

/* --------------------------------------------------------------------------
   Navigation groups
   -------------------------------------------------------------------------- */

function renderNavGroups(badges, capabilities) {
  const collapsed = getCollapsedGroups();
  const cap = key => capabilities?.get(key);

  return WORKSPACE_GROUPS.map(group => {
    const isCollapsed = collapsed.includes(group.collapseKey);

    const items = group.items.map(item => {
      const capEntry    = item.capability ? cap(item.capability) : null;
      const unavailable = capEntry && !capEntry.available;
      const badgeCount  = item.badge ? (_badges[item.badge] ?? null) : null;
      const badge       = badgeCount !== null && badgeCount > 0
                        ? badgePill(badgeCount, item.badgeTone) : "";
      const isActive    = _activeNavId === item.id;

      return `
<button class="nav-item${isActive ? " nav-item--active" : ""}"
        type="button"
        data-nav="${esc(item.id)}"
        data-nav-label="${esc(item.label)}"
        aria-current="${isActive ? "page" : "false"}"
        ${unavailable ? `title="${esc(capEntry?.reason ?? "Not available")}"` : ""}>
  ${icon(item.icon, 15)}
  <span class="nav-item__label">${esc(item.label)}</span>
  ${badge}
</button>`;
    }).join("");

    return `
<div class="nav-group" data-group="${esc(group.collapseKey)}">
  <button class="nav-group__header" type="button"
          data-collapse="${esc(group.collapseKey)}"
          aria-expanded="${!isCollapsed}"
          aria-controls="ng-${esc(group.collapseKey)}">
    <span class="nav-group__label">${esc(group.group)}</span>
    <svg class="nav-group__caret${isCollapsed ? "" : " nav-group__caret--open"}"
         width="11" height="11" viewBox="0 0 24 24"
         fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
      <path d="M6 9l6 6 6-6"/>
    </svg>
  </button>
  <div class="nav-group__body" id="ng-${esc(group.collapseKey)}"
       ${isCollapsed ? "hidden" : ""}>
    ${items}
  </div>
</div>`;
  }).join("");
}

/* --------------------------------------------------------------------------
   Next Action card
   -------------------------------------------------------------------------- */

function renderNextAction(meta, badges) {
  const run    = meta?.run;
  const review = badges.reviewPapers ?? 0;
  const gaps   = badges.gapsCount    ?? 0;

  let iconName, text, label, navId, cls;

  if (!run) {
    iconName = "warn";  text = "No training run loaded";
    label = "View System";  navId = "admin";  cls = "next-action--critical";
  } else if (review > 0) {
    iconName = "flag";
    text  = `${review} paper${review === 1 ? "" : "s"} need${review === 1 ? "s" : ""} review`;
    label = "Review Now";  navId = "review";  cls = "next-action--warn";
  } else if (gaps > 0) {
    iconName = "gap";
    text  = `${gaps} research gap${gaps === 1 ? "" : "s"} identified`;
    label = "Analyze Gaps";  navId = "gaps";  cls = "";
  } else if (!run.model_ready) {
    iconName = "warn";  text = "Model unavailable — run degraded";
    label = "View Models";  navId = "models";  cls = "next-action--warn";
  } else {
    iconName = "trend";  text = "Research progressing well";
    label = "View Trends";  navId = "trends";  cls = "next-action--good";
  }

  return `
<div class="next-action ${cls}" id="next-action-card">
  <div class="next-action__hd">
    ${icon(iconName, 12)}
    <span>Next Action</span>
  </div>
  <p class="next-action__text">${esc(text)}</p>
  <button class="next-action__cta" type="button"
          data-nav="${esc(navId)}" data-nav-label="${esc(label)}">
    ${esc(label)} →
  </button>
</div>`;
}

/* --------------------------------------------------------------------------
   Workspace picker popup
   -------------------------------------------------------------------------- */

function showWorkspacePicker() {
  document.getElementById("ws-picker")?.remove();
  const btn = document.getElementById("ws-project-btn");
  if (!btn) return;

  const picker = document.createElement("div");
  picker.id = "ws-picker";
  picker.className = "ws-picker";
  picker.setAttribute("role", "dialog");
  picker.setAttribute("aria-label", "Research workspace selector");

  picker.innerHTML = `
<div class="ws-picker__header">
  <span class="ws-picker__title">Research Workspace</span>
  <button class="ws-picker__close" type="button" aria-label="Close">&times;</button>
</div>
<div class="ws-picker__body">
  <div class="ws-picker__section-label">Active Project</div>
  <div class="ws-picker__item ws-picker__item--active">
    ${icon("trend", 13)}
    <span>Academic Paper Classification</span>
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2.5" stroke-linecap="round"><path d="M20 6L9 17l-5-5"/></svg>
  </div>
  <div class="ws-picker__section-label" style="margin-top:0.75rem">Other Projects</div>
  <div class="ws-picker__note">Multi-project support is not available in this build.</div>
  <div class="ws-picker__footer">
    <button class="ws-picker__new" type="button" disabled>
      ${icon("plus", 13)} New Research Project
    </button>
    <button class="ws-picker__new" type="button" disabled>
      ${icon("database", 13)} Import Project
    </button>
  </div>
</div>`;

  document.body.appendChild(picker);

  const rect = btn.getBoundingClientRect();
  picker.style.left = `${rect.left}px`;
  picker.style.top  = `${rect.bottom + 6}px`;
  btn.setAttribute("aria-expanded", "true");

  const close = () => {
    picker.remove();
    btn.setAttribute("aria-expanded", "false");
    document.removeEventListener("click", outsideClick, true);
  };
  picker.querySelector(".ws-picker__close").addEventListener("click", close);
  const outsideClick = e => { if (!picker.contains(e.target) && e.target !== btn) close(); };
  setTimeout(() => document.addEventListener("click", outsideClick, true), 10);
}

/* --------------------------------------------------------------------------
   Sidebar minimize
   -------------------------------------------------------------------------- */

function applySidebarMinimized(minimized) {
  const sidebar = document.getElementById("sidebar");
  const btn     = document.getElementById("sidebar-minimize-btn");
  if (!sidebar) return;
  sidebar.classList.toggle("sidebar--minimized", minimized);
  if (btn) {
    const tip = minimized ? "Expand sidebar" : "Collapse sidebar";
    btn.title = tip;
    btn.setAttribute("aria-label", tip);
    btn.innerHTML = minimized
      ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M9 18l6-6-6-6"/></svg>'
      : '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M15 18l-6-6 6-6"/></svg>';
  }
}

/* --------------------------------------------------------------------------
   Interaction wiring
   -------------------------------------------------------------------------- */

function wireSidebarInteractions(onNavClick) {
  const sidebar = document.getElementById("sidebar");

  // Workspace project button
  document.getElementById("ws-project-btn")?.addEventListener("click", e => {
    e.stopPropagation();
    showWorkspacePicker();
  });

  // Progress toggle
  sidebar?.addEventListener("click", e => {
    if (e.target.closest("#ws-progress-toggle")) {
      const detail = document.getElementById("ws-progress-detail");
      const toggle = document.getElementById("ws-progress-toggle");
      if (!detail) return;
      const nowHidden = !detail.hidden;
      detail.hidden = nowHidden;
      toggle?.setAttribute("aria-expanded", String(!nowHidden));
      const caret = toggle?.querySelector(".ws-progress__caret");
      caret?.classList.toggle("ws-progress__caret--open", !nowHidden);
      return;
    }

    // Group collapse
    const collapseBtn = e.target.closest("[data-collapse]");
    if (collapseBtn) {
      const key  = collapseBtn.dataset.collapse;
      const body = document.getElementById(`ng-${key}`);
      if (!body) return;
      const willCollapse = !body.hidden;
      body.hidden = willCollapse;
      collapseBtn.setAttribute("aria-expanded", String(!willCollapse));
      const caret = collapseBtn.querySelector(".nav-group__caret");
      caret?.classList.toggle("nav-group__caret--open", !willCollapse);
      setCollapsedGroup(key, willCollapse);
      return;
    }

    // Nav item
    const navBtn = e.target.closest("[data-nav]");
    if (navBtn && !e.target.closest("[data-collapse]") && !e.target.closest("#ws-project-btn")) {
      e.preventDefault();
      const navId    = navBtn.dataset.nav;
      const navLabel = navBtn.dataset.navLabel || navId;
      if (onNavClick) onNavClick(navId, navLabel);
    }
  });

  // Minimize toggle
  document.getElementById("sidebar-minimize-btn")?.addEventListener("click", () => {
    const minimized = !isSidebarMinimized();
    setSidebarMinimized(minimized);
    applySidebarMinimized(minimized);
  });
}

/* --------------------------------------------------------------------------
   Badge update (live data)
   -------------------------------------------------------------------------- */

export async function updateSidebarBadges() {
  const results = await Promise.allSettled([
    listPapers({ limit: 1 }),
    listPapers({ needs_review: true, limit: 1 }),
    getGaps(),
  ]);

  const [totalRes, reviewRes, gapsRes] = results;
  if (totalRes.status  === "fulfilled") _badges.totalPapers  = totalRes.value?.total  ?? null;
  if (reviewRes.status === "fulfilled") _badges.reviewPapers = reviewRes.value?.total ?? null;
  if (gapsRes.status   === "fulfilled") {
    const cats = gapsRes.value?.categories ?? [];
    _badges.gapsCount = cats.reduce((s, c) => s + (c.count ?? 0), 0);
  }

  // Patch badge pills in-place
  const nav = document.getElementById("nav");
  if (nav) {
    WORKSPACE_GROUPS.forEach(group => {
      group.items.forEach(item => {
        if (!item.badge) return;
        const count = _badges[item.badge] ?? null;
        // Find the specific button by id + label combination
        const btns = nav.querySelectorAll(`[data-nav="${item.id}"][data-nav-label="${item.label}"]`);
        btns.forEach(btn => {
          btn.querySelector(".nav-badge")?.remove();
          if (count !== null && count > 0) {
            const pill = document.createElement("span");
            pill.className = item.badgeTone === "critical" ? "nav-badge nav-badge--critical"
                           : item.badgeTone === "warn"     ? "nav-badge nav-badge--warn"
                           :                                  "nav-badge";
            pill.textContent = count > 999 ? "999+" : String(count);
            btn.appendChild(pill);
          }
        });
      });
    });
  }

  // Refresh next-action card
  const naOld = document.getElementById("next-action-card");
  if (naOld && _meta) {
    const temp = document.createElement("div");
    temp.innerHTML = renderNextAction(_meta, _badges);
    naOld.replaceWith(temp.firstElementChild);
  }

  // Refresh progress bar
  const progOld = document.getElementById("ws-progress");
  if (progOld && _meta) {
    const temp = document.createElement("div");
    temp.innerHTML = renderProgress(_meta, _badges);
    progOld.replaceWith(temp.firstElementChild);
    // Re-wire toggle click (event delegation on sidebar handles it)
  }
}

/* --------------------------------------------------------------------------
   Active nav
   -------------------------------------------------------------------------- */

export function setActiveNav(navId) {
  _activeNavId = navId;
  const nav = document.getElementById("nav");
  if (!nav) return;
  nav.querySelectorAll(".nav-item").forEach(btn => {
    const active = btn.dataset.nav === navId;
    btn.classList.toggle("nav-item--active", active);
    btn.setAttribute("aria-current", active ? "page" : "false");
  });
}

/* --------------------------------------------------------------------------
   Init
   -------------------------------------------------------------------------- */

export function initSidebar(meta, capabilities, { onNavClick } = {}) {
  _meta         = meta;
  _capabilities = capabilities || new Map();

  const workspaceMount  = document.getElementById("sidebar-workspace");
  const nav             = document.getElementById("nav");
  const progressMount   = document.getElementById("sidebar-progress");
  const nextActionMount = document.getElementById("sidebar-next-action");

  if (workspaceMount) workspaceMount.innerHTML = renderWorkspaceHeader(meta);
  if (progressMount)  progressMount.innerHTML  = renderProgress(meta, _badges);
  if (nav)            nav.innerHTML            = renderNavGroups(_badges, capabilities);
  if (nextActionMount) nextActionMount.innerHTML = renderNextAction(meta, _badges);

  applySidebarMinimized(isSidebarMinimized());
  wireSidebarInteractions(onNavClick || (() => {}));
}
