"""System endpoints: health, capability manifest, and run listing.

``/api/health`` is the only unauthenticated route, and it is the only one that
answers while the server is otherwise unusable — no run, no dataset, unloadable
model. Everything here is read-only.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, status

from src.api import capabilities as caps
from src.api.deps import ActiveRun, SettingsDep, StoreDep
from src.api.runstore import LoadedRun, RunUnavailableError
from src.api.schemas import (
    DatasetDiagnosticsResponse,
    DatasetInfo,
    DatasetProvenance,
    HealthResponse,
    MetaResponse,
    RunDetail,
    RunListResponse,
    RunSummaryOut,
    SplitSummary,
    StorageInfo,
    UserInfo,
)
from src.preprocessing.text import normalize_for_matching
from src.utils.logging import get_logger

__all__ = ["public_router", "router", "run_detail"]

logger = get_logger(__name__)

#: Routes reachable without an API key. Only liveness qualifies: a monitoring
#: probe that needs the secret is a probe that stops working the moment the
#: secret rotates.
public_router = APIRouter(tags=["system"])

router = APIRouter(tags=["system"])

#: Headline metrics lifted out of ``metrics.json`` for the run banner. The full
#: file is mostly per-class detail, which the dashboard does not show.
_HEADLINE_KEYS = ("accuracy", "balanced_accuracy", "n_samples", "n_classes")


def _headline_metrics(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reduce a full metrics payload to the numbers the dashboard displays."""
    summary: dict[str, dict[str, Any]] = {}
    for split, payload in metrics.items():
        if not isinstance(payload, dict):
            continue
        entry = {key: payload.get(key) for key in _HEADLINE_KEYS if key in payload}
        averages = payload.get("averages")
        if isinstance(averages, dict):
            for average in ("macro", "micro", "weighted"):
                scores = averages.get(average)
                if isinstance(scores, dict):
                    if "f1" in scores:
                        entry[f"{average}_f1"] = scores["f1"]
                    if "precision" in scores:
                        entry[f"{average}_precision"] = scores["precision"]
                    if "recall" in scores:
                        entry[f"{average}_recall"] = scores["recall"]
        # The confusion matrix travels with the headline numbers so the Model
        # Tracker can render Actual-vs-Predicted without re-reading metrics.json.
        matrix = payload.get("confusion_matrix")
        if isinstance(matrix, dict) and matrix.get("counts") is not None:
            entry["confusion_matrix"] = {
                key: matrix.get(key) for key in ("labels", "counts", "normalized")
            }
        primary = payload.get("primary_metric")
        if isinstance(primary, dict):
            entry["primary_metric"] = primary
        confidence = payload.get("confidence")
        if isinstance(confidence, dict):
            entry["confidence"] = {
                key: confidence.get(key)
                for key in ("kind", "available", "statistic", "mean", "median", "caveat")
            }
        summary[split] = entry
    return summary


def _dataset_info(run: LoadedRun) -> DatasetInfo:
    """Describe the corpus behind a run, degrading if it is unreadable.

    A missing dataset does not make the run undescribable — its manifest still
    records what it trained on — so the fields that come from the manifest are
    reported either way, and only the on-disk checks are skipped.
    """
    recorded = run.manifest.get("dataset", {})
    labels = run.manifest.get("labels", {})
    info = DatasetInfo(
        source=recorded.get("source"),
        directory=recorded.get("directory"),
        split_sizes=dict(recorded.get("split_sizes") or {}),
        n_classes=labels.get("n_classes"),
        classes=run.classes,
        built_at=recorded.get("build_created_at"),
        is_synthetic=run.is_synthetic_corpus,
        integrity_findings=list(recorded.get("integrity_findings") or []),
    )
    try:
        info.is_stale = run.dataset_is_stale
    except RunUnavailableError as exc:
        logger.warning("api | dataset unreadable for run %s: %s", run.run_id, exc)
    return info


def run_detail(run: LoadedRun) -> RunDetail:
    """Convert a loaded run into its API representation."""
    warnings = list(run.warnings)
    model_ready = True
    try:
        run.pipeline  # noqa: B018 - loading it is the readiness check
    except RunUnavailableError as exc:
        model_ready = False
        warnings.append(str(exc))

    labels = run.manifest.get("labels", {})
    return RunDetail(
        run_id=run.run_id,
        model_name=run.model_name,
        model_display_name=run.model_display_name,
        created_at=run.manifest.get("created_at"),
        finished_at=run.manifest.get("finished_at"),
        seed=run.manifest.get("seed"),
        git_commit=run.manifest.get("git_commit"),
        label_mode=labels.get("mode"),
        taxonomy_level=labels.get("taxonomy_level"),
        classes=run.classes,
        confidence_kind=run.confidence_kind,
        primary_split=run.primary_split,
        metrics=_headline_metrics(run.metrics),
        timing=dict(run.manifest.get("timing") or {}),
        dataset=_dataset_info(run),
        model_ready=model_ready,
        warnings=warnings,
    )


@public_router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness, plus what the server actually managed to load",
)
def health(settings: SettingsDep, store: StoreDep) -> HealthResponse:
    """Report process health and the state of the active run.

    Returns ``degraded`` rather than an error status when a run is missing or
    partly unusable: the process is serving correctly, and a 500 here would make
    an ordinary "nothing trained yet" state look like a crash.
    """
    warnings: list[str] = []
    run_id: str | None = None
    dataset_ready = False
    model_ready = False

    try:
        run = store.active()
        run_id = run.run_id
        try:
            run.dataset  # noqa: B018 - the load is the check
            dataset_ready = True
        except RunUnavailableError as exc:
            warnings.append(str(exc))
        try:
            run.pipeline  # noqa: B018 - the load is the check
            model_ready = True
        except RunUnavailableError as exc:
            warnings.append(str(exc))
        warnings.extend(run.warnings)
    except RunUnavailableError as exc:
        warnings.append(str(exc))

    return HealthResponse(
        status="ok" if (dataset_ready and model_ready) else "degraded",
        app_name=settings.app.project.name,
        version=settings.app.project.version,
        environment=settings.env.aris_env,
        run_id=run_id,
        dataset_ready=dataset_ready,
        model_ready=model_ready,
        warnings=warnings,
    )


@router.get(
    "/meta",
    response_model=MetaResponse,
    summary="Everything the dashboard shell needs to render once",
)
def meta(settings: SettingsDep, store: StoreDep) -> MetaResponse:
    """Return identity, storage, the active run, and the capability table.

    Tolerates a missing run: ``run`` is null and every model-dependent capability
    reports unavailable, which is what lets the dashboard render an honest empty
    state instead of a spinner that never resolves.
    """
    try:
        run: LoadedRun | None = store.active()
    except RunUnavailableError as exc:
        logger.info("api | /meta with no active run: %s", exc)
        run = None

    usage = store.storage_usage()
    return MetaResponse(
        app_name=settings.app.project.name,
        version=settings.app.project.version,
        environment=settings.env.aris_env,
        # No accounts exist, so this describes the local operator and says as
        # much through is_authenticated (master spec §40).
        user=UserInfo(
            first_name="Researcher",
            full_name="Local Researcher",
            role="Local session — no account",
            initials="LR",
            is_authenticated=False,
        ),
        storage=StorageInfo(**usage),
        run=run_detail(run) if run is not None else None,
        capabilities=caps.capabilities_for(run),
        caveats=caps.caveats_for(run),
    )


@router.get("/runs", response_model=RunListResponse, summary="Every discoverable run")
def list_runs(store: StoreDep) -> RunListResponse:
    """List runs newest-first, marking which one the dashboard is showing."""
    try:
        active_id: str | None = store.active().run_id
    except RunUnavailableError:
        active_id = None

    return RunListResponse(
        runs=[
            RunSummaryOut(
                run_id=summary.run_id,
                model_name=summary.model_name,
                model_display_name=summary.model_display_name,
                created_at=summary.created_at,
                finished_at=summary.finished_at,
                primary_metric_name=summary.primary_metric_name,
                primary_metric_value=summary.primary_metric_value,
                n_classes=summary.n_classes,
                split_sizes=summary.split_sizes,
                is_complete=summary.is_complete,
                is_active=summary.run_id == active_id,
            )
            for summary in store.summaries()
        ],
        active_run_id=active_id,
    )


@router.get("/runs/active", response_model=RunDetail, summary="The run being displayed")
def active_run(run: ActiveRun) -> RunDetail:
    """Return full detail for the active run."""
    return run_detail(run)


@router.get("/runs/{run_id}", response_model=RunDetail, summary="One run by id")
def get_run(run_id: str, store: StoreDep) -> RunDetail:
    """Return full detail for one run.

    Raises:
        HTTPException: 404 when the id is unknown or malformed. The store
            validates the id as a bare token before touching the filesystem, so a
            traversal attempt lands here rather than reading an arbitrary path.
    """
    try:
        return run_detail(store.load(run_id))
    except RunUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get(
    "/dataset/diagnostics",
    response_model=DatasetDiagnosticsResponse,
    summary="Data-quality facts about the active run's corpus",
)
def dataset_diagnostics(run: ActiveRun) -> DatasetDiagnosticsResponse:
    """Compute the corpus diagnostics table from the loaded dataset records.

    Every value is measured, none estimated: duplicate detection re-runs the
    same normalisation the deduplicator used, lengths are word counts over the
    composed record text, and the leakage counts are the ones the split stage
    audited at build time (a non-zero count there means the build should have
    failed, so seeing one here is a bug worth surfacing loudly).
    """
    entries = list(run.papers.values())
    texts = [entry.record.text for entry in entries]

    lengths = [len(text.split()) for text in texts]
    normalized = [normalize_for_matching(text) for text in texts]
    # A duplicate is a normalised text seen more than once; the count reports
    # the surplus copies, so a corpus with one repeated text twice reports 1.
    seen: set[str] = set()
    duplicates = 0
    for text in normalized:
        if not text:
            continue
        if text in seen:
            duplicates += 1
        seen.add(text)

    label_counts: dict[str, int] = {}
    for entry in entries:
        if entry.record.label:
            label_counts[entry.record.label] = label_counts.get(entry.record.label, 0) + 1

    splits = [
        SplitSummary(
            name=name,
            n_records=len(data.records),
            label_counts=dict(data.class_counts),
        )
        for name, data in sorted(run.dataset.splits.items())
    ]

    # Leakage counts recorded by the split audit at dataset build time.
    leakage = (
        (run.dataset.manifest.get("stages", {}).get("split", {}) or {}).get(
            "leakage_checks"
        )
        or {}
    )

    # Field-completeness diagnostics, measured over the same records. Each
    # count is a simple absence check on the structured fields the dataset
    # pipeline carries; a missing field means the source export lacked it.
    missing_titles = sum(1 for entry in entries if not (entry.record.title or "").strip())
    missing_abstracts = sum(1 for entry in entries if not (entry.record.abstract or "").strip())
    missing_authors = sum(1 for entry in entries if not entry.record.authors)
    missing_years = sum(1 for entry in entries if entry.record.year is None)

    # Outliers are relative to the corpus's own length distribution: the
    # shortest and longest deciles. That is a measured definition (stated in
    # the panel), not an absolute rule imposed on every corpus.
    ordered_lengths = sorted(lengths)
    short_papers = 0
    long_papers = 0
    if ordered_lengths:
        decile = max(1, len(ordered_lengths) // 10)
        short_cutoff = ordered_lengths[decile - 1]
        long_cutoff = ordered_lengths[-decile]
        short_papers = sum(1 for n in lengths if n <= short_cutoff)
        long_papers = sum(1 for n in lengths if n >= long_cutoff)

    # Duplicate titles use the same normalisation as the deduplicator so the
    # count agrees with what the build stage removed (or failed to remove).
    title_counts: dict[str, int] = {}
    for entry in entries:
        normalized_title = normalize_for_matching(entry.record.title or "")
        if normalized_title:
            title_counts[normalized_title] = title_counts.get(normalized_title, 0) + 1
    duplicate_titles = sum(1 for count in title_counts.values() if count > 1)

    # Provenance comes straight from the dataset manifest; absent keys stay
    # absent rather than being guessed.
    dataset_manifest = run.dataset.manifest
    split_config = (dataset_manifest.get("config", {}).get("split", {}) or {})
    split_ratios = {
        key: float(split_config[key])
        for key in ("train_ratio", "val_ratio", "test_ratio")
        if key in split_config
    }
    provenance = DatasetProvenance(
        source=dataset_manifest.get("source"),
        created_at=dataset_manifest.get("created_at"),
        git_commit=dataset_manifest.get("git_commit"),
        seed=dataset_manifest.get("seed"),
        is_synthetic=run.is_synthetic_corpus,
        split_ratios=split_ratios,
    )

    return DatasetDiagnosticsResponse(
        total_papers=len(entries),
        n_classes=len(run.classes),
        classes=run.classes,
        splits=splits,
        class_distribution=dict(sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        duplicate_papers=duplicates,
        empty_papers=sum(1 for text in normalized if not text),
        missing_labels=sum(1 for entry in entries if not entry.record.label),
        average_paper_length_words=round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        min_paper_length_words=min(lengths) if lengths else 0,
        max_paper_length_words=max(lengths) if lengths else 0,
        is_synthetic=run.is_synthetic_corpus,
        leakage_checks={str(key): int(value) for key, value in leakage.items()},
        missing_titles=missing_titles,
        missing_abstracts=missing_abstracts,
        missing_authors=missing_authors,
        missing_years=missing_years,
        short_papers=short_papers,
        long_papers=long_papers,
        duplicate_titles=duplicate_titles,
        provenance=provenance,
    )
