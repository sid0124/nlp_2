"""Request and response models for the dashboard API (master spec §26).

Two jobs, and the second is the interesting one.

The first is ordinary: validate input. Every request model sets
``extra="forbid"`` and bounds every field, so a malformed or oversized body is a
422 produced by the schema rather than an exception raised somewhere inside
scikit-learn.

The second is to make *unavailability* a first-class part of the contract. This
system is partly built. Section attention needs the hierarchical model from
Milestone 3; grounded question answering needs a retrieval index; there is no PDF
parser yet. A response shape that could only express *answers* would force each
missing feature to be either omitted silently or filled with something
plausible-looking. So :class:`Capability` and the ``available``/``reason`` pairs
below exist to let the API say "not built, and here is why" in a form the UI can
render — which is the difference between a dashboard that is honest about its
state and one that looks finished.

Confidence carries its ``kind`` everywhere for the same reason. A logistic
regression's 0.82 is a probability; a LinearSVC's 0.82 is an uncalibrated margin
between the top two classes. Dropping the distinction would make the two
indistinguishable in the UI while meaning entirely different things.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "AskRequest",
    "Capability",
    "CitationEdgeSchema",
    "CitationGraphResponse",
    "CitationNodeSchema",
    "ClassifyRequest",
    "ClassifyResponse",
    "ClassifyResult",
    "DatasetDiagnosticsResponse",
    "DatasetInfo",
    "DistributionResponse",
    "DistributionSlice",
    "ErrorResponse",
    "ExplanationResponse",
    "GapCategory",
    "GapExample",
    "GapsResponse",
    "HealthResponse",
    "LabelScore",
    "MetaResponse",
    "MethodologyBucket",
    "MethodologyItem",
    "MethodologyResponse",
    "PaperDetail",
    "PaperListResponse",
    "PaperSummary",
    "RunDetail",
    "RunListResponse",
    "RunSummaryOut",
    "SectionAttention",
    "SplitSummary",
    "SentenceEvidence",
    "SimilarItem",
    "SimilarResponse",
    "StatTile",
    "StatsResponse",
    "StorageInfo",
    "TermWeight",
    "TrendSeries",
    "TrendsResponse",
    "UnavailableFeature",
    "UserInfo",
]

#: Requests reject unknown keys: a typo in a client payload is a bug worth a 422
#: rather than a field silently ignored.
_REQUEST = ConfigDict(extra="forbid", str_strip_whitespace=True)

#: Responses are constructed only by this package, so they need no input
#: strictness — but they do forbid extras, which turns a renamed field into a
#: server-side error instead of a key the frontend quietly stops receiving.
#:
#: ``protected_namespaces`` is cleared because several fields legitimately begin
#: with ``model_`` — they describe the *trained model*, which is the domain
#: object here, not pydantic's own configuration namespace.
_RESPONSE = ConfigDict(extra="forbid", protected_namespaces=())


def _max_text_chars() -> int:
    """Return the configured ceiling for free-text input.

    Imported lazily: this module is imported while the settings module is still
    being wired up by :mod:`src.api.app`, and the limit is only needed when a
    request actually arrives.
    """
    from src.api.deps import get_settings

    return get_settings().api.security.max_text_chars


# ---------------------------------------------------------------------------
# Shared value objects
# ---------------------------------------------------------------------------
class LabelScore(BaseModel):
    """One class and the model's score for it."""

    model_config = _RESPONSE

    label: str
    score: float


class UnavailableFeature(BaseModel):
    """A feature the UI expects that this build does not provide.

    Returned with HTTP 200 in place of the feature's own payload, so the panel
    can render an explicit "not built" state. A 404 would be wrong: the resource
    exists conceptually and the client asked correctly.
    """

    model_config = _RESPONSE

    available: Literal[False] = False
    #: Why it is unavailable, in words a user can act on.
    reason: str
    #: The milestone or prerequisite that would make it available.
    requires: str | None = None


class Capability(BaseModel):
    """Whether one named feature is implemented in this build."""

    model_config = _RESPONSE

    key: str
    label: str
    available: bool
    #: Populated when ``available`` is false. Rendered by the UI verbatim.
    reason: str | None = None


# ---------------------------------------------------------------------------
# System
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Liveness plus a summary of what the server managed to load.

    Deliberately unauthenticated and deliberately not a bare ``{"ok": true}``: a
    process that is up but has no run, no dataset, or an unloadable model is the
    failure mode worth reporting, and it is invisible to a plain liveness ping.
    """

    model_config = _RESPONSE

    status: Literal["ok", "degraded"]
    app_name: str
    version: str
    environment: str
    run_id: str | None = None
    dataset_ready: bool = False
    model_ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class DatasetInfo(BaseModel):
    """Provenance of the corpus behind a run."""

    model_config = _RESPONSE

    source: str | None = None
    directory: str | None = None
    split_sizes: dict[str, int] = Field(default_factory=dict)
    n_classes: int | None = None
    classes: list[str] = Field(default_factory=list)
    built_at: str | None = None
    #: True when the corpus is the generated test fixture rather than real
    #: papers. The UI must surface this: the fixture is separable by
    #: construction, so its metrics are a wiring check, not a research result.
    is_synthetic: bool = False
    #: True when the dataset on disk no longer hashes to what the run trained on.
    is_stale: bool = False
    integrity_findings: list[dict[str, Any]] = Field(default_factory=list)


class RunSummaryOut(BaseModel):
    """One run, as listed in the run picker."""

    model_config = _RESPONSE

    run_id: str
    model_name: str | None = None
    model_display_name: str | None = None
    created_at: str | None = None
    finished_at: str | None = None
    primary_metric_name: str | None = None
    primary_metric_value: float | None = None
    n_classes: int | None = None
    split_sizes: dict[str, int] = Field(default_factory=dict)
    is_complete: bool = False
    is_active: bool = False


class RunListResponse(BaseModel):
    """Every discoverable run, newest first."""

    model_config = _RESPONSE

    runs: list[RunSummaryOut]
    active_run_id: str | None = None


class RunDetail(BaseModel):
    """Full description of one run, including its headline metrics."""

    model_config = _RESPONSE

    run_id: str
    model_name: str
    model_display_name: str
    created_at: str | None = None
    finished_at: str | None = None
    seed: int | None = None
    git_commit: str | None = None
    label_mode: str | None = None
    taxonomy_level: str | None = None
    classes: list[str] = Field(default_factory=list)
    confidence_kind: str = "unavailable"
    primary_split: str = "val"
    #: ``{split: {accuracy, macro_f1, n_samples, ...}}`` — the headline numbers
    #: only, since the full ``metrics.json`` is large and mostly per-class.
    metrics: dict[str, dict[str, Any]] = Field(default_factory=dict)
    #: ``{fit_seconds, total_seconds}`` from the run manifest, when recorded.
    timing: dict[str, Any] = Field(default_factory=dict)
    dataset: DatasetInfo
    model_ready: bool = False
    warnings: list[str] = Field(default_factory=list)


class UserInfo(BaseModel):
    """Who the dashboard says you are.

    There is no authentication layer yet (master spec §40 calls for
    authentication-*ready*, which is what the API-key dependency provides). So
    this describes a local operator, and ``is_authenticated`` is false — the UI
    must not imply a signed-in account that does not exist.
    """

    model_config = _RESPONSE

    first_name: str
    full_name: str
    role: str
    initials: str
    is_authenticated: bool = False


class StorageInfo(BaseModel):
    """Measured disk usage against the configured quota."""

    model_config = _RESPONSE

    used_bytes: int
    used_gb: float
    quota_gb: float
    percent: float
    measured: list[str] = Field(default_factory=list)


class MetaResponse(BaseModel):
    """Everything the dashboard needs before it can render anything.

    One request rather than five, because the shell — greeting, nav, run banner,
    storage meter — is useless in pieces, and five parallel fetches would each
    need their own failure state.
    """

    model_config = _RESPONSE

    app_name: str
    version: str
    environment: str
    user: UserInfo
    storage: StorageInfo
    run: RunDetail | None = None
    capabilities: list[Capability] = Field(default_factory=list)
    #: Standing caveats the UI must keep visible (master spec §14/§17/§23).
    caveats: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Dashboard aggregates
# ---------------------------------------------------------------------------
class StatTile(BaseModel):
    """One headline number.

    ``note`` replaces the period-over-period delta a dashboard of this shape
    usually shows. There is only ever one run in view, so there is no previous
    period to compare against, and an invented trend arrow is exactly the kind of
    decoration that reads as data.
    """

    model_config = _RESPONSE

    id: str
    label: str
    value: str
    #: Short qualifier under the value: what it counts, or which split it is from.
    note: str | None = None
    icon: str
    hue: str


class StatsResponse(BaseModel):
    """The stat row, derived entirely from the active run."""

    model_config = _RESPONSE

    tiles: list[StatTile]
    run_id: str
    split: str


class DistributionSlice(BaseModel):
    """One wedge of the domain donut."""

    model_config = _RESPONSE

    label: str
    count: int
    share: float


class DistributionResponse(BaseModel):
    """Corpus composition by class."""

    model_config = _RESPONSE

    total: int
    unit: str
    slices: list[DistributionSlice]
    #: Which labels were counted — ground truth from the source taxonomy, not
    #: model predictions. The two differ, and the distinction matters.
    basis: str
    note: str | None = None


class TrendSeries(BaseModel):
    """One class's counts, aligned to the shared year axis."""

    model_config = _RESPONSE

    label: str
    values: list[int]


class TrendsResponse(BaseModel):
    """Publication counts per class per year."""

    model_config = _RESPONSE

    years: list[int]
    series: list[TrendSeries]
    #: Series omitted to keep the chart legible, reported rather than dropped
    #: silently.
    dropped_series: int = 0
    basis: str
    note: str | None = None


# ---------------------------------------------------------------------------
# Papers
# ---------------------------------------------------------------------------
class PaperSummary(BaseModel):
    """One row of the papers table."""

    model_config = _RESPONSE

    paper_id: str
    title: str
    authors_short: str | None = None
    year: int | None = None
    #: Which evaluation split this paper belongs to — ``train``, ``val``,
    #: ``test`` — or ``uploaded`` for a user-supplied inference paper.
    split: str
    #: Ground-truth label from the source taxonomy.
    true_label: str | None = None
    true_labels: list[str] = Field(default_factory=list)
    #: The run's prediction. ``None`` for training-split papers, which the model
    #: was fitted on and therefore has no honest score for.
    predicted_label: str | None = None
    predicted_labels: list[str] = Field(default_factory=list)
    correct: bool | None = None
    confidence: float | None = None
    confidence_kind: str = "unavailable"
    #: Server-derived (master spec §15). The client must not recompute it.
    needs_review: bool | None = None
    #: Server-derived band: ``"high"``, ``"review_recommended"``, or
    #: ``"review_required"``. Thresholds live in ``configs/api.yaml`` only.
    review_band: str | None = None


class PaperDetail(PaperSummary):
    """One paper in full, for the preview panel."""

    model_config = _RESPONSE

    text: str
    labels: list[str] = Field(default_factory=list)
    venue: str | None = None
    n_authors: int | None = None
    n_references: int | None = None
    predicted_scores: list[LabelScore] = Field(default_factory=list)


class PaperListResponse(BaseModel):
    """A page of papers."""

    model_config = _RESPONSE

    items: list[PaperSummary]
    total: int
    limit: int
    offset: int
    splits: list[str]
    query: str | None = None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
class ClassifyRequest(BaseModel):
    """Text to classify with the active run's model.

    Title and abstract are separate fields because the vectorizer was fitted on
    the configured field order; the server joins them the same way the dataset
    build did, so an ad-hoc classification goes through the same composition as
    every training example.
    """

    model_config = _REQUEST

    title: Annotated[str, Field(min_length=1, max_length=1000)]
    abstract: Annotated[str, Field(default="", max_length=200_000)] = ""

    @field_validator("abstract")
    @classmethod
    def _within_configured_limit(cls, value: str) -> str:
        """Reject text above ``security.max_text_chars``.

        Enforced against configuration rather than a literal so the ceiling has
        one definition (master spec §32). The static ``max_length`` above is only
        a backstop that bounds memory before this runs.
        """
        limit = _max_text_chars()
        if len(value) > limit:
            raise ValueError(f"abstract is {len(value)} characters; the limit is {limit}")
        return value


class ClassifyResult(BaseModel):
    """One classification outcome.

    The score fields are deliberately separated by what they mathematically are:

    * ``probabilities`` — calibrated per-class probabilities in [0, 1] summing
      to 1 across classes. The only field the UI may render as percentages.
    * ``decision_scores`` — raw decision-function values: signed, unbounded,
      never percentages and never a distribution.
    * ``confidence`` — ``max(probabilities)`` when probabilities exist; the
      top-1-minus-top-2 margin for an uncalibrated model; ``None`` when the
      model exposes no score at all. ``confidence_kind`` names which.
    """

    model_config = _RESPONSE

    predicted_label: str
    predicted_labels: list[str] = Field(default_factory=list)
    confidence: float | None = None
    confidence_kind: str = "unavailable"
    scores: list[LabelScore] = Field(default_factory=list)
    #: Calibrated probabilities keyed by class, present only when
    #: ``confidence_kind == "probability"``. Sums to ~1.0 across classes.
    probabilities: dict[str, float] = Field(default_factory=dict)
    #: Raw decision-function values keyed by class, present only when
    #: ``confidence_kind == "decision"``. Signed and unbounded.
    decision_scores: dict[str, float] = Field(default_factory=dict)
    needs_review: bool | None = None
    review_band: str | None = None
    #: Server-composed sentence citing the configured threshold, e.g.
    #: "Review recommended because confidence is 68.4%."
    review_reason: str | None = None
    #: Always ``"inference"``: this endpoint never feeds evaluation metrics.
    split: str = "inference"
    model_name: str | None = None
    model_version: str | None = None


class ClassifyResponse(BaseModel):
    """Result of a real forward pass through the run's fitted pipeline."""

    model_config = _RESPONSE

    result: ClassifyResult
    run_id: str
    model_display_name: str
    #: What the score is and is not. Always populated.
    caveat: str


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------
class SimilarItem(BaseModel):
    """One neighbour of a query paper."""

    model_config = _RESPONSE

    paper_id: str
    title: str
    score: float
    label: str | None = None
    split: str | None = None
    year: int | None = None


class SimilarResponse(BaseModel):
    """Nearest neighbours by cosine distance in the run's TF-IDF space."""

    model_config = _RESPONSE

    query_paper_id: str
    items: list[SimilarItem]
    #: ``"tfidf_cosine"``. Named in the payload so the UI cannot describe this as
    #: semantic or embedding-based similarity, which it is not.
    method: str
    #: Master spec §17: shared vocabulary is not methodological equivalence.
    caveat: str


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------
class TermWeight(BaseModel):
    """One term's contribution to a linear model's decision."""

    model_config = _RESPONSE

    term: str
    contribution: float
    tfidf: float
    #: Contribution as a share of the largest in this list, for bar widths. The
    #: server computes it so every client scales the bars identically.
    weight: float


class SectionWeight(BaseModel):
    """Importance weight for a canonical paper section."""

    model_config = _RESPONSE

    name: str
    canonical_name: str
    weight: float


class SectionAttention(BaseModel):
    """Per-section weights, labelled by how they were produced.

    ``kind`` names the evidence so the UI can never present heuristic weights
    as neural attention:

    * ``"hierarchical_section_attention"`` — the trained HAN's own attention
      layer output. Only this may be called "Section Attention".
    * ``"projected_section_evidence"`` — term-level contributions summed over
      detected canonical sections. A projection, not attention.
    """

    model_config = _RESPONSE

    available: bool = True
    sections: list[SectionWeight] = Field(default_factory=list)
    #: One of the two kinds documented above.
    kind: str = "projected_section_evidence"
    reason: str | None = None
    requires: str | None = None


class SentenceEvidence(BaseModel):
    """One sentence the model attended to, with its provenance and weight.

    Only the hierarchical model produces these: the weight is the trained
    network's own sentence-attention output, joined back to the sentence text
    and the section it came from.
    """

    model_config = _RESPONSE

    section_name: str
    canonical_name: str
    text: str
    weight: float


class ExplanationResponse(BaseModel):
    """Why the model assigned the label it did."""

    model_config = _RESPONSE

    paper_id: str
    predicted_label: str | None = None
    #: ``"linear_term_contributions"`` or ``"han_hierarchical_attention"``.
    method: str
    #: The explanation family, named for the UI so a linear model's feature
    #: evidence is never presented as hierarchical attention:
    #: ``"Linear model feature evidence"`` or ``"Hierarchical model attention"``.
    evidence_label: str | None = None
    terms: list[TermWeight] = Field(default_factory=list)
    #: Decision value for the predicted class, so the term list is checkable
    #: against the model's own arithmetic rather than taken on trust.
    decision_value: float | None = None
    section_attention: SectionAttention
    #: Highest-attention sentences with their section provenance. Populated
    #: only by the hierarchical model; empty for linear models, which have no
    #: sentence-level representation to weight.
    sentences: list[SentenceEvidence] = Field(default_factory=list)
    #: Master spec §14: a weight is not a causal claim.
    caveat: str


# ---------------------------------------------------------------------------
# Dataset diagnostics
# ---------------------------------------------------------------------------
class SplitSummary(BaseModel):
    """Composition of one evaluation split."""

    model_config = _RESPONSE

    name: str
    n_records: int
    label_counts: dict[str, int] = Field(default_factory=dict)


class DatasetDiagnosticsResponse(BaseModel):
    """Data-quality facts about the corpus behind the active run.

    Every value is computed from the loaded dataset records — nothing is
    estimated. This is the dissertation's data-quality table, served rather
    than transcribed.
    """

    model_config = _RESPONSE

    total_papers: int
    n_classes: int
    classes: list[str] = Field(default_factory=list)
    splits: list[SplitSummary] = Field(default_factory=list)
    class_distribution: dict[str, int] = Field(default_factory=dict)
    #: Papers whose normalised text is byte-identical to another's.
    duplicate_papers: int
    #: Papers with no usable text after normalisation.
    empty_papers: int
    #: Papers carrying no label.
    missing_labels: int
    average_paper_length_words: float
    min_paper_length_words: int
    max_paper_length_words: int
    #: True when the corpus is the generated development fixture.
    is_synthetic: bool
    #: Leakage-audit counts recorded at split time (all must be 0).
    leakage_checks: dict[str, int] = Field(default_factory=dict)
    # -- Field-completeness diagnostics (all measured from the same records) --
    #: Papers whose title field is absent or blank.
    missing_titles: int = 0
    #: Papers with no abstract text stored.
    missing_abstracts: int = 0
    #: Papers with no author metadata.
    missing_authors: int = 0
    #: Papers with no publication year.
    missing_years: int = 0
    #: Papers in the shortest decile of the corpus's own word-count
    #: distribution (a relative outlier definition, not an absolute rule).
    short_papers: int = 0
    #: Papers in the longest decile of the same distribution.
    long_papers: int = 0
    #: Distinct normalized titles shared by more than one paper.
    duplicate_titles: int = 0
    #: Build-time provenance from the dataset manifest, when it was recorded.
    provenance: DatasetProvenance | None = None


class DatasetProvenance(BaseModel):
    """Where the corpus came from, exactly as the dataset manifest recorded it.

    Every field is optional because older manifests may not carry it; the UI
    shows "unavailable" rather than a guess for anything absent.
    """

    model_config = _RESPONSE

    source: str | None = None
    created_at: str | None = None
    git_commit: str | None = None
    seed: int | None = None
    is_synthetic: bool = False
    split_ratios: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Question answering
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    """A question about one paper."""

    model_config = _REQUEST

    question: Annotated[str, Field(min_length=1, max_length=2000)]


class PassageEvidence(BaseModel):
    """Evidence passage supporting a Q&A answer."""

    model_config = _RESPONSE

    source_section: str
    passage: str
    confidence: float


class AskResponse(BaseModel):
    """Answer payload returned by the paper Q&A engine."""

    model_config = _RESPONSE

    paper_id: str
    question: str
    answer: str
    source: str | None = None
    evidence: list[PassageEvidence] = Field(default_factory=list)
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# Per-paper methodology extraction
# ---------------------------------------------------------------------------
class MethodologyEvidenceItem(BaseModel):
    """One extracted methodology signal, with where it came from.

    ``statement`` is a verbatim sentence from the paper and ``section`` is the
    parsed section it was found in — never generated. Fields with no matches
    simply carry an empty ``items`` list; the UI renders that as unavailable.
    """

    model_config = _RESPONSE

    value: str
    section: str
    statement: str


class MethodologyField(BaseModel):
    """All extracted signals for one methodology field of one paper."""

    model_config = _RESPONSE

    field: str
    items: list[MethodologyEvidenceItem] = Field(default_factory=list)


class PaperMethodologyResponse(BaseModel):
    """Structured methodology extraction for one paper, with provenance."""

    model_config = _RESPONSE

    paper_id: str
    title: str
    fields: list[MethodologyField] = Field(default_factory=list)
    n_sections_parsed: int = 0
    n_passages_scanned: int = 0
    run_id: str
    basis: str = (
        "Regex pattern extraction over the parsed paper sections "
        "(src/analytics/methodology.py, src/analytics/gaps.py). "
        "Every item quotes the paper verbatim; unmatched fields are omitted."
    )





# ---------------------------------------------------------------------------
# Full analysis aggregation
# ---------------------------------------------------------------------------
class ConfusionMatrixSchema(BaseModel):
    """Confusion matrix for one evaluation split."""

    model_config = _RESPONSE

    labels: list[str] = Field(default_factory=list)
    counts: list[list[int]] = Field(default_factory=list)
    normalized: list[list[float]] = Field(default_factory=list)


class SplitMetrics(BaseModel):
    """Metrics for one evaluation split."""

    model_config = _RESPONSE

    n_samples: int
    n_classes: int
    accuracy: float | None = None
    balanced_accuracy: float | None = None
    macro_precision: float | None = None
    macro_recall: float | None = None
    macro_f1: float | None = None
    weighted_f1: float | None = None
    per_class: dict[str, dict[str, float]] = Field(default_factory=dict)
    confusion_matrix: ConfusionMatrixSchema | None = None


class RunMetricsSummary(BaseModel):
    """Metrics across splits for the active run."""

    model_config = _RESPONSE

    val: SplitMetrics | None = None
    test: SplitMetrics | None = None
    train_samples: int = 0


class PaperMetadata(BaseModel):
    """Paper metadata derived from the record."""

    model_config = _RESPONSE

    paper_id: str
    title: str
    authors_short: str | None = None
    year: int | None = None
    venue: str | None = None
    word_count: int = 0
    n_sections: int = 0
    n_references: int = 0
    split: str
    true_label: str | None = None
    abstract: str | None = None
    keywords: list[str] = Field(default_factory=list)


class PredictionInfo(BaseModel):
    """The model's prediction for this paper."""

    model_config = _RESPONSE

    predicted_label: str | None = None
    confidence: float | None = None
    confidence_kind: str = "probability"
    needs_review: bool | None = None
    review_band: str | None = None
    review_reason: str | None = None
    correct: bool | None = None
    scores: list = Field(default_factory=list)


class ReviewAssessment(BaseModel):
    """Derived review-assessment facts computed server-side."""

    model_config = _RESPONSE

    needs_review: bool
    review_reason: str | None = None
    top_label: str | None = None
    top_score: float | None = None
    second_label: str | None = None
    second_score: float | None = None
    confidence_gap: float | None = None


class TokenHighlight(BaseModel):
    """A token with its attention weight for highlighting."""

    model_config = _RESPONSE

    token: str
    weight: float


class AnalysisPaper(BaseModel):
    """Paper metadata for the analysis page."""

    model_config = _RESPONSE

    paper_id: str
    title: str | None = None
    authors_short: str | None = None
    year: int | None = None
    venue: str | None = None
    split: str | None = None
    true_label: str | None = None
    n_authors: int | None = None
    n_references: int | None = None
    n_sections: int | None = None
    word_count: int = 0
    text_length: int = 0


class ClassProbability(BaseModel):
    """One class probability in the prediction ranking."""

    model_config = _RESPONSE

    label: str
    score: float


class AnalysisPrediction(BaseModel):
    """The model's prediction for this paper."""

    model_config = _RESPONSE

    predicted_label: str | None = None
    confidence: float | None = None
    confidence_kind: str = "probability"
    needs_review: bool | None = None
    review_band: str | None = None
    correct: bool | None = None
    probabilities: list[ClassProbability] = Field(default_factory=list)


class AttentionTerm(BaseModel):
    """One term's contribution to the decision."""

    model_config = _RESPONSE

    term: str
    contribution: float
    weight: float


class AnalysisAttention(BaseModel):
    """Attention and evidence data for the analysis page."""

    model_config = _RESPONSE

    method: str
    evidence_label: str | None = None
    terms: list[AttentionTerm] = Field(default_factory=list)
    section_attention: SectionAttention
    sentences: list[SentenceEvidence] = Field(default_factory=list)
    tokens: list[AttentionTerm] = Field(default_factory=list)
    caveat: str | None = None


class AnalysisModel(BaseModel):
    """Model and metrics data for the analysis page."""

    model_config = _RESPONSE

    run_id: str
    model_name: str | None = None
    model_display_name: str | None = None
    confidence_kind: str = "probability"
    is_han: bool = False
    similarity_method: str = "tfidf_cosine"
    classes: list[str] = Field(default_factory=list)
    #: Paper counts per dataset split, from the run manifest (``dataset.split_sizes``).
    split_sizes: dict[str, int] = Field(default_factory=dict)
    #: Split the run's headline metric describes (from the manifest).
    primary_split: str | None = None
    val_metrics: SplitMetrics | None = None
    test_metrics: SplitMetrics | None = None
    train_metrics: SplitMetrics | None = None


class GroundTruth(BaseModel):
    """Ground truth comparison for the analysis page."""

    model_config = _RESPONSE

    true_label: str | None = None
    predicted_label: str | None = None
    correct: bool | None = None
    is_unlabelled: bool = True


class EmbeddingPoint(BaseModel):
    """2D embedding coordinates for a paper in the semantic space."""

    model_config = _RESPONSE

    paper_id: str
    title: str
    x: float
    y: float
    domain: str | None = None
    is_current: bool = False


class AnalysisEmbedding(BaseModel):
    """Transformer embedding metadata and 2D semantic projection."""

    model_config = _RESPONSE

    model_name: str = "allenai/scibert_scivocab_uncased"
    dimension: int = 768
    representation: str = "Document-level Transformer embedding"
    method: str = "2D Semantic Projection"
    points: list[EmbeddingPoint] = Field(default_factory=list)
    caveat: str | None = None


class AnalysisInsights(BaseModel):
    """Extracted scientific and methodological insights for the paper."""

    model_config = _RESPONSE

    research_domain: str | None = None
    primary_topic: str | None = None
    key_concepts: list[str] = Field(default_factory=list)
    methodology: str | None = None
    main_contribution: str | None = None
    potential_limitations: list[str] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    """Complete analysis payload for the Full Paper Analysis page.

    Aggregates the paper record, its prediction, explanation evidence,
    similar-paper neighbours, and run-level model metrics into one
    response so the analysis page renders from a single fetch.
    """

    model_config = _RESPONSE

    paper: AnalysisPaper
    prediction: AnalysisPrediction
    review: ReviewAssessment
    attention: AnalysisAttention
    model: AnalysisModel
    similar_papers: SimilarResponse
    ground_truth: GroundTruth
    embedding: AnalysisEmbedding | None = None
    insights: AnalysisInsights | None = None

class ErrorResponse(BaseModel):
    """Uniform error body.

    Every failure the API produces on purpose has this shape, so the client has
    one error path instead of guessing whether a body holds ``detail``,
    ``message``, or an HTML page from a proxy.
    """

    model_config = _RESPONSE

    error: str
    detail: str
    #: Populated when there is a concrete next step, e.g. the command to run.
    hint: str | None = None


# ---------------------------------------------------------------------------
# Research analytics responses (Phase B)
# ---------------------------------------------------------------------------


class GapExample(BaseModel):
    """A single sentence the gap detector matched against a category."""

    model_config = _RESPONSE

    category: str
    paper_id: str
    title: str
    statement: str
    confidence: float = Field(ge=0.0, le=1.0)
    #: Where in the record the sentence came from. The detector scans title +
    #: abstract only (that is what the dataset stores as model input), so this
    #: is a fact about the scan scope, not a page reference.
    source: str = "Abstract"


class GapCategory(BaseModel):
    """A gap category with its paper count and supporting examples."""

    model_config = _RESPONSE

    name: str
    count: int = Field(ge=0)
    examples: list[GapExample] = Field(default_factory=list)


class GapsResponse(BaseModel):
    """Aggregate of detected research gaps across the active corpus."""

    model_config = _RESPONSE

    categories: list[GapCategory] = Field(default_factory=list)
    n_papers_scanned: int = Field(ge=0)
    n_papers_with_gaps: int = Field(ge=0)
    run_id: str
    basis: str = "Pattern-matched against the corpus's paper text. See src/analytics/gaps.py for the regex set."


class MethodologyItem(BaseModel):
    """One ranked term inside a methodology bucket."""

    model_config = _RESPONSE

    name: str
    count: int = Field(ge=0)


class MethodologyBucket(BaseModel):
    """One of the four methodology buckets (datasets, metrics, architectures, algorithms)."""

    model_config = _RESPONSE

    name: Literal["datasets", "metrics", "architectures", "algorithms"]
    items: list[MethodologyItem] = Field(default_factory=list)


class MethodologyResponse(BaseModel):
    """Aggregate of the four methodology buckets across the active corpus."""

    model_config = _RESPONSE

    buckets: list[MethodologyBucket] = Field(default_factory=list)
    n_papers_scanned: int = Field(ge=0)
    run_id: str
    basis: str = "Regex extraction over the corpus's paper text. See src/analytics/methodology.py for the term lists."


class CitationNodeSchema(BaseModel):
    """A paper node in the citation graph response."""

    model_config = _RESPONSE

    id: str
    label: str
    domain: str | None = None


class CitationEdgeSchema(BaseModel):
    """A directed reference link between two papers in the corpus."""

    model_config = _RESPONSE

    source: str
    target: str


class CitationGraphResponse(BaseModel):
    """Citation graph plus summary stats for the dashboard's Citation Network panel."""

    model_config = _RESPONSE

    nodes: list[CitationNodeSchema] = Field(default_factory=list)
    edges: list[CitationEdgeSchema] = Field(default_factory=list)
    n_nodes: int = Field(ge=0)
    n_edges: int = Field(ge=0)
    n_components: int = Field(ge=0)
    run_id: str
    basis: str = "Paper-to-paper reference links present in the active run's records."

