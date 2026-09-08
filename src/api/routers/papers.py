"""Paper endpoints: listing, detail, classification, similarity, explanation, ask.

Two of these do real work rather than reading a file. ``POST /classify`` runs new
text through the run's fitted pipeline, and ``/similar`` computes cosine distance
in that pipeline's own TF-IDF space. The rest read what the training run already
wrote, so a number shown in the dashboard is the same number in ``report.md``.

``POST /{paper_id}/ask`` answers questions about one paper: passage retrieval
over that paper's own text, then grounded synthesis through Groq when a key is
configured. A generated response with no retrieval behind it would be the one
outcome master spec §20 forbids, so the passages are always retrieved first and
returned as evidence alongside the answer.
"""

from __future__ import annotations

import re
from typing import Annotated

import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from src.api.capabilities import (
    ATTENTION_UNAVAILABLE_REASON,
    EXPLANATION_CAVEAT,
    HAN_EXPLANATION_CAVEAT,
    SEMANTIC_SIMILARITY_CAVEAT,
    SIMILARITY_CAVEAT,
    classification_caveat,
)
from src.api.deps import ActiveRun, Pagination, SettingsDep
from src.analytics.gaps import ResearchGapDetector
from src.analytics.methodology import MethodologyExtractor
from src.api.retrieval import GroqSynthesisError, PaperQAEngine
from src.api.runstore import HELD_OUT_SPLITS, LoadedRun, PaperEntry, RunUnavailableError
from src.api.schemas import (
    AnalysisAttention,
    AnalysisEmbedding,
    AnalysisInsights,
    AnalysisModel,
    AnalysisPaper,
    AnalysisPrediction,
    AnalysisResponse,
    AskRequest,
    AskResponse,
    AttentionTerm,
    ClassifyRequest,
    ClassifyResponse,
    ClassifyResult,
    ClassProbability,
    ConfusionMatrixSchema,
    EmbeddingPoint,
    ExplanationResponse,
    GroundTruth,
    LabelScore,
    MethodologyEvidenceItem,
    MethodologyField,
    PaperMethodologyResponse,
    PaperDetail,
    PaperListResponse,
    PaperSummary,
    ReviewAssessment,
    SectionAttention,
    SectionWeight,
    SentenceEvidence,
    SimilarItem,
    SimilarResponse,
    SplitMetrics,
    TermWeight,
)
from src.ingestion.pdf_parser import PDFPaperParser
from src.preprocessing.sections import parse_text_into_sections
from src.preprocessing.sentence_splitter import split_sentences
from src.schemas.paper import DatasetRecord
from src.utils.logging import get_logger

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(prefix="/papers", tags=["papers"])

#: Accepted values for the ``split`` query parameter. ``held_out`` is the default
#: because a prediction on the training split is not evidence: the model was
#: fitted on those rows, so listing them beside held-out rows would quietly mix
#: training accuracy into what reads as a results table.
_SPLIT_FILTERS: dict[str, tuple[str, ...]] = {
    "held_out": HELD_OUT_SPLITS,
    "val": ("val",),
    "test": ("test",),
    "train": ("train",),
    "all": (),
}


def _summary(run: LoadedRun, entry: PaperEntry) -> PaperSummary:
    """Build the table row for one paper."""
    prediction = entry.prediction or {}
    confidence = prediction.get("confidence")
    confidence = float(confidence) if isinstance(confidence, int | float) else None
    kind = str(prediction.get("confidence_kind") or run.confidence_kind)

    return PaperSummary(
        paper_id=entry.paper_id,
        title=entry.record.title or entry.paper_id,
        authors_short=entry.authors_short,
        year=entry.year,
        split=entry.split,
        true_label=entry.record.label,
        predicted_label=prediction.get("predicted_label"),
        correct=prediction.get("correct"),
        confidence=confidence,
        confidence_kind=kind,
        # Derived here, never on the client: the threshold has one home
        # (master spec §15).
        needs_review=run.needs_review(confidence, kind) if entry.prediction else None,
        review_band=run.review_band(confidence, kind) if entry.prediction else None,
    )


def _detail(run: LoadedRun, entry: PaperEntry) -> PaperDetail:
    """Build the preview payload for one paper."""
    base = _summary(run, entry)
    prediction = entry.prediction or {}
    scores = prediction.get("top_scores")
    # Older artifacts stored only the winning label. Re-score those records so
    # the detail view always has ranked per-class scores.
    if not scores:
        try:
            live = run.classify([entry.record.text])[0]
            scores = live.get("scores")
        except RunUnavailableError:
            scores = []
    meta = entry.record.meta

    return PaperDetail(
        **base.model_dump(),
        text=entry.record.text,
        labels=list(entry.record.labels),
        venue=meta.get("venue") if isinstance(meta.get("venue"), str) else None,
        n_authors=meta.get("n_authors") if isinstance(meta.get("n_authors"), int) else None,
        n_references=entry.n_references,
        predicted_scores=[
            LabelScore(label=str(item["label"]), score=float(item["score"]))
            for item in (scores or [])
            if isinstance(item, dict) and "label" in item and "score" in item
        ],
    )


def _require_paper(run: LoadedRun, paper_id: str) -> PaperEntry:
    """Look up a paper or raise 404.

    The lookup is a dictionary hit against the loaded corpus. No part of
    ``paper_id`` reaches the filesystem, so a traversal attempt returns 404 like
    any other unknown id (master spec §40).
    """
    entry = run.paper(paper_id)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No paper '{paper_id}' in run '{run.run_id}'.",
        )
    return entry


@router.get("", response_model=PaperListResponse, summary="Browse the corpus")
def list_papers(
    run: ActiveRun,
    page: Pagination,
    split: str = Query(
        "held_out",
        description="Which splits to include. 'held_out' covers val and test.",
    ),
    q: str | None = Query(None, max_length=200, description="Case-insensitive title match."),
    needs_review: bool | None = Query(
        None, description="Restrict to predictions above or below the review threshold."
    ),
) -> PaperListResponse:
    """Return a page of papers, filtered and searched.

    Raises:
        HTTPException: 422 when ``split`` is not one of the accepted values.
    """
    if split not in _SPLIT_FILTERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"split must be one of {sorted(_SPLIT_FILTERS)}; got '{split}'.",
        )

    wanted = _SPLIT_FILTERS[split]
    entries = run.entries(wanted or None)

    rows = [_summary(run, entry) for entry in entries]

    if q:
        needle = q.strip().lower()
        rows = [
            row
            for row in rows
            if needle in row.title.lower()
            or needle in row.paper_id.lower()
            or (row.true_label and needle in row.true_label.lower())
        ]
    if needs_review is not None:
        rows = [row for row in rows if row.needs_review is needs_review]

    total = len(rows)
    window = rows[page.offset : page.offset + page.limit]
    return PaperListResponse(
        items=window,
        total=total,
        limit=page.limit,
        offset=page.offset,
        splits=list(wanted) if wanted else sorted({row.split for row in rows}),
        query=q,
    )


@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify new text with the active run's model",
)
def classify(payload: ClassifyRequest, run: ActiveRun, settings: SettingsDep) -> ClassifyResponse:
    """Run a real forward pass over submitted title and abstract.

    The fields are joined in the order ``text.fields`` specifies, which is the
    same composition the dataset build used, so an ad-hoc classification reaches
    the vectorizer in the form it was fitted on. Joining them in some other order
    would feed the model a distribution it never saw.

    Raises:
        HTTPException: 503 when the run has no saved model to load.
    """
    parts = {"title": payload.title, "abstract": payload.abstract}
    composed = "\n\n".join(
        parts[field] for field in settings.app.text.fields if parts.get(field)
    ).strip()
    if not composed:
        # Reachable when text.fields excludes every field the request supplied.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No usable text: this run was built from fields "
                f"{settings.app.text.fields}, and none of them were supplied."
            ),
        )

    try:
        outcome = run.classify([composed])[0]
    except RunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return ClassifyResponse(
        result=ClassifyResult(
            predicted_label=outcome["predicted_label"],
            confidence=outcome["confidence"],
            confidence_kind=outcome["confidence_kind"],
            scores=[LabelScore(**score) for score in outcome["scores"]],
            probabilities=dict(outcome.get("probabilities") or {}),
            decision_scores=dict(outcome.get("decision_scores") or {}),
            needs_review=outcome["needs_review"],
            review_band=outcome.get("review_band"),
            review_reason=outcome.get("review_reason"),
            split="inference",
            model_name=run.model_display_name,
            model_version=run.run_id,
        ),
        run_id=run.run_id,
        model_display_name=run.model_display_name,
        caveat=classification_caveat(run),
    )


@router.get("/{paper_id}", response_model=PaperDetail, summary="One paper in full")
def get_paper(paper_id: str, run: ActiveRun) -> PaperDetail:
    """Return one paper's record and the run's prediction for it."""
    return _detail(run, _require_paper(run, paper_id))


@router.get(
    "/{paper_id}/similar",
    response_model=SimilarResponse,
    summary="Lexically nearest papers in the corpus",
)
def similar_papers(
    paper_id: str,
    run: ActiveRun,
    limit: int | None = Query(None, ge=1, le=50, description="Neighbours to return."),
) -> SimilarResponse:
    """Return the corpus papers with the most vocabulary in common.

    Cosine distance between TF-IDF vectors in the run's own fitted space. This is
    lexical overlap, and ``method`` plus ``caveat`` say so in the payload so the
    UI cannot present it as semantic similarity (master spec §17).

    Raises:
        HTTPException: 404 for an unknown paper; 503 when the model is unloadable.
    """
    _require_paper(run, paper_id)
    return _build_similar(run, paper_id, limit or 5, raise_on_error=True)


@router.get(
    "/{paper_id}/explanation",
    response_model=ExplanationResponse,
    summary="Which terms drove the predicted label",
)
def explanation(paper_id: str, run: ActiveRun) -> ExplanationResponse:
    """Decompose the model's decision for one paper into per-term contributions.

    ``section_attention`` is always the unavailable marker. The dashboard has a
    section-attention panel, and this endpoint is where it would be filled; it is
    not filled, because a bag-of-words model has no section representation to
    weight. Splitting the abstract into thirds and summing term weights would draw
    the same chart while meaning something else entirely.

    Raises:
        HTTPException: 404 for an unknown paper; 503 when the model is unloadable.
    """
    entry = _require_paper(run, paper_id)
    prediction = entry.prediction or {}
    pred_label = prediction.get("predicted_label")
    return _build_explanation(run, entry, pred_label, run.confidence_kind)


def _han_explanation(run: LoadedRun, entry: PaperEntry) -> ExplanationResponse:
    """Serve the HAN's real attention for one paper, aligned to its text.

    The weights come from ``HANClassifier.explain`` — the trained network's own
    additive attention at both levels of the hierarchy, joined back to the
    sentences and sections they weighted. Nothing here is heuristic.
    """
    try:
        explained = run.han_model.explain(entry.record.text)
    except RunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    sections = [
        SectionWeight(
            name=str(item["name"]),
            canonical_name=str(item["canonical_name"]),
            weight=round(float(item["weight"]), 6),
        )
        for item in explained["sections"]
    ]
    sentences = [
        SentenceEvidence(
            section_name=str(item["section_name"]),
            canonical_name=str(item["canonical_name"]),
            text=str(item["text"]),
            weight=round(float(item["weight"]), 6),
        )
        for item in explained["sentences"]
    ]

    return ExplanationResponse(
        paper_id=entry.paper_id,
        predicted_label=str(explained["predicted_label"]),
        method="han_hierarchical_attention",
        evidence_label="Hierarchical model attention",
        section_attention=SectionAttention(
            available=True,
            sections=sections,
            kind="hierarchical_section_attention",
        ),
        sentences=sentences,
        caveat=HAN_EXPLANATION_CAVEAT,
    )


def _decision_value(run: LoadedRun, text: str, label: str) -> float | None:
    """Return the raw decision value for ``label``, or ``None`` if unavailable.

    Reported alongside the term list so the contributions can be checked against
    the model's own arithmetic instead of taken on trust.
    """
    classifier = run.classifier
    if not hasattr(classifier, "decision_function"):
        return None
    classes = [str(name) for name in getattr(classifier, "classes_", run.classes)]
    if label not in classes:
        return None

    values = np.asarray(classifier.decision_function(run.vectorizer.transform([text])))
    row = values[0] if values.ndim > 1 else values
    if np.ndim(row) == 0:
        # Binary: one signed value, oriented toward classes_[1].
        return float(row) if classes.index(label) == 1 else -float(row)
    return float(row[classes.index(label)])


@router.post(
    "/{paper_id}/ask",
    response_model=AskResponse,
    summary="Answer questions about a paper using passage retrieval",
)
def ask(paper_id: str, payload: AskRequest, run: ActiveRun) -> AskResponse:
    """Answer a question about a paper using passage retrieval plus Groq synthesis.

    Segments the paper into candidate section/paragraph passages, scores passage
    relevance against the question, and returns a grounded answer alongside the
    evidence passages and their section provenance. The Groq key stays in the
    server process: the client never sees it, only the synthesized answer.
    """
    entry = _require_paper(run, paper_id)
    engine = PaperQAEngine(
        paper_id=entry.paper_id,
        title=entry.record.title or entry.paper_id,
        text=entry.record.text,
        groq_api_key=run.settings.env.groq_api_key,
        groq_model=run.settings.env.groq_model,
    )
    comparison_context = _comparison_context(run, paper_id, payload.question)
    try:
        return engine.answer_question(
            payload.question, comparison_context=comparison_context
        )
    except GroqSynthesisError as exc:
        raise _groq_http_error(exc) from exc


def _comparison_context(
    run: LoadedRun, paper_id: str, question: str
) -> list[dict[str, object]]:
    """Corpus neighbours for a comparison question, or an empty list.

    The run's own similar-papers feature (``run.similar``) is reused rather
    than re-implemented: the neighbours are the same lexical matches the
    Similar Papers panel shows. A short excerpt of each neighbour's stored text
    travels along so the model can say something concrete about the comparison
    instead of only listing titles.
    """
    if not re.search(
        r"\b(compare|comparison|similar|versus|vs\.?|differ)\b", question, re.IGNORECASE
    ):
        return []
    try:
        neighbours = run.similar(paper_id, top_k=3)
    except RunUnavailableError as exc:
        logger.warning("Ask | similar-papers lookup failed for comparison: %s", exc)
        return []

    context: list[dict[str, object]] = []
    for neighbour in neighbours:
        neighbour_id = str(neighbour.get("paper_id", ""))
        entry = run.paper(neighbour_id)
        context.append(
            {
                "title": neighbour.get("title") or neighbour_id,
                "label": neighbour.get("label"),
                "score": neighbour.get("score", 0.0),
                "excerpt": (entry.record.text or "")[:600] if entry else "",
            }
        )
    return context


def _groq_http_error(exc: GroqSynthesisError) -> HTTPException:
    """Map a synthesis failure onto a user-facing error.

    The wording distinguishes what the user can do about it: a credentials
    problem is an operator fix, a rate limit is "wait", and anything else is
    "retry". The Groq status and message stay in the server log.
    """
    if exc.status_code in (401, 403):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "AI analysis is not configured correctly. The Groq API rejected "
                "the server's credentials. Please check the GROQ_API_KEY."
            ),
        )
    if exc.status_code == 429:
        return HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="The AI service is temporarily rate limited. Please try again shortly.",
        )
    logger.warning("Ask | Groq synthesis failed: %s", exc)
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail=(
            "The paper was retrieved successfully, but the AI synthesis step "
            "failed. Please try again."
        ),
    )


@router.post(
    "/upload",
    response_model=PaperDetail,
    summary="Upload and parse a paper or PDF document",
)
async def upload_paper(
    run: ActiveRun, file: Annotated[UploadFile, File(...)]
) -> PaperDetail:
    """Upload a PDF or text paper document, parse its canonical sections, and return the detail."""
    content = await file.read()
    parser = PDFPaperParser()
    doc = parser.parse_bytes(content, filename=file.filename or "uploaded.pdf")

    full_text = doc.full_text or doc.text_for(("title", "abstract")) or doc.title

    # An uploaded paper has no ground-truth label: the model's prediction is the
    # only label in play, and it is carried by the prediction fields below —
    # never by ``record.label``, which means ground truth everywhere else.
    record = DatasetRecord(
        paper_id=doc.paper_id,
        text=full_text,
        title=doc.title,
        label=None,
        split=None,
        meta={
            "year": doc.publication_year,
            "first_author": doc.authors[0].name if doc.authors else None,
            "n_authors": len(doc.authors),
            "n_references": len(doc.references),
            "abstract": doc.abstract,
            "keywords": doc.keywords,
        },
    )

    # Classify uploaded text through the same fitted-model path as /classify.
    classified_label = "Uncategorized"
    outcome: dict[str, object] = {}
    try:
        outcome = run.classify([full_text])[0]
        classified_label = str(outcome["predicted_label"])
    except RunUnavailableError:
        outcome = {}

    entry = PaperEntry(
        record=record,
        split="uploaded",
        prediction={
            "paper_id": doc.paper_id,
            "predicted_label": classified_label,
            "confidence": outcome.get("confidence"),
            "confidence_kind": outcome.get("confidence_kind", run.confidence_kind),
            "top_scores": outcome.get("scores", []),
            "needs_review": outcome.get("needs_review"),
            "review_band": outcome.get("review_band"),
        },
    )
    run.papers[doc.paper_id] = entry

    return PaperDetail(
        paper_id=doc.paper_id,
        title=doc.title,
        text=full_text,
        authors_short=entry.authors_short,
        year=doc.publication_year,
        split="uploaded",
        true_label=None,
        predicted_label=classified_label,
        confidence=outcome.get("confidence"),
        needs_review=outcome.get("needs_review"),
        review_band=outcome.get("review_band"),
        confidence_kind=str(outcome.get("confidence_kind") or run.confidence_kind),
        labels=[],
        n_references=len(doc.references),
        predicted_scores=[
            LabelScore(label=str(item["label"]), score=float(item["score"]))
            for item in outcome.get("scores", [])
            if isinstance(item, dict) and "label" in item and "score" in item
        ],
    )


@router.get(
    "/{paper_id}/analysis",
    response_model=AnalysisResponse,
    summary="Complete analysis payload for the full analysis page",
)
def analysis(paper_id: str, run: ActiveRun, limit: int = Query(5, ge=1, le=20)) -> AnalysisResponse:
    """Aggregate paper detail, explanation, similarity, and metrics into one payload.

    The full analysis page consumes this single response rather than firing
    multiple requests, so the page renders consistently even when individual
    components (section attention, similarity) are unavailable.
    """
    entry = _require_paper(run, paper_id)
    detail = _detail(run, entry)
    explanation = _build_explanation(run, entry, detail.predicted_label, detail.confidence_kind)
    similar = _build_similar(run, paper_id, limit=limit)

    metrics = run.metrics or {}
    val_metrics = _split_metrics(metrics.get("val"))
    test_metrics = _split_metrics(metrics.get("test"))
    train_metrics = _split_metrics(metrics.get("train"))

    scores = detail.predicted_scores
    confidence_gap = None
    review_reason = None
    if scores and len(scores) >= 2:
        top = scores[0].score if scores else None
        second = scores[1].score if len(scores) > 1 else None
        if top is not None and second is not None and detail.confidence_kind == "probability":
            confidence_gap = round(top - second, 4)
            if confidence_gap < 0.15 and top < 0.7:
                review_reason = "The top prediction has moderate confidence and is relatively close to the second-highest prediction."
            elif top < 0.5:
                review_reason = "The top prediction has low confidence, indicating high uncertainty."

    # 2D Embedding projection. The projection must match the label it carries:
    # the SciBERT run gets a SciBERT document vector reduced to 2D (semantic
    # proximity in the encoder's own space), the baseline gets TF-IDF + SVD
    # (lexical proximity in the fitted vectorizer's space). Mixing one with
    # the other's label is the bug this replaces.
    embedding_obj = None
    try:
        from sklearn.decomposition import TruncatedSVD

        sample_papers = list(run.papers.values())[:60]
        texts = [p.record.text or p.record.title or "" for p in sample_papers]
        if len(texts) < 2:
            raise ValueError("Need at least two papers to project.")

        coords: np.ndarray
        representation: str
        model_name: str
        dimension: int
        if run.is_han:
            # The same frozen encoder the HAN consumed: mean-pooled
            # sentence vectors, then SVD to 2D. Reuses the disk-cached
            # encoder, so repeat projections do not re-encode.
            try:
                documents = run.han_model.embed_documents(texts)
            except RunUnavailableError as exc:
                logger.warning("HAN encoder unavailable for 2D projection: %s", exc)
                raise
            doc_vecs = [
                v if v.size else np.zeros(run.han_model.encoder.embedding_dim, dtype=np.float32)
                for v in (
                    np.mean(
                        np.asarray(sentence, dtype=np.float32),
                        axis=0,
                    )
                    for section in documents
                    for sentence in section
                )
            ]
            matrix = np.vstack(doc_vecs) if doc_vecs else np.zeros((0, 0), dtype=np.float32)
            dimension = int(matrix.shape[1]) if matrix.ndim == 2 else 0
            if matrix.size == 0 or matrix.shape[0] < 2:
                raise ValueError("Insufficient document vectors for 2D projection.")
            svd = TruncatedSVD(n_components=2, random_state=42)
            coords = svd.fit_transform(matrix)
            model_name = "allenai/scibert_scivocab_uncased"
            representation = "Document-level Transformer embedding"
        else:
            if not hasattr(run, "vectorizer") or run.vectorizer is None:
                raise ValueError("No fitted vectorizer on this run.")
            X = run.vectorizer.transform(texts)
            dimension = int(X.shape[1])
            svd = TruncatedSVD(n_components=2, random_state=42)
            coords = svd.fit_transform(X)
            model_name = "TF-IDF + TruncatedSVD"
            representation = "Corpus-level lexical vector (TF-IDF + SVD)"

        max_abs = float(np.max(np.abs(coords))) or 1.0
        coords = coords / max_abs
        points = [
            EmbeddingPoint(
                paper_id=p.paper_id,
                title=p.record.title or p.paper_id,
                x=round(float(coords[i, 0]), 4),
                y=round(float(coords[i, 1]), 4),
                domain=p.record.label
                or (p.prediction.get("predicted_label") if p.prediction else None),
                is_current=(p.paper_id == paper_id),
            )
            for i, p in enumerate(sample_papers)
        ]
        embedding_obj = AnalysisEmbedding(
            model_name=model_name,
            dimension=dimension,
            representation=representation,
            method="2D SVD / PCA Projection",
            points=points,
            caveat=(
                "2D projection reduces the high-dimensional embedding space to 2 "
                "principal components for visualization only. Proximity in this "
                "plot is proximity in the projected space, not in the original."
            ),
        )
    except RunUnavailableError:
        # Surface as unavailable rather than silently labelling the run as
        # having a plot it cannot produce.
        embedding_obj = None
    except Exception as exc:
        logger.warning("Failed to compute embedding projection: %s", exc)

    # Scientific Insights
    import re
    full_text = entry.record.text or ""
    arch_matches = re.findall(
        r"\b(Transformer|SciBERT|BERT|RoBERTa|Attention|Multi-Head Attention|Self-Attention|BiGRU|Bi-GRU|GRU|LSTM|CNN|GNN|LinearSVC|LogisticRegression)\b",
        full_text,
        re.IGNORECASE,
    )
    methodology_str = (
        ", ".join(sorted(set(m.title() for m in arch_matches))[:4])
        if arch_matches
        else ("Hierarchical Attention Network (HAN)" if run.is_han else "TF-IDF + Linear Classification")
    )

    parsed_sections = parse_text_into_sections(full_text, title=entry.record.title)
    contribution_sentence = None
    abstract_sec = next(
        (s for s in parsed_sections if s.canonical_name == "abstract" or "abstract" in s.section_name.lower()),
        None,
    )
    if abstract_sec and abstract_sec.text:
        sents = split_sentences(abstract_sec.text)
        for s in sents:
            if re.search(r"\b(propose|present|introduce|show|demonstrate|investigate|develop|classify)\b", s, re.I):
                contribution_sentence = s.strip()
                break
        if not contribution_sentence and sents:
            contribution_sentence = sents[0].strip()
    if not contribution_sentence:
        sents = split_sentences(full_text[:2000])
        contribution_sentence = sents[0].strip() if sents else "Classified research document within the academic corpus."

    limitations = []
    for s in split_sentences(full_text):
        if re.search(r"\b(limitation|bottleneck|future work|open challenge|computational cost|scalability|interpretability)\b", s, re.I):
            limitations.append(s.strip())
            if len(limitations) >= 3:
                break
    if not limitations:
        limitations = ["No explicit limitations or open challenges extracted from paper text."]

    key_terms = [t.term for t in explanation.terms if t.contribution > 0][:6]
    primary_topic = " / ".join(key_terms[:2]) if key_terms else (detail.predicted_label or "Academic Classification")

    insights_obj = AnalysisInsights(
        research_domain=detail.predicted_label or detail.true_label,
        primary_topic=primary_topic,
        key_concepts=key_terms,
        methodology=methodology_str,
        main_contribution=contribution_sentence[:300] if contribution_sentence else None,
        potential_limitations=limitations,
    )

    return AnalysisResponse(
        paper=AnalysisPaper(
            paper_id=detail.paper_id,
            title=detail.title,
            authors_short=detail.authors_short,
            year=detail.year,
            venue=detail.venue,
            split=detail.split,
            true_label=detail.true_label,
            n_authors=detail.n_authors,
            n_references=detail.n_references,
            n_sections=(
                len(explanation.section_attention.sections)
                if explanation.section_attention and explanation.section_attention.sections
                else None
            ),
            word_count=len(entry.record.text.split()) if entry.record.text else 0,
            text_length=len(entry.record.text) if entry.record.text else 0,
        ),
        prediction=AnalysisPrediction(
            predicted_label=detail.predicted_label,
            confidence=detail.confidence,
            confidence_kind=detail.confidence_kind,
            needs_review=detail.needs_review,
            review_band=detail.review_band,
            correct=detail.correct,
            probabilities=[
                ClassProbability(label=s.label, score=s.score)
                for s in scores
            ],
        ),
        review=ReviewAssessment(
            needs_review=detail.needs_review or False,
            confidence_gap=confidence_gap,
            review_reason=review_reason,
            top_label=scores[0].label if scores else None,
            top_score=scores[0].score if scores else None,
            second_label=scores[1].label if len(scores) > 1 else None,
            second_score=scores[1].score if len(scores) > 1 else None,
        ),
        attention=AnalysisAttention(
            method=explanation.method,
            evidence_label=explanation.evidence_label,
            terms=[
                AttentionTerm(
                    term=t.term,
                    contribution=t.contribution,
                    weight=t.weight,
                )
                for t in explanation.terms
            ],
            section_attention=explanation.section_attention,
            sentences=explanation.sentences,
            tokens=[
                AttentionTerm(
                    term=t.term,
                    contribution=t.contribution,
                    weight=t.weight,
                )
                for t in explanation.terms
            ],
            caveat=explanation.caveat,
        ),
        model=AnalysisModel(
            run_id=run.run_id,
            model_name=run.model_name,
            model_display_name=run.model_display_name,
            confidence_kind=run.confidence_kind,
            is_han=run.is_han,
            similarity_method=run.similarity_method,
            classes=run.classes,
            split_sizes={
                str(name): int(size)
                for name, size in (run.manifest.get("dataset", {}).get("split_sizes", {}) or {}).items()
                if isinstance(size, (int, float))
            },
            primary_split=run.primary_split,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            train_metrics=train_metrics,
        ),
        similar_papers=similar,
        ground_truth=GroundTruth(
            true_label=detail.true_label,
            predicted_label=detail.predicted_label,
            correct=detail.correct,
            is_unlabelled=detail.true_label is None,
        ),
        embedding=embedding_obj,
        insights=insights_obj,
    )


def _build_explanation(
    run: LoadedRun, entry: PaperEntry, pred_label: str | None, kind: str
) -> ExplanationResponse:
    """Build the explanation payload reused by both ``/explanation`` and ``/analysis``."""
    if run.is_han:
        return _han_explanation(run, entry)

    label = pred_label or entry.record.label
    attention = SectionAttention(reason=ATTENTION_UNAVAILABLE_REASON)

    if not label:
        return ExplanationResponse(
            paper_id=entry.paper_id,
            method="linear_term_contributions",
            evidence_label="Linear model feature evidence",
            section_attention=attention,
            caveat="This paper has no predicted or ground-truth label to explain.",
        )

    try:
        contributions = run.term_contributions(entry.record.text, label)
        decision = _decision_value(run, entry.record.text, label)
    except RunUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    largest = max((abs(item.contribution) for item in contributions), default=0.0)
    terms = [
        TermWeight(
            term=item.term,
            contribution=round(item.contribution, 6),
            tfidf=round(item.tfidf, 6),
            weight=round(abs(item.contribution) / largest, 4) if largest else 0.0,
        )
        for item in contributions
    ]

    parsed_sections = parse_text_into_sections(
        entry.record.text, title=entry.record.title
    )
    sec_weights: list[SectionWeight] = []
    term_dict = {item.term.lower(): abs(item.contribution) for item in contributions}
    sentence_evidence: list[SentenceEvidence] = []

    for sec in parsed_sections:
        sec_text = sec.text.lower()
        score = sum(val for term, val in term_dict.items() if term in sec_text)
        if score == 0:
            score = len(sec.text.split()) * 0.001
        sec_weights.append(
            SectionWeight(
                name=sec.section_name,
                canonical_name=sec.canonical_name or "other",
                weight=score,
            )
        )

        sents = split_sentences(sec.text)
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) < 12:
                continue
            s_lower = s_clean.lower()
            s_score = sum(val for term, val in term_dict.items() if term in s_lower)
            sentence_evidence.append(
                SentenceEvidence(
                    section_name=sec.section_name,
                    canonical_name=sec.canonical_name or "other",
                    text=s_clean,
                    weight=round(s_score, 4),
                )
            )

    max_sec_w = max((s.weight for s in sec_weights), default=1.0) or 1.0
    for s in sec_weights:
        s.weight = round(s.weight / max_sec_w, 4)

    max_sent_w = max((se.weight for se in sentence_evidence), default=1.0) or 1.0
    for se in sentence_evidence:
        se.weight = round(se.weight / max_sent_w, 4) if max_sent_w > 0 else 0.1

    attention = SectionAttention(
        available=True,
        sections=sec_weights,
        kind="projected_section_evidence",
    )

    return ExplanationResponse(
        paper_id=entry.paper_id,
        predicted_label=label,
        method="linear_term_contributions",
        evidence_label="Linear model feature evidence",
        terms=terms,
        decision_value=decision,
        section_attention=attention,
        sentences=sentence_evidence,
        caveat=EXPLANATION_CAVEAT,
    )


def _build_similar(
    run: LoadedRun,
    paper_id: str,
    limit: int = 5,
    *,
    raise_on_error: bool = False,
) -> SimilarResponse:
    """Build the similar-papers payload reused by ``/similar`` and ``/analysis``.

    Args:
        run: The active loaded run.
        paper_id: Target paper identifier.
        limit: Maximum number of neighbours to return.
        raise_on_error: When ``True``, a :class:`RunUnavailableError` is
            re-raised as an HTTP 503 so the standalone ``/similar`` endpoint
            can satisfy the contract test.  When ``False`` (default) the error
            is suppressed and an empty neighbour list is returned, which lets
            ``/analysis`` degrade gracefully without a full 503 response.
    """
    caveat = SEMANTIC_SIMILARITY_CAVEAT if run.is_han else SIMILARITY_CAVEAT
    try:
        neighbours = run.similar(paper_id, top_k=limit)
    except RunUnavailableError as exc:
        if raise_on_error:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc
        neighbours = []

    return SimilarResponse(
        query_paper_id=paper_id,
        items=[SimilarItem(**item) for item in neighbours],
        method=run.similarity_method,
        caveat=caveat,
    )


def _split_metrics(data: dict | None) -> SplitMetrics | None:
    """Build a ``SplitMetrics`` from a metrics.json split block, or ``None``."""
    if not data:
        return None
    averages = data.get("averages", {}) or {}
    macro = averages.get("macro", {}) or {}
    weighted = averages.get("weighted", {}) or {}
    cm_raw = data.get("confusion_matrix")
    cm = None
    if cm_raw:
        cm = ConfusionMatrixSchema(
            labels=cm_raw.get("labels", []),
            counts=cm_raw.get("counts", []),
            normalized=cm_raw.get("normalized", []),
        )
    return SplitMetrics(
        n_samples=int(data.get("n_samples", 0)),
        n_classes=int(data.get("n_classes", 0)),
        accuracy=data.get("accuracy"),
        balanced_accuracy=data.get("balanced_accuracy"),
        macro_precision=macro.get("precision"),
        macro_recall=macro.get("recall"),
        macro_f1=macro.get("f1"),
        weighted_f1=weighted.get("f1"),
        per_class=data.get("per_class", {}) or {},
        confusion_matrix=cm,
    )


# ---------------------------------------------------------------------------
# Per-paper methodology extraction
# ---------------------------------------------------------------------------

#: Sentence patterns for the fields the term extractor cannot cover. Each one
#: targets explicit academic phrasing so a match is a quote, not a paraphrase.
_OBJECTIVE_RE = re.compile(
    r"\b(we (?:propose|present|introduce|investigate|aim)|this paper "
    r"(?:proposes|presents|introduces|investigates|studies)|the (?:aim|goal|"
    r"objective) of (?:this|the) (?:paper|study|work))\b",
    re.IGNORECASE,
)
_FUTURE_WORK_RE = re.compile(
    r"\b(future work|future research|future directions|further work|"
    r"we plan to|we intend to|as future)\b",
    re.IGNORECASE,
)
_RESULTS_SECTION_RE = re.compile(
    r"\b(result|experiment|evaluation|finding)s?\b", re.IGNORECASE
)
_MAX_ITEMS_PER_FIELD = 8


@router.get(
    "/{paper_id}/methodology",
    response_model=PaperMethodologyResponse,
    summary="Structured methodology extraction for one paper, with section evidence",
)
def paper_methodology(paper_id: str, run: ActiveRun) -> PaperMethodologyResponse:
    """Extract methodology signals from one paper, quoting the source.

    Reuses the corpus-level :class:`MethodologyExtractor` and
    :class:`ResearchGapDetector` services sentence-by-sentence over the paper's
    parsed sections, so every extracted item carries the section it came from
    and the verbatim supporting sentence. Fields with no matches return an
    empty list — the UI marks them unavailable rather than filling them in.
    """
    entry = _require_paper(run, paper_id)
    parsed_sections = parse_text_into_sections(
        entry.record.text, title=entry.record.title or entry.paper_id
    )
    extractor = MethodologyExtractor()
    gap_detector = ResearchGapDetector()
    return _extract_methodology(
        entry, parsed_sections, extractor, gap_detector, run.run_id
    )


def _extract_methodology(
    entry: PaperEntry,
    parsed_sections: list,
    extractor: MethodologyExtractor,
    gap_detector: ResearchGapDetector,
    run_id: str,
) -> PaperMethodologyResponse:
    """Scan the parsed sections sentence-by-sentence and collect evidence.

    A private helper only because the sentence loop is long; everything it
    uses comes from the shared analytics services.
    """
    collected: dict[str, list[MethodologyEvidenceItem]] = {
        name: [] for name in (
            "objective", "datasets", "metrics", "architectures",
            "algorithms", "results", "limitations", "future_work",
        )
    }
    seen: dict[str, set[tuple[str, str]]] = {name: set() for name in collected}
    n_passages = 0

    def _add(field: str, value: str, section: str, statement: str) -> None:
        key = (value.strip().lower(), section)
        if key in seen[field] or len(collected[field]) >= _MAX_ITEMS_PER_FIELD:
            return
        seen[field].add(key)
        collected[field].append(
            MethodologyEvidenceItem(
                value=value.strip()[:200],
                section=section,
                statement=statement.strip()[:400],
            )
        )

    for section in parsed_sections:
        section_name = section.section_name or section.canonical_name or "Body"
        for paragraph in section.paragraphs:
            if not paragraph.text.strip():
                continue
            n_passages += 1
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", paragraph.text)
                if len(s.strip()) > 20
            ]
            for sentence in sentences:
                extracted = extractor.extract(sentence)
                for term in extracted.datasets:
                    _add("datasets", term, section_name, sentence)
                for term in extracted.metrics:
                    _add("metrics", term, section_name, sentence)
                for term in extracted.architectures:
                    _add("architectures", term, section_name, sentence)
                for term in extracted.algorithms:
                    _add("algorithms", term, section_name, sentence)

                if _OBJECTIVE_RE.search(sentence):
                    _add("objective", "Research objective", section_name, sentence)
                if _FUTURE_WORK_RE.search(sentence):
                    _add("future_work", "Future work", section_name, sentence)
                if _RESULTS_SECTION_RE.search(section_name) and re.search(r"\d", sentence):
                    _add("results", "Reported result", section_name, sentence)

                # Limitations reuse the gap detector's categories so the labels
                # here match what Research Gaps reports for the same sentence.
                for gap in gap_detector.detect(sentence):
                    _add(
                        "limitations",
                        gap.category.replace("_", " ").title(),
                        section_name,
                        sentence,
                    )

    fields = [MethodologyField(field=name, items=items) for name, items in collected.items()]
    return PaperMethodologyResponse(
        paper_id=entry.paper_id,
        title=entry.record.title or entry.paper_id,
        fields=fields,
        n_sections_parsed=len(parsed_sections),
        n_passages_scanned=n_passages,
        run_id=run_id,
    )
