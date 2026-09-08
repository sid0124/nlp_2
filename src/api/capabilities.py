"""What this build can and cannot do, in one place.

The dashboard was designed against the finished system: it has a section-attention
panel, a research-assistant composer, an upload button, and a semantic-similarity
list. Most of that is not built. Every one of those panels therefore has two
possible honest renderings — the real thing, or an explicit "not built yet" — and
exactly one dishonest one, which is a plausible-looking placeholder.

Keeping the availability table here rather than scattered through the routers
means the UI's honesty has a single source. When Milestone 2 lands embeddings, one
entry changes from ``False`` to a real check, and the panel that was showing "not
available" starts showing data — without a hunt through the frontend for a
hard-coded string that says the feature is missing.

The ``reason`` strings are user-facing and are rendered verbatim. They name the
prerequisite rather than saying "coming soon", because a user looking at an empty
panel wants to know whether it is broken or unbuilt.
"""

from __future__ import annotations

from src.api.runstore import LoadedRun, RunUnavailableError
from src.api.schemas import Capability
from src.models.baselines import linear_coefficients

__all__ = [
    "ATTENTION_UNAVAILABLE_REASON",
    "EXPLANATION_CAVEAT",
    "HAN_EXPLANATION_CAVEAT",
    "PROJECTED_SECTION_EVIDENCE_CAVEAT",
    "RAG_UNAVAILABLE_REASON",
    "REVIEW_CAVEAT",
    "SEMANTIC_SIMILARITY_CAVEAT",
    "SIMILARITY_CAVEAT",
    "capabilities_for",
    "caveats_for",
    "classification_caveat",
]

#: Master spec §17. Shown beside every similar-papers list.
SIMILARITY_CAVEAT = (
    "Similarity here is lexical: cosine distance between TF-IDF vectors, which "
    "measures shared vocabulary. Two papers using the same words are not "
    "necessarily methodologically equivalent, and two papers solving the same "
    "problem in different vocabularies will score low."
)

#: Named in the payload so the UI cannot describe this as embedding similarity.
SIMILARITY_METHOD = "tfidf_cosine"

#: Shown beside every similar-papers list when the active run is the HAN, whose
#: neighbours come from SciBERT embedding space rather than TF-IDF overlap.
SEMANTIC_SIMILARITY_CAVEAT = (
    "Similarity here is semantic: cosine distance between SciBERT document "
    "embeddings from the same frozen encoder the classifier consumes. Learned "
    "representations can match papers that solve the same problem in different "
    "words, but embedding closeness is still not methodological equivalence."
)

#: Shown beside HAN attention. Master spec §14: attention is evidence, not proof.
HAN_EXPLANATION_CAVEAT = (
    "These are the trained network's own attention weights — one per section and "
    "one per sentence — produced by the model's additive-attention layers at "
    "inference. They show where the model looked while deciding, not why the "
    "paper belongs to the class: attention is model evidence, not a causal "
    "explanation, and it should be read alongside the prediction's confidence."
)

#: Master spec §14. Shown beside term contributions.
EXPLANATION_CAVEAT = (
    "These weights are the terms of the model's own decision sum "
    "(TF-IDF value x coefficient), so they faithfully describe how this "
    "classifier reached its label. They do not explain the paper, and they are "
    "not evidence that the term caused anything."
)

#: Shown beside the per-section panel of a NON-hierarchical model. The weights
#: are term-level contributions projected onto detected canonical sections —
#: a heuristic summary of where the model's vocabulary evidence sits, never
#: the output of an attention mechanism. Naming the distinction here is what
#: keeps the panel from quietly becoming "attention" before the HAN exists.
PROJECTED_SECTION_EVIDENCE_CAVEAT = (
    "Projected Section Evidence, not attention. These weights are the linear "
    "model's per-term contributions summed over detected canonical sections — "
    "a projection of feature evidence onto the section layout. This model has "
    "no attention mechanism; real hierarchical attention arrives with the "
    "SciBERT + HAN model."
)

#: Master spec §15.
REVIEW_CAVEAT = (
    "Low-confidence predictions are flagged for human review. The threshold is "
    "applied on the server; the flag is not a quality judgement about the paper."
)

ATTENTION_UNAVAILABLE_REASON = (
    "Section-level attention needs the hierarchical attention network, which is "
    "not the active model. The current model is a bag-of-words classifier and "
    "has no representation of a section at all, so there is nothing to weight. "
    "Train the HAN (scripts/train_han.py) and pin it as the active run."
)

RAG_UNAVAILABLE_REASON = (
    "Answering questions about a paper needs a retrieval index over its full "
    "text. This build has abstracts only and no retriever, so any answer would "
    "be generated without a source to ground it."
)

_PDF_UNAVAILABLE_REASON = (
    "PDF parsing is not implemented, so there is no upload endpoint. Papers enter "
    "the corpus through scripts/build_dataset.py."
)

_SEMANTIC_UNAVAILABLE_REASON = (
    "Semantic similarity needs scientific-transformer embeddings, which are "
    "Milestone 2. Lexical TF-IDF similarity is available in the meantime and is "
    "labelled as such."
)

#: The citation graph needs paper-to-paper reference metadata, which only
#: arrives with the PDF parser or a real OpenAlex fetch. The committed
#: synthetic fixture has no references, so the graph is correctly empty.
_CITATION_UNAVAILABLE_REASON = (
    "Citation graph needs paper reference metadata (outbound `references` "
    "on each record), which the committed synthetic fixture does not carry. "
    "The endpoint returns a correct empty graph rather than fabricating one."
)


def _is_calibrated(run: LoadedRun) -> bool:
    """Whether the run's model carries post-hoc probability calibration."""
    calibration = run.manifest.get("model", {}).get("calibration") or {}
    return bool(calibration.get("applied"))


def classification_caveat(run: LoadedRun) -> str:
    """Describe what a classification score from ``run`` does and does not mean.

    Three texts, because three quantities are in play: calibrated probabilities,
    uncalibrated (raw softmax/sigmoid) probabilities, and decision margins.
    Conflating them would let the UI print "82% confident" for a number that is
    not a probability.
    """
    if run.confidence_kind == "probability":
        if _is_calibrated(run):
            return (
                f"Predicted by {run.model_display_name} over {len(run.classes)} classes. "
                "The scores are calibrated probabilities produced by Platt scaling "
                "fitted on the training split only: each is in [0, 1] and they sum "
                "to 1 across classes. Confidence is the maximum class probability."
            )
        return (
            f"Predicted by {run.model_display_name} over {len(run.classes)} classes. "
            "The score is the model's predicted probability, which is uncalibrated: "
            "treat it as a ranking, not as a measured likelihood of being correct."
        )
    if run.confidence_kind == "decision":
        return (
            f"Predicted by {run.model_display_name} over {len(run.classes)} classes. "
            "This model has no probability output, so the score is the margin "
            "between the top two classes on an unbounded scale. It is comparable "
            "between predictions from this run and meaningless as a percentage."
        )
    return (
        f"Predicted by {run.model_display_name} over {len(run.classes)} classes. "
        "This model exposes no confidence score, so predictions cannot be ranked "
        "by certainty and none are flagged for review."
    )


def _model_is_linear(run: LoadedRun) -> bool:
    """Whether the run's classifier exposes per-feature coefficients.

    A calibrated SVM holds one base estimator per calibration fold rather than a
    bare ``coef_``; :func:`linear_coefficients` unwraps the ensemble, so feature
    evidence survives calibration instead of silently disappearing.
    """
    try:
        return linear_coefficients(run.classifier) is not None
    except RunUnavailableError:
        return False


def _model_loads(run: LoadedRun) -> bool:
    """Whether the run's model artifact can actually be loaded and served.

    Baseline runs load a scikit-learn pipeline; HAN runs load the neural model
    (which also constructs the SciBERT encoder). Each kind is checked through
    its own path, so a HAN run is not misreported as model-less just because it
    has no TF-IDF pipeline.
    """
    try:
        if run.is_han:
            run.han_model  # noqa: B018 - the load itself is the check
        else:
            run.pipeline  # noqa: B018 - the load itself is the check
        return True
    except RunUnavailableError:
        return False


def capabilities_for(run: LoadedRun | None) -> list[Capability]:
    """Build the capability table for the active run.

    Args:
        run: The active run, or ``None`` when none could be loaded — in which case
            everything that needs a model reports unavailable rather than the API
            pretending a feature exists that has nothing behind it.

    Returns:
        One entry per dashboard feature, in the order the UI presents them.
    """
    has_model = False
    if run is not None:
        has_model = _model_loads(run)

    no_run = "No completed training run is loaded, so this needs a run first."
    no_model = "The active run has no saved model file, so it cannot score new text."
    model_reason = None if has_model else (no_run if run is None else no_model)

    entries: list[Capability] = [
        Capability(
            key="corpus",
            label="Corpus browsing",
            available=run is not None,
            reason=None if run is not None else no_run,
        ),
        Capability(
            key="classification",
            label="Domain classification",
            available=has_model,
            reason=model_reason,
        ),
        Capability(
            key="confidence",
            label="Confidence scoring",
            available=bool(run and run.confidence_kind != "unavailable"),
            reason=(
                None
                if run and run.confidence_kind != "unavailable"
                else "This model exposes no per-prediction score."
            ),
        ),
        Capability(
            key="similarity_lexical",
            label="Lexical similarity",
            # Lexical similarity needs a fitted TF-IDF vectorizer; HAN runs
            # do not have one, so this is baseline-only.
            available=bool(run is not None and not run.is_han and has_model),
            reason=(
                None
                if (run is not None and not run.is_han and has_model)
                else (
                    "The active run is the SciBERT + HAN model: lexical "
                    "similarity over TF-IDF is not the run's native "
                    "neighbourhood metric. Use semantic similarity instead."
                    if run is not None and run.is_han
                    else model_reason
                )
            ),
        ),
        Capability(
            key="similarity_semantic",
            label="Semantic similarity",
            # Semantic similarity requires a HAN run whose SciBERT encoder
            # is loadable. Baseline runs do not produce document vectors.
            available=bool(run is not None and run.is_han and has_model),
            reason=(
                None
                if (run is not None and run.is_han and has_model)
                else (
                    "The active run is a TF-IDF baseline: there is no "
                    "learned document embedding to be semantic about. "
                    "Use lexical similarity (cosine over the fitted TF-IDF "
                    "space) for this run."
                    if run is not None and not run.is_han
                    else model_reason
                )
            ),
        ),
        Capability(
            key="explanation_terms",
            label="Term contributions",
            available=has_model and run is not None and _model_is_linear(run),
            reason=(
                model_reason
                or (
                    None
                    if run and _model_is_linear(run)
                    else "This classifier exposes no coefficients to decompose."
                )
            ),
        ),
        Capability(
            key="section_attention",
            label="Section attention",
            # Only the HAN has a real attention mechanism. For a linear
            # baseline the panel renders projected term evidence under a
            # different title; that rendering is not "section attention"
            # and the capability must say so.
            available=bool(run is not None and run.is_han and has_model),
            reason=(
                None
                if (run is not None and run.is_han and has_model)
                else (
                    ATTENTION_UNAVAILABLE_REASON
                    if (run is not None and not run.is_han)
                    else model_reason
                )
            ),
        ),
        Capability(
            key="trends",
            label="Publication trends",
            available=bool(run and run.counts_by_year),
            reason=(
                None
                if run and run.counts_by_year
                else "No paper in the corpus recorded a publication year."
            ),
        ),
        Capability(
            key="rag_ask",
            label="Research assistant",
            # The /ask endpoint works: it does TF-IDF passage retrieval and
            # optionally calls Groq. The old text said "no retriever", which
            # was wrong; the new text names what it actually does so a user
            # who clicks in knows what to expect.
            available=bool(run is not None and has_model),
            reason=(
                None
                if (run is not None and has_model)
                else (model_reason or no_run)
            ),
        ),
        Capability(
            key="pdf_upload",
            label="PDF upload",
            available=run is not None,
            reason=None if run is not None else _PDF_UNAVAILABLE_REASON,
        ),
        Capability(
            key="comparison",
            label="Paper comparison",
            available=run is not None,
            reason=None if run is not None else no_run,
        ),
        Capability(
            key="research_gaps",
            label="Research gaps",
            available=run is not None,
            reason=None if run is not None else no_run,
        ),
        Capability(
            key="methodology",
            label="Methodology extraction",
            available=run is not None,
            reason=None if run is not None else no_run,
        ),
        Capability(
            key="citations",
            label="Citation network",
            available=run is not None,
            reason=None if run is not None else _CITATION_UNAVAILABLE_REASON,
        ),
        Capability(
            key="authentication",
            label="User accounts",
            available=False,
            reason=(
                "There are no user accounts. The API supports an optional shared "
                "API key, so it is authentication-ready but not authenticated."
            ),
        ),
    ]
    return entries


def caveats_for(run: LoadedRun | None) -> list[str]:
    """Standing caveats the UI must keep visible.

    The corpus-provenance caveat comes first when it applies, because it changes
    how every other number on the page should be read: metrics over the generated
    fixture describe the wiring, not the science.
    """
    caveats: list[str] = []
    if run is None:
        return [
            "No training run is loaded, so every panel on this page is empty by "
            "necessity rather than by error."
        ]

    if run.is_synthetic_corpus:
        caveats.append(
            "This run was trained on the generated test corpus, not on real "
            "papers. That corpus is separable by construction, so near-perfect "
            "scores here confirm the pipeline works and say nothing about "
            "real-world accuracy."
        )
    try:
        if run.dataset_is_stale:
            caveats.append(
                "The dataset on disk has changed since this run was trained, so "
                "the papers listed here are not exactly the papers that produced "
                "these metrics. Re-train to bring them back into agreement."
            )
    except RunUnavailableError:
        pass

    total = sum(run.summary().split_sizes.values())
    if total and total < 500:
        caveats.append(
            f"The corpus holds {total} papers. Per-class metrics on a corpus this "
            "small have wide confidence intervals; treat differences of a few "
            "points as noise."
        )

    caveats.extend(
        [
            # What the confidence column actually holds. Without this the UI has
            # to decide for itself whether a score is a probability or an
            # unbounded margin, which would put the distinction in two places.
            classification_caveat(run),
            REVIEW_CAVEAT,
            SIMILARITY_CAVEAT,
            EXPLANATION_CAVEAT,
        ]
    )
    return caveats
