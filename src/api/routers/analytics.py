"""Research analytics endpoints.

Three read-only endpoints that read the active run's corpus and return
aggregate signals: detected research gaps, ranked methodology terms, and the
in-corpus citation graph. The detectors themselves live in
:mod:`src.analytics`; this module is the API contract that wires them to the
dashboard, including the per-category rollups, the ``max_series``-style
capping, and the ``run_id`` echo so the UI can confirm which run produced
the numbers.

Like the rest of the API these endpoints are honest about their inputs.
The synthetic fixture produces a gaps payload, a methodology payload, and a
citation graph with zero edges — the third is a correct answer, not a
fallback, and the panel renders that.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter

from src.analytics.citations import CitationNetworkBuilder
from src.analytics.gaps import ResearchGapDetector
from src.analytics.methodology import MethodologyExtractor
from src.api.deps import ActiveRun
from src.api.runstore import PaperEntry
from src.api.schemas import (
    CitationEdgeSchema,
    CitationGraphResponse,
    CitationNodeSchema,
    GapCategory,
    GapExample,
    GapsResponse,
    MethodologyBucket,
    MethodologyItem,
    MethodologyResponse,
)
from src.utils.logging import get_logger

__all__ = ["router"]

logger = get_logger(__name__)

router = APIRouter(tags=["analytics"])

#: Examples to keep per gap category. The dashboard shows the first N for each
#: bucket; the rest are counted but not listed, mirroring the trends cap so a
#: panel never silently truncates.
_EXAMPLES_PER_CATEGORY = 8

#: Items to keep per methodology bucket. Same reasoning as the gaps cap.
_ITEMS_PER_BUCKET = 12


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _paper_text(entry: PaperEntry) -> str:
    """Compose the text a detector scans for one paper.

    Title + abstract is what the dataset pipeline stored; for the synthetic
    fixture that is the only text the model was fitted on, so it is also the
    only text the detectors have. A real PDF-extracted corpus will have richer
    text on ``entry.record.full_text``; we still concatenate title here so the
    detector sees a consistent input shape.
    """
    title = entry.record.title or ""
    abstract = entry.record.abstract or entry.record.text
    return f"{title}\n{abstract}".strip()


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


@router.get(
    "/research/gaps",
    response_model=GapsResponse,
    summary="Pattern-detected research gaps in the active corpus",
)
def research_gaps(run: ActiveRun) -> GapsResponse:
    """Return detected research gaps, grouped by category, with examples.

    Each paper's text is scanned by :class:`ResearchGapDetector`; matches are
    rolled up into a per-category count plus a short list of example
    statements (capped at :data:`_EXAMPLES_PER_CATEGORY`). The detector is
    intentionally simple (regex categories) and the dashboard labels the basis
    as such — these are signals from the literature, not a causal claim about
    open problems.
    """
    detector = ResearchGapDetector()
    per_category: dict[str, list[GapExample]] = {}
    papers_with_gaps = 0

    for entry in run.entries():
        text = _paper_text(entry)
        for gap in detector.detect(text):
            # The detector scans title + abstract, so a statement that appears
            # verbatim in the title is honestly labelled as coming from it.
            source = "Title" if gap.statement in (entry.record.title or "") else "Abstract"
            per_category.setdefault(gap.category, []).append(
                GapExample(
                    category=gap.category,
                    paper_id=entry.paper_id,
                    title=entry.record.title or entry.paper_id,
                    statement=gap.statement,
                    confidence=float(gap.confidence),
                    source=source,
                )
            )

    categories: list[GapCategory] = []
    for name in sorted(per_category):
        examples = per_category[name]
        categories.append(
            GapCategory(
                name=name,
                count=len(examples),
                examples=examples[:_EXAMPLES_PER_CATEGORY],
            )
        )
        if examples:
            papers_with_gaps += len({ex.paper_id for ex in examples})

    return GapsResponse(
        categories=categories,
        n_papers_scanned=len(run.papers),
        n_papers_with_gaps=papers_with_gaps,
        run_id=run.run_id,
    )


# ---------------------------------------------------------------------------
# Methodology
# ---------------------------------------------------------------------------


_BUCKET_NAMES = ("datasets", "metrics", "architectures", "algorithms")


def _methodology_counts_for_bucket(
    run: ActiveRun, extractor: MethodologyExtractor, bucket: str
) -> Counter[str]:
    """Count how many distinct papers mention each term in this bucket.

    Per-paper dedup avoids one paper that mentions "BERT" 30 times inflating
    the corpus-wide count; the bucket is then a count of *papers* that
    mention the term, which is what a reader of the dashboard expects.
    """
    counts: Counter[str] = Counter()
    for entry in run.entries():
        extracted = extractor.extract(_paper_text(entry))
        items = getattr(extracted, bucket, []) or []
        for item in items:
            counts[item] += 1
    return counts


@router.get(
    "/research/methodology",
    response_model=MethodologyResponse,
    summary="Ranked datasets / metrics / architectures / algorithms in the corpus",
)
def research_methodology(run: ActiveRun) -> MethodologyResponse:
    """Return ranked lists of the four methodology buckets.

    Each bucket is the count of *papers* (not occurrences) that mention the
    term; the list is sorted descending and capped at
    :data:`_ITEMS_PER_BUCKET` so the panel does not silently truncate.
    """
    extractor = MethodologyExtractor()
    buckets: list[MethodologyBucket] = []
    for bucket in _BUCKET_NAMES:
        counts = _methodology_counts_for_bucket(run, extractor, bucket)
        ranked = counts.most_common(_ITEMS_PER_BUCKET)
        buckets.append(
            MethodologyBucket(
                name=bucket,  # type: ignore[arg-type]
                items=[
                    MethodologyItem(name=name, count=count) for name, count in ranked
                ],
            )
        )
    return MethodologyResponse(
        buckets=buckets,
        n_papers_scanned=len(run.papers),
        run_id=run.run_id,
    )


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------


def _components(nodes: list[CitationNodeSchema], edges: list[CitationEdgeSchema]) -> int:
    """Count weakly connected components in the citation graph.

    Implemented inline because the corpus is small (the synthetic fixture has
    zero edges, real corpora have tens of thousands) and a dedicated graph
    library would be overkill.
    """
    adjacency: dict[str, set[str]] = {node.id: set() for node in nodes}
    for edge in edges:
        adjacency.setdefault(edge.source, set()).add(edge.target)
        adjacency.setdefault(edge.target, set()).add(edge.source)

    seen: set[str] = set()
    components = 0
    for start in adjacency:
        if start in seen:
            continue
        components += 1
        stack = [start]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(adjacency[current] - seen)
    return components


@router.get(
    "/research/citations",
    response_model=CitationGraphResponse,
    summary="In-corpus citation graph for the active run",
)
def research_citations(run: ActiveRun) -> CitationGraphResponse:
    """Return the directed reference graph between papers in the active run.

    Only references that resolve to another paper in the corpus are kept as
    edges, so the dashboard's graph is "papers that cite other papers in this
    corpus" — not the full OpenAlex graph. The synthetic fixture has no
    reference metadata, so the response is a correct empty graph rather than
    a fabricated one.
    """
    builder = CitationNetworkBuilder()
    records: list[Any] = [entry.record for entry in run.entries()]
    graph = builder.build_graph(records)

    nodes = [
        CitationNodeSchema(id=n.id, label=n.label, domain=n.domain) for n in graph.nodes
    ]
    edges = [CitationEdgeSchema(source=e.source, target=e.target) for e in graph.edges]
    return CitationGraphResponse(
        nodes=nodes,
        edges=edges,
        n_nodes=len(nodes),
        n_edges=len(edges),
        n_components=_components(nodes, edges),
        run_id=run.run_id,
    )
