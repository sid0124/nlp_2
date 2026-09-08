"""Extractive + LLM-synthesized paper Q&A retrieval engine.

Segments paper text into candidate section/paragraph passages, scores passage relevance against
the input question using TF-IDF cosine and stemmed token overlap, and produces grounded answers
with section provenance. When a Groq API key is configured, the top retrieved passages are sent
to Groq server-side and the answer is synthesized under a strict grounding contract: the model
may use only the supplied passages and must say so when they are insufficient. Without a key the
engine degrades to a deterministic extractive answer and refuses when nothing matches (master
spec §20).
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from src.api.schemas import AskResponse, PassageEvidence
from src.preprocessing.sections import parse_text_into_sections
from src.schemas.paper import PaperDocument, PaperSection
from src.utils.logging import get_logger

__all__ = ["GroqSynthesisError", "PaperQAEngine"]

logger = get_logger(__name__)

#: Groq's OpenAI-compatible chat endpoint. The key travels only in this
#: server-side request; it is never serialized into an API response.
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

#: System instruction for the grounded synthesis step. Grounding beats fluency:
#: the model is told, in effect, that an honest "the passages do not say" is a
#: correct answer and an invented one is not.
GROQ_SYSTEM_PROMPT = (
    "You are an academic research assistant answering a question about ONE "
    "specific academic paper.\n"
    "Use ONLY the provided paper passages as evidence.\n"
    "Do not use outside knowledge. Do not invent facts, citations, results, "
    "methodologies, datasets, page numbers, or conclusions.\n"
    "If the provided passages do not contain enough information to answer the "
    "question, say clearly that the available paper passages do not provide "
    "enough evidence, and name what is missing.\n"
    "When possible: answer directly; identify which section the evidence comes "
    "from; distinguish reported evidence from your interpretation; preserve "
    "technical terminology; mention uncertainty when appropriate."
)

#: Question words that describe the paper as a whole rather than a specific
#: fact. Such questions share little vocabulary with the passages ("what is
#: this about?" appears nowhere verbatim), so a weak lexical score must not
#: refuse them — the grounded model sees the strongest passages and decides.
_GENERAL_QUESTION_PATTERN = re.compile(
    r"\b(about|topic|overview|summaries|summary|summarize|purpose|objective|"
    r"research question|main idea|contributions?|methods?|methodology|"
    r"approach(?:es)?|techniques?|datasets?|data|limitations?|strengths?|"
    r"weakness(?:es)?|findings?|results?|conclusions?|compare|comparison|"
    r"similar|differences?|differ|versus|vs\.?|sections?|classified|"
    r"classification|domain|category|evidence|influence|prediction|why)\b",
    re.IGNORECASE,
)

#: Deterministic refusal for questions with no lexical foothold in the paper
#: and no general intent (master spec §20's exact wording).
_REFUSAL = "Information not found in the provided paper."

#: How many ranked passages are handed to the model, and the character budget
#: that keeps one verbose section from crowding out the rest.
_MAX_CONTEXT_PASSAGES = 6
_MAX_PASSAGE_CHARS = 1500
_MAX_CONTEXT_CHARS = 7000


class GroqSynthesisError(Exception):
    """The paper was retrieved, but the Groq synthesis step failed.

    ``status_code`` is the HTTP status Groq returned (0 for a transport
    failure); the router maps it to a user-facing message and logs the detail
    server-side. The API key itself is never included in the message.
    """

    def __init__(self, message: str, status_code: int = 0) -> None:
        """Carry the Groq HTTP status alongside the technical message."""
        super().__init__(message)
        self.status_code = status_code


def _stem(token: str) -> str:
    """A deliberately small suffix stripper.

    Real stemming (Porter, Snowball) would add a dependency-shaped surface for
    bugs; four rules cover the morphological gap that actually matters for
    matching a question to prose: plural -s, -ies, participial -ed and gerund
    -ing.
    """
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 5 and token.endswith("ed"):
        return token[:-2]
    return token


def _content_tokens(text: str) -> set[str]:
    """Lowercased, stemmed content tokens: stopwords and fragments dropped.

    Without the stopword filter, function words ("the", "is", "for") would
    manufacture overlap between any question and any paragraph, which would
    let off-topic questions slip past the deterministic refusal.
    """
    return {
        _stem(token)
        for token in re.findall(r"[a-z][a-z0-9'-]+", text.lower())
        if len(token) > 1 and token not in ENGLISH_STOP_WORDS
    }


class PaperQAEngine:
    """Passage-level retrieval and Q&A engine for an academic paper."""

    def __init__(
        self,
        paper_id: str,
        title: str,
        text: str,
        sections: Sequence[PaperSection] | None = None,
        groq_api_key: str | None = None,
        groq_model: str = "openai/gpt-oss-120b",
    ) -> None:
        """Build the engine from the paper's id, title, raw text, and Groq config."""
        self.paper_id = paper_id
        self.title = title
        self.text = text
        self.sections = parse_text_into_sections(text, title=title, existing_sections=sections)
        self.groq_api_key = groq_api_key
        self.groq_model = groq_model

    @classmethod
    def from_document(cls, paper: PaperDocument) -> PaperQAEngine:
        """Instantiate engine directly from a PaperDocument."""
        return cls(
            paper_id=paper.paper_id,
            title=paper.title,
            text=paper.full_text or paper.text_for(("title", "abstract")),
            sections=paper.sections,
        )

    def answer_question(
        self,
        question: str,
        min_confidence: float = 0.08,
        comparison_context: Sequence[dict[str, object]] | None = None,
    ) -> AskResponse:
        """Answer a question using passage retrieval, then grounded synthesis.

        Args:
            question: The user's natural language question.
            min_confidence: Minimum combined relevance score required before a
                question with a specific lexical footprint is answered at all.
                Questions about the paper in general bypass it, because their
                vocabulary legitimately does not occur in the text.
            comparison_context: Optional corpus neighbours of this paper, built
                by the router from the run's existing similar-papers feature.
                Used only for explicit comparison questions and clearly labelled
                as coming from *other* papers, never as evidence about this one.

        Returns:
            An :class:`src.api.schemas.AskResponse` payload.

        Raises:
            GroqSynthesisError: A key is configured but every synthesis attempt
                failed. Retrieval itself succeeded; the router turns this into
                a user-facing error.
        """
        passages: list[tuple[str, str]] = []  # (section_name, passage_text)
        for sec in self.sections:
            sec_name = sec.section_name or sec.canonical_name or "Body"
            for para in sec.paragraphs:
                if para.text.strip():
                    passages.append((sec_name, para.text.strip()))

        if not passages:
            passages = [("Paper Content", self.text[:4000] if self.text else self.title)]

        scores = self._score_passages(question, passages)
        best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
        best_score = float(scores[best_idx]) if len(scores) > 0 else 0.0

        # Questions with a specific lexical footprint that matches nothing in
        # the paper are refused deterministically — that is cheaper than a
        # model call and honest about the miss. Questions about the paper in
        # general (contribution, methodology, limitations, comparison, ...)
        # legitimately share no vocabulary with the prose, so they proceed to
        # retrieval + synthesis and the grounding contract handles insufficiency.
        general_question = bool(_GENERAL_QUESTION_PATTERN.search(question))
        if best_score < min_confidence and not general_question:
            return AskResponse(
                paper_id=self.paper_id,
                question=question,
                answer=_REFUSAL,
                confidence=0.0,
            )

        context = self._build_context(passages, scores)
        source_section, best_passage = context[0][0], context[0][1]

        # Start with a deterministic local answer. Groq improves the wording
        # when configured, but a provider outage must not make the panel blank
        # in the keyless extractive mode.
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", best_passage) if s.strip()]
        answer_text = sentences[0] if sentences else best_passage

        if self.groq_api_key:
            answer_text = self._synthesize(question, context, comparison_context or [])

        evidence = [
            PassageEvidence(
                source_section=section,
                passage=passage,
                confidence=round(score, 4),
            )
            for section, passage, score in context
        ]

        return AskResponse(
            paper_id=self.paper_id,
            question=question,
            answer=answer_text,
            source=f"Section: {source_section}",
            evidence=evidence,
            confidence=round(best_score, 4),
        )

    # -- retrieval ----------------------------------------------------------

    def _score_passages(
        self, question: str, passages: list[tuple[str, str]]
    ) -> np.ndarray:
        """Combined lexical relevance of every passage to the question.

        Two complementary signals, both cheap and local:

        * TF-IDF cosine (the original signal) rewards distinctive vocabulary.
        * Stemmed token overlap rewards plain shared words that TF-IDF's idf
          weighting can zero out when a word appears in many passages.

        The final score is the max of the two, so a question is refused only
        when *neither* signal finds anything.
        """
        corpus = [p[1] for p in passages]
        try:
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            tfidf_matrix = vectorizer.fit_transform(corpus)
            query_vec = vectorizer.transform([question])
            tfidf_scores = (tfidf_matrix * query_vec.T).toarray().ravel()
        except Exception:
            tfidf_scores = np.zeros(len(passages))

        query_tokens = _content_tokens(question)
        if not query_tokens:
            return np.asarray(tfidf_scores, dtype=float)

        overlap_scores = np.zeros(len(passages))
        for idx, passage in enumerate(corpus):
            passage_tokens = _content_tokens(passage)
            if not passage_tokens:
                continue
            overlap_scores[idx] = len(query_tokens & passage_tokens) / len(query_tokens)

        return np.maximum(np.asarray(tfidf_scores, dtype=float), 0.5 * overlap_scores)

    def _build_context(
        self,
        passages: list[tuple[str, str]],
        scores: np.ndarray,
    ) -> list[tuple[str, str, float]]:
        """Select the passages handed to the model, best first.

        Ranked passages above zero come first, capped by count and character
        budget. When nothing scores (a general question about a paper whose
        vocabulary simply differs), a spread of passages across the document
        is taken instead so the synthesizer still sees representative content
        rather than an arbitrary slice.
        """
        order = sorted(range(len(passages)), key=lambda i: float(scores[i]), reverse=True)
        ranked = [(i, float(scores[i])) for i in order if float(scores[i]) > 0.0]

        selected: list[tuple[str, str, float]] = []
        used_chars = 0
        for idx, score in ranked[:_MAX_CONTEXT_PASSAGES]:
            section, text = passages[idx]
            text = text[:_MAX_PASSAGE_CHARS]
            if used_chars + len(text) > _MAX_CONTEXT_CHARS and selected:
                break
            selected.append((section, text, score))
            used_chars += len(text)

        if not selected:
            # No lexical signal at all: sample across the document instead of
            # handing over a random head slice.
            step = max(1, len(passages) // _MAX_CONTEXT_PASSAGES)
            for idx in range(0, len(passages), step):
                section, text = passages[idx]
                text = text[:_MAX_PASSAGE_CHARS]
                if used_chars + len(text) > _MAX_CONTEXT_CHARS and selected:
                    break
                selected.append((section, text, 0.0))
                used_chars += len(text)
                if len(selected) >= _MAX_CONTEXT_PASSAGES:
                    break

        return selected

    # -- grounded synthesis -------------------------------------------------

    def _synthesize(
        self,
        question: str,
        context: list[tuple[str, str, float]],
        comparison_context: Sequence[dict[str, object]],
    ) -> str:
        """Call Groq server-side and return the grounded answer.

        Every failure is logged with its status and a short body summary —
        never the API key — and mapped onto a :class:`GroqSynthesisError`
        carrying the Groq status so the router can produce the right
        user-facing message.
        """
        context_text = "\n\n".join(
            f"[Section: {section} | relevance: {score:.2f}]\n{passage}"
            for section, passage, score in context
        )
        prompt = (
            f"Paper title: {self.title}\n"
            f"User question: {question}\n\n"
            f"Retrieved paper passages from THIS paper:\n{context_text}"
        )

        if comparison_context:
            neighbour_lines = "\n".join(
                f"- {str(item.get('title', 'Untitled'))} "
                f"(domain: {item.get('label') or 'unknown'}; "
                f"lexical similarity {float(item.get('score', 0.0)):.2f})"
                f"\n  Excerpt: {str(item.get('excerpt', ''))[:500]}"
                for item in comparison_context
            )
            prompt += (
                "\n\nCorpus neighbours of this paper, retrieved by the system's "
                "similar-paper feature. Use these ONLY for comparison questions, "
                "and always attribute statements about them to those papers — "
                "they are NOT evidence about the paper above:\n"
                f"{neighbour_lines}"
            )

        models_to_try = [self.groq_model]
        for fallback in ("openai/gpt-oss-120b", "openai/gpt-oss-20b"):
            if fallback not in models_to_try:
                models_to_try.append(fallback)
        # Two attempts maximum: a bad key or a dead endpoint will not improve
        # on retry, and the request must stay inside the client's timeout.
        models_to_try = models_to_try[:2]

        last_error: GroqSynthesisError | None = None
        for model_name in models_to_try:
            try:
                response = requests.post(
                    GROQ_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self.groq_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": GROQ_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 800,
                    },
                    timeout=20,
                )
            except requests.RequestException as exc:
                logger.warning(
                    "PaperQAEngine | Groq transport failure (model %s): %s", model_name, exc
                )
                last_error = GroqSynthesisError("Groq request failed", status_code=0)
                continue

            if response.status_code == 200:
                content = (
                    response.json().get("choices", [{}])[0].get("message", {}).get("content")
                )
                if isinstance(content, str) and content.strip():
                    return content.strip()
                last_error = GroqSynthesisError(
                    "Groq returned an empty answer", status_code=200
                )
                continue

            # Failure: log the status plus a redacted body snippet — never the
            # key, and never the full request payload.
            logger.warning(
                "PaperQAEngine | Groq HTTP %s from model %s: %s",
                response.status_code,
                model_name,
                response.text[:300],
            )
            last_error = GroqSynthesisError(
                f"Groq returned HTTP {response.status_code}",
                status_code=response.status_code,
            )

        raise last_error or GroqSynthesisError("Groq synthesis failed")
