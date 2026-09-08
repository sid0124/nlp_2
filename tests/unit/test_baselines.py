"""Baseline construction from ``configs/model.yaml``.

The property under test is that **no hyper-parameter lives in Python** (master
spec §32). These tests read the expected values out of the configuration object
rather than restating them, so a config change moves the assertion with it and a
value hard-coded back into ``src/models/baselines.py`` fails here.

The second theme is the capability gap between the two configured classifiers.
``LinearSVC`` has no ``predict_proba``, and the whole evaluation layer depends on
that being reported honestly rather than papered over.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from sklearn.calibration import CalibratedClassifierCV

from src.config.settings import ClassifierConfig, Settings, VectorizerConfig
from src.models.baselines import (
    CLASSIFIER_STEP,
    VECTORIZER_STEP,
    UnknownEstimatorError,
    build_baseline,
    build_classifier,
    build_vectorizer,
    linear_coefficients,
    prediction_scores,
    resolved_params,
    supports_probabilities,
)

MODEL_NAMES = ["tfidf_logreg", "tfidf_svm"]

#: A tiny, trivially separable corpus. Enough to fit both classifiers so their
#: score-function behaviour can be probed on a *fitted* estimator.
TEXTS = [
    "reward policy agent learns reward",
    "policy gradient reward agent",
    "pixel segmentation image occlusion",
    "image pixel convolutional segmentation",
]
LABELS = ["ai", "ai", "vision", "vision"]


@pytest.fixture
def fitted(request: pytest.FixtureRequest, settings: Settings) -> Pipeline:
    """A pipeline for the requested baseline, fitted on :data:`TEXTS`."""
    # The tiny 4-document corpus has 2 samples per class, so the configured
    # 5-fold calibration CV cannot stratify; clamp to 2 folds exactly as the
    # trainer does for a small training split.
    pipeline = build_baseline(
        settings.model, request.param, seed=settings.seed, calibration_cv=2
    )
    # min_df from configuration would drop nearly every term in a 4-document
    # corpus, so it is relaxed for this fixture only.
    pipeline.set_params(**{f"{VECTORIZER_STEP}__min_df": 1})
    return pipeline.fit(TEXTS, LABELS)


def uncalibrated_model(settings: Settings):
    """The model configuration with the SVM's calibration switched off.

    Calibration is a configured property, so the uncalibrated behaviour is
    tested against the same configuration with only that flag flipped — the
    same way a user would turn it off.
    """
    model = settings.model.model_copy(deep=True)
    model.baselines["tfidf_svm"].calibration.enabled = False
    return model


# ---------------------------------------------------------------------------
# Configuration is the single source of hyper-parameters
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_pipeline_has_the_two_named_steps(name: str, settings: Settings) -> None:
    pipeline = build_baseline(settings.model, name, seed=settings.seed)
    assert list(pipeline.named_steps) == [VECTORIZER_STEP, CLASSIFIER_STEP]
    assert isinstance(pipeline.named_steps[VECTORIZER_STEP], TfidfVectorizer)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_every_configured_param_reaches_the_estimator(name: str, settings: Settings) -> None:
    """Each configured value is read back off the constructed object."""
    pipeline = build_baseline(settings.model, name, seed=settings.seed)
    for step, config in (
        (VECTORIZER_STEP, settings.model.vectorizer_for(name)),
        (CLASSIFIER_STEP, settings.model.baseline(name).classifier),
    ):
        step_estimator = pipeline.named_steps[step]
        params = step_estimator.get_params()
        # A calibrated head holds the configured classifier one level down, so
        # the configured parameters are read off that inner estimator.
        if isinstance(step_estimator, CalibratedClassifierCV):
            params = step_estimator.estimator.get_params()
        for key, expected in config.params.items():
            # YAML lists become tuples where scikit-learn requires one.
            wanted = tuple(expected) if isinstance(expected, list) else expected
            assert params[key] == wanted, f"{name}.{step}.{key}"


@pytest.mark.parametrize(
    "name,classifier_type", [("tfidf_logreg", LogisticRegression), ("tfidf_svm", LinearSVC)]
)
def test_classifier_type_follows_config(
    name: str, classifier_type: type, settings: Settings
) -> None:
    pipeline = build_baseline(settings.model, name, seed=settings.seed)
    head = pipeline.named_steps[CLASSIFIER_STEP]
    if isinstance(head, CalibratedClassifierCV):
        # The configured classifier sits inside the calibration wrapper.
        head = head.estimator
    assert isinstance(head, classifier_type)


def test_ngram_range_is_coerced_to_a_tuple(settings: Settings) -> None:
    """YAML has no tuple type, and scikit-learn validates this one as a tuple."""
    vectorizer = build_vectorizer(settings.model.vectorizer_for("tfidf_logreg"))
    assert isinstance(vectorizer.ngram_range, tuple)
    assert vectorizer.ngram_range == tuple(
        settings.model.vectorizer_for("tfidf_logreg").params["ngram_range"]
    )


def test_list_valued_params_are_not_blanket_converted(settings: Settings) -> None:
    """Only ``ngram_range`` is tuple-typed; ``stop_words`` must stay as configured.

    A blanket list-to-tuple conversion would be wrong here: ``stop_words`` accepts
    a string or a list, and a tuple is neither.
    """
    vectorizer = build_vectorizer(settings.model.vectorizer_for("tfidf_logreg"))
    assert not isinstance(vectorizer.stop_words, tuple)
    assert vectorizer.stop_words == "english"


def test_unknown_param_fails_at_construction() -> None:
    """A config typo surfaces immediately, not deep inside a fit call."""
    with pytest.raises(TypeError):
        build_vectorizer(VectorizerConfig(type="tfidf", params={"not_a_real_param": 1}))


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_run_seed_is_injected_when_config_omits_it(name: str, settings: Settings) -> None:
    """The seed is a run-level property, so it is not repeated per model entry."""
    assert "random_state" not in settings.model.baseline(name).classifier.params
    pipeline = build_baseline(settings.model, name, seed=1234)
    head = pipeline.named_steps[CLASSIFIER_STEP]
    if isinstance(head, CalibratedClassifierCV):
        head = head.estimator
    assert head.random_state == 1234


def test_explicit_random_state_in_config_wins() -> None:
    """A deliberately pinned model is never silently overridden by the run seed."""
    config = ClassifierConfig(type="logistic_regression", params={"random_state": 7})
    assert build_classifier(config, seed=99).random_state == 7


def test_seed_is_skipped_for_estimators_that_reject_it() -> None:
    """Injection is signature-checked, so it cannot raise on a seedless estimator."""
    vectorizer = build_vectorizer(VectorizerConfig(type="tfidf", params={}))
    assert not hasattr(vectorizer, "random_state")


# ---------------------------------------------------------------------------
# Registry errors
# ---------------------------------------------------------------------------
def test_unknown_vectorizer_type_lists_the_registry() -> None:
    with pytest.raises(UnknownEstimatorError, match="tfidf"):
        build_vectorizer(VectorizerConfig(type="word2vec", params={}))


def test_unknown_classifier_type_lists_the_registry() -> None:
    with pytest.raises(UnknownEstimatorError, match="linear_svc"):
        build_classifier(ClassifierConfig(type="random_forest", params={}), seed=0)


def test_unknown_baseline_name_lists_available(settings: Settings) -> None:
    with pytest.raises(KeyError, match="tfidf_logreg"):
        build_baseline(settings.model, "no_such_baseline", seed=settings.seed)


def test_multilabel_wraps_the_configured_classifier(settings: Settings) -> None:
    """Multi-label mode uses one independent classifier per label."""
    pipeline = build_baseline(settings.model, "tfidf_logreg", seed=settings.seed, multilabel=True)
    classifier = pipeline.named_steps[CLASSIFIER_STEP]
    assert isinstance(classifier, OneVsRestClassifier)
    assert isinstance(classifier.estimator, LogisticRegression)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_resolved_params_include_untouched_defaults(name: str, settings: Settings) -> None:
    """The manifest records what ran, including defaults config never mentioned."""
    pipeline = build_baseline(settings.model, name, seed=settings.seed)
    resolved = resolved_params(pipeline)
    assert set(resolved) == {VECTORIZER_STEP, CLASSIFIER_STEP}
    assert "analyzer" in resolved[VECTORIZER_STEP]  # never set in configs/model.yaml
    for values in resolved.values():
        for value in values.values():
            assert value is None or isinstance(value, bool | int | float | str | list)


# ---------------------------------------------------------------------------
# The capability gap: LinearSVC has no predict_proba
# ---------------------------------------------------------------------------
def test_logreg_advertises_probabilities(settings: Settings) -> None:
    pipeline = build_baseline(settings.model, "tfidf_logreg", seed=settings.seed)
    assert supports_probabilities(pipeline)


def test_calibrated_svm_advertises_probabilities(settings: Settings) -> None:
    """The configured SVM is calibrated, so its pipeline exposes predict_proba.

    ``Pipeline`` forwards the capability from its final estimator, so asking the
    pipeline is an accurate test of what the training code will see.
    """
    pipeline = build_baseline(settings.model, "tfidf_svm", seed=settings.seed)
    assert isinstance(pipeline.named_steps[CLASSIFIER_STEP], CalibratedClassifierCV)
    assert supports_probabilities(pipeline)


def test_uncalibrated_svm_does_not_advertise_probabilities(settings: Settings) -> None:
    """With calibration disabled the SVM degrades to decision margins only."""
    pipeline = build_baseline(uncalibrated_model(settings), "tfidf_svm", seed=settings.seed)
    assert not supports_probabilities(pipeline)
    assert not hasattr(pipeline.named_steps[CLASSIFIER_STEP], "predict_proba")


@pytest.mark.parametrize("fitted", ["tfidf_svm"], indirect=True)
def test_calibrated_svm_scores_are_genuine_probabilities(fitted: Pipeline) -> None:
    """Calibration turns the SVM's margins into real probabilities.

    Each score is in [0, 1] and each row sums to 1 across classes — the two
    properties that let the UI print percentages and a confidence honestly.
    """
    scores, kind = prediction_scores(fitted, TEXTS)
    assert kind == "probability"
    assert scores is not None
    assert scores.shape == (len(TEXTS), 2)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-6)


def test_calibrated_svm_feature_evidence_is_reachable(settings: Settings) -> None:
    """linear_coefficients unwraps the calibration ensemble to feature weights."""
    pipeline = build_baseline(
        settings.model, "tfidf_svm", seed=settings.seed, calibration_cv=2
    )
    pipeline.set_params(**{f"{VECTORIZER_STEP}__min_df": 1})
    pipeline.fit(TEXTS, LABELS)
    coefficients = linear_coefficients(pipeline.named_steps[CLASSIFIER_STEP])
    assert coefficients is not None
    n_features = len(pipeline.named_steps[VECTORIZER_STEP].vocabulary_)
    assert coefficients.shape[-1] == n_features


@pytest.mark.parametrize("fitted", ["tfidf_logreg"], indirect=True)
def test_logreg_scores_are_probabilities(fitted: Pipeline) -> None:
    scores, kind = prediction_scores(fitted, TEXTS)
    assert kind == "probability"
    assert scores is not None
    assert scores.shape == (len(TEXTS), 2)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    np.testing.assert_allclose(scores.sum(axis=1), 1.0, rtol=1e-6)


def test_uncalibrated_svm_degrades_to_a_labelled_decision_margin(settings: Settings) -> None:
    """The SVM path must not crash, and must not pass a margin off as a probability."""
    pipeline = build_baseline(uncalibrated_model(settings), "tfidf_svm", seed=settings.seed)
    pipeline.set_params(**{f"{VECTORIZER_STEP}__min_df": 1})
    pipeline.fit(TEXTS, LABELS)
    scores, kind = prediction_scores(pipeline, TEXTS)
    assert kind == "decision"
    assert scores is not None
    assert scores.shape == (len(TEXTS), 2)
    # The margin is unbounded and signed: nothing here is on a probability scale.
    assert not np.allclose(scores.sum(axis=1), 1.0)
    assert scores.min() < 0.0


def test_binary_decision_output_is_widened_to_two_columns(settings: Settings) -> None:
    """A binary ``decision_function`` returns one column; consumers need two.

    Without the mirror, a two-class run would index a one-column matrix by class
    position and raise partway through evaluation.
    """
    pipeline = build_baseline(uncalibrated_model(settings), "tfidf_svm", seed=settings.seed)
    pipeline.set_params(**{f"{VECTORIZER_STEP}__min_df": 1})
    pipeline.fit(TEXTS, LABELS)
    raw = np.asarray(pipeline.decision_function(TEXTS))
    assert raw.ndim == 1
    scores, _ = prediction_scores(pipeline, TEXTS)
    assert scores is not None
    np.testing.assert_allclose(scores[:, 1], raw)
    np.testing.assert_allclose(scores[:, 0], -raw)


@pytest.mark.parametrize("fitted", MODEL_NAMES, indirect=True)
def test_argmax_of_scores_agrees_with_predict(fitted: Pipeline) -> None:
    """Whatever the score kind, it must rank the same class ``predict`` chose."""
    scores, _ = prediction_scores(fitted, TEXTS)
    assert scores is not None
    classes = fitted.named_steps[CLASSIFIER_STEP].classes_
    assert list(classes[scores.argmax(axis=1)]) == list(fitted.predict(TEXTS))


def test_estimator_with_neither_score_function_returns_unavailable() -> None:
    """Confidence is omitted, not faked, when an estimator exposes no scores."""

    class LabelOnly(BaseEstimator):
        """Minimal estimator exposing only ``predict``."""

        def fit(self, x: Any, y: Any) -> LabelOnly:  # noqa: D102
            return self

        def predict(self, x: Any) -> list[str]:  # noqa: D102
            return ["ai"] * len(x)

    scores, kind = prediction_scores(LabelOnly(), TEXTS)
    assert scores is None
    assert kind == "unavailable"
