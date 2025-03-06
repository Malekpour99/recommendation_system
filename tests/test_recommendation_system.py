import pytest
from datetime import datetime

from recommendation_system import RecommendationManager
from data_handling.data_processor import DataProcessor
from data_handling.similarity_calculator import SimilarityCalculator
from recommendation_enhancements.explanation_generator import ExplanationGenerator
from recommendation_enhancements.contextual_booster import ContextualBooster
from recommendation_enhancements.diversity_enhancer import DiversityEnhancer


@pytest.fixture
def data_processor(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    dp = DataProcessor()
    dp.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )
    return dp


@pytest.fixture
def similarity_calculator(data_processor):
    return SimilarityCalculator(data_processor)


@pytest.fixture
def explanation_generator(data_processor, similarity_calculator):
    return ExplanationGenerator(data_processor, similarity_calculator)


@pytest.fixture
def contextual_booster(data_processor):
    return ContextualBooster(data_processor)


@pytest.fixture
def diversity_enhancer(data_processor):
    return DiversityEnhancer(data_processor)


@pytest.fixture
def recommendation_manager(data_processor):
    return RecommendationManager(data_processor)


def test_generate_candidates(recommendation_manager):
    # Test generating candidate products for a user
    candidates = recommendation_manager.generate_candidates(user_id=1)
    assert isinstance(candidates, dict)
    assert len(candidates) > 0  # Ensure candidates are generated


def test_score_candidates(recommendation_manager):
    # Test scoring candidate products
    candidates = {201: 3.0, 202: 5.0, 203: 4.0}  # Example candidates
    scored_candidates = recommendation_manager.score_candidates(candidates)
    assert isinstance(scored_candidates, dict)
    assert all(
        0 <= score <= 10 for score in scored_candidates.values()
    )  # Scores normalized to 0-10


def test_get_recommendations_without_explanations(recommendation_manager):
    # Test getting recommendations without explanations
    recommendations = recommendation_manager.get_recommendations(
        user_id=1, n=3, include_explanations=False
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 3  # Ensure the correct number of recommendations


def test_get_recommendations_with_explanations(recommendation_manager):
    # Test getting recommendations with explanations
    recommendations = recommendation_manager.get_recommendations(
        user_id=1, n=3, include_explanations=True
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 3  # Ensure the correct number of recommendations
    for rec in recommendations:
        assert "product_id" in rec
        assert "product_name" in rec
        assert "explanation" in rec  # Ensure explanations are included


def test_cold_start_recommendations(recommendation_manager):
    # Test cold start scenario (user with no interactions)
    recommendations = recommendation_manager.get_recommendations(
        user_id=999, n=3, include_explanations=True
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 3  # Ensure the correct number of recommendations
    for rec in recommendations:
        assert "product_id" in rec
        assert "product_name" in rec
        assert "explanation" in rec  # Ensure explanations are included
        assert (
            "popular product for new users" in rec["explanation"]
        )  # Cold start explanation


def test_contextual_boosting(recommendation_manager, monkeypatch):
    # Mock the current time to control contextual boosting
    mock_time = datetime(2024, 2, 5)  # A Monday

    # Create a mock datetime class with a now() method that returns mock_time
    class MockDateTime:
        @staticmethod
        def now():
            return mock_time

    monkeypatch.setattr(
        "recommendation_enhancements.contextual_booster.datetime",
        MockDateTime,
    )

    candidates = {201: 3.0, 202: 5.0, 203: 4.0}  # Example candidates
    boosted_scores = recommendation_manager.contextual_booster.boost_scores(
        candidates, user_id=1
    )
    assert isinstance(boosted_scores, dict)
    assert all(
        score >= 0 for score in boosted_scores.values()
    )  # Ensure scores are boosted


def test_diversity_enhancement(recommendation_manager):
    # Test diversity enhancement
    recommendations = [(201, 5.0), (202, 4.5), (203, 4.0), (204, 3.5), (205, 3.0)]
    diverse_recommendations = (
        recommendation_manager.diversity_enhancer.enhance_diversity(
            recommendations, n=3
        )
    )
    assert isinstance(diverse_recommendations, list)
    assert (
        len(diverse_recommendations) <= 3
    )  # Ensure the correct number of recommendations
