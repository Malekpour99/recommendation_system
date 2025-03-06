import pytest
from datetime import datetime

from data_handling.data_processor import DataProcessor
from data_handling.similarity_calculator import SimilarityCalculator
from recommendation_enhancements.explanation_generator import ExplanationGenerator


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


def test_generate_explanation_user_based(explanation_generator):
    explanation = explanation_generator.generate_explanation(
        user_id=1, product_id=204, recommendation_source="user_based"
    )
    assert (
        explanation == "Recommended because users similar to you purchased Smartwatch."
    )


def test_generate_explanation_item_based(explanation_generator):
    explanation = explanation_generator.generate_explanation(
        user_id=1, product_id=204, recommendation_source="item_based"
    )
    # Assuming the user has interacted with similar products
    assert "Recommended because it's similar to" in explanation


def test_generate_explanation_trending(explanation_generator):
    explanation = explanation_generator.generate_explanation(
        user_id=1, product_id=204, recommendation_source="trending"
    )
    assert explanation == "Recommended because it's trending in Wearable Tech."


def test_generate_explanation_viewed_not_purchased(explanation_generator):
    explanation = explanation_generator.generate_explanation(
        user_id=1, product_id=203, recommendation_source="viewed_not_purchased"
    )
    assert explanation == "You viewed Running Shoes but haven't purchased it yet."


def test_generate_explanation_contextual(explanation_generator, monkeypatch):
    # Mock the current time to control the day of the week
    mock_time = datetime(2024, 2, 5)  # A Monday

    # Create a mock datetime class with a now() method that returns mock_time
    class MockDateTime:
        @staticmethod
        def now():
            return mock_time

    monkeypatch.setattr(
        "recommendation_enhancements.explanation_generator.datetime",
        MockDateTime,
    )

    explanation = explanation_generator.generate_explanation(
        user_id=1, product_id=203, recommendation_source="contextual"
    )
    assert explanation == "Popular Sports choice for Monday."


def test_generate_explanation_default(explanation_generator):
    explanation = explanation_generator.generate_explanation(
        user_id=1, product_id=204, recommendation_source="unknown_source"
    )
    assert explanation == "Recommended for you."
