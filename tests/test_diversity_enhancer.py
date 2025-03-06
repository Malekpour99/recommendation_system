import pytest

from recommendation_enhancements.diversity_enhancer import DiversityEnhancer


@pytest.fixture
def diversity_enhancer(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    from data_handling.data_processor import DataProcessor

    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )
    return DiversityEnhancer(processor)


def test_enhance_diversity_basic(diversity_enhancer):
    # Given a list of product recommendations with scores
    recommendations = [
        (201, 4.8),  # Gaming Mouse (Electronics)
        (202, 4.6),  # Noise Cancelling Headphones (Audio)
        (203, 4.5),  # Running Shoes (Sports)
        (204, 4.7),  # Smartwatch (Wearable Tech)
        (205, 4.9),  # Ergonomic Chair (Furniture)
    ]

    # Enhance diversity and get top 5 products
    diverse_recommendations = diversity_enhancer.enhance_diversity(recommendations, n=5)

    # Check if we have exactly 5 recommendations
    assert len(diverse_recommendations) == 5

    # Check that we get one product from each category
    categories = {
        "Electronics": 0,
        "Audio": 0,
        "Sports": 0,
        "Wearable Tech": 0,
        "Furniture": 0,
    }

    for product_id in diverse_recommendations:
        category = diversity_enhancer.data_processor.products[product_id]["category"]
        categories[category] += 1

    # Assert each category is represented at least once
    for category, count in categories.items():
        assert count == 1


def test_enhance_diversity_more_than_available_categories(diversity_enhancer):
    # Given a list of product recommendations with scores, where some categories will not be represented
    recommendations = [
        (201, 4.8),  # Gaming Mouse (Electronics)
        (202, 4.6),  # Noise Cancelling Headphones (Audio)
        (203, 4.5),  # Running Shoes (Sports)
    ]

    # Enhance diversity and get top 5 products
    diverse_recommendations = diversity_enhancer.enhance_diversity(recommendations, n=5)

    # Check if we have exactly 3 recommendations
    assert len(diverse_recommendations) == 3


def test_enhance_diversity_empty_list(diversity_enhancer):
    # Given an empty list of recommendations
    recommendations = []

    # Enhance diversity (should return an empty list)
    diverse_recommendations = diversity_enhancer.enhance_diversity(recommendations, n=5)

    # Assert that the result is an empty list
    assert diverse_recommendations == []


def test_enhance_diversity_single_category(diversity_enhancer):
    # Given a list of product recommendations from a single category
    recommendations = [
        (201, 4.8),  # Gaming Mouse (Electronics)
        (202, 4.6),  # Noise Cancelling Headphones (Audio)
    ]

    # Enhance diversity and get top 2 products
    diverse_recommendations = diversity_enhancer.enhance_diversity(recommendations, n=2)

    # Check that we have 2 recommendations, and both are from the top products
    assert len(diverse_recommendations) == 2
    assert diverse_recommendations == [201, 202]


def test_enhance_diversity_diverse_recommendations_count(diversity_enhancer):
    # Given a list of product recommendations with scores
    recommendations = [
        (201, 4.8),  # Gaming Mouse (Electronics)
        (202, 4.6),  # Noise Cancelling Headphones (Audio)
        (203, 4.5),  # Running Shoes (Sports)
        (204, 4.7),  # Smartwatch (Wearable Tech)
        (205, 4.9),  # Ergonomic Chair (Furniture)
    ]

    # Enhance diversity and get top 3 products
    diverse_recommendations = diversity_enhancer.enhance_diversity(recommendations, n=3)

    # Ensure only 3 recommendations are returned
    assert len(diverse_recommendations) == 3

    # Assert diverse categories are selected
    categories = set(
        diversity_enhancer.data_processor.products[product_id]["category"]
        for product_id in diverse_recommendations
    )
    assert len(categories) == 3  # Ensure diversity with at least 3 different categories
