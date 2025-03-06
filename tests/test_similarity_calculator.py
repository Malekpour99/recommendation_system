import pytest
import pandas as pd
from data_handling.data_processor import DataProcessor
from data_handling.similarity_calculator import SimilarityCalculator


def test_calculate_user_similarities(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    similarity_calculator = SimilarityCalculator(processor)
    user_similarities = similarity_calculator.calculate_user_similarities()

    assert isinstance(user_similarities, pd.DataFrame)
    assert user_similarities.shape == (4, 4)  # Since we have 4 users
    assert user_similarities.iloc[0, 0] == 1.0  # Self-similarity should be 1


def test_calculate_product_similarities(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    similarity_calculator = SimilarityCalculator(processor)
    product_similarities = similarity_calculator.calculate_product_similarities()

    assert isinstance(product_similarities, pd.DataFrame)
    assert product_similarities.shape == (5, 5)  # Since we have 5 products
    # prevent floating point precision issues due to memory storage
    assert product_similarities.iloc[0, 0] == pytest.approx(
        1.0, abs=1e-9
    )  # Self-similarity should be 1


def test_get_similar_users(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    similarity_calculator = SimilarityCalculator(processor)
    similarity_calculator.calculate_user_similarities()
    similar_users = similarity_calculator.get_similar_users(1, n=2)

    assert isinstance(similar_users, list)
    assert len(similar_users) <= 2  # Should return at most 2 similar users


def test_get_similar_products(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    similarity_calculator = SimilarityCalculator(processor)
    similarity_calculator.calculate_product_similarities()
    similar_products = similarity_calculator.get_similar_products(201, n=2)

    assert isinstance(similar_products, list)
    assert len(similar_products) <= 2  # Should return at most 2 similar products


def test_calculate_product_content_similarity(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    similarity_calculator = SimilarityCalculator(processor)
    similarity_score = similarity_calculator.calculate_product_content_similarity(
        201, 202
    )

    assert isinstance(similarity_score, float)
    assert 0.0 <= similarity_score <= 1.0  # Similarity should be in range [0,1]
