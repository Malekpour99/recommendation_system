import pandas as pd
from datetime import datetime

from data_handling.data_processor import DataProcessor


def test_load_data(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    assert len(processor.users) == 4
    assert len(processor.products) == 5
    assert len(processor.browsing_history) == 5
    assert len(processor.purchase_history) == 4
    assert len(processor.contextual_signals) == 4


def test_convert_timestamps(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    for interaction in processor.browsing_history + processor.purchase_history:
        assert isinstance(interaction["timestamp"], datetime)


def test_get_user_interactions(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    interactions = processor.get_user_interactions(1)
    assert len(interactions["browsing"]) == 2
    assert len(interactions["purchases"]) == 1


def test_get_product_interactions(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    interactions = processor.get_product_interactions(201)
    assert len(interactions["browsing"]) == 1
    assert len(interactions["purchases"]) == 1


def test_create_user_product_matrix(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )

    matrix = processor.create_user_product_matrix()
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape == (4, 5)
    assert matrix.loc[1, 201] == 1  # 1 browsing event
    assert matrix.loc[2, 205] == 10  # (5 * 2) purchases
