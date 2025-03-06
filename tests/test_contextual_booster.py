import pytest
from datetime import datetime

from recommendation_enhancements.contextual_booster import ContextualBooster


@pytest.fixture
def contextual_booster(sample_data):
    users_data, products_data, browsing_data, purchase_data, contextual_data = (
        sample_data
    )
    from data_handling.data_processor import DataProcessor

    processor = DataProcessor()
    processor.load_data(
        users_data, products_data, browsing_data, purchase_data, contextual_data
    )
    return ContextualBooster(processor)


def test_boost_scores_day_of_week(contextual_booster):
    product_scores = {201: 1.0, 202: 1.0, 203: 1.0}
    boosted_scores = contextual_booster.boost_scores(
        product_scores, user_id=1, current_time=datetime(2024, 2, 3)
    )

    # Saturday is a peak day for Electronics (category of product 201) - 1.2
    # And February is considered in winter - 1.3
    assert boosted_scores[201] == pytest.approx(1.2 * 1.3)

    # No peak day boost for products 202 & 203
    assert boosted_scores[202] == pytest.approx(1.0)
    assert boosted_scores[203] == pytest.approx(1.0)


def test_boost_scores_seasonal(contextual_booster):
    product_scores = {201: 1.0, 202: 1.0, 203: 1.0, 204: 1.0}
    boosted_scores = contextual_booster.boost_scores(
        product_scores, user_id=1, current_time=datetime(2024, 12, 1)
    )

    # Winter is the peak season for Electronics (category of product 201) - 1.3
    # 2024-12-01 is Sunday -> Peak day - 1.2
    assert boosted_scores[201] == pytest.approx(1.2 * 1.3)

    # Wearable Tech is always in season (product 204)
    assert boosted_scores[204] == pytest.approx(1.3)

    # No seasonal boost for other products
    assert boosted_scores[202] == pytest.approx(1.0)
    assert boosted_scores[203] == pytest.approx(1.0)


def test_boost_scores_device(contextual_booster):
    product_scores = {201: 1.0, 205: 1.0}

    # User 2 is on mobile, Electronics should get a boost
    boosted_scores_mobile = contextual_booster.boost_scores(
        product_scores, user_id=2, current_time=datetime(2024, 4, 8)
    )
    assert boosted_scores_mobile[201] == pytest.approx(1.1)
    assert boosted_scores_mobile[205] == pytest.approx(1.0)

    # User 1 is on desktop, Office Supplies should get a boost (but we have Furniture, so no boost)
    boosted_scores_desktop = contextual_booster.boost_scores(
        product_scores, user_id=1, current_time=datetime(2024, 4, 8)
    )
    assert boosted_scores_desktop[201] == pytest.approx(1.0)
    assert boosted_scores_desktop[205] == pytest.approx(1.0)


def test_combined_boosts(contextual_booster):
    product_scores = {201: 1.0}  # Electronics
    boosted_scores = contextual_booster.boost_scores(
        product_scores, user_id=2, current_time=datetime(2024, 12, 1)
    )

    # Winter season boost (1.3), mobile device boost (1.1) and Sunday Peak day (1.2)
    expected_boost = 1.0 * 1.2 * 1.3 * 1.1
    assert boosted_scores[201] == pytest.approx(expected_boost)
