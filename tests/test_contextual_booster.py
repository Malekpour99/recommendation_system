import pytest
from datetime import datetime

from recommendation_enhancements.contextual_booster import ContextualBooster
from common.const import SEASON_BOOST_WEIGHT, PEAK_DAY_BOOST_WEIGHT, DEVICE_BOOST_WEIGHT


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

    # Saturday is a peak day for Electronics (category of product 201)
    # And February is considered in winter
    assert boosted_scores[201] == pytest.approx(
        PEAK_DAY_BOOST_WEIGHT * SEASON_BOOST_WEIGHT
    )

    # No peak day boost for products 202 & 203
    assert boosted_scores[202] == pytest.approx(1.0)
    assert boosted_scores[203] == pytest.approx(1.0)


def test_boost_scores_seasonal(contextual_booster):
    product_scores = {201: 1.0, 202: 1.0, 203: 1.0, 204: 1.0}
    boosted_scores = contextual_booster.boost_scores(
        product_scores, user_id=1, current_time=datetime(2024, 12, 1)
    )

    # Winter is the peak season for Electronics (category of product 201)
    # 2024-12-01 is Sunday -> Peak day
    assert boosted_scores[201] == pytest.approx(
        PEAK_DAY_BOOST_WEIGHT * SEASON_BOOST_WEIGHT
    )

    # Wearable Tech is always in season (product 204)
    assert boosted_scores[204] == pytest.approx(SEASON_BOOST_WEIGHT)

    # No seasonal boost for other products
    assert boosted_scores[202] == pytest.approx(1.0)
    assert boosted_scores[203] == pytest.approx(1.0)


def test_boost_scores_device(contextual_booster):
    product_scores = {201: 1.0, 205: 1.0}

    # User 2 is on mobile, Electronics should get a boost
    boosted_scores_mobile = contextual_booster.boost_scores(
        product_scores, user_id=2, current_time=datetime(2024, 4, 8)
    )
    assert boosted_scores_mobile[201] == pytest.approx(DEVICE_BOOST_WEIGHT)
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

    # Winter season boost, mobile device boost and Sunday Peak day
    expected_boost = (
        1.0 * SEASON_BOOST_WEIGHT * PEAK_DAY_BOOST_WEIGHT * DEVICE_BOOST_WEIGHT
    )
    assert boosted_scores[201] == pytest.approx(expected_boost)
