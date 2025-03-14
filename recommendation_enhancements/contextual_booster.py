from datetime import datetime
from typing import Dict, Optional

from data_handling.data_processor import DataProcessor
from common.const import SEASON_BOOST_WEIGHT, PEAK_DAY_BOOST_WEIGHT, DEVICE_BOOST_WEIGHT


class ContextualBooster:
    """Adjusts recommendation scores based on contextual signals."""

    def __init__(self, data_processor: DataProcessor) -> None:
        self.data_processor = data_processor

    def boost_scores(
        self,
        product_scores: Dict[int, float],
        user_id: int,
        current_time: Optional[datetime] = None,
    ) -> Dict[int, float]:
        """
        Boost product scores based on contextual signals.

        Parameters:
        - product_scores: Dictionary of product IDs to scores
        - user_id: The ID of the user
        - current_time: Current datetime (defaults to now if None)

        Returns:
        - Dictionary of product IDs to boosted scores
        """
        if current_time is None:
            current_time = datetime.now()

        # Get user device
        user_device = self.data_processor.users[user_id]["device"]

        # Get day of week
        day_of_week = current_time.strftime("%A")

        # Get season (simplified)
        month = current_time.month
        if 3 <= month <= 5:
            season = "Spring"
        elif 6 <= month <= 8:
            season = "Summer"
        elif 9 <= month <= 11:
            season = "Fall"
        else:
            # 12, 1, 2
            season = "Winter"

        # For simplicity, map Back-to-School to Fall and Holiday to Winter
        season_mapping = {"Back-to-School": "Fall", "Holiday": "Winter"}

        # Boost scores based on contextual signals
        boosted_scores = dict(product_scores)

        for product_id in product_scores:
            product = self.data_processor.products[product_id]
            category = product["category"]

            # Boost based on contextual signals if available for the category
            if category in self.data_processor.contextual_signals:
                signal = self.data_processor.contextual_signals[category]

                # Boost if current day is a peak day for the category
                if day_of_week in signal["peak_days"]:
                    boosted_scores[product_id] *= PEAK_DAY_BOOST_WEIGHT

                # Boost if current season matches the category's peak season
                mapped_season = season_mapping.get(season, season)
                if signal["season"] == mapped_season or signal["season"] == "All Year":
                    boosted_scores[product_id] *= SEASON_BOOST_WEIGHT

            # Boost based on device type (simplified)
            # Assume mobile users prefer certain categories
            if user_device == "mobile":
                if category in ["Electronics", "Accessories"]:
                    boosted_scores[product_id] *= DEVICE_BOOST_WEIGHT
            else:  # desktop
                if category in ["Office Supplies"]:
                    boosted_scores[product_id] *= DEVICE_BOOST_WEIGHT

        return boosted_scores
