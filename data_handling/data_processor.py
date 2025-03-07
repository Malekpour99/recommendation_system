import logging
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any

from caching.cache_decorator import cache_result

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Loads and preprocesses the e-commerce data for the recommendation system."""

    def __init__(self) -> None:
        self.users: Dict[int, Dict[str, Any]] = {}
        self.products: Dict[int, Dict[str, Any]] = {}
        self.browsing_history: List[Dict[str, Any]] = []
        self.purchase_history: List[Dict[str, Any]] = []
        self.contextual_signals: Dict[str, Dict[str, Any]] = {}

    def load_data(
        self,
        users_data: List[Dict[str, Any]],
        products_data: List[Dict[str, Any]],
        browsing_data: List[Dict[str, Any]],
        purchase_data: List[Dict[str, Any]],
        contextual_data: List[Dict[str, Any]],
    ) -> None:
        """Load data from the provided datasets."""
        # Load users
        for user in users_data:
            self.users[user["user_id"]] = user

        # Load products
        for product in products_data:
            self.products[product["product_id"]] = product

        # Load browsing history
        self.browsing_history = browsing_data

        # Load purchase history
        self.purchase_history = purchase_data

        # Load contextual signals
        for signal in contextual_data:
            self.contextual_signals[signal["category"]] = signal

        # Ensure timestamps are converted to datetime
        self._convert_timestamps()

        logger.info(f"Loaded {len(self.users)} users, {len(self.products)} products")
        logger.info(
            f"Loaded {len(self.browsing_history)} browsing events, {len(self.purchase_history)} purchase events"
        )

    def _convert_timestamps(self) -> None:
        """Convert all timestamps to datetime objects."""
        for interaction in self.browsing_history + self.purchase_history:
            # If timestamp is already a datetime, skip
            if isinstance(interaction["timestamp"], datetime):
                continue

            # If timestamp is a string, convert to datetime
            if isinstance(interaction["timestamp"], str):
                try:
                    interaction["timestamp"] = datetime.strptime(
                        interaction["timestamp"], "%Y-%m-%d %H:%M:%S"
                    )
                except ValueError:
                    logger.warning(
                        f"Could not parse timestamp: {interaction['timestamp']}"
                    )
                    # Optionally, set to a default timestamp or current time
                    interaction["timestamp"] = datetime.now()

    @cache_result(ttl_seconds=3600)  # 1 hour
    def get_user_interactions(self, user_id: int) -> Dict[str, List[Dict[str, Any]]]:
        """Get all interactions for a specific user."""

        browsing = [b for b in self.browsing_history if b["user_id"] == user_id]
        purchases = [p for p in self.purchase_history if p["user_id"] == user_id]

        return {"browsing": browsing, "purchases": purchases}

    @cache_result(ttl_seconds=3600)  # 1 hour
    def get_product_interactions(
        self, product_id: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get all interactions for a specific product."""

        browsing = [b for b in self.browsing_history if b["product_id"] == product_id]
        purchases = [p for p in self.purchase_history if p["product_id"] == product_id]

        return {"browsing": browsing, "purchases": purchases}

    @cache_result(ttl_seconds=3600)  # 1 hour
    def create_user_product_matrix(self, interaction_type: str = "all") -> pd.DataFrame:
        """
        Create a user-product interaction matrix.

        Parameters:
        - interaction_type: 'browsing', 'purchase', or 'all'

        Returns:
        - user_product_matrix: DataFrame with users as rows, products as columns, values as interaction strength
        """
        interactions = []

        if interaction_type in ["browsing", "all"]:
            for interaction in self.browsing_history:
                interactions.append(
                    {
                        "user_id": interaction["user_id"],
                        "product_id": interaction["product_id"],
                        "value": 1,  # Value of 1 for a view
                    }
                )

        if interaction_type in ["purchase", "all"]:
            for interaction in self.purchase_history:
                interactions.append(
                    {
                        "user_id": interaction["user_id"],
                        "product_id": interaction["product_id"],
                        "value": 5
                        * interaction.get(
                            "quantity", 1
                        ),  # Value of 5 * quantity for a purchase
                    }
                )

        # Create DataFrame from interactions
        interactions_data_frame = pd.DataFrame(interactions)

        # If there are multiple interactions between the same user and product, aggregate them
        interactions_data_frame = (
            interactions_data_frame.groupby(["user_id", "product_id"])
            .sum()
            .reset_index()
        )

        # Create the matrix
        user_product_matrix = interactions_data_frame.pivot(
            index="user_id", columns="product_id", values="value"
        ).fillna(0)

        return user_product_matrix
