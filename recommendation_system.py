from datetime import datetime
from collections import defaultdict
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses the e-commerce data for the recommendation system."""

    def __init__(self):
        self.users = {}
        self.products = {}
        self.browsing_history = []
        self.purchase_history = []
        self.contextual_signals = {}

    def load_data(
        self, users_data, products_data, browsing_data, purchase_data, contextual_data
    ):
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

    def _convert_timestamps(self):
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

    def get_user_interactions(self, user_id):
        """Get all interactions for a specific user."""
        # Ensure timestamps are converted
        self._convert_timestamps()

        browsing = [b for b in self.browsing_history if b["user_id"] == user_id]
        purchases = [p for p in self.purchase_history if p["user_id"] == user_id]

        return {"browsing": browsing, "purchases": purchases}

    def get_product_interactions(self, product_id):
        """Get all interactions for a specific product."""
        # Ensure timestamps are converted
        self._convert_timestamps()

        browsing = [b for b in self.browsing_history if b["product_id"] == product_id]
        purchases = [p for p in self.purchase_history if p["product_id"] == product_id]

        return {"browsing": browsing, "purchases": purchases}

