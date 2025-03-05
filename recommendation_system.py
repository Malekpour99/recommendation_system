import heapq
import logging
import pandas as pd
from datetime import datetime
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Set, Any


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

    def create_user_item_matrix(self, interaction_type="all"):
        """
        Create a user-item interaction matrix.

        Parameters:
        - interaction_type: 'browsing', 'purchase', or 'all'

        Returns:
        - user_item_matrix: DataFrame with users as rows, items as columns, values as interaction strength
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
        df = pd.DataFrame(interactions)

        # If there are multiple interactions between the same user and product, take the sum
        df = df.groupby(["user_id", "product_id"]).sum().reset_index()

        # Create the matrix
        user_item_matrix = df.pivot(
            index="user_id", columns="product_id", values="value"
        ).fillna(0)

        return user_item_matrix


class SimilarityCalculator:
    """Calculates similarities between users and between products."""

    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.user_similarity_matrix = None
        self.product_similarity_matrix = None

    def calculate_user_similarities(self):
        """Calculate similarities between users based on their interactions."""
        user_item_matrix = self.data_loader.create_user_item_matrix()

        # If matrix is empty, return empty similarity matrix
        if user_item_matrix.empty:
            logger.warning(
                "User-item matrix is empty, cannot calculate user similarities"
            )
            return pd.DataFrame()

        # Calculate cosine similarity
        user_similarity = cosine_similarity(user_item_matrix)

        # Convert to DataFrame for easier manipulation
        self.user_similarity_matrix = pd.DataFrame(
            user_similarity,
            index=user_item_matrix.index,
            columns=user_item_matrix.index,
        )

        return self.user_similarity_matrix

    def calculate_product_similarities(self):
        """Calculate similarities between products based on user interactions."""
        user_item_matrix = self.data_loader.create_user_item_matrix()

        # If matrix is empty, return empty similarity matrix
        if user_item_matrix.empty:
            logger.warning(
                "User-item matrix is empty, cannot calculate product similarities"
            )
            return pd.DataFrame()

        # Calculate cosine similarity between products (transpose the matrix first)
        product_similarity = cosine_similarity(user_item_matrix.T)

        # Convert to DataFrame for easier manipulation
        self.product_similarity_matrix = pd.DataFrame(
            product_similarity,
            index=user_item_matrix.columns,
            columns=user_item_matrix.columns,
        )

        return self.product_similarity_matrix

    def get_similar_users(self, user_id, n=5):
        """Get top n users similar to the given user."""
        if self.user_similarity_matrix is None:
            self.calculate_user_similarities()

        if user_id not in self.user_similarity_matrix.index:
            logger.warning(f"User {user_id} not found in similarity matrix")
            return []

        # Get similarities for the user
        similarities = self.user_similarity_matrix.loc[user_id]

        # Sort and get top n (excluding the user itself)
        similar_users = (
            similarities.sort_values(ascending=False)
            .drop(user_id, errors="ignore")
            .head(n)
        )

        return similar_users.index.tolist()

    def get_similar_products(self, product_id, n=5):
        """Get top n products similar to the given product."""
        if self.product_similarity_matrix is None:
            self.calculate_product_similarities()

        if product_id not in self.product_similarity_matrix.index:
            logger.warning(f"Product {product_id} not found in similarity matrix")
            return []

        # Get similarities for the product
        similarities = self.product_similarity_matrix.loc[product_id]

        # Sort and get top n (excluding the product itself)
        similar_products = (
            similarities.sort_values(ascending=False)
            .drop(product_id, errors="ignore")
            .head(n)
        )

        return similar_products.index.tolist()

    def calculate_product_content_similarity(self, product1, product2):
        """Calculate content-based similarity between two products."""
        p1 = self.data_loader.products[product1]
        p2 = self.data_loader.products[product2]

        # Calculate similarity based on tags
        tags1 = set(p1["tags"])
        tags2 = set(p2["tags"])

        tag_similarity = len(tags1.intersection(tags2)) / max(
            len(tags1.union(tags2)), 1
        )

        # Category similarity (1 if same, 0 if different)
        category_similarity = 1 if p1["category"] == p2["category"] else 0

        # Rating similarity
        rating_similarity = 1 - abs(p1["rating"] - p2["rating"]) / 5

        # Weighted combination
        similarity = (
            0.5 * tag_similarity + 0.3 * category_similarity + 0.2 * rating_similarity
        )

        return similarity


class UserBasedRecommender:
    """Generates recommendations based on similar users' behaviors."""

    def __init__(self, data_loader, similarity_calculator):
        self.data_loader = data_loader
        self.similarity_calculator = similarity_calculator

    def recommend(self, user_id, n=5, exclude_viewed=False, exclude_purchased=True):
        """
        Generate recommendations for a user based on similar users' interactions.

        Parameters:
        - user_id: The ID of the user to recommend for
        - n: Number of recommendations to generate
        - exclude_viewed: Whether to exclude products the user has already viewed
        - exclude_purchased: Whether to exclude products the user has already purchased

        Returns:
        - List of recommended product IDs
        """
        # Get user interactions
        user_interactions = self.data_loader.get_user_interactions(user_id)

        # Products the user has already interacted with
        viewed_products = set(b["product_id"] for b in user_interactions["browsing"])
        purchased_products = set(
            p["product_id"] for p in user_interactions["purchases"]
        )

        # Products to exclude
        excluded_products = set()
        if exclude_viewed:
            excluded_products.update(viewed_products)
        if exclude_purchased:
            excluded_products.update(purchased_products)

        # Get similar users
        similar_users = self.similarity_calculator.get_similar_users(user_id, n=10)

        # If no similar users found, return empty list
        if not similar_users:
            logger.warning(f"No similar users found for user {user_id}")
            return []

        # Get products that similar users have interacted with
        product_scores = defaultdict(float)

        for similar_user_id in similar_users:
            similar_user_interactions = self.data_loader.get_user_interactions(
                similar_user_id
            )

            # Score viewed products
            for interaction in similar_user_interactions["browsing"]:
                product_id = interaction["product_id"]
                if product_id not in excluded_products:
                    product_scores[product_id] += 1

            # Score purchased products (with higher weight)
            for interaction in similar_user_interactions["purchases"]:
                product_id = interaction["product_id"]
                if product_id not in excluded_products:
                    product_scores[product_id] += 5 * interaction.get("quantity", 1)

        # Sort products by score and get top n
        recommended_products = heapq.nlargest(
            n, product_scores.items(), key=lambda x: x[1]
        )

        return [product_id for product_id, score in recommended_products]


class ItemBasedRecommender:
    """Generates recommendations based on similar products."""

    def __init__(self, data_loader, similarity_calculator):
        self.data_loader = data_loader
        self.similarity_calculator = similarity_calculator

    def recommend(self, user_id, n=5, exclude_viewed=False, exclude_purchased=True):
        """
        Generate recommendations for a user based on similar products.

        Parameters:
        - user_id: The ID of the user to recommend for
        - n: Number of recommendations to generate
        - exclude_viewed: Whether to exclude products the user has already viewed
        - exclude_purchased: Whether to exclude products the user has already purchased

        Returns:
        - List of recommended product IDs
        """
        # Get user interactions
        user_interactions = self.data_loader.get_user_interactions(user_id)

        # Products the user has already interacted with
        viewed_products = set(b["product_id"] for b in user_interactions["browsing"])
        purchased_products = set(
            p["product_id"] for p in user_interactions["purchases"]
        )

        # Products to exclude
        excluded_products = set()
        if exclude_viewed:
            excluded_products.update(viewed_products)
        if exclude_purchased:
            excluded_products.update(purchased_products)

        # Products the user has interacted with (to find similar products)
        interacted_products = viewed_products.union(purchased_products)

        # If user hasn't interacted with any products, return empty list
        if not interacted_products:
            logger.warning(f"User {user_id} has no interactions")
            return []

        # Get similar products
        product_scores = defaultdict(float)

        for product_id in interacted_products:
            # Get weight based on interaction type
            weight = 5 if product_id in purchased_products else 1

            # Get similar products
            similar_products = self.similarity_calculator.get_similar_products(
                product_id, n=10
            )

            for similar_product_id in similar_products:
                if similar_product_id not in excluded_products:
                    # Calculate content-based similarity to further refine the score
                    content_similarity = (
                        self.similarity_calculator.calculate_product_content_similarity(
                            product_id, similar_product_id
                        )
                    )

                    # Add weighted score
                    product_scores[similar_product_id] += weight * content_similarity

        # Sort products by score and get top n
        recommended_products = heapq.nlargest(
            n, product_scores.items(), key=lambda x: x[1]
        )

        return [product_id for product_id, score in recommended_products]


class ContextualBooster:
    """Adjusts recommendation scores based on contextual signals."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def boost_scores(self, product_scores, user_id, current_time=None):
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
        user_device = self.data_loader.users[user_id]["device"]

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
            season = "Winter"

        # For simplicity, map Back-to-School to Fall and Holiday to Winter
        season_mapping = {"Fall": "Back-to-School", "Winter": "Holiday"}

        # Boost scores based on contextual signals
        boosted_scores = dict(product_scores)

        for product_id in product_scores:
            product = self.data_loader.products[product_id]
            category = product["category"]

            # Boost based on contextual signals if available for the category
            if category in self.data_loader.contextual_signals:
                signal = self.data_loader.contextual_signals[category]

                # Boost if current day is a peak day for the category
                if day_of_week in signal["peak_days"]:
                    boosted_scores[product_id] *= 1.2

                # Boost if current season matches the category's peak season
                mapped_season = season_mapping.get(season, season)
                if signal["season"] == mapped_season or signal["season"] == "All Year":
                    boosted_scores[product_id] *= 1.3

            # Boost based on device type (simplified)
            # Assume mobile users prefer certain categories
            if user_device == "mobile":
                if category in ["Electronics", "Accessories"]:
                    boosted_scores[product_id] *= 1.1
            else:  # desktop
                if category in ["Office Supplies"]:
                    boosted_scores[product_id] *= 1.1

        return boosted_scores


class DiversityEnhancer:
    """Ensures diversity in the final recommendation list."""

    def __init__(self, data_loader):
        self.data_loader = data_loader

    def enhance_diversity(self, recommendations, n=5):
        """
        Enhance diversity in the recommendation list.

        Parameters:
        - recommendations: List of (product_id, score) tuples
        - n: Number of recommendations to return

        Returns:
        - List of product IDs with enhanced diversity
        """
        if not recommendations:
            return []

        # Group products by category
        categories = defaultdict(list)

        for product_id, score in recommendations:
            category = self.data_loader.products[product_id]["category"]
            categories[category].append((product_id, score))

        # Sort products within each category by score
        for category in categories:
            categories[category].sort(key=lambda x: x[1], reverse=True)

        # Select products in a round-robin fashion from each category
        diverse_recommendations = []
        category_index = 0
        category_keys = list(categories.keys())

        while len(diverse_recommendations) < n and category_keys:
            category = category_keys[category_index % len(category_keys)]

            if categories[category]:
                product_id, score = categories[category].pop(0)
                diverse_recommendations.append(product_id)

                # If category is empty, remove it
                if not categories[category]:
                    category_keys.remove(category)

            category_index += 1

        # If we still need more recommendations, add the highest-scoring remaining products
        if len(diverse_recommendations) < n:
            remaining_products = []
            for category in categories:
                remaining_products.extend(categories[category])

            remaining_products.sort(key=lambda x: x[1], reverse=True)

            for product_id, _ in remaining_products[: n - len(diverse_recommendations)]:
                diverse_recommendations.append(product_id)

        return diverse_recommendations

