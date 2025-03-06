import logging
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

from data_handling.data_processor import DataProcessor

# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """Calculates similarities between users and between products."""

    def __init__(self, data_processor: DataProcessor) -> None:
        self.data_processor = data_processor
        self.user_similarity_matrix: Optional[pd.DataFrame] = None
        self.product_similarity_matrix: Optional[pd.DataFrame] = None

    def calculate_user_similarities(self) -> pd.DataFrame:
        """Calculate similarities between users based on their interactions."""
        user_product_matrix = self.data_processor.create_user_product_matrix()

        # If matrix is empty, return empty similarity matrix
        if user_product_matrix.empty:
            logger.warning(
                "User-item matrix is empty, cannot calculate user similarities"
            )
            return pd.DataFrame()

        # Calculate cosine similarity
        user_similarity = cosine_similarity(user_product_matrix)

        # Convert to DataFrame for easier manipulation
        self.user_similarity_matrix = pd.DataFrame(
            user_similarity,
            index=user_product_matrix.index,
            columns=user_product_matrix.index,
        )

        return self.user_similarity_matrix

    def calculate_product_similarities(self) -> pd.DataFrame:
        """Calculate similarities between products based on user interactions."""
        user_product_matrix = self.data_processor.create_user_product_matrix()

        # If matrix is empty, return empty similarity matrix
        if user_product_matrix.empty:
            logger.warning(
                "User-item matrix is empty, cannot calculate product similarities"
            )
            return pd.DataFrame()

        # Calculate cosine similarity between products (transpose the matrix first)
        product_similarity = cosine_similarity(user_product_matrix.T)

        # Convert to DataFrame for easier manipulation
        self.product_similarity_matrix = pd.DataFrame(
            product_similarity,
            index=user_product_matrix.columns,
            columns=user_product_matrix.columns,
        )

        return self.product_similarity_matrix

    def get_similar_users(self, user_id: int, n: int = 5) -> List[int]:
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

    def get_similar_products(self, product_id: int, n: int = 5) -> List[int]:
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

    def calculate_product_content_similarity(self, product1: int, product2: int):
        """Calculate content-based similarity between two products."""
        p1 = self.data_processor.products[product1]
        p2 = self.data_processor.products[product2]

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
