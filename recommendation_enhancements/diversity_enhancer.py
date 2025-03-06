from typing import List, Tuple
from collections import defaultdict

from data_handling.data_processor import DataProcessor


class DiversityEnhancer:
    """Ensures diversity in the final recommendation list."""

    def __init__(self, data_processor: DataProcessor) -> None:
        self.data_processor = data_processor

    def enhance_diversity(
        self,
        recommendations: List[Tuple[int, float]],
        n: int = 5,
    ) -> List[int]:
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
            category = self.data_processor.products[product_id]["category"]
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
