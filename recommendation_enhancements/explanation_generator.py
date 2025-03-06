from datetime import datetime

from data_handling.data_processor import DataProcessor
from data_handling.similarity_calculator import SimilarityCalculator


class ExplanationGenerator:
    """Generates explanations for recommended products."""

    def __init__(
        self,
        data_processor: DataProcessor,
        similarity_calculator: SimilarityCalculator,
    ):
        self.data_processor = data_processor
        self.similarity_calculator = similarity_calculator

    def generate_explanation(
        self,
        user_id: int,
        product_id: int,
        recommendation_source: str,
    ) -> str:
        """
        Generate an explanation for why a product was recommended.

        Parameters:
        - user_id: The ID of the user
        - product_id: The ID of the recommended product
        - recommendation_source: Source of the recommendation (e.g., 'user_based', 'item_based', 'trending')

        Returns:
        - Explanation string
        """
        product = self.data_processor.products[product_id]

        if recommendation_source == "user_based":
            return (
                f"Recommended because users similar to you purchased {product['name']}."
            )

        elif recommendation_source == "item_based":
            # Find a product the user interacted with that's similar to this one
            user_interactions = self.data_processor.get_user_interactions(user_id)
            interacted_products = set(
                b["product_id"] for b in user_interactions["browsing"]
            ).union(set(p["product_id"] for p in user_interactions["purchases"]))

            max_similarity = 0
            most_similar_product = None

            for interacted_product_id in interacted_products:
                similarity = (
                    self.similarity_calculator.calculate_product_content_similarity(
                        interacted_product_id, product_id
                    )
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_product = interacted_product_id

            if most_similar_product:
                similar_product = self.data_processor.products[most_similar_product]
                return f"Recommended because it's similar to {similar_product['name']} that you viewed earlier."
            else:
                return f"Recommended because it matches your browsing preferences."

        elif recommendation_source == "trending":
            return f"Recommended because it's trending in {product['category']}."

        elif recommendation_source == "viewed_not_purchased":
            return f"You viewed {product['name']} but haven't purchased it yet."

        elif recommendation_source == "contextual":
            # Get current time and day
            current_time = datetime.now()
            day_of_week = current_time.strftime("%A")

            if product["category"] in self.data_processor.contextual_signals:
                signal = self.data_processor.contextual_signals[product["category"]]

                if day_of_week in signal["peak_days"]:
                    return f"Popular {product['category']} choice for {day_of_week}."
                else:
                    return f"Recommended based on seasonal trends in {product['category']}."
            else:
                return f"Recommended based on current trends."

        else:
            return f"Recommended for you."
