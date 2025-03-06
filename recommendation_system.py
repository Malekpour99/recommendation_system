import heapq
import logging
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from data_handling.data_processor import DataProcessor
from data_handling.similarity_calculator import SimilarityCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class UserBasedRecommender:
    """Generates recommendations based on similar users' behaviors."""

    def __init__(
        self,
        data_processor: DataProcessor,
        similarity_calculator: SimilarityCalculator,
    ) -> None:
        self.data_processor = data_processor
        self.similarity_calculator = similarity_calculator

    def recommend(
        self,
        user_id: int,
        n: int = 5,
        exclude_viewed: bool = False,
        exclude_purchased: bool = True,
    ) -> List[int]:
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
        user_interactions = self.data_processor.get_user_interactions(user_id)

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
            similar_user_interactions = self.data_processor.get_user_interactions(
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

    def __init__(
        self,
        data_processor: DataProcessor,
        similarity_calculator: SimilarityCalculator,
    ) -> None:
        self.data_processor = data_processor
        self.similarity_calculator = similarity_calculator

    def recommend(
        self,
        user_id: int,
        n: int = 5,
        exclude_viewed: bool = False,
        exclude_purchased: bool = True,
    ) -> List[int]:
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
        user_interactions = self.data_processor.get_user_interactions(user_id)

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
            season = "Winter"

        # For simplicity, map Back-to-School to Fall and Holiday to Winter
        season_mapping = {"Fall": "Back-to-School", "Winter": "Holiday"}

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


class RecommendationManager:
    """Orchestrates the entire recommendation process."""

    def __init__(self, data_processor: DataProcessor) -> None:
        self.data_processor = data_processor
        self.similarity_calculator = SimilarityCalculator(data_processor)
        self.user_based_recommender = UserBasedRecommender(
            data_processor, self.similarity_calculator
        )
        self.item_based_recommender = ItemBasedRecommender(
            data_processor, self.similarity_calculator
        )
        self.contextual_booster = ContextualBooster(data_processor)
        self.diversity_enhancer = DiversityEnhancer(data_processor)
        self.explanation_generator = ExplanationGenerator(
            data_processor, self.similarity_calculator
        )

        # Pre-compute similarities
        logger.info("Pre-computing user similarities...")
        self.similarity_calculator.calculate_user_similarities()

        logger.info("Pre-computing product similarities...")
        self.similarity_calculator.calculate_product_similarities()

    def generate_candidates(self, user_id: int) -> Dict[int, float]:
        """
        Generate candidate products for recommendations.

        Parameters:
        - user_id: The ID of the user

        Returns:
        - Dictionary of candidate product IDs with initial scores
        """
        # Get user interactions
        user_interactions = self.data_processor.get_user_interactions(user_id)

        # Initialize candidate dictionary
        candidates = {}

        # 1. Viewed but not purchased products
        viewed_products = set(b["product_id"] for b in user_interactions["browsing"])
        purchased_products = set(
            p["product_id"] for p in user_interactions["purchases"]
        )
        viewed_not_purchased = viewed_products - purchased_products

        for product_id in viewed_not_purchased:
            candidates[product_id] = 3.0  # Moderate initial score

        # 2. User-based recommendations
        user_based_recs = self.user_based_recommender.recommend(
            user_id, n=5, exclude_viewed=True, exclude_purchased=True
        )
        for product_id in user_based_recs:
            candidates[product_id] = candidates.get(product_id, 0) + 5.0

        # 3. Item-based recommendations
        item_based_recs = self.item_based_recommender.recommend(
            user_id, n=5, exclude_viewed=True, exclude_purchased=True
        )
        for product_id in item_based_recs:
            candidates[product_id] = candidates.get(product_id, 0) + 4.0

        # 4. Popular products in user's preferred categories
        if user_interactions["purchases"]:
            preferred_categories = set(
                self.data_processor.products[p["product_id"]]["category"]
                for p in user_interactions["purchases"]
            )

            for product_id, product in self.data_processor.products.items():
                if (
                    product["category"] in preferred_categories
                    and product_id not in candidates
                ):
                    candidates[product_id] = 2.0

        # 5. Trending products (simplified)
        for product_id, product in self.data_processor.products.items():
            product_interactions = self.data_processor.get_product_interactions(
                product_id
            )
            interaction_count = len(product_interactions["browsing"]) + len(
                product_interactions["purchases"]
            )

            if product_id not in candidates and interaction_count > 0:
                candidates[product_id] = 1.5 * interaction_count

        return candidates

    def score_candidates(self, candidates: Dict[int, float]) -> Dict[int, float]:
        """
        Score and refine candidate products.

        Parameters:
        - candidates: Dictionary of candidate product IDs with initial scores

        Returns:
        - Dictionary of product IDs to final scores
        """
        # Normalize and adjust candidate scores
        max_score = max(candidates.values()) if candidates else 1

        scored_candidates = {
            product_id: score / max_score * 10  # Normalize to 0-10 scale
            for product_id, score in candidates.items()
        }

        return scored_candidates

    def get_recommendations(
        self,
        user_id: int,
        n: int = 5,
        include_explanations: bool = True,
    ):
        """
        Get personalized recommendations for a user.

        Parameters:
        - user_id: The ID of the user
        - n: Number of recommendations to return
        - include_explanations: Whether to include explanations

        Returns:
        - List of recommended products (with explanations if requested)
        """
        # Check if user exists
        if user_id not in self.data_processor.users:
            logger.warning(f"User {user_id} not found")
            return []

        # Get user interactions
        user_interactions = self.data_processor.get_user_interactions(user_id)

        # Handle cold start scenario for new users
        if not user_interactions["browsing"] and not user_interactions["purchases"]:
            logger.info(f"Cold start: User {user_id} has no interactions")

            # Recommend popular products across categories
            candidates = {}
            for product_id, product in self.data_processor.products.items():
                product_interactions = self.data_processor.get_product_interactions(
                    product_id
                )
                interaction_count = len(product_interactions["browsing"]) + len(
                    product_interactions["purchases"]
                )
                candidates[product_id] = interaction_count

            recommended_product_ids = sorted(
                candidates, key=candidates.get, reverse=True
            )[:n]

            if include_explanations:
                return [
                    {
                        "product_id": pid,
                        "product_name": self.data_processor.products[pid]["name"],
                        "explanation": "Recommended as a popular product for new users",
                    }
                    for pid in recommended_product_ids
                ]
            else:
                return recommended_product_ids

        # Generate candidates
        candidates = self.generate_candidates(user_id)

        # Score candidates
        candidate_scores = self.score_candidates(candidates)

        # Boost scores based on contextual signals
        boosted_scores = self.contextual_booster.boost_scores(candidate_scores, user_id)

        # Convert to (product_id, score) tuples
        recommendations = [
            (product_id, score) for product_id, score in boosted_scores.items()
        ]

        # Sort by score
        recommendations.sort(key=lambda x: x[1], reverse=True)

        # Enhance diversity
        diverse_recommendations = self.diversity_enhancer.enhance_diversity(
            recommendations, n
        )

        # Add explanations if requested
        if include_explanations:
            result = []
            for product_id in diverse_recommendations:
                product = self.data_processor.products[product_id]

                # Determine the source of the recommendation
                source = "item_based"  # Default source
                if product_id in [
                    b["product_id"]
                    for b in user_interactions["browsing"]
                    if b["product_id"]
                    not in [p["product_id"] for p in user_interactions["purchases"]]
                ]:
                    source = "viewed_not_purchased"

                # Generate explanation
                explanation = self.explanation_generator.generate_explanation(
                    user_id, product_id, source
                )

                result.append(
                    {
                        "product_id": product_id,
                        "product_name": product["name"],
                        "explanation": explanation,
                    }
                )

            return result
        else:
            return diverse_recommendations
