import logging

from data_handling.data_processor import DataProcessor
from recommendation_system import RecommendationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Sample data from the provided document
users_data = [
    {"user_id": 1, "name": "Alice", "location": "New York", "device": "mobile"},
    {"user_id": 2, "name": "Bob", "location": "Los Angeles", "device": "desktop"},
    {"user_id": 3, "name": "Charlie", "location": "Chicago", "device": "mobile"},
    {"user_id": 4, "name": "Diana", "location": "San Francisco", "device": "desktop"},
]

products_data = [
    {
        "product_id": 101,
        "name": "Wireless Earbuds",
        "category": "Electronics",
        "tags": ["audio", "wireless", "Bluetooth"],
        "rating": 4.5,
    },
    {
        "product_id": 102,
        "name": "Smartphone Case",
        "category": "Accessories",
        "tags": ["phone", "protection", "case"],
        "rating": 4.2,
    },
    {
        "product_id": 103,
        "name": "Yoga Mat",
        "category": "Fitness",
        "tags": ["exercise", "mat", "yoga"],
        "rating": 4.7,
    },
    {
        "product_id": 104,
        "name": "Electric Toothbrush",
        "category": "Personal Care",
        "tags": ["hygiene", "electric", "toothbrush"],
        "rating": 4.3,
    },
    {
        "product_id": 105,
        "name": "Laptop Stand",
        "category": "Office Supplies",
        "tags": ["work", "laptop", "stand"],
        "rating": 4.6,
    },
]

browsing_history = [
    {"user_id": 1, "product_id": 101, "timestamp": "2023-10-01 10:00:00"},
    {"user_id": 1, "product_id": 103, "timestamp": "2023-10-01 10:05:00"},
    {"user_id": 2, "product_id": 102, "timestamp": "2023-10-02 11:30:00"},
    {"user_id": 3, "product_id": 104, "timestamp": "2023-10-03 14:20:00"},
    {"user_id": 4, "product_id": 105, "timestamp": "2023-10-04 16:45:00"},
]

purchase_history = [
    {
        "user_id": 1,
        "product_id": 104,
        "quantity": 1,
        "timestamp": "2023-10-10 15:20:00",
    },
    {
        "user_id": 2,
        "product_id": 105,
        "quantity": 2,
        "timestamp": "2023-10-12 12:00:00",
    },
    {
        "user_id": 3,
        "product_id": 103,
        "quantity": 1,
        "timestamp": "2023-10-15 09:30:00",
    },
    {
        "user_id": 4,
        "product_id": 101,
        "quantity": 1,
        "timestamp": "2023-10-16 10:15:00",
    },
]

contextual_signals = [
    {
        "category": "Electronics",
        "peak_days": ["Friday", "Saturday"],
        "season": "Holiday",
    },
    {"category": "Fitness", "peak_days": ["Monday", "Wednesday"], "season": "Summer"},
    {
        "category": "Office Supplies",
        "peak_days": ["Tuesday", "Thursday"],
        "season": "Back-to-School",
    },
    {"category": "Personal Care", "peak_days": ["Sunday"], "season": "All Year"},
]


def run_recommendation_tests():
    """Run comprehensive tests on the recommendation system."""

    # Initialize data loader
    data_processor = DataProcessor()

    # Ensure all timestamps are strings
    processed_browsing_history = [
        {
            **item,
            "timestamp": (
                item["timestamp"]
                if isinstance(item["timestamp"], str)
                else item["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            ),
        }
        for item in browsing_history
    ]

    processed_purchase_history = [
        {
            **item,
            "timestamp": (
                item["timestamp"]
                if isinstance(item["timestamp"], str)
                else item["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            ),
        }
        for item in purchase_history
    ]

    data_processor.load_data(
        users_data,
        products_data,
        processed_browsing_history,
        processed_purchase_history,
        contextual_signals,
    )

    # Initialize recommendation manager
    recommendation_manager = RecommendationManager(data_processor)

    # Test recommendations for each user
    print("\n--- Recommendation Tests ---")
    for user_id in [1, 2, 3, 4]:
        print(f"\nRecommendations for User {user_id}:")
        try:
            recommendations = recommendation_manager.get_recommendations(
                user_id, n=3, include_explanations=True
            )

            for rec in recommendations:
                print(f"- Product: {rec['product_name']}")
                print(f"  Explanation: {rec['explanation']}")
        except Exception as e:
            logger.error(f"Error generating recommendations for user {user_id}: {e}")

    # Test cold start scenario (with a new user)
    print("\n--- Cold Start Test ---")
    cold_start_user_id = 5
    data_processor.users[cold_start_user_id] = {
        "user_id": 5,
        "name": "Eve",
        "location": "Denver",
        "device": "mobile",
    }

    try:
        cold_start_recommendations = recommendation_manager.get_recommendations(
            cold_start_user_id, n=3, include_explanations=True
        )

        print("Cold Start Recommendations:")
        for rec in cold_start_recommendations:
            print(f"- Product: {rec['product_name']}")
            print(f"  Explanation: {rec['explanation']}")
    except Exception as e:
        logger.error(f"Error generating cold start recommendations: {e}")


def main():
    run_recommendation_tests()


if __name__ == "__main__":
    main()
