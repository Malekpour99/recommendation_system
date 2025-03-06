import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from recommendation_system import DataProcessor, RecommendationManager


def generate_large_dataset(
    num_users: int, num_products: int, num_interactions: int
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    """
    Generate a large synthetic dataset for performance testing of the recommendation system

    Parameters:
        num_users (int): The number of users in the dataset.
        num_products (int): The number of products in the dataset.
        num_interactions (int): The number of user-product interactions to generate.

    Returns:
        tuple: A tuple containing lists of users, products, browsing history, purchase history,
               and contextual signals.
    """
    users_data = [
        {
            "user_id": i,
            "name": f"User_{i}",
            "location": random.choice(
                ["New York", "Los Angeles", "Chicago", "San Francisco"]
            ),
            "device": random.choice(["mobile", "desktop"]),
        }
        for i in range(1, num_users + 1)
    ]

    products_data = [
        {
            "product_id": i,
            "name": f"Product_{i}",
            "category": random.choice(
                [
                    "Electronics",
                    "Fitness",
                    "Personal Care",
                    "Office Supplies",
                    "Accessories",
                ]
            ),
            "tags": [f"tag_{random.randint(1, 10)}" for _ in range(3)],
            "rating": round(random.uniform(3.0, 5.0), 1),
        }
        for i in range(1, num_products + 1)
    ]

    browsing_history = []
    purchase_history = []

    for _ in range(num_interactions):
        user_id = random.randint(1, num_users)
        product_id = random.randint(1, num_products)
        timestamp = f"2023-10-{random.randint(1, 30):02d} {random.randint(0, 23):02d}:{random.randint(0, 59):02d}:00"

        if random.random() < 0.7:  # 70% browsing, 30% purchase
            browsing_history.append(
                {"user_id": user_id, "product_id": product_id, "timestamp": timestamp}
            )
        else:
            purchase_history.append(
                {
                    "user_id": user_id,
                    "product_id": product_id,
                    "quantity": random.randint(1, 3),
                    "timestamp": timestamp,
                }
            )

    contextual_signals = [
        {
            "category": "Electronics",
            "peak_days": ["Friday", "Saturday"],
            "season": "Holiday",
        },
        {
            "category": "Fitness",
            "peak_days": ["Monday", "Wednesday"],
            "season": "Summer",
        },
        {
            "category": "Office Supplies",
            "peak_days": ["Tuesday", "Thursday"],
            "season": "Back-to-School",
        },
        {"category": "Personal Care", "peak_days": ["Sunday"], "season": "All Year"},
    ]

    return (
        users_data,
        products_data,
        browsing_history,
        purchase_history,
        contextual_signals,
    )


def benchmark_recommendation_performance(
    users_data: List[Dict[str, object]],
    products_data: List[Dict[str, object]],
    browsing_history: List[Dict[str, object]],
    purchase_history: List[Dict[str, object]],
    contextual_signals: List[Dict[str, object]],
) -> List[float]:
    """
    Benchmark recommendation system performance.

    Parameters:
        users_data (list): A list of user data dictionaries.
        products_data (list): A list of product data dictionaries.
        browsing_history (list): A list of browsing interactions.
        purchase_history (list): A list of purchase interactions.
        contextual_signals (list): A list of contextual data affecting recommendations.

    Returns:
        list: A list of recommendation generation times for sample users.
    """
    data_processor = DataProcessor()
    data_processor.load_data(
        users_data,
        products_data,
        browsing_history,
        purchase_history,
        contextual_signals,
    )

    recommendation_manager = RecommendationManager(data_processor)

    # Measure recommendation generation time
    recommendation_times = []
    for user_id in range(1, 11):  # Test first 10 users
        start_time = time.time()
        recommendation_manager.get_recommendations(user_id, n=5)
        end_time = time.time()
        recommendation_times.append(end_time - start_time)

    return recommendation_times


def plot_performance_benchmark(
    dataset_sizes: List[int],
    performance_data: List[float],
) -> None:
    """
    Plot recommendation generation time for different dataset sizes

    Parameters:
        dataset_sizes (list): A list of dataset sizes (number of interactions).
        performance_data (list): A list of average recommendation times corresponding to dataset sizes.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(dataset_sizes, performance_data, marker="o")
    plt.title("Recommendation System Performance")
    plt.xlabel("Number of Interactions")
    plt.ylabel("Average Recommendation Time (seconds)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("recommendation_performance.png")
    plt.close()


def main():
    # Performance testing across different dataset sizes
    dataset_sizes = [1000, 5000, 10000, 50000, 100000]
    performance_data = []

    for num_interactions in dataset_sizes:
        print(f"\nBenchmarking with {num_interactions} interactions...")
        (
            users_data,
            products_data,
            browsing_history,
            purchase_history,
            contextual_signals,
        ) = generate_large_dataset(
            num_users=1000, num_products=500, num_interactions=num_interactions
        )

        recommendation_times = benchmark_recommendation_performance(
            users_data,
            products_data,
            browsing_history,
            purchase_history,
            contextual_signals,
        )

        avg_recommendation_time = np.mean(recommendation_times)
        performance_data.append(avg_recommendation_time)

        print(f"Average Recommendation Time: {avg_recommendation_time:.4f} seconds")

    # Plot performance benchmark
    plot_performance_benchmark(dataset_sizes, performance_data)


if __name__ == "__main__":
    main()
