import pytest


@pytest.fixture
def sample_data():
    users_data = [
        {"user_id": 1, "name": "Eve", "location": "Boston", "device": "desktop"},
        {"user_id": 2, "name": "Frank", "location": "Seattle", "device": "mobile"},
        {"user_id": 3, "name": "Grace", "location": "Austin", "device": "desktop"},
        {"user_id": 4, "name": "Henry", "location": "Denver", "device": "mobile"},
    ]

    products_data = [
        {
            "product_id": 201,
            "name": "Gaming Mouse",
            "category": "Electronics",
            "tags": ["gaming", "mouse", "wireless"],
            "rating": 4.8,
        },
        {
            "product_id": 202,
            "name": "Noise Cancelling Headphones",
            "category": "Audio",
            "tags": ["headphones", "noise cancelling", "wireless"],
            "rating": 4.6,
        },
        {
            "product_id": 203,
            "name": "Running Shoes",
            "category": "Sports",
            "tags": ["shoes", "running", "fitness"],
            "rating": 4.5,
        },
        {
            "product_id": 204,
            "name": "Smartwatch",
            "category": "Wearable Tech",
            "tags": ["smartwatch", "fitness", "tracking"],
            "rating": 4.7,
        },
        {
            "product_id": 205,
            "name": "Ergonomic Chair",
            "category": "Furniture",
            "tags": ["chair", "ergonomic", "office"],
            "rating": 4.9,
        },
    ]

    browsing_data = [
        {"user_id": 1, "product_id": 201, "timestamp": "2024-02-01 09:00:00"},
        {"user_id": 1, "product_id": 203, "timestamp": "2024-02-01 09:10:00"},
        {"user_id": 2, "product_id": 202, "timestamp": "2024-02-02 10:15:00"},
        {"user_id": 3, "product_id": 204, "timestamp": "2024-02-03 13:45:00"},
        {"user_id": 4, "product_id": 205, "timestamp": "2024-02-04 15:30:00"},
    ]

    purchase_data = [
        {
            "user_id": 1,
            "product_id": 204,
            "quantity": 1,
            "timestamp": "2024-02-10 14:00:00",
        },
        {
            "user_id": 2,
            "product_id": 205,
            "quantity": 2,
            "timestamp": "2024-02-12 11:00:00",
        },
        {
            "user_id": 3,
            "product_id": 203,
            "quantity": 1,
            "timestamp": "2024-02-15 08:45:00",
        },
        {
            "user_id": 4,
            "product_id": 201,
            "quantity": 1,
            "timestamp": "2024-02-16 09:30:00",
        },
    ]

    contextual_data = [
        {
            "category": "Electronics",
            "peak_days": ["Saturday", "Sunday"],
            "season": "Winter",
        },
        {"category": "Sports", "peak_days": ["Monday", "Thursday"], "season": "Spring"},
        {
            "category": "Furniture",
            "peak_days": ["Wednesday", "Friday"],
            "season": "Summer",
        },
        {"category": "Wearable Tech", "peak_days": ["Tuesday"], "season": "All Year"},
    ]

    return users_data, products_data, browsing_data, purchase_data, contextual_data
