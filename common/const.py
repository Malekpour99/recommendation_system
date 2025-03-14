# Boost Weights
SEASON_BOOST_WEIGHT = 1.3
DEVICE_BOOST_WEIGHT = 1.1
PEAK_DAY_BOOST_WEIGHT = 1.2

# Boost Season Mapping
# For simplicity, map Holiday to Winter & Back-to-School to Fall
SEASON_MAPPING = {
    "Holiday": "Winter",
    "Back-to-School": "Fall",
}

# Boost - Device user interest categories
# For simplicity, assume device users prefer certain categories
MOBILE_USER_INTEREST_CATEGORIES = [
    "Electronics",
    "Accessories",
]
DESKTOP_USER_INTEREST_CATEGORIES = [
    "Office Supplies",
]

# Similarity Weights
TAG_SIMILARITY_WEIGHT = 0.5
RATING_SIMILARITY_WEIGHT = 0.2
CATEGORY_SIMILARITY_WEIGHT = 0.3

# User-Product Matrix
VIEW_VALUE = 1  # Value for a single view
PURCHASE_VALUE = 5  # Value for a single item purchase (quantity = 1)
