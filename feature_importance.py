import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/kc_house_data.csv")

# Features used in model
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement"
]

# Load trained model
model = joblib.load("house_price_model.pkl")

# Get feature importance
importance = model.feature_importances_

# Plot graph
plt.figure(figsize=(10, 6))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()

# Save image
plt.savefig("static/feature_importance.png")

print("Feature Importance Graph Saved!")