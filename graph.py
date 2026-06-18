import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("dataset/kc_house_data.csv")

# Take first 500 rows for faster plotting
df = df.head(500)

# Create graph
plt.figure(figsize=(8,5))
plt.scatter(df["sqft_living"], df["price"])

plt.title("House Price vs Living Area")
plt.xlabel("Living Area (sq ft)")
plt.ylabel("Price")

plt.savefig("static/price_graph.png")
print("Graph Created Successfully!")