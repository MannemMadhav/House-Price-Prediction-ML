from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

model = joblib.load("house_price_model.pkl")
accuracy = 0.51

@app.route("/")
def home():
    return render_template(
    "index.html",
    accuracy=round(accuracy * 100, 2)
)

@app.route("/predict", methods=["POST"])
def predict():
    bedrooms = int(request.form["bedrooms"])
    bathrooms = float(request.form["bathrooms"])
    sqft_living = int(request.form["sqft_living"])

    new_house = pd.DataFrame({
    "bedrooms":[bedrooms],
    "bathrooms":[bathrooms],
    "sqft_living":[sqft_living]
})

    prediction = model.predict(new_house)

    return render_template(
        "index.html",
       prediction_text=f"Estimated House Price: ${prediction[0]:,.2f}"
    )
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)