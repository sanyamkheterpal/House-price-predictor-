
from flask import Flask, request, jsonify, send_file
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Generate sample training data
np.random.seed(42)
X = []
y = []

for i in range(1500):
    area = np.random.randint(400, 5001)
    bedrooms = np.random.randint(1, 7)
    bathrooms = np.random.randint(1, 5)
    age = np.random.randint(0, 51)

    # Sample price in lakhs
    price = (
        area * 0.035
        + bedrooms * 2
        + bathrooms * 1.5
        - age * 0.250
        + np.random.normal(0, 3)
    )

    X.append([area, bedrooms, bathrooms, age])
    y.append(max(price, 5))

# Train the machine learning model
model = LinearRegression()
model.fit(X, y)


@app.route("/")
def home():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        area = float(data["area"])
        bedrooms = float(data["bedrooms"])
        bathrooms = float(data["bathrooms"])
        age = float(data["age"])

        values = [area, bedrooms, bathrooms, age]

        if not all(np.isfinite(v) for v in values):
            return jsonify({"error": "Invalid input values"}), 400

        if not (100 <= area <= 10000):
            return jsonify({"error": "Area must be 100–10000 sq. ft."}), 400

        if not (0 <= bedrooms <= 20 and bedrooms.is_integer()):
            return jsonify({"error": "Invalid bedroom count"}), 400

        if not (0 <= bathrooms <= 20 and bathrooms.is_integer()):
            return jsonify({"error": "Invalid bathroom count"}), 400

        if not (0 <= age <= 200 and age.is_integer()):
            return jsonify({"error": "Invalid property age"}), 400

        prediction = model.predict([values])[0]

        return jsonify({
            "price_lakhs": round(max(float(prediction), 1), 2)
        })

    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Please enter valid house details"}), 400

    except Exception:
        return jsonify({"error": "Unable to predict price"}), 500


if __name__ == "__main__":
    print("House Price Predictor is running!")
    app.run(debug=True)