from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model (make sure 'diabetes_model.pkl' is in the same directory)
with open("diabetes_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)
    output = "Positive" if prediction[0] == 1 else "Negative"
    return render_template("index.html", prediction_text=f"Diabetes Prediction: {output}")

if __name__ == "__main__":
    app.run(debug=True)
