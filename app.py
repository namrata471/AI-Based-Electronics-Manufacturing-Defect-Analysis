from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained model
model = joblib.load("defect_prediction_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        voltage = float(request.form["voltage"])
        current = float(request.form["current"])
        ai_risk = float(request.form["ai_risk"])

        input_data = pd.DataFrame({
            "Temperature": [temperature],
            "Humidity": [humidity],
            "Voltage": [voltage],
            "Current": [current],
            "AI_Risk_Score": [ai_risk]
        })

        prediction = model.predict(input_data)[0]

        severity_labels = {
            0: "High",
            1: "Low",
            2: "Medium"
        }

        prediction = severity_labels.get(prediction, "Unknown")

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)