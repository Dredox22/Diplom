import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    prediction = model.predict()
    prediction = int(prediction.label[0])
    return render_template('index.html', prediction_text = '{}'.format(prediction))

if __name__ == "__main__":
    # Run the app on local host and port 8989
    flask_app.run(host='localhost', port=8080, debug=True)