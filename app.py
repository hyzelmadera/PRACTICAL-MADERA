import pickle

import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import sklearn

app = Flask(__name__)
model = pickle.load(open('model/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    feature = [float(x) for x in request.form.values()]
    features = [np.array(feature)]
    prediction = model.predict(features)
    print(prediction)

    if round(prediction[0], 2) == 1:
        result = 'You might have diabetes.'
    else:
        result = 'You migh be safe of diabetes'

    return render_template('index.html', prediction_output = result)

if __name__ == "__main__":
    app.run()
