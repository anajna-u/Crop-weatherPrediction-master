from flask import Flask, url_for
from flask import request, render_template
# from flask_cors import cross_origin
import pickle
import pandas as pd
import sklearn
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from model import Model, use as model_use
import pickle


m = model_use("model.pkl")

app = Flask(__name__)


@app.route("/")
# @cross_origin()
def home():
    return render_template('home.html', value="shreyas")


@app.route("/generate")
def generate():
    m.generate()
    m.save("trained-model.pkl")

    return "DONE."


@app.route("/predict", methods=["POST"])
# @cross_origin()
def predict():
    # TODO: Normalise the parameters

    tmp = float(request.form["temp"])
    fert_nit = float(request.form["nitro"])
    fert_pot = float(request.form["pot"])
    fert_phos = float(request.form["phos"])
    rain = float(request.form["rain"])
    humid = float(request.form["hum"])
    ph = float(request.form["ph"])




    data = [[tmp, fert_nit, fert_pot, fert_phos, rain, humid, ph]]
    scaler=MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    normalized_data = scaler.fit_transform(data)
    normalized_tmp = normalized_data[0][0]
    normalized_fert_nit = normalized_data[0][1]
    normalized_fert_pot = normalized_data[0][2]
    normalized_fert_phos = normalized_data[0][3]
    normalized_rain = normalized_data[0][4]
    normalized_humid = normalized_data[0][5]
    normalized_ph = normalized_data[0][6]


    # prediction = m.predict(
    #     [tmp, fert_nit, fert_pot, fert_phos, rain, humid, ph])
    # statename, cropname = m.labels[np.argmax(prediction)].split(" / ")


    prediction = m.predict(
        [normalized_tmp, normalized_fert_nit, normalized_fert_pot, normalized_fert_phos, normalized_rain, normalized_humid, normalized_ph])
    statename, cropname = m.labels[np.argmax(prediction)].split(" / ")

    return render_template('home.html', statename=statename, cropname=cropname)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
