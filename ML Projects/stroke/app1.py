# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:37:20 2023

@author: 91974
"""

from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model.plk','rb'))
@app.route("/")
def home():
    return render_template("stroke.html")
@app.route("/predict",methods=['POST'])
def predict():
    gender=(request.values['text1'])
    age=(request.values['text2'])
    hypertension=(request.values['text3'])
    heart_disease=(request.values['text4'])
    ever_married=(request.values['text5'])
    avg_glucos_level=(request.values['text6'])
    bmi=(request.values['text7'])
    smoking_status=(request.values['text8'])
    x=pd.DataFrame({"gender":[gender],
                    "age":[age],
                    "hypertension":[hypertension],
                    "heart_disease":[heart_disease],
                    "ever_married":[ever_married],
                    "avg_glucos_level":[avg_glucos_level],
                    "bmi":[bmi],
                    "smoking_status":[smoking_status]})
    print(x)
    y_pred=model.predict(x)
    print(y_pred)
    if y_pred==([0]):
        prediction="stroke"
         return render_template("result.html", prediction_text="The Patient is {}".format(prediction))
    else:
        prediction="safe"
        return render_template("result2.html", prediction_text="The Patient is {}".format(prediction))
if __name__=="__main__":
        app.run()