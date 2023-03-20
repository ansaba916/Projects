# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 12:14:47 2023

@author: 91974
"""
from flask import Flask,request,render_template
import pickle
import numpy as np
import pandas as pd

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
@app.route("/")
def home():
    return render_template("body.html")
@app.route("/predict",methods=['POST'])
def predict():
    Density=(request.values['text1'])
    Age=(request.values['text2'])
    Weight=(request.values['text3'])
    Height=(request.values['text4'])
    Neck=(request.values['text5'])
    Chest=(request.values['text6'])
    Abdomen=(request.values['text7'])
    Hip=(request.values['text8'])
    Thigh=(request.values['text9'])
    Knee=(request.values['text10'])
    Ankle=(request.values['text11'])
    Biceps=(request.values['text12'])
    Forearm=(request.values['text12'])
    Wrist=(request.values['text14'])
    x=pd.DataFrame({"Density":[Density],
                    "Age":[Age],
                    "Weight":[Weight],
                    "Height":[Height],
                    "Neck":[Neck],
                    "Chest":[Chest],
                    "Abdomen":[Abdomen],
                    "Hip":[Hip],
                    "Thigh":[Thigh],
                    "Knee":[Knee],
                    "Ankle":[Ankle],
                    "Biceps":[Biceps],
                    "Forearm":[Forearm],
                    "Wrist":[Wrist]})

    print(x)
    y_pred=model.predict(x)
    print(y_pred)
    return render_template("result.html", prediction_text="The BodyFat prediction is {}".format(y_pred))
if __name__=="__main__":
        app.run()
