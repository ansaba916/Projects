# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:08:08 2023

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
    return render_template("home.html")
@app.route("/upload1")
def home1():
    return render_template("form.html")
@app.route("/predict",methods=['POST'])
def predict():
    Species=(request.values['text1'])
    Length1=(request.values['text2'])
    Length2=(request.values['text3'])
    Length3=(request.values['text4'])
    Height=(request.values['text5'])
    Width=(request.values['text6'])
    x=pd.DataFrame({"Species":[Species],
                    "Length1":[Length1],
                    "Length2":[Length2],
                    "Length3":[Length3],
                    "Height":[Height],
                    "Width":[Width]})

    print(x)
    y_pred=model.predict(x)
    print(y_pred)
    return render_template("result.html", prediction_text="The Weight prediction is {}".format(y_pred))
if __name__=="__main__":
        app.run()
