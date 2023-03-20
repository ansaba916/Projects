# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:06:22 2023

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
    return render_template("breast.html")
@app.route("/predict",methods=['POST'])
def predict():
    age=(request.values['text1'])
    tumor_size=(request.values['text2'])
    inv_nodes=(request.values['text3'])
    degmalig=(request.values['text4'])
    breast=(request.values['text5'])
    Class=(request.values['text6'])
    a=pd.DataFrame({"age":[age],
                    "tumor_size":[tumor_size],
                    "inv-nodes":[inv_nodes],
                    "deg-malig":[degmalig],
                    "breast":[breast],
                    "Class":[Class]})
    print(a)
    y_pred=model.predict(a)
    print(y_pred)
    if y_pred==([0]):
        prediction="irradiate"
    else:
        prediction="radiate"
        return render_template("breast.html", prediction_text="The Patient is {}".format(prediction))
if __name__=="__main__":
        app.run()
    