# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:02:43 2023

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
    return render_template("index.html")
@app.route("/upload2")
def home2():
    return render_template("jwellery.html")
@app.route("/upload3")
def home3():
    return render_template("fill.html")
@app.route("/predict",methods=['POST'])
def predict():
    carat=(request.values['text1'])
    color=(request.values['text2'])
    clarity=(request.values['text3'])
    depth=(request.values['text4'])
    table=(request.values['text5'])
    price=(request.values['text6'])
    x=(request.values['text7'])
    y=(request.values['text8'])
    z=(request.values['text9'])
    
    x=pd.DataFrame({"carat":[carat],
                    "color":[color],
                    "clarity":[clarity],
                    "depth":[depth],
                    "table":[table],
                    "price":[price],
                    "x":[x],
                    "y":[y],
                    "z":[z]})
    print(x)
    y_pred=model.predict(x)
    print(y_pred)
    
    if y_pred==([0]):
        prediction="Fair"
    elif y_pred==([1]):
        prediction="Good"
    elif y_pred==([2]):
        prediction="ideal"
    elif y_pred==([3]):
        prediction="Premium"
    else:
        prediction="Very Good"
        
        return render_template("result.html", prediction_text="The prediction is {}".format(prediction))
if __name__=="__main__":
    app.run()
    
    
    